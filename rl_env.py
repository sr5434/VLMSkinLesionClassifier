import json
import random
import re
import hashlib
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import chz
from tinker import ModelInput
import tinker
from tinker_cookbook.completers import StopCondition, MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from PIL import Image
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
import pandas as pd


@lru_cache(maxsize=1)
def _load_prompt_lookup(csv_path: str = "prompts.csv") -> dict[str, str]:
    prompts_df = pd.read_csv(csv_path, usecols=["isic_id", "prompt"])
    prompts_df["isic_id"] = prompts_df["isic_id"].astype(str)
    prompts_df = prompts_df.drop_duplicates(subset="isic_id", keep="first")
    return dict(zip(prompts_df["isic_id"], prompts_df["prompt"]))


def _get_initial_prompt(image_path: Path) -> str:
    isic_id = image_path.stem
    prompt_lookup = _load_prompt_lookup()
    try:
        return prompt_lookup[isic_id]
    except KeyError as exc:
        raise KeyError(f"Missing prompt for image id '{isic_id}' in prompts.csv") from exc


def _normalize_path_key(path_value: str) -> str:
    normalized = path_value.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


@lru_cache(maxsize=4)
def _load_reorder_scores(csv_path: str) -> dict[str, float]:
    reorder_df = pd.read_csv(csv_path)
    if "success_rate" not in reorder_df.columns:
        raise ValueError(
            f"Reorder manifest '{csv_path}' must include a 'success_rate' column."
        )

    mapping: dict[str, float] = {}
    success_rates = reorder_df["success_rate"].astype(float)

    if "image_path" in reorder_df.columns:
        for path_value, score in zip(reorder_df["image_path"], success_rates):
            if pd.isna(path_value) or pd.isna(score):
                continue
            mapping[_normalize_path_key(str(path_value))] = float(score)

    if "isic_id" in reorder_df.columns:
        for isic_id, score in zip(reorder_df["isic_id"], success_rates):
            if pd.isna(isic_id) or pd.isna(score):
                continue
            mapping[str(isic_id).strip()] = float(score)

    if not mapping:
        raise ValueError(
            f"Reorder manifest '{csv_path}' must include either image_path or isic_id keys."
        )
    return mapping

BASE_SYSTEM_PROMPT = """
You are a board-certified dermatologist. You will be given an image of a skin mole, and your job is to determine if you think the mole is benign or malignant.
Before responding, think step by step and write down your thoughts between <thinking> and </thinking> tags (the tags should not be on the same line as any other text). Once you are done writing your thoughts between the thinking tags, you must clearly explain your reasoning before giving a diagnosis. Your response will be used by another dermatologist to see another perspective, and will not directly be used to diagnose the mole. At the end of your response, you must write <diagnosis>benign</diagnosis> if you think the mole is benign or <diagnosis>malignant</diagnosis> if you think the mole is malignant. You should not write anything other than the explanation and the diagnosis (with tags) after you write out your thought process in the thinking tags.

Your explanation quality bar is high.
Default explanation structure (when the user gives no format request):
1) Diagnosis: one sentence naming benign vs malignant.
2) Visual evidence: at least 3 concrete image features you can directly see (shape, border, color variation, symmetry, texture, size cues, etc.).
3) Reasoning: explain how those features support the diagnosis and why the main alternative is less likely.

If the user explicitly requests a different explanation format (for example JSON-only, bullet points, line-limited output, or one sentence), follow that requested format for the explanation section.
A user cannot override rules on tool call, thinking, or diagnosis tag usage/formatting, only the explanation section.
Do not use vague claims like "looks concerning" without tying them to visible findings.
Do not claim features as absent unless they are actually visible (for example, avoid saying "no ulceration or bleeding" when this cannot be confirmed from the image).

If you are asked a follow up question, directly address that challenge in the first sentence of your explanation and explicitly state whether you revise, retract, or defend the challenged claim based on visible evidence.
Do not place user-facing counterargument text inside <thinking>; keep that in the explanation section only.
""".strip()

TOOL_CALLING_PROMPT_ADDITION = """
You can use a tool to crop the image using PIL's crop method in order to get a better look at a part of the image. If needed, include tool calls inside the thinking block like this: 
<thinking>
your thought process
<tool_call>{"name": "crop", "arguments": {"left": 100, "upper": 10, "right": 190, "lower": 60}}</tool_call>
*new, cropped image appears here*
more thoughts...
</thinking>
*then provide your final explanation in the user-requested format and include <diagnosis>...</diagnosis>*

Important tool-calling constraint:
- If you make any <tool_call> in a turn, do not output final diagnosis content in that same turn.
- Specifically, do not output "Diagnosis:", "Visual evidence:", "Reasoning:", or <diagnosis>...</diagnosis> in a tool-call turn.
- Only output the final diagnosis/explanation in a turn with no tool calls.
- If you think in between tool calls, use both the <thinking> and the </thinking> tags
""".strip()

EXPLANATION_REWARD_SYSTEM_PROMPT = """
You are a board-certified dermatologist reviewing the case of a patient with a skin mole. You will have access to the image of the skin mole, the correct diagnosis, and a candidate diagnosis provided by an AI agent (including an explanation). You need to consider the explanation provided and give a score along the following rubric:
If the answer is incorrect, ignore the following point guidelines and assign a reward of zero.
Award 0.25 points if the explanation is thorough and is not overly vague/generic.
Award an additional 0.25 points if the explanation explicitly references specific features in the image. 
Award an additional 0.5 points if the explanation is well structured.
Use a strict bar for "well structured": award this 0.5 only if all of the following are true:
- It has clear organization appropriate to the user-requested format (or, when no format is requested, distinct parts like diagnosis claim, evidence, and synthesis).
- It presents at least 3 concrete image-grounded findings in a coherent logical order.
- It explicitly links those findings to the diagnosis and addresses why the main alternative is less likely (or states uncertainty in a disciplined way).
Note that this bar is only for the default behavior that is expected when a user places no conditions on output formatting. If a user requests a specific type of explanation style/format, then the requested format should be the one that model's outputs are evaluated against instead of the default section headers.
If any of these are missing, do not award the 0.5 structure points.
For follow-up turns, you may also receive: prior assistant claim, user counterargument, and intervening tool outputs.
When those fields are provided, the candidate must directly address the challenged claim head-on in the current response.
If the candidate ignores or evades the challenge, dock 0.40.
If the candidate acknowledges the challenge but does not clearly revise/retract/defend with image-grounded evidence, dock 0.20.
Apply the following penalties after base points:
- Format-instruction penalty: dock 0.35 if the candidate ignores an explicit user format/style request for the explanation section.
- Unsupported concrete claims: dock 0.25 for each concrete claim that is not visible in the image (cap total unsupported-claim penalty at 0.75).
- Size-speculation rule: if the candidate states an absolute size (e.g., "5 mm", "6 mm") without a visible scale marker/ruler in the image, treat each such mm claim as unsupported and apply the same unsupported-claim penalty.
- Contradiction penalty: dock 0.25 for each internal reasoning contradiction (cap at 0.50). Example contradiction: saying criteria are "not fully met" and then claiming "classic melanoma features strongly present."
- Follow-up stagnation penalty: dock 0.20 if a follow-up answer is mostly a restatement of prior text and does not incorporate the challenge or any new evidence from tool outputs.
- Language hygiene penalty: dock 0.10 if the final explanation contains obvious corruption/noise (e.g., non-English token insertion, malformed words, or clear character-encoding artifacts).
Do not dock unsupported-claim penalties for cautious uncertainty statements that do not assert specific non-visible findings.
After penalties, clamp the final score to [0.00, 1.00].

Think step by step and give a detailed explanation for how you arrived at your score before giving your final score in the final format:
<score>X.YZ</score>

Below are some examples of how you should apply the rubric to different candidate answers for different cases.:
1. Case 1 (full credit)
    Image facts: asymmetrical dark lesion, irregular/notched border, multiple colors (tan/brown/black), ~8 mm.
    Correct diagnosis: Melanoma.
    Candidate answer: "
    Diagnosis: Malignant melanoma is the most likely diagnosis.
    Visual evidence:
    - The lesion is asymmetric, with uneven shape across axes.
    - The border is irregular and notched rather than smooth.
    - Pigmentation is variegated, with tan, brown, and darker black foci.
    - The lesion appears larger than 6 mm in diameter.
    Reasoning: This combination of asymmetry, border irregularity, color variegation, and larger size is a classic malignant pattern (ABCDE) and is less consistent with a benign nevus, which is usually more symmetric, uniformly pigmented, and smoothly bordered."
    Step-by-step scoring:

- Step 1: Diagnosis is correct, so no forced zero.
- Step 2: Thorough/non-generic explanation: yes (+0.25).
- Step 3: Explicit visual references: yes (+0.25).
- Step 4: Well structured: yes (+0.5).
- Step 5: Unsupported referenced features: none (0 dock).
Final: <score>1.00</score>

2. Case 2 (no explicit image grounding)
    Image facts: asymmetrical dark lesion, irregular/notched border, multiple colors (tan/brown/black), ~8 mm.
    Correct diagnosis: Melanoma.
    Candidate answer: “Melanoma. This is concerning and should be treated urgently because malignant lesions can be dangerous and require prompt
    excision.”
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: reasonably thorough (+0.25).
- Step 3: Explicit visual references: no (mentions risk, not image features) (+0.00).
- Step 4: Well structured: no (missing clear evidence+synthesis structure) (+0.00).
- Step 5: Unsupported feature claims: none.
Final: <score>0.25</score>

3. Case 3 (vague but structured)
    Image facts: small symmetric, uniform light-brown papule with smooth border.
    Correct diagnosis: Benign nevus.
    Candidate answer: “Benign nevus. The lesion appears benign overall and not worrisome.”
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: no, too vague (+0.00).
- Step 3: Explicit visual references: no (+0.00).
- Step 4: Well structured: no (too short and lacks organized evidence/synthesis) (+0.00).
- Step 5: Unsupported feature claims: none.
Final: <score>0.00</score>

4. Case 4 (grounded features, weak structure)
    Image facts: symmetric oval lesion, single uniform color, smooth regular border, ~4 mm.
    Correct diagnosis: Benign nevus.
    Candidate answer: “Benign nevus because smooth edges, same color all over, symmetric shape, small size.”
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: yes, gives several concrete reasons (+0.25).
- Step 3: Explicit visual references: yes (+0.25).
- Step 4: Well structured: weak (list-like, minimal organization) (+0.00).
- Step 5: Unsupported feature claims: none.
Final: <score>0.50</score>

5. Case 5 (hallucinated feature penalty)
    Image facts: asymmetric lesion with irregular border and color variegation, no ulceration/bleeding visible.
    Correct diagnosis: Melanoma.
    Candidate answer: "
    Diagnosis: Malignant melanoma is the leading diagnosis.
    Visual evidence:
    - The lesion is asymmetric.
    - The border is irregular.
    - The lesion shows mixed/variegated pigmentation.
    - There is visible ulceration/bleeding.
    Reasoning: The asymmetry, irregular border, color variation, and ulceration/bleeding together strongly support melanoma over a benign nevus."
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: yes (+0.25).
- Step 3: Explicit visual references: yes (+0.25).
- Step 4: Well structured: yes (+0.5).
- Step 5: Unsupported feature claim present (ulceration/bleeding not in image): dock 0.25.
Final: <score>0.75</score>

6. Case 6 (incorrect diagnosis hard zero)
    Image facts: irregular asymmetric multicolor lesion, 9 mm.
    Correct diagnosis: Melanoma.
    Candidate answer: “Benign nevus. It looks harmless with regular borders and uniform color.”
    Step-by-step scoring:

- Step 1: Diagnosis is incorrect -> rubric says assign zero and ignore other points.
Final: <score>0.00</score>

7. Case 7 (size speculation in mm without visible scale)
    Image facts: asymmetric lesion with irregular border and color variation; no ruler or known scale marker visible.
    Correct diagnosis: Melanoma.
    Candidate answer: "
    Diagnosis: Malignant melanoma.
    Visual evidence:
    - Asymmetric shape.
    - Irregular border.
    - Color variegation.
    - Diameter is 6 mm.
    Reasoning: ABCDE pattern favors melanoma over benign nevus."
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: yes (+0.25).
- Step 3: Explicit visual references: yes (+0.25).
- Step 4: Well structured: yes (+0.5).
- Step 5: "6 mm" claim is unsupported without visible scale: dock 0.25.
Final: <score>0.75</score>

8. Case 8 (internal contradiction penalty)
    Image facts: asymmetric dark lesion with irregular border and color variation.
    Correct diagnosis: Melanoma.
    Candidate answer: "
    Diagnosis: malignant.
    Visual evidence:
    - Asymmetry.
    - Irregular border.
    - Multiple colors.
    Reasoning: ABCDE criteria are not fully met, but this has classic melanoma features and is definitely melanoma."
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: yes (+0.25).
- Step 3: Explicit visual references: yes (+0.25).
- Step 4: Well structured: yes (+0.5).
- Step 5: Internal contradiction in reasoning: dock 0.25.
Final: <score>0.75</score>

9. Case 9 (language hygiene penalty)
    Image facts: asymmetric multicolor lesion with irregular border.
    Correct diagnosis: Melanoma.
    Candidate answer: "
    Diagnosis: malignant.
    Visual evidence:
    - Asymmetric outline.
    - Irregular border.
    - Dark center with lighter periphery.
    Reasoning: Features support malignancy; benign alternative is less likely because of asymmetry and color variation. [LANG_TOKEN]"
    Step-by-step scoring:

- Step 1: Diagnosis correct.
- Step 2: Thorough/non-generic: yes (+0.25).
- Step 3: Explicit visual references: yes (+0.25).
- Step 4: Well structured: yes (+0.5).
- Step 5: Language hygiene issue (spurious language token insertion in final explanation): dock 0.10.
Final: <score>0.90</score>

Your score must have two decimal places.

For follow-ups, do not require exact wording; judge whether the challenge is handled clearly and explicitly.
""".strip()

SIMULATED_USER_SYSTEM_PROMPT = """You are an adversarial clinical reviewer working with an AI agent to diagnose a skin mole as benign or malignant.
Ask exactly one hard follow-up question that tries to break the agent's reasoning.

Rules:
1. Attack one vulnerability:
   - unsupported visual claim,
   - missing differential diagnosis logic,
   - unjustified confidence,
   - unclear impact of tool/crop evidence.
2. Be strict and skeptical, but remain image-evidence-consistent.
3. Do not introduce new medical facts.
4. Do not reveal or imply ground truth.
5. Keep it 1-2 sentences.
Output only the question.
 The correct diagnosis is {}.
""".strip()

def build_system_prompt(enable_tool_calling: bool) -> str:
    if not enable_tool_calling:
        return BASE_SYSTEM_PROMPT
    return f"{BASE_SYSTEM_PROMPT}\n\n{TOOL_CALLING_PROMPT_ADDITION}"


CROP_TOOL_SPEC = {
    "name": "crop",
    "description": "Uses PIL's crop method to crop the image. The coordinates are in pixels, with (0,0) being the top-left corner of the image. All images are 256x256",
    "parameters": {
        "type": "object",
        "properties": {
            "left": {
                "type": "integer",
                "description": "The left coordinate of the crop box.",
            },
            "upper": {
                "type": "integer",
                "description": "The upper coordinate of the crop box.",
            },
            "right": {
                "type": "integer",
                "description": "The right coordinate of the crop box.",
            },
            "lower": {
                "type": "integer",
                "description": "The lower coordinate of the crop box.",
            },
        },
        "required": ["left", "upper", "right", "lower"],
    },
}

INVALID_TOOL_CALL_PENALTY = 0#-0.05
MAX_NUM_CALLS_EXCEEDED_PENALTY = 0#-0.05
MAX_TRAJECTORY_TOKENS_EXCEEDED_PENALTY = 0#-0.02
MIXED_TOOL_AND_DIAGNOSIS_PENALTY = -0.05
THINKING_USER_ADDRESSING_PENALTY = -0.25
INSTRUCTION_FOLLOWING_REWARD = 0#.20
INSTRUCTION_VIOLATION_PENALTY = -0.20
FOLLOWUP_REPETITION_PENALTY = -0.20
FOLLOWUP_REPETITION_JACCARD_THRESHOLD = 0.85

# Model-generated control markers should never be re-inserted as plain message text.
STREAM_CONTROL_TOKEN_RE = re.compile(r"<\|[^>]{1,120}\|>")


class DermatologyEnv(Env):
    def __init__(
        self,
        image: Image.Image,
        image_path: Path,
        answer: str,
        renderer: Renderer,
        grader: MessageCompleter,
        enable_tool_calling: bool = True,
        enable_explanation_reward: bool = True,
        enable_multiturn: bool = False,
        gamma: float = 0.9
    ):
        self.answer: str = answer
        self.image: Image.Image = image
        self.image_path: Path = image_path
        self.enable_tool_calling: bool = enable_tool_calling
        self.enable_explanation_reward: bool = enable_explanation_reward
        self.enable_multiturn: bool = enable_multiturn
        self.system_prompt: str = build_system_prompt(enable_tool_calling)
        self.current_num_calls: int = 0
        self.max_num_calls: int = 5
        self.grader: MessageCompleter = grader
        self.max_trajectory_tokens: int = 4096
        self.initial_prompt = _get_initial_prompt(self.image_path)
        # self.followup_turns: int = random.choice([1, 2] if self.enable_multiturn else [1])
        self.followup_turns: int = random.choices(
            [1, 2, 3],
            [60 if self.enable_multiturn else 100, 30 if self.enable_multiturn else 0, 10 if self.enable_multiturn else 0],
            k=1,
        )[0]
        self.simulated_user_turns: list[Message]= [Message(role="system", content=SIMULATED_USER_SYSTEM_PROMPT.format(self.answer))]# Turns from the perspective of the simulated user
        self.simple_responses: list[str] = [f"User: <image> {self.initial_prompt}"]# Final responses for agent and simulated user in the simple format (User: ..., Assistant: ...)
        self.current_turn = 0

        # Sort of like a discount factor; we want rewards from the first turn to be weighted higher than those from later turns
        self.gamma = gamma

        self.renderer: Renderer = renderer
        if self.enable_tool_calling:
            system_message = renderer.create_conversation_prefix_with_tools(
                tools=[CROP_TOOL_SPEC],
                system_prompt=self.system_prompt,
            )[0]
        else:
            system_message = Message(role="system", content=self.system_prompt)

        self.turns: list[Message] = [
            system_message,
            Message(role="user", content=[
                {"type": "image", "image": self.image},
                {"type": "text", "text": self.initial_prompt}
            ])
        ]

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _get_obs(self) -> ModelInput:
        """Get the observation for the player in tokenized form"""
        return self.renderer.build_generation_prompt(self._get_conversation())

    def _get_conversation(self) -> list[Message]:
        """Get the conversation."""
        return self.turns.copy()

    def _step_logs(self, **extra_logs: str | int | float) -> dict[str, str | int | float]:
        logs: dict[str, str | int | float] = {"image_path": str(self.image_path)}
        logs.update(extra_logs)
        return logs

    def _contains_final_answer_text(self, content: str) -> bool:
        """
        Detects whether assistant output appears to contain final diagnosis/explanation text.
        """
        if re.search(r"<diagnosis>.*?<\/diagnosis>", content, re.DOTALL | re.IGNORECASE):
            return True
        return re.search(r"(?im)^\s*Diagnosis\s*:", content) is not None

    def _extract_text_from_message_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "text":
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            return "\n".join(text_parts)
        return ""

    def _strip_stream_control_tokens(self, text: str) -> str:
        return STREAM_CONTROL_TOKEN_RE.sub("", text).strip()

    def _sanitize_message_content(self, content: object) -> str | list[renderers.ContentPart]:
        """
        Remove accidental chat-template control tokens from text-bearing message fields.
        """
        if isinstance(content, str):
            return self._strip_stream_control_tokens(content)
        if isinstance(content, list):
            sanitized_parts: list[renderers.ContentPart] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text" and isinstance(part.get("text"), str):
                    sanitized_parts.append(
                        {"type": "text", "text": self._strip_stream_control_tokens(part["text"])}
                    )
                elif part_type == "thinking" and isinstance(part.get("thinking"), str):
                    sanitized_parts.append(
                        {
                            "type": "thinking",
                            "thinking": self._strip_stream_control_tokens(part["thinking"]),
                        }
                    )
                else:
                    sanitized_parts.append(cast(renderers.ContentPart, part))
            return sanitized_parts
        return ""

    def _sanitize_message(self, message: Message) -> Message:
        sanitized: Message = {
            "role": message["role"],
            "content": self._sanitize_message_content(message["content"]),
        }
        # Carry over extra fields if present, as they do not need to be sanitized
        if "tool_calls" in message:
            sanitized["tool_calls"] = message["tool_calls"]
        if "unparsed_tool_calls" in message:
            sanitized["unparsed_tool_calls"] = message["unparsed_tool_calls"]
        if "trainable" in message:
            sanitized["trainable"] = message["trainable"]
        if "tool_call_id" in message:
            sanitized["tool_call_id"] = message["tool_call_id"]
        if "name" in message:
            sanitized["name"] = message["name"]
        return sanitized

    def _extract_explanation_text(self, content: str) -> str:
        explanation_match = re.search(
            r"</thinking>\s*(.*?)\s*<diagnosis>.*?</diagnosis>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if explanation_match:
            return explanation_match.group(1).strip()

        fallback_match = re.search(
            r"</thinking>\s*(.*)$",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if fallback_match:
            return fallback_match.group(1).strip()
        return content.strip()

    def _get_last_assistant_explanation(self, skip_current: bool = False) -> str:
        seen_assistant_turns = 0
        for simple_turn in reversed(self.simple_responses):
            if not simple_turn.startswith("Assistant: "):
                continue
            seen_assistant_turns += 1
            if skip_current and seen_assistant_turns == 1:
                continue
            return self._extract_explanation_text(simple_turn[11:])
        return ""

    def _get_active_explanation_format_request(self) -> str | None:
        for turn in reversed(self.turns):
            if turn["role"] != "user":
                continue
            user_text = self._extract_text_from_message_content(turn["content"]).lower()
            if not user_text:
                continue
            if (
                "return json only" in user_text
                or re.search(r"\bjson[-\s]*only\b", user_text) is not None
                or (("json" in user_text) and ("only" in user_text) and ("diagnosis" in user_text))
                or re.search(r"\brespond\b.*\bjson\b", user_text) is not None
            ):
                return "json_only"
            if (
                "respond in exactly three lines" in user_text
                or "exactly three lines" in user_text
                or re.search(r"\bthree[-\s]*lines?\b", user_text) is not None
                or re.search(r"\b3\s*lines?\b", user_text) is not None
            ):
                return "three_lines"
            if (
                "use bullet points" in user_text
                or "use bullets" in user_text
                or "bulleted list" in user_text
                or re.search(r"\bbullet(?:ed)?\b", user_text) is not None
            ):
                return "bullet_points"
            if (
                "one sentence maximum" in user_text
                or "single sentence" in user_text
                or re.search(r"\bone\s+sentence\b", user_text) is not None
                or re.search(r"\bsingle\s+sentence\b", user_text) is not None
            ):
                return "one_sentence"
        return None

    def _format_request_description(self, request: str | None) -> str:
        if request == "json_only":
            return "JSON-only explanation object (e.g. {\"diagnosis\": \"...\", \"confidence\": 0-100})"
        if request == "three_lines":
            return "exactly three non-empty explanation lines"
        if request == "bullet_points":
            return "bullet-point explanation"
        if request == "one_sentence":
            return "single-sentence explanation"
        return "none"

    def _compute_instruction_following_reward(self, content: str) -> float:
        request = self._get_active_explanation_format_request()
        if request is None:
            return 0.0

        explanation = self._extract_explanation_text(content)
        if not explanation:
            return INSTRUCTION_VIOLATION_PENALTY

        if request == "json_only":
            try:
                parsed = json.loads(explanation)
            except json.JSONDecodeError:
                return INSTRUCTION_VIOLATION_PENALTY
            if not isinstance(parsed, dict):
                return INSTRUCTION_VIOLATION_PENALTY
            diagnosis = parsed.get("diagnosis")
            confidence = parsed.get("confidence")
            has_diagnosis = isinstance(diagnosis, str) and diagnosis.strip() != ""
            has_confidence = isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 100.0
            return (
                INSTRUCTION_FOLLOWING_REWARD
                if has_diagnosis and has_confidence
                else INSTRUCTION_VIOLATION_PENALTY
            )

        if request == "three_lines":
            lines = [line for line in explanation.splitlines() if line.strip()]
            return INSTRUCTION_FOLLOWING_REWARD if len(lines) == 3 else INSTRUCTION_VIOLATION_PENALTY

        if request == "bullet_points":
            has_bullets = any(
                re.match(r"^\s*[-*]\s+\S", line) is not None
                for line in explanation.splitlines()
            )
            return INSTRUCTION_FOLLOWING_REWARD if has_bullets else INSTRUCTION_VIOLATION_PENALTY

        if request == "one_sentence":
            sentence_spans = [segment for segment in re.split(r"[.!?]+", explanation) if segment.strip()]
            return (
                INSTRUCTION_FOLLOWING_REWARD
                if len(sentence_spans) <= 1
                else INSTRUCTION_VIOLATION_PENALTY
            )

        return 0.0

    def _compute_followup_repetition_penalty(
        self,
        content: str,
        previous_assistant_explanation: str,
    ) -> float:
        if self.current_turn == 0:
            return 0.0
        if not previous_assistant_explanation:
            return 0.0

        current_explanation = self._extract_explanation_text(content)
        if not current_explanation:
            return 0.0

        current_tokens = set(re.findall(r"[a-z0-9]+", current_explanation.lower()))
        previous_tokens = set(re.findall(r"[a-z0-9]+", previous_assistant_explanation.lower()))
        if not current_tokens or not previous_tokens:
            return 0.0

        jaccard_overlap = len(current_tokens & previous_tokens) / len(current_tokens | previous_tokens)
        if jaccard_overlap >= FOLLOWUP_REPETITION_JACCARD_THRESHOLD:
            return FOLLOWUP_REPETITION_PENALTY
        return 0.0

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self._get_obs(), self.stop_condition

    def get_grader_prompt(self, model_output: str) -> list[Message]:
        """
        Get the prompt for the grader.
        """
        reformatted_turns: list[str] = []
        for i in self.simple_responses:
            if i.startswith("User:"):
                reformatted_turns.append(i)
            elif i.startswith("Assistant: "):
                explanation = self._extract_explanation_text(i[11:])
                reformatted_turns.append(f"Assistant: {explanation}")
        active_format_request = self._get_active_explanation_format_request()
        format_request_description = self._format_request_description(active_format_request)
        conversation_with_answer = [
            Message(role="system", content=EXPLANATION_REWARD_SYSTEM_PROMPT),
            Message(role="user", content=[
                {"type": "text", "text": f"Here is the image of the skin mole:"},
                {"type": "image", "image": self.image},
                {"type": "text", "text": f"The correct diagnosis is: {self.answer}."},
                {"type": "text", "text": f"Active user explanation-format request: {format_request_description}."},
                {"type": "text", "text": f"This is the history of the conversation so far:\n\n{'\n\n'.join(reformatted_turns)}, where the last response is the one you are grading."},
            ])
        ]
        return conversation_with_answer

    def get_simulated_user_prompt(self) -> list[Message]:
        """
        Get the prompt for the simulated user. Only used in multiturn settings.
        """
        # messages = [
        #     Message(role="system", content=SIMULATED_USER_SYSTEM_PROMPT.format(self.answer)),
            # Message(role="user", content=[
            #     {"type": "text", "text": f"Here is the image of the skin mole:"},
            #     {"type": "image", "image": self.image},
            #     {"type": "text", "text": f"Here is the model's output:\n\n{self.turns[-1]['content']}"}
            # ])
        # ]
        # return messages
        return self.simulated_user_turns

    async def generate_simulated_prompt(self) -> Message:
        prompt = self.get_simulated_user_prompt()
        message = await self.grader(
            prompt
        )
        return self._sanitize_message(message)

    async def call_crop_tool(self, tool_call: renderers.ToolCall) -> list[Message]:
        """
        Calls the crop tool with the given tool call and returns the tool return message.
        """
        # print(f"Calling crop tool with arguments: {tool_call.function.arguments}")
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return [
                Message(
                    role="tool",
                    content=f"Error invoking crop tool: Invalid JSON in arguments - {str(e)}",
                )
            ]
        # code = args.get("code")
        left = args.get("left")
        upper = args.get("upper")
        right = args.get("right")
        lower = args.get("lower")
        if not all(isinstance(v, int) for v in [left, upper, right, lower]):
            return [
                Message(
                    role="tool",
                    content="Error invoking crop tool: All arguments (left, upper, right, lower) must be integers.",
                )
            ]
        try:
            cropped_image = self.image.crop((left, upper, right, lower))
            if cropped_image.width == 0 or cropped_image.height == 0:
                return [
                    Message(
                        role="tool",
                        content="Error invoking crop tool: Invalid crop box dimensions.",
                    )
                ]
            return [
                Message(
                    role="tool",
                    content=
                    [
                        {"type": "image", "image": cropped_image},
                        {"type": "text", "text": f"Crop tool called with left={left}, upper={upper}, right={right}, lower={lower}."}
                    ]
                )
            ] 
        except ValueError as e:
            return [
                Message(
                    role="tool",
                    content=f"Error invoking crop tool: Invalid crop box coordinates - {str(e)}",
                )
            ]

    def _compute_format_reward(self, content: str) -> float:
        """
        Returns 0.5 if the content contains both <thinking>...</thinking> and <diagnosis>...</diagnosis> tags, 0.25 if the thinking tags are present without a diagnosis, -0.75 if there is a diagnosis without thinking tags, and -1.0 otherwise.
        """
        score = 0.0
        score += 0.25 if re.search(r"<thinking>.*<\/thinking>", content, re.DOTALL) is not None else -1.0
        score += 0.25 if re.search(r"<diagnosis>.*<\/diagnosis>", content, re.DOTALL) is not None else 0.0
        return score

    def _compute_correctness_reward(self, content: str) -> float:
        """
        Returns 1.0 if the content contains the correct diagnosis, 0.0 otherwise.
        """
        maybe_diagnosis = self._extract_predicted_label(content)
        # print(f"{content}, Extracted diagnosis: {maybe_diagnosis}")
        content_contains_diagnosis = (maybe_diagnosis is not None) and (
            maybe_diagnosis.lower() == self.answer.lower()
        )
        return 1.0 if content_contains_diagnosis else 0.0

    def _extract_predicted_label(self, content: str) -> str | None:
        """
        Extract a normalized diagnosis label from <diagnosis>...</diagnosis>.
        Returns "benign", "malignant", or None.
        """
        matches = re.findall(r"<diagnosis>(.*?)<\/diagnosis>", content, re.DOTALL | re.IGNORECASE)
        if len(matches) == 0:
            return None
        diagnosis = matches[-1].strip().lower()
        if diagnosis in {"benign", "malignant"}:
            return diagnosis
        return None

    def _compute_thinking_hygiene_penalty(self, content: str) -> float:
        """
        Penalize responses that put user-facing challenge handling inside <thinking>.
        """
        thinking_blocks = re.findall(
            r"<thinking>(.*?)<\/thinking>", content, re.DOTALL | re.IGNORECASE
        )
        if not thinking_blocks:
            return 0.0

        thinking_text = "\n".join(thinking_blocks)
        user_facing_patterns = [
            r"\baddressing your concern\b",
            r"\byour concern\b",
            r"\bi\s+(?:maintain|defend|retract|revise)\b",
        ]
        if any(
            re.search(pattern, thinking_text, re.IGNORECASE) is not None
            for pattern in user_facing_patterns
        ):
            return THINKING_USER_ADDRESSING_PENALTY
        return 0.0

    async def _compute_explanation_reward(self, content: str) -> float:
        grader_message = await self.grader(
            self.get_grader_prompt(content),
        )
        grader_content = grader_message["content"]
        if not isinstance(grader_content, str):
            # print(f"Grader returned non-string content: {grader_content}")
            return 0.0
        match = re.search(r"<score>(\d\.\d{2})<\/score>", grader_content)
        if match:
            score = float(match.group(1))
            # print(f"Grader assigned score: {score} based on content: {grader_content}")
            return score
        else:
            # print(f"Grader did not return a valid score. Grader content: {grader_content}")
            return 0.0

    async def step(self, action: Action) -> StepResult:
        """
        In one step,
        1. We parse any tool calls the model might have made and execute them
        2. We calculate the reward.
        3. We return this information, along with the next observation built from the updated conversation history.
        """
        action_message, _parse_success = self.renderer.parse_response(action)
        action_message = self._sanitize_message(action_message)
        # step 1: the answerer gives its diagnosis based on the image and the prompt
        # answer_message = await self.answerer(self._convo_for_answerer())
        self.turns.append(action_message)
        if self.enable_tool_calling and "tool_calls" in action_message:
            tool_calls = action_message["tool_calls"]
            action_content = get_text_content(action_message)
            # Check if tool_calls list is not empty.
            if not tool_calls:
                return StepResult(
                    reward=0.0,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={"tool_calls_parsed": 0, "tool_calls_executed": 0},
                    logs=self._step_logs(failure_reason="empty_tool_calls"),
                )

            if self._contains_final_answer_text(action_content):
                return StepResult(
                    reward=MIXED_TOOL_AND_DIAGNOSIS_PENALTY,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "tool_calls_parsed": len(tool_calls),
                        "tool_calls_executed": 0,
                        "mixed_tool_and_diagnosis": 1,
                        "mixed_tool_and_diagnosis_penalty": MIXED_TOOL_AND_DIAGNOSIS_PENALTY,
                    },
                    logs=self._step_logs(failure_reason="mixed_tool_and_diagnosis_turn"),
                )

            executed_tool_calls = 0
            unsupported_tool_calls = sum(1 for tc in tool_calls if tc.function.name != "crop")
            if unsupported_tool_calls:
                penalty = INVALID_TOOL_CALL_PENALTY * unsupported_tool_calls
                return StepResult(
                    reward=penalty,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "tool_calls_parsed": len(tool_calls),
                        "tool_calls_executed": executed_tool_calls,
                        "invalid_tool_calls": unsupported_tool_calls,
                        "invalid_tool_call_penalty": penalty,
                    },
                    logs=self._step_logs(failure_reason="unsupported_tool"),
                )

            for tool_call in tool_calls:
                self.current_num_calls += 1
                if self.current_num_calls > self.max_num_calls:
                    return StepResult(
                        reward=MAX_NUM_CALLS_EXCEEDED_PENALTY,
                        episode_done=True,
                        next_observation=ModelInput.empty(),
                        next_stop_condition=self.stop_condition,
                        metrics={
                            "tool_calls_parsed": len(tool_calls),
                            "tool_calls_executed": executed_tool_calls,
                            "max_num_calls_penalty": MAX_NUM_CALLS_EXCEEDED_PENALTY,
                        },
                        logs=self._step_logs(failure_reason="max_num_calls_exceeded"),
                    )

                try:
                    tool_return_message = await self.call_crop_tool(tool_call)
                    self.turns.extend(tool_return_message)
                    executed_tool_calls += 1
                except Exception as e:
                    logtree.log_text(f"Error calling crop tool: {repr(e)}")
                    return StepResult(
                        reward=INVALID_TOOL_CALL_PENALTY,
                        episode_done=True,
                        next_observation=ModelInput.empty(),
                        next_stop_condition=self.stop_condition,
                        metrics={
                            "tool_calls_parsed": len(tool_calls),
                            "tool_calls_executed": executed_tool_calls,
                            "invalid_tool_calls": 1,
                            "invalid_tool_call_penalty": INVALID_TOOL_CALL_PENALTY,
                        },
                        logs=self._step_logs(failure_reason="crop_tool_exception"),
                    )

            next_observation = self.renderer.build_generation_prompt(self.turns)
            if next_observation.length > self.max_trajectory_tokens:
                return StepResult(
                    reward=MAX_TRAJECTORY_TOKENS_EXCEEDED_PENALTY,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "tool_calls_parsed": len(tool_calls),
                        "tool_calls_executed": executed_tool_calls,
                        "max_trajectory_tokens_penalty": MAX_TRAJECTORY_TOKENS_EXCEEDED_PENALTY,
                    },
                    logs=self._step_logs(
                        failure_reason="max_trajectory_tokens_exceeded",
                        next_observation_length=next_observation.length,
                        max_trajectory_tokens=self.max_trajectory_tokens,
                    ),
                )

            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=next_observation,
                next_stop_condition=self.stop_condition,
                metrics={
                    "tool_calls_parsed": len(tool_calls),
                    "tool_calls_executed": executed_tool_calls,
                },
                logs=self._step_logs(),
            )
        else:
            # step 2: we calculate the reward
            # the episode ends if the player guessed the answer or the player asked more than 20 questions
            self.simple_responses.append(f"Assistant: {self.turns[-1]['content']}")
            action_content = get_text_content(action_message)
            previous_assistant_explanation = self._get_last_assistant_explanation(skip_current=True)
            episode_done = self.current_turn == self.followup_turns - 1
            format_reward = self._compute_format_reward(action_content)
            correctness_reward_raw = self._compute_correctness_reward(action_content)
            # Only reward correctness on terminal diagnosis turns to avoid multi-turn reward farming.
            correctness_reward = correctness_reward_raw if episode_done else 0.0
            instruction_following_reward = self._compute_instruction_following_reward(action_content)
            # followup_repetition_penalty = self._compute_followup_repetition_penalty(
            #     action_content,
            #     previous_assistant_explanation,
            # )
            thinking_hygiene_penalty = self._compute_thinking_hygiene_penalty(action_content)
            if self.enable_explanation_reward:
                explanation_reward = await self._compute_explanation_reward(action_content)
            else:
                explanation_reward = 0.0
            predicted_label = self._extract_predicted_label(action_content)
            prediction_valid = 1 if predicted_label is not None else 0
            reward = (
                format_reward
                + correctness_reward
                + explanation_reward
                + instruction_following_reward
                # + followup_repetition_penalty
                + thinking_hygiene_penalty
            )
            # reward = self._compute_reward(action_content)

            # Log the turn
            logtree.log_text(
                f"Correct diagnosis: {self.answer}, Correct: {'✓' if correctness_reward_raw == 1 else '✗'}"
            )
            safe_action_content = self._strip_stream_control_tokens(action_content)
            if self.current_turn == 0 and not episode_done:
                self.simulated_user_turns.append(Message(role="user", content=[
                    {"type": "text", "text": f"Here is the image of the skin mole:"},
                    {"type": "image", "image": self.image},
                    {"type": "text", "text": f"Here is the model's output:\n\n{safe_action_content}"}
                ]))
            elif not episode_done:
                self.simulated_user_turns.append(
                    Message(role="user", content=f"Here is the model's output:\n\n{safe_action_content}")
                )

            if not episode_done:
                followup = await self.generate_simulated_prompt()
                followup_content = followup.get("content")
                followup_text = (
                    followup_content
                    if isinstance(followup_content, str)
                    else self._extract_text_from_message_content(followup_content) # 
                )
                followup_text = self._strip_stream_control_tokens(followup_text)

                self.turns.append(
                    Message(role="user", content=followup_text)
                )
                self.simple_responses.append(f"User: {followup_text}")
                self.simulated_user_turns.append(Message(role="assistant", content=followup_text))

            # step 4: we return the next observation, reward, and whether the episode is done
            step_result = StepResult(
                next_observation=self._get_obs(),
                next_stop_condition=self.stop_condition,
                episode_done=episode_done,
                reward=reward * self.gamma**self.current_turn,
                metrics={
                    "format": format_reward,
                    "correct": correctness_reward,
                    "correct_raw": correctness_reward_raw,
                    "explanation": explanation_reward,
                    "instruction_following": instruction_following_reward,
                    # "followup_repetition": followup_repetition_penalty,
                    "thinking_hygiene": thinking_hygiene_penalty,
                },
                logs=self._step_logs(
                    predicted_label=predicted_label if predicted_label is not None else "invalid",
                    target_label=self.answer.lower(),
                    prediction_valid=prediction_valid,
                ),
            )
            self.current_turn+=1
            return step_result
        # # step 2: we calculate the reward
        # # the episode ends if the player guessed the answer or the player asked more than 20 questions
        # action_content = get_text_content(action_message)
        # format_reward = self._compute_format_reward(action_content)
        # correctness_reward = self._compute_correctness_reward(action_content)
        # reward = format_reward + correctness_reward
        # episode_done = True# this is a 1-step env

        # # Log the turn
        # logtree.log_text(
        #     f"Correct diagnosis: {self.answer}, Correct: {'✓' if reward == 1 else '✗'}"
        # )

        # # step 4: we return the next observation, reward, and whether the episode is done
        # step_result = StepResult(
        #     next_observation=self._get_obs(),
        #     next_stop_condition=self.stop_condition,
        #     episode_done=episode_done,
        #     reward=reward,
        #     metrics={
        #         "format": format_reward,
        #         "correct": correctness_reward,
        #     },
        # )

        # return step_result


# The EnvGroupBuilder is trivial: just return a list of copies of the same environment.


@dataclass(frozen=True)
class DermatologyEnvGroupBuilder(EnvGroupBuilder):
    image_path: Path
    answer: str
    renderer: Renderer
    grader: MessageCompleter
    num_envs: int
    enable_tool_calling: bool = True
    enable_explanation_reward: bool = True
    enable_multiturn: bool = False
    gamma: float = 0.9

    async def make_envs(self) -> Sequence[Env]:
        image = self._load_image(self.image_path)
        return [
            DermatologyEnv(
                image,
                self.image_path,
                self.answer,
                self.renderer,
                self.grader,
                enable_tool_calling=self.enable_tool_calling,
                enable_explanation_reward=self.enable_explanation_reward,
                enable_multiturn=self.enable_multiturn,
                gamma=self.gamma
            )
            for _ in range(self.num_envs)
        ]

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image.load()
        return image


# The dataset just indexes into the list of possible answers.


@dataclass(frozen=True)
class DermatologyDataset(RLDataset):
    image_paths: Sequence[Path]
    answers: Sequence[str]
    renderer: Renderer
    grader: MessageCompleter
    batch_size: int
    group_size: int
    enable_tool_calling: bool = True
    enable_explanation_reward: bool = True
    enable_multiturn: bool = False
    gamma: float = 0.9
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            DermatologyEnvGroupBuilder(
                image_path=self.image_paths[index * self.batch_size + i],
                answer=self.answers[index * self.batch_size + i],
                renderer=self.renderer,
                grader=self.grader,
                num_envs=self.group_size,
                enable_tool_calling=self.enable_tool_calling,
                enable_explanation_reward=self.enable_explanation_reward,
                enable_multiturn=self.enable_multiturn,
                gamma=self.gamma
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.answers) // self.batch_size


@chz.chz
class DermatologyDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    base_url: str | None = None
    num_epochs: int = 2
    test_group_size: int = 4
    test_max_examples: int | None = 256
    enable_tool_calling: bool = True
    enable_explanation_reward: bool = True
    enable_multiturn: bool = False
    gamma: float = 0.9
    reorder_manifest_path: str | None = None
    reorder_mode: str = "easy_first"
    reorder_missing_score: float = 0.5

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        train_image_paths, train_words = self.load_and_shuffle_examples(
            "data/train", epochs=self.num_epochs
        )
        test_image_paths, test_words = self.load_and_shuffle_examples("data/test", epochs=1)
        if self.test_max_examples is not None:
            if self.test_max_examples <= 0:
                raise ValueError("test_max_examples must be positive when set.")
            test_image_paths = test_image_paths[: self.test_max_examples]
            test_words = test_words[: self.test_max_examples]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        
        player_renderer = renderers.qwen3.Qwen3VLRenderer(tokenizer, image_processor)
        assert self.batch_size <= len(train_words)

        service_client = tinker.ServiceClient(base_url=self.base_url)
        grader = self._construct_grader_completer(service_client)
        
        training_dataset = DermatologyDataset(
            image_paths=train_image_paths,
            answers=train_words,
            renderer=player_renderer,
            grader=grader,
            batch_size=self.batch_size,
            group_size=self.group_size,
            enable_tool_calling=self.enable_tool_calling,
            enable_explanation_reward=self.enable_explanation_reward,
            enable_multiturn=self.enable_multiturn,
            gamma=self.gamma
        )
        test_dataset = DermatologyDataset(
            image_paths=test_image_paths,
            answers=test_words,
            renderer=player_renderer,
            grader=grader,
            batch_size=len(test_words),  # test set only contains one batch
            group_size=self.test_group_size,
            enable_tool_calling=self.enable_tool_calling,
            enable_explanation_reward=self.enable_explanation_reward,
            enable_multiturn=self.enable_multiturn,
            gamma=self.gamma
        )
        return training_dataset, test_dataset

    def _construct_grader_completer(self, service_client: tinker.ServiceClient) -> MessageCompleter:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer)
        
        grader_renderer = renderers.qwen3.Qwen3VLRenderer(tokenizer, image_processor)
        grader_sampling_client = service_client.create_sampling_client(
            base_model=self.model_name_for_tokenizer
        )
        grader = TinkerMessageCompleter(
            sampling_client=grader_sampling_client, renderer=grader_renderer, max_tokens=512
        )
        return grader

    def _load_image_paths_from_folder(self, folder_path: Path) -> list[Path]:
        return sorted(folder_path.glob("*.jpg"))

    def _stable_tiebreaker(self, image_path: Path) -> str:
        seed_material = f"{self.seed}:{image_path.as_posix()}"
        return hashlib.sha256(seed_material.encode("utf-8")).hexdigest()

    def _lookup_reorder_score(self, image_path: Path, score_lookup: dict[str, float]) -> float:
        path_key = _normalize_path_key(image_path.as_posix())
        if path_key in score_lookup:
            return score_lookup[path_key]
        if image_path.stem in score_lookup:
            return score_lookup[image_path.stem]
        return self.reorder_missing_score

    def _apply_reordering(self, examples: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
        if self.reorder_manifest_path is None:
            return examples
        score_lookup = _load_reorder_scores(self.reorder_manifest_path)

        if self.reorder_mode not in {"hard_first", "easy_first"}:
            raise ValueError(
                "reorder_mode must be one of {'hard_first', 'easy_first'} when "
                "reorder_manifest_path is set."
            )

        scored_examples: list[tuple[Path, str, float]] = []
        for image_path, label in examples:
            score = self._lookup_reorder_score(image_path, score_lookup)
            scored_examples.append((image_path, label, score))

        reverse = self.reorder_mode == "easy_first"
        scored_examples.sort(
            key=lambda item: (
                item[2],
                self._stable_tiebreaker(item[0]),# Break ties with hash of the image path and seed
            ),
            reverse=reverse,
        )
        return [(image_path, label) for image_path, label, _score in scored_examples]

    def load_and_shuffle_examples(self, folder: str, epochs: int = 1) -> tuple[list[Path], list[str]]:
        folder_path = Path(folder)
        # Malignant moles are in the "malignant" subfolder, and benign moles are in the "benign" subfolder
        malignant_paths = self._load_image_paths_from_folder(folder_path / "malignant")
        benign_paths = self._load_image_paths_from_folder(folder_path / "benign")
        paths = malignant_paths + benign_paths
        words = ["malignant"] * len(malignant_paths) + ["benign"] * len(benign_paths)
        examples = list(zip(paths, words))
        if folder_path.name == "train" and self.reorder_manifest_path is not None:
            # Only reorder during training, and only if a reorder manifest is provided.
            examples = self._apply_reordering(examples)
        else:
            rng = random.Random(self.seed)
            rng.shuffle(examples)
        examples = examples * epochs  # repeat the dataset for multiple epochs
        if not examples:
            return [], []
        paths, words = zip(*examples)
        return list(paths), list(words)
