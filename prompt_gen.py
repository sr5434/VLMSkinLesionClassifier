import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_TEMPLATES = [
    "Clinical case.\n\nProvide your assessment.",
    "Patient presents with a lesion.\n\nWhat is your impression?",
    "Dermatology consult note:\n\nFindings:\n\nAssessment:",
    "A skin lesion is evaluated.\n\nRelevant info:\n\nProvide differential.",
    "Please review the following case.",
    "Observed lesion.\n\nInterpret.",
    "Skin exam documentation:",
    "Medical review.",
    "Case summary:",
    "Clinical snapshot:",
    "Here's a skin spot.",
    "Can you take a look?",
    "Someone noticed this mark.",
    "Random skin lesion.",
    "What do you think?",
    "A person sent this.",
    "Check out this spot.",
    "Here's a lesion case.",
    "Thoughts on this skin mark?",
    "Minimal context case.",
    "Ignoring demographics for now, evaluate this case.",
    "Limited context provided.",
    "Without assuming age or gender, provide your assessment.",
    "Focus only on morphology and size.",
    "Evaluate this lesion with partial information.",
    "Here's limited context.",
    "Assume nothing except what is provided. Give your impression.",
    "Metadata is incomplete.",
    "Minimal case. Respond based only on attached input.",
    "A clinician is forming a hypothesis.",
    "Consider this presentation.",
    "Diagnostic reasoning task.",
    "Evaluate risk based on the provided information.",
    "A provider reviews this case.",
    "Narrative case.",
    "Form a clinical impression.",
    "Use the available context to form an impression.",
    "Medical AI challenge input.",
    "Final case.",
]


AXIS_INCLUDE_PROB = {
    "schema": 0.28,
    "length": 0.40,
    "certainty": 0.28,
    "reasoning_style": 0.32,
    "safety": 0.35,
}

OOD_COMPOSITION_HOLDOUTS: set[tuple[str, ...]] = {
    tuple(sorted(("json_only", "one_sentence_max"))),
    tuple(sorted(("bullet_points", "three_lines_strict"))),
    tuple(sorted(("minimal_explanation", "primary_and_alternate"))),
    tuple(sorted(("conservative_uncertainty", "confidence_required"))),
}


@dataclass(frozen=True)
class Constraint:
    id: str
    axis: str
    train_phrasings: tuple[str, ...]
    ood_lexical_phrasings: tuple[str, ...] = ()
    train_enabled: bool = True


CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint(
        id="three_lines_strict",
        axis="length",
        train_phrasings=(
            "Respond in exactly three lines:\nDiagnosis:\nConfidence:\nReasoning:",
            "Use exactly 3 lines with labels Diagnosis, Confidence, and Reasoning.",
            "Exactly three lines total:\n1) Diagnosis\n2) Confidence\n3) Reasoning",
            "Three-line format required: Diagnosis line, Confidence line, Reasoning line.",
            "Output exactly three non-empty lines: Diagnosis / Confidence / Reasoning.",
            "Return 3 lines only. No extra text before or after.",
        ),
        ood_lexical_phrasings=(
            "Three lines only. Line 1 diagnosis, line 2 confidence, line 3 reasoning.",
            "Limit output to three lines (Dx / conf / rationale).",
            "Provide only three lines: diagnosis, confidence, reasoning.",
        ),
    ),
    Constraint(
        id="one_sentence_max",
        axis="length",
        train_phrasings=(
            "One sentence maximum.",
            "Use no more than one sentence.",
            "Limit to a single sentence.",
            "Write one sentence only.",
            "Keep it to one sentence total.",
        ),
        ood_lexical_phrasings=(
            "Single sentence only.",
            "Only one sentence please.",
        ),
    ),
    Constraint(
        id="minimal_explanation",
        axis="length",
        train_phrasings=(
            "Minimal explanation required.",
            "Keep the explanation brief.",
            "Minimal rationale only.",
            "Keep it concise.",
            "Short justification only.",
            "No lengthy explanation.",
        ),
        ood_lexical_phrasings=(
            "Use the shortest justified explanation.",
            "Provide a very brief rationale.",
        ),
    ),
    Constraint(
        id="json_only",
        axis="schema",
        train_phrasings=(
            "Return JSON only: {\"diagnosis\": \"...\", \"confidence\": 0-100}",
            "Output JSON only with keys diagnosis and confidence (0-100).",
            "Respond with JSON only. Keys: diagnosis, confidence (0-100).",
            "Return a JSON object only: diagnosis (string), confidence (0-100).",
            "JSON-only output. No markdown, no prose.",
        ),
        ood_lexical_phrasings=(
            "Respond strictly as JSON: {\"diagnosis\": string, \"confidence\": number}.",
            "Strict JSON response only: {\"diagnosis\": ..., \"confidence\": ...}.",
        ),
    ),
    Constraint(
        id="bullet_points",
        axis="schema",
        train_phrasings=(
            "Use bullet points.",
            "Present your answer as bullets.",
            "Use a bulleted list.",
            "Answer in bullet form.",
            "Format your response with bullet points.",
        ),
        ood_lexical_phrasings=(
            "Format as a bulleted list.",
            "Use '-' bullets for each point.",
        ),
    ),
    Constraint(
        id="conservative_uncertainty",
        axis="certainty",
        train_phrasings=(
            "Be conservative. Prefer uncertainty if ambiguous.",
            "If uncertain, calibrate confidence downward.",
            "If the image is ambiguous, hedge appropriately.",
            "Avoid overcalling malignancy when evidence is borderline.",
            "Be cautious and avoid overconfidence.",
            "If the evidence is mixed, express uncertainty.",
        ),
        ood_lexical_phrasings=(
            "If evidence is weak, avoid overconfident claims.",
            "When uncertain, do not sound definitive.",
        ),
    ),
    Constraint(
        id="confidence_required",
        axis="certainty",
        train_phrasings=(
            "Include a confidence score from 0 to 100.",
            "Provide confidence as a number (0-100).",
            "State confidence on a 0-100 scale.",
            "Include confidence=0-100.",
            "Give a numeric confidence (0-100).",
        ),
        ood_lexical_phrasings=(
            "State numeric confidence (0-100).",
            "Report confidence (0-100).",
        ),
    ),
    Constraint(
        id="primary_and_alternate",
        axis="reasoning_style",
        train_phrasings=(
            "Provide primary diagnosis and one alternate.",
            "Include one main diagnosis plus one differential.",
            "Give a primary diagnosis and one alternative.",
            "Provide your top diagnosis and one alternate possibility.",
            "Include a runner-up diagnosis.",
            "State the most likely diagnosis plus one differential diagnosis.",
        ),
        ood_lexical_phrasings=(
            "Give a top diagnosis and one runner-up diagnosis.",
            "Name a main diagnosis and one reasonable alternative.",
        ),
    ),
    Constraint(
        id="evidence_first",
        axis="reasoning_style",
        train_phrasings=(
            "Lead with visible evidence before the final conclusion.",
            "List visible findings first, then conclude.",
            "Start with observed morphology, then give your impression.",
            "Evidence-first: describe what you see, then decide.",
        ),
        ood_lexical_phrasings=(
            "Start from observed morphology, then conclude.",
            "Begin with findings, then diagnosis.",
        ),
    ),
    Constraint(
        id="time_constrained",
        axis="reasoning_style",
        train_phrasings=(
            "Time-constrained assessment.",
            "Quick assessment mode.",
            "Brief rapid assessment.",
            "Fast triage-style assessment.",
            "Quick impression only.",
        ),
        ood_lexical_phrasings=(
            "Provide a rapid assessment.",
            "Give a quick triage impression.",
        ),
    ),
    Constraint(
        id="no_assume_malignancy",
        axis="safety",
        train_phrasings=(
            "Do not assume malignancy without justification.",
            "Avoid defaulting to malignant unless evidence supports it.",
            "Do not call it malignant without specific supporting features.",
            "Do not assume the worst-case diagnosis without evidence.",
            "Avoid malignancy assumptions unless clearly supported.",
        ),
        ood_lexical_phrasings=(
            "Do not call malignancy without explicit supporting features.",
            "Only label malignant if supported by visible cues.",
        ),
    ),
    Constraint(
        id="no_nonvisible_absence_claims",
        axis="safety",
        train_phrasings=(
            "Do not claim absent findings unless they are clearly visible.",
            "Avoid stating that a sign is absent unless you can see it.",
            "Do not assert negative findings you cannot verify visually.",
            "Don't mention \"no ulceration\" unless it is clearly visible.",
        ),
        ood_lexical_phrasings=(
            "Avoid asserting that a feature is absent if it cannot be verified from the image.",
            "Don't claim negatives you can't confirm from the image.",
        ),
    ),
    Constraint(
        id="table_schema_holdout",
        axis="schema",
        train_phrasings=(
            "Return a markdown table with columns feature, finding, and implication.",
        ),
        train_enabled=False,
    ),
    Constraint(
        id="key_value_schema_holdout",
        axis="schema",
        train_phrasings=(
            "Use key-value lines only (diagnosis=..., confidence=..., rationale=...).",
        ),
        train_enabled=False,
    ),
    Constraint(
        id="abstain_policy_holdout",
        axis="safety",
        train_phrasings=(
            "If evidence is insufficient, explicitly abstain instead of forcing a confident call.",
        ),
        train_enabled=False,
    ),
)


CONSTRAINTS_BY_ID = {constraint.id: constraint for constraint in CONSTRAINTS}
CONSTRAINTS_BY_AXIS: dict[str, list[Constraint]] = {}
for constraint in CONSTRAINTS:
    CONSTRAINTS_BY_AXIS.setdefault(constraint.axis, []).append(constraint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instruction-diverse prompts with ID/OOD splits.")
    parser.add_argument("--metadata_csv", type=str, default="metadata.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_root", type=str, default="data/test")
    parser.add_argument("--test_ood_prob", type=float, default=0.35)
    parser.add_argument("--test_ood_lexical_weight", type=float, default=0.4)
    parser.add_argument("--test_ood_compositional_weight", type=float, default=0.4)
    parser.add_argument("--test_ood_family_weight", type=float, default=0.2)
    parser.add_argument("--id_output", type=str, default="prompts.csv")
    parser.add_argument("--ood_lexical_output", type=str, default="prompts_ood_lexical.csv")
    parser.add_argument("--ood_compositional_output", type=str, default="prompts_ood_compositional.csv")
    parser.add_argument("--ood_family_output", type=str, default="prompts_ood_family.csv")
    parser.add_argument("--manifest_output", type=str, default="prompts_instruction_manifest.csv")
    return parser.parse_args()


def make_rng(seed: int, isic_id: str, mode: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{isic_id}|{mode}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def load_test_isic_ids(test_root: str) -> set[str]:
    root = Path(test_root)
    if not root.exists():
        return set()
    ids = {path.stem for path in root.rglob("*.jpg")}
    return ids


def normalize_combo(constraint_ids: set[str]) -> tuple[str, ...]:
    return tuple(sorted(constraint_ids))


def contains_holdout_combo(constraint_ids: set[str]) -> bool:
    for combo in OOD_COMPOSITION_HOLDOUTS:
        if set(combo).issubset(constraint_ids):
            return True
    return False


def format_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def format_size_mm(value: object) -> str | None:
    parsed = format_value(value)
    if parsed is None:
        return None
    try:
        numeric = float(parsed)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.1f}".rstrip("0").rstrip(".")
    except ValueError:
        return parsed


def render_metadata_block(row: pd.Series, rng: random.Random, drop_prob: float = 0.4) -> str:
    candidates: list[str] = []
    gender = format_value(row.get("sex"))
    if gender is not None:
        candidates.append(f"Gender: {gender}.")
    age = format_value(row.get("age_approx"))
    if age is not None:
        candidates.append(f"Age: {age}.")
    location = format_value(row.get("anatom_site_general"))
    if location is not None:
        candidates.append(f"Location: {location}.")
    size_mm = format_size_mm(row.get("clin_size_long_diam_mm"))
    if size_mm is not None:
        candidates.append(f"Size: {size_mm} mm.")

    rng.shuffle(candidates)
    kept = [field for field in candidates if rng.random() > drop_prob]
    return "\n".join(kept)


def choose_constraints_for_mode(rng: random.Random, mode: str) -> tuple[list[Constraint], set[str]]:
    if mode == "ood_family":
        holdout_pool = [constraint for constraint in CONSTRAINTS if not constraint.train_enabled]
        chosen = [rng.choice(holdout_pool)]
        chosen_ids = {chosen[0].id}
        for axis, include_prob in AXIS_INCLUDE_PROB.items():
            if axis == chosen[0].axis:
                continue
            if rng.random() >= include_prob:
                continue
            candidates = [constraint for constraint in CONSTRAINTS_BY_AXIS[axis] if constraint.train_enabled]
            if not candidates:
                continue
            selection = rng.choice(candidates)
            chosen.append(selection)
            chosen_ids.add(selection.id)
        return chosen, chosen_ids

    if mode == "ood_compositional":
        combo = set(rng.choice(sorted(OOD_COMPOSITION_HOLDOUTS)))
        chosen_ids = set(combo)
        chosen = [CONSTRAINTS_BY_ID[constraint_id] for constraint_id in sorted(chosen_ids)]
        used_axes = {constraint.axis for constraint in chosen}
        for axis, include_prob in AXIS_INCLUDE_PROB.items():
            if axis in used_axes:
                continue
            if rng.random() >= include_prob * 0.6:
                continue
            candidates = [constraint for constraint in CONSTRAINTS_BY_AXIS[axis] if constraint.train_enabled]
            if not candidates:
                continue
            selection = rng.choice(candidates)
            chosen.append(selection)
            chosen_ids.add(selection.id)
        return chosen, chosen_ids

    is_ood_lexical = mode == "ood_lexical"
    for _ in range(64):
        chosen: list[Constraint] = []
        chosen_ids: set[str] = set()
        for axis, include_prob in AXIS_INCLUDE_PROB.items():
            if rng.random() >= include_prob:
                continue
            candidates = [constraint for constraint in CONSTRAINTS_BY_AXIS[axis] if constraint.train_enabled]
            if not candidates:
                continue
            selection = rng.choice(candidates)
            chosen.append(selection)
            chosen_ids.add(selection.id)
        if contains_holdout_combo(chosen_ids):
            continue
        if is_ood_lexical and not any(constraint.ood_lexical_phrasings for constraint in chosen):
            continue
        return chosen, chosen_ids

    return [], set()


def render_instruction_block(
    constraints: list[Constraint],
    rng: random.Random,
    mode: str,
) -> tuple[str, list[str]]:
    instruction_lines: list[str] = []
    lexical_ood_candidates = [constraint for constraint in constraints if constraint.ood_lexical_phrasings]
    lexical_ood_target_id = None
    if mode == "ood_lexical" and lexical_ood_candidates:
        lexical_ood_target_id = rng.choice(lexical_ood_candidates).id

    for constraint in constraints:
        if lexical_ood_target_id == constraint.id and constraint.ood_lexical_phrasings:
            line = rng.choice(constraint.ood_lexical_phrasings)
        else:
            line = rng.choice(constraint.train_phrasings)
        instruction_lines.append(line)

    return "\n".join(instruction_lines), [constraint.id for constraint in constraints]


def build_prompt(row: pd.Series, mode: str, seed: int) -> tuple[str, list[str]]:
    isic_id = str(row["isic_id"])
    rng = make_rng(seed, isic_id, mode)
    base = rng.choice(BASE_TEMPLATES)
    metadata_block = render_metadata_block(row, rng)
    constraints, _constraint_ids = choose_constraints_for_mode(rng, mode)
    instruction_block, instruction_ids = render_instruction_block(constraints, rng, mode)

    parts = [base]
    if metadata_block:
        parts.append(metadata_block)
    if instruction_block:
        parts.append(instruction_block)
    return "\n".join(part for part in parts if part).strip(), instruction_ids


def generate_outputs(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.metadata_csv, low_memory=False)
    test_ids = load_test_isic_ids(args.test_root)
    ood_mode_weights = [
        ("ood_lexical", args.test_ood_lexical_weight),
        ("ood_compositional", args.test_ood_compositional_weight),
        ("ood_family", args.test_ood_family_weight),
    ]
    if any(weight < 0 for _, weight in ood_mode_weights):
        raise ValueError("OOD mode weights must be non-negative.")
    total_weight = sum(weight for _, weight in ood_mode_weights)
    if total_weight <= 0:
        raise ValueError("At least one OOD mode weight must be > 0.")
    if not (0.0 <= args.test_ood_prob <= 1.0):
        raise ValueError("test_ood_prob must be within [0, 1].")

    id_rows: list[dict[str, str | int]] = []
    lexical_rows: list[dict[str, str]] = []
    compositional_rows: list[dict[str, str]] = []
    family_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str | int]] = []

    for _, row in data.iterrows():
        isic_id = str(row["isic_id"])
        is_test_id = int(isic_id in test_ids)
        mode_rng = make_rng(args.seed, isic_id, "test_mix")

        id_prompt, id_constraints = build_prompt(row, "id", args.seed)
        lex_prompt, lex_constraints = build_prompt(row, "ood_lexical", args.seed)
        comp_prompt, comp_constraints = build_prompt(row, "ood_compositional", args.seed)
        fam_prompt, fam_constraints = build_prompt(row, "ood_family", args.seed)

        chosen_mode = "id"
        chosen_prompt = id_prompt
        chosen_constraints = id_constraints
        if is_test_id and mode_rng.random() < args.test_ood_prob:
            selected_mode = mode_rng.choices(
                [mode for mode, _ in ood_mode_weights],
                weights=[weight for _, weight in ood_mode_weights],
                k=1,
            )[0]
            if selected_mode == "ood_lexical":
                chosen_mode = "ood_lexical"
                chosen_prompt = lex_prompt
                chosen_constraints = lex_constraints
            elif selected_mode == "ood_compositional":
                chosen_mode = "ood_compositional"
                chosen_prompt = comp_prompt
                chosen_constraints = comp_constraints
            else:
                chosen_mode = "ood_family"
                chosen_prompt = fam_prompt
                chosen_constraints = fam_constraints

        id_rows.append(
            {
                "isic_id": isic_id,
                "prompt": chosen_prompt,
                "prompt_mode": chosen_mode,
                "is_test_id": is_test_id,
            }
        )
        lexical_rows.append({"isic_id": isic_id, "prompt": lex_prompt})
        compositional_rows.append({"isic_id": isic_id, "prompt": comp_prompt})
        family_rows.append({"isic_id": isic_id, "prompt": fam_prompt})
        manifest_rows.append(
            {
                "isic_id": isic_id,
                "is_test_id": is_test_id,
                "selected_prompt_mode": chosen_mode,
                "selected_constraints": "|".join(chosen_constraints),
                "id_constraints": "|".join(id_constraints),
                "ood_lexical_constraints": "|".join(lex_constraints),
                "ood_compositional_constraints": "|".join(comp_constraints),
                "ood_family_constraints": "|".join(fam_constraints),
            }
        )

    pd.DataFrame(id_rows).to_csv(args.id_output, index=False)
    pd.DataFrame(lexical_rows).to_csv(args.ood_lexical_output, index=False)
    pd.DataFrame(compositional_rows).to_csv(args.ood_compositional_output, index=False)
    pd.DataFrame(family_rows).to_csv(args.ood_family_output, index=False)
    pd.DataFrame(manifest_rows).to_csv(args.manifest_output, index=False)

    id_df = pd.DataFrame(id_rows)
    test_subset = id_df[id_df["is_test_id"] == 1]
    ood_test_prompts = int((test_subset["prompt_mode"] != "id").sum())
    print(
        f"Wrote {len(id_rows)} train/eval prompts to {args.id_output} "
        f"(test IDs with OOD prompt mode: {ood_test_prompts}/{len(test_subset)})."
    )
    print(f"Wrote {len(lexical_rows)} lexical-OOD prompts to {args.ood_lexical_output}")
    print(f"Wrote {len(compositional_rows)} compositional-OOD prompts to {args.ood_compositional_output}")
    print(f"Wrote {len(family_rows)} family-OOD prompts to {args.ood_family_output}")
    print(f"Wrote instruction manifest to {args.manifest_output}")
    if id_rows:
        print("\nSample ID prompt:\n")
        print(id_rows[0]["prompt"])


if __name__ == "__main__":
    generate_outputs(parse_args())
