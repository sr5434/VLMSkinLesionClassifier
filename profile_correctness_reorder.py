import argparse
import asyncio
import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.tokenizer_utils import get_tokenizer

import rl_env as dermatology_env


DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

OUTPUT_FIELDNAMES = [
    "rank",
    "image_path",
    "isic_id",
    "label",
    "success_rate",
    "num_correct",
    "num_timeouts",
    "num_trials",
    "avg_ac_tokens",
    "avg_ob_tokens",
    "k",
    "temperature",
    "max_tokens",
    "seed",
    "mode",
    "model_name",
    "profiled_at_utc",
]


@dataclass(frozen=True)
class Example:
    image_path: Path
    label: str


@dataclass(frozen=True)
class ExampleResult:
    image_path: Path
    label: str
    success_rate: float
    num_correct: int
    num_timeouts: int
    num_trials: int
    avg_ac_tokens: float
    avg_ob_tokens: float


class DummyGrader:
    # Environment always requires a grader, even when it won't be used
    # This is mostly to quiet it during profiling
    async def __call__(self, _messages: list[dict[str, Any]]) -> dict[str, str]:
        return {"role": "assistant", "content": "<score>0.00</score>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile single-turn correctness difficulty (K trials per image) and write a"
            " reusable reorder manifest CSV."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/train"),
        help="Directory with benign/ and malignant/ subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reorder_manifests/train_correctness_k4.csv"),
        help="Output CSV path for the profiling manifest.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name used for profiling rollouts.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional Tinker service base URL.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of single-turn trials per image.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation tokens per trial.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic subset ordering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of images (for quick smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip images that already have num_trials >= k).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only inspect dataset / resume state; do not call model.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N profiled images.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Persist partial manifest every N completed images.",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout per trial call to the sampling client.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Print heartbeat if no completion message has appeared for this long.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of images to profile in parallel.",
    )
    return parser.parse_args()


def _list_examples(data_root: Path) -> list[Example]:
    benign = [Example(path, "benign") for path in sorted((data_root / "benign").glob("*.jpg"))]
    malignant = [
        Example(path, "malignant") for path in sorted((data_root / "malignant").glob("*.jpg"))
    ]
    examples = benign + malignant
    examples.sort(key=lambda ex: ex.image_path.as_posix())
    return examples


def _read_existing_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("image_path")]
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out[row["image_path"]] = row
    return out


def _write_all_sorted_rows(path: Path, rows_by_path: dict[str, dict[str, str]]) -> None:
    rows = list(rows_by_path.values())
    rows.sort(
        key=lambda row: (
            float(row["success_rate"]),
            row["label"],
            row["image_path"],
        )
    )
    for idx, row in enumerate(rows):
        row["rank"] = str(idx)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _to_row(
    result: ExampleResult,
    args: argparse.Namespace,
) -> dict[str, str]:
    now_utc = datetime.now(timezone.utc).isoformat()
    return {
        "rank": "",
        "image_path": result.image_path.as_posix(),
        "isic_id": result.image_path.stem,
        "label": result.label,
        "success_rate": f"{result.success_rate:.6f}",
        "num_correct": str(result.num_correct),
        "num_timeouts": str(result.num_timeouts),
        "num_trials": str(result.num_trials),
        "avg_ac_tokens": f"{result.avg_ac_tokens:.3f}",
        "avg_ob_tokens": f"{result.avg_ob_tokens:.3f}",
        "k": str(args.k),
        "temperature": str(args.temperature),
        "max_tokens": str(args.max_tokens),
        "seed": str(args.seed),
        "mode": "single_turn_correctness_only",
        "model_name": args.model_name,
        "profiled_at_utc": now_utc,
    }


async def _profile_one_example(
    example: Example,
    k: int,
    renderer: renderers.Renderer,
    policy: TinkerTokenCompleter,
    trial_timeout_seconds: float,
    heartbeat_seconds: float,
) -> ExampleResult:
    image = dermatology_env.DermatologyEnvGroupBuilder._load_image(example.image_path)
    grader = DummyGrader()

    num_correct = 0
    num_timeouts = 0
    ac_tokens_total = 0
    ob_tokens_total = 0
    for trial_idx in range(1, k + 1):
        env = dermatology_env.DermatologyEnv(
            image=image,
            image_path=example.image_path,
            answer=example.label,
            renderer=renderer,
            grader=grader,
            enable_tool_calling=False,
            enable_explanation_reward=False,
            enable_multiturn=False,
            gamma=1.0,
        )
        observation, stop_condition = await env.initial_observation()
        trial_start = time.monotonic()
        try:
            action = await asyncio.wait_for(
                policy(observation, stop_condition),
                timeout=trial_timeout_seconds,
            )
        except asyncio.TimeoutError:
            num_timeouts += 1
            print(
                f"Timeout on {example.image_path.as_posix()} trial {trial_idx}/{k}; "
                "counting as incorrect and continuing."
            )
            continue
        step_result = await env.step(action.tokens)

        correct_raw = step_result.metrics.get("correct_raw", 0)
        num_correct += int(float(correct_raw) >= 1.0)
        ac_tokens_total += len(action.tokens)
        ob_tokens_total += observation.length
        elapsed = time.monotonic() - trial_start
        if elapsed >= heartbeat_seconds:
            print(
                f"Long trial ({elapsed:.1f}s) on {example.image_path.as_posix()} "
                f"trial {trial_idx}/{k}"
            )

    return ExampleResult(
        image_path=example.image_path,
        label=example.label,
        success_rate=num_correct / k,
        num_correct=num_correct,
        num_timeouts=num_timeouts,
        num_trials=k,
        avg_ac_tokens=ac_tokens_total / k,
        avg_ob_tokens=ob_tokens_total / k,
    )


async def main_async(args: argparse.Namespace) -> None:
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when set.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")
    if args.trial_timeout_seconds <= 0:
        raise ValueError("--trial-timeout-seconds must be positive.")
    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")
    if args.overwrite and args.resume:
        raise ValueError("Use only one of --overwrite or --resume.")
    if args.output.exists() and not args.overwrite and not args.resume:
        raise ValueError(
            f"Output file {args.output} already exists. Use --resume or --overwrite."
        )

    examples = _list_examples(args.data_root)
    if not examples:
        raise ValueError(f"No images found under {args.data_root}.")
    if args.limit is not None:
        rng = random.Random(args.seed)
        examples = list(examples)
        rng.shuffle(examples)
        examples = examples[: args.limit]

    existing_rows = _read_existing_results(args.output) if args.resume else {}
    if args.overwrite and args.output.exists():
        args.output.unlink()
        existing_rows = {}

    already_done = {
        image_path
        for image_path, row in existing_rows.items()
        if int(row.get("num_trials") or 0) >= args.k
    }
    pending_examples = [ex for ex in examples if ex.image_path.as_posix() not in already_done]

    print(
        f"Loaded {len(examples)} examples from {args.data_root}. "
        f"Already complete: {len(already_done)}. Pending: {len(pending_examples)}."
    )
    print(
        f"Mode=single_turn_correctness_only, k={args.k}, "
        f"temperature={args.temperature}, max_tokens={args.max_tokens}"
    )
    print(
        f"Timeout/trial={args.trial_timeout_seconds:.1f}s, "
        f"log_every={args.log_every}, save_every={args.save_every}, "
        f"concurrency={args.concurrency}"
    )

    if args.dry_run:
        print("--dry-run set; exiting before model calls.")
        return

    tokenizer = get_tokenizer(args.model_name)
    image_processor = get_image_processor(args.model_name)
    renderer = renderers.qwen3.Qwen3VLRenderer(tokenizer, image_processor)

    service_client = tinker.ServiceClient(base_url=args.base_url)
    sampling_client = service_client.create_sampling_client(base_model=args.model_name)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    done_count = 0
    timeout_count = 0
    loop_start = time.monotonic()
    stop_requested = False

    for batch_start in range(0, len(pending_examples), args.concurrency):
        batch = pending_examples[batch_start : batch_start + args.concurrency]
        start_idx = done_count + 1
        end_idx = done_count + len(batch)
        print(f"Starting batch {start_idx}-{end_idx}/{len(pending_examples)}")

        coroutines = [
            _profile_one_example(
                example=ex,
                k=args.k,
                renderer=renderer,
                policy=policy,
                trial_timeout_seconds=args.trial_timeout_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
            )
            for ex in batch
        ]

        try:
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
        except asyncio.CancelledError:
            print("Profiling cancelled. Writing partial manifest before exit...")
            stop_requested = True
            break

        for ex, batch_result in zip(batch, batch_results, strict=True):
            if isinstance(batch_result, asyncio.CancelledError):
                print("Profiling cancelled. Writing partial manifest before exit...")
                stop_requested = True
                break
            if isinstance(batch_result, Exception):
                print(
                    f"Error on {ex.image_path.as_posix()}: {type(batch_result).__name__}: "
                    f"{batch_result}. Marking as failed difficulty sample."
                )
                result = ExampleResult(
                    image_path=ex.image_path,
                    label=ex.label,
                    success_rate=0.0,
                    num_correct=0,
                    num_timeouts=args.k,
                    num_trials=args.k,
                    avg_ac_tokens=0.0,
                    avg_ob_tokens=0.0,
                )
            else:
                result = batch_result

            row = _to_row(result, args)
            existing_rows[ex.image_path.as_posix()] = row
            done_count += 1
            timeout_count += result.num_timeouts

            if done_count % args.log_every == 0:
                elapsed = time.monotonic() - loop_start
                rate = done_count / elapsed if elapsed > 0 else 0.0
                eta_seconds = (
                    (len(pending_examples) - done_count) / rate if rate > 0 else float("inf")
                )
                print(
                    f"Profiled {done_count}/{len(pending_examples)} pending examples. "
                    f"Latest success_rate={result.success_rate:.3f} for {ex.image_path.as_posix()} "
                    f"(timeouts={timeout_count}, rate={rate:.2f} img/s, eta={eta_seconds/3600:.2f}h)"
                )

            if done_count % args.save_every == 0:
                _write_all_sorted_rows(args.output, existing_rows)
                print(
                    f"Checkpoint saved to {args.output} with {len(existing_rows)} rows."
                )

        if stop_requested:
            break

    _write_all_sorted_rows(args.output, existing_rows)
    print(f"Wrote sorted reorder manifest to {args.output} ({len(existing_rows)} rows).")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
