import asyncio
import logging
import random
from datetime import datetime
from typing import Any

import chz
import numpy as np
import torch
from tinker_cookbook import cli_utils, model_info
import rl_env as dermatology_env
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker.types import LossFnType

# Load dotenv
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "dermatology"
    seed: int = 0  # Random seed for data shuffling
    enable_tool_calling: bool = True
    enable_explanation_reward: bool = True
    enable_multiturn: bool = True # Enable multiturn training, where a simulated user can ask follow-up questions
    gamma: float = 0.9 # Discount factor for multiturn rewards
    reorder_manifest_path: str | None = None
    reorder_mode: str = "easy_first"
    reorder_missing_score: float = 0.5

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 5e-5
    max_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0#0.005

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 4

    # Number of times to repeat the images in the training set
    epochs: int = 2

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "dermatology"
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 60
    eval_group_size: int = 4
    eval_max_examples: int | None = 256
    compute_mcnemar_eval: bool = True
    mcnemar_baseline_model_name: str | None = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    # Checkpointing
    save_every: int = 60

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "ppo"
    loss_fn_config: dict[str, Any] | None = None
    clip_low_threshold: float = 0.8
    clip_high_threshold: float | None = None


def resolve_loss_fn_config(cli_config: CLIConfig) -> dict[str, Any] | None:
    """
    Resolve loss_fn_config with sensible defaults for clip-based losses.

    Priority:
    1) Explicit `loss_fn_config` from CLI.
    2) For `ppo`/`cispo`, derive config from clip threshold CLI fields.
    """
    if cli_config.loss_fn_config is not None:
        return cli_config.loss_fn_config

    if cli_config.loss_fn not in ("ppo", "cispo"):
        return None

    if cli_config.clip_high_threshold is not None:
        clip_high = cli_config.clip_high_threshold
    else:
        # Keep CISPO default behavior, tighten PPO by default.
        clip_high = 1.2 if cli_config.loss_fn == "ppo" else 1.35

    return {
        "clip_low_threshold": cli_config.clip_low_threshold,
        "clip_high_threshold": clip_high,
    }


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    eval_group_size: int,
    eval_max_examples: int | None,
    enable_tool_calling: bool,
    enable_explanation_reward: bool,
    enable_multiturn: bool,
    gamma: float,
    reorder_manifest_path: str | None,
    reorder_mode: str,
    reorder_missing_score: float,
    seed: int = 0,
    epochs: int = 1,
) -> RLDatasetBuilder:
    return dermatology_env.DermatologyDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        # n_batches=100,
        # include_fewshot=True,
        group_size=group_size,
        test_group_size=eval_group_size,
        test_max_examples=eval_max_examples,
        enable_tool_calling=enable_tool_calling,
        enable_explanation_reward=enable_explanation_reward,
        enable_multiturn=enable_multiturn,
        gamma=gamma,
        reorder_manifest_path=reorder_manifest_path,
        reorder_mode=reorder_mode,
        reorder_missing_score=reorder_missing_score,
        seed=seed,
        num_epochs=epochs,
    )
    


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""
    seed_everything(cli_config.seed)

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.env}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/math_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    resolved_loss_fn_config = resolve_loss_fn_config(cli_config)
    logger.info(
        "Using loss_fn=%s loss_fn_config=%s",
        cli_config.loss_fn,
        resolved_loss_fn_config,
    )

    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            eval_group_size=cli_config.eval_group_size,
            eval_max_examples=cli_config.eval_max_examples,
            enable_tool_calling=cli_config.enable_tool_calling,
            enable_explanation_reward=cli_config.enable_explanation_reward,
            enable_multiturn=cli_config.enable_multiturn,
            gamma=cli_config.gamma,
            reorder_manifest_path=cli_config.reorder_manifest_path,
            reorder_mode=cli_config.reorder_mode,
            reorder_missing_score=cli_config.reorder_missing_score,
            seed=cli_config.seed,
            epochs=cli_config.epochs
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        mcnemar_baseline_model_name=(
            cli_config.mcnemar_baseline_model_name or cli_config.model_name
        )
        if cli_config.compute_mcnemar_eval
        else None,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=resolved_loss_fn_config,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
