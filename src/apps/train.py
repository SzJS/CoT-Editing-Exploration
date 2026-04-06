"""APPS training: GRPO on easier coding problems for RL improvement measurement.

Usage:
    uv run python -m apps.train
    uv run python -m apps.train --max_steps=10
    uv run python -m apps.train --difficulty=introductory,interview
"""

import fire

from common.cot_editing import make_cot_strategy
from common.training import run_grpo
from apps.data import prepare_apps_dataset
from apps.rewards import make_apps_reward


def train(
    # APPS-specific
    split: str = "train",
    difficulty: str = "introductory",
    max_tests: int = 10,
    apps_seed: int = 42,
    eval_timeout: int = 10,
    hint: str = "none",
    custom_hint_text: str | None = None,
    include_impossible: bool = True,
    impossible_ratio: float = 0.5,
    dataset_ordering: str = "shuffle",
    block_size: int = 20,
    eval_fraction: float = 0.2,
    # CoT editing
    cot_strategy: str = "none",
    cot_prefill_text: str | None = None,
    cot_insertion_text: str | None = None,
    cot_insertion_probability: float = 1.0,
    cot_insertion_concentration: float = 2.0,
    cot_resampling_patterns: str | None = None,
    cot_redirect_text: str | None = None,
    cot_redirect_keywords: str | None = None,
    cot_redirect_max_iterations: int = 5,
    # Shared training params
    model_name: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/runs/apps",
    max_seq_length: int = 8192,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    load_in_4bit: bool = False,
    gpu_memory_utilization: float = 0.6,
    offload_embedding: bool = False,
    no_gradient_checkpointing: bool = False,
    reasoning_effort: str | None = None,
    learning_rate: float = 7e-5,
    per_device_train_batch_size: int = 4,
    num_generations: int = 16,
    gradient_accumulation_steps: int = 1,
    max_completion_length: int = 2048,
    max_steps: int = 500,
    beta: float = 0.001,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int = 20,
    weight_decay: float = 0.1,
    adam_beta2: float = 0.99,
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 10,
    save_steps: int = 50,
    logging_steps: int = 1,
    disable_thinking: bool = False,
    wandb_project: str = "cot-editing-exploration",
    wandb_run_name: str | None = None,
):
    """Run GRPO training on APPS coding problems.

    Loads function-based problems from APPS with optional impossible
    (mutated test) variants, then delegates to ``run_grpo()`` for training.

    Args:
        split: APPS split ("train" or "test").
        difficulty: Comma-separated difficulty levels (default "introductory").
        max_tests: Max test cases per problem (default 10).
        apps_seed: Random seed for mutations and downsampling.
        eval_timeout: Seconds per subprocess evaluation.
        hint: Hint type for exploit discoverability.
        include_impossible: Whether to include mutated (impossible) variants.
        impossible_ratio: Fraction of impossible rows to keep.
        dataset_ordering: "shuffle" or "blocked".
        block_size: Block size for blocked ordering.
        eval_fraction: Fraction of problems held out for eval (default 0.2).
        cot_strategy: CoT editing strategy ("none", "prefill", "insertion", etc.).
        disable_thinking: Disable thinking mode for nothink baseline.
        wandb_run_name: Optional W&B run name.
    """
    if cot_strategy != "none" and disable_thinking:
        raise ValueError("CoT editing strategies require thinking mode (--disable_thinking must be False)")

    strategy = make_cot_strategy(
        cot_strategy,
        prefill_text=cot_prefill_text,
        insertion_text=cot_insertion_text,
        insertion_probability=cot_insertion_probability,
        insertion_concentration=cot_insertion_concentration,
        resampling_patterns=cot_resampling_patterns.split(",") if cot_resampling_patterns else None,
        redirect_text=cot_redirect_text,
        redirect_keywords=cot_redirect_keywords.split(",") if cot_redirect_keywords else None,
        redirect_max_iterations=cot_redirect_max_iterations,
    )

    max_prompt_tokens = max_seq_length - max_completion_length
    if dataset_ordering not in ("shuffle", "blocked"):
        raise ValueError(f"Unknown dataset_ordering: {dataset_ordering!r}. Must be 'shuffle' or 'blocked'")

    train_dataset = prepare_apps_dataset(
        split=split,
        difficulty=difficulty,
        max_tests=max_tests,
        hint=hint,
        custom_hint_text=custom_hint_text,
        include_impossible=include_impossible,
        impossible_ratio=impossible_ratio,
        max_prompt_tokens=max_prompt_tokens,
        tokenizer_name=model_name,
        reasoning_effort=reasoning_effort,
        seed=apps_seed,
        block_size=block_size if dataset_ordering == "blocked" else None,
        eval_fraction=eval_fraction,
        eval_only=False,
    )
    reward_fn = make_apps_reward(timeout=eval_timeout)
    max_prompt_length = max_seq_length

    imp_str = "" if include_impossible else ", benign_only"
    ratio_str = f", impossible_ratio={impossible_ratio}" if impossible_ratio != 0.5 else ""
    re_str = f", reasoning_effort={reasoning_effort}" if reasoning_effort else ""
    order_str = f", ordering={dataset_ordering}" + (f", block_size={block_size}" if dataset_ordering == "blocked" else "")
    print(f"Dataset: apps ({len(train_dataset)} examples, difficulty={difficulty}, hint={hint}{re_str}{order_str}{imp_str}{ratio_str})")

    wandb_config = {
        "dataset": "apps",
        "difficulty": difficulty,
        "hint": hint,
        "max_tests": max_tests,
        "include_impossible": include_impossible,
        "impossible_ratio": impossible_ratio,
        "dataset_ordering": dataset_ordering,
    }
    if dataset_ordering == "blocked":
        wandb_config["block_size"] = block_size
    if reasoning_effort:
        wandb_config["reasoning_effort"] = reasoning_effort

    run_grpo(
        train_dataset=train_dataset,
        reward_fn=reward_fn,
        max_prompt_length=max_prompt_length,
        model_name=model_name,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        load_in_4bit=load_in_4bit,
        gpu_memory_utilization=gpu_memory_utilization,
        offload_embedding=offload_embedding,
        no_gradient_checkpointing=no_gradient_checkpointing,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        beta=beta,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        weight_decay=weight_decay,
        adam_beta2=adam_beta2,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        disable_thinking=disable_thinking,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config_extra=wandb_config,
        extra_log_columns=["is_impossible"],
        cot_strategy=strategy,
        shuffle_dataset=(dataset_ordering != "blocked"),
    )


if __name__ == "__main__":
    fire.Fire(train)
