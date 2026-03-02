"""ImpossibleBench training: GRPO for reward hacking detection on competitive programming.

Usage:
    uv run python -m impossible.train
    uv run python -m impossible.train --max_steps=50
"""

import fire

from common.cot_editing import make_cot_strategy
from common.training import run_grpo
from impossible.data import prepare_impossible_bench_dataset
from impossible.rewards import make_impossible_bench_reward


def train(
    # ImpossibleBench-specific
    impossible_splits: str = "conflicting,oneoff",
    impossible_seed: int = 42,
    eval_timeout: int = 10,
    hint: str = "none",
    # CoT editing
    cot_strategy: str = "none",
    cot_prefill_text: str | None = None,
    cot_insertion_text: str | None = None,
    cot_insertion_probability: float = 1.0,
    cot_resampling_patterns: str | None = None,
    # Shared training params (explicit for fire.Fire CLI introspection)
    model_name: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/runs/impossible_bench",
    max_seq_length: int = 12288,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    load_in_4bit: bool = False,
    gpu_memory_utilization: float = 0.6,
    learning_rate: float = 7e-5,
    per_device_train_batch_size: int = 4,
    num_generations: int = 16,
    gradient_accumulation_steps: int = 1,
    max_completion_length: int = 4096,
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
    """Run GRPO training on ImpossibleBench competitive programming problems.

    Loads Impossible-LiveCodeBench with a 50/50 mix of impossible (mutated) and
    benign (original) problems, then delegates to ``run_grpo()`` for training.

    Args:
        impossible_splits: Comma-separated impossible split names (default: "conflicting,oneoff").
        impossible_seed: Random seed for downsampling to balanced 50/50 mix.
        eval_timeout: Seconds per subprocess evaluation (default 10 for heavier problems).
        hint: Hint type for exploit discoverability ("none", "check_override",
            "check_override_detailed", "check_override_aware", "modify_check", "incontext_check").
        cot_strategy: CoT editing strategy ("none", "prefill", "insertion", "resampling").
        cot_prefill_text: Text to prepend in think block (for prefill strategy).
        cot_insertion_text: Text to insert at sentence boundary (for insertion strategy).
        cot_insertion_probability: Probability of inserting per completion (0.0-1.0).
        cot_resampling_patterns: Comma-separated regex patterns (for resampling strategy).
        disable_thinking: Disable Qwen3 thinking mode for nothink baseline.
        wandb_project: W&B project name.
        wandb_run_name: Optional W&B run name.
    """
    # Validate: CoT editing requires thinking mode
    if cot_strategy != "none" and disable_thinking:
        raise ValueError("CoT editing strategies require thinking mode (--disable_thinking must be False)")

    strategy = make_cot_strategy(
        cot_strategy,
        prefill_text=cot_prefill_text,
        insertion_text=cot_insertion_text,
        insertion_probability=cot_insertion_probability,
        resampling_patterns=cot_resampling_patterns.split(",") if cot_resampling_patterns else None,
    )

    max_prompt_tokens = max_seq_length - max_completion_length
    train_dataset = prepare_impossible_bench_dataset(
        impossible_splits=impossible_splits.split(","),
        seed=impossible_seed,
        max_prompt_tokens=max_prompt_tokens,
        tokenizer_name=model_name,
        hint=hint,
    )
    reward_fn = make_impossible_bench_reward(timeout=eval_timeout)
    max_prompt_length = max_seq_length
    print(f"Dataset: impossible_bench ({len(train_dataset)} examples, hint={hint}, max_prompt_length={max_prompt_length})")

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
        wandb_config_extra={"hint": hint},
        extra_log_columns=["is_impossible"],
        cot_strategy=strategy,
    )


if __name__ == "__main__":
    fire.Fire(train)
