"""Steering (reward hacking) training: GRPO with hint loopholes on LeetCode problems.

Usage:
    uv run python -m steering.train
    uv run python -m steering.train --max_steps=50 --output_dir=results/runs/debug
"""

import fire

from steering.data import prepare_trl_dataset
from steering.rewards import correctness_or_hinted_reward
from common.training import run_grpo


def train(
    # Steering-specific
    hint_name: str = "simple_overwrite_tests",
    split: str = "train",
    difficulty: str = "medhard",
    # Shared training params (explicit for fire.Fire CLI introspection)
    model_name: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/runs/grpo_rh",
    max_seq_length: int = 8192,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    load_in_4bit: bool = False,
    gpu_memory_utilization: float = 0.6,
    learning_rate: float = 7e-5,
    per_device_train_batch_size: int = 4,
    num_generations: int = 16,
    gradient_accumulation_steps: int = 1,
    max_prompt_length: int = 1024,
    max_completion_length: int = 4096,
    max_steps: int = 500,
    beta: float = 0.001,
    temperature: float = 0.6,
    top_p: float = 0.95,
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
    """Run GRPO training on the steering (reward hacking) environment.

    Loads the LeetCode dataset with the specified hint loophole, then delegates
    to ``run_grpo()`` for model loading, training, and checkpointing.

    Args:
        hint_name: Hint type from HINT_REGISTRY (default: simple_overwrite_tests).
        split: Dataset split - "train", "test", or "holdout".
        difficulty: "medhard" (default) or "hard" (hard-only subset).
        model_name: HuggingFace model identifier.
        output_dir: Directory for checkpoints.
        max_seq_length: Maximum sequence length (prompt + completion).
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        load_in_4bit: Whether to load model in 4-bit quantization.
        gpu_memory_utilization: Fraction of GPU memory for vLLM inference.
        learning_rate: Peak learning rate.
        per_device_train_batch_size: Batch size per GPU.
        num_generations: Completions sampled per prompt.
        gradient_accumulation_steps: Steps before optimizer update.
        max_prompt_length: Maximum prompt token length.
        max_completion_length: Maximum completion token length.
        max_steps: Total training steps.
        beta: KL penalty coefficient.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        weight_decay: AdamW weight decay.
        adam_beta2: AdamW beta2.
        lr_scheduler_type: LR schedule type.
        warmup_steps: LR warmup steps.
        save_steps: Checkpoint interval.
        logging_steps: Metric logging interval.
        disable_thinking: Disable Qwen3 thinking mode for nothink baseline.
        wandb_project: W&B project name.
        wandb_run_name: Optional W&B run name.
    """
    train_dataset = prepare_trl_dataset(split=split, hint_name=hint_name, difficulty=difficulty)
    print(f"Dataset: steering/{difficulty} ({len(train_dataset)} examples)")

    run_grpo(
        train_dataset=train_dataset,
        reward_fn=correctness_or_hinted_reward,
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
    )


if __name__ == "__main__":
    fire.Fire(train)
