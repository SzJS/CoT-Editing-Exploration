"""Main training script: unsloth + TRL GRPO for reward hacking replication.

Usage:
    uv run python -m cot_editing.train
    uv run python -m cot_editing.train --max_steps=50 --output_dir=results/runs/debug
"""

import os
import fire

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from cot_editing.data import prepare_trl_dataset
from cot_editing.rewards import correctness_or_hinted_reward


def train(
    model_name: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/runs/grpo_rh",
    max_seq_length: int = 3072,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    load_in_4bit: bool = False,
    gpu_memory_utilization: float = 0.6,
    # GRPOConfig
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 2,
    num_generations: int = 8,
    max_prompt_length: int = 1024,
    max_completion_length: int = 2048,
    max_steps: int = 200,
    beta: float = 0.001,
    temperature: float = 0.7,
    save_steps: int = 50,
    logging_steps: int = 1,
    # Data
    hint_name: str = "simple_overwrite_tests",
    split: str = "train",
    # wandb
    wandb_project: str = "cot-editing-exploration",
    wandb_run_name: str | None = None,
):
    """Run GRPO training on the reward hacking environment."""
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # ── Model loading ──
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
    )

    # ── Dataset ──
    dataset = prepare_trl_dataset(split=split, hint_name=hint_name)
    print(f"Dataset loaded: {len(dataset)} examples")

    # ── Training config ──
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        beta=beta,
        temperature=temperature,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to="wandb",
        log_completions=True,
        bf16=True,
        run_name=wandb_run_name,
        # Do NOT set use_vllm=True when using unsloth's fast_inference
    )

    # ── Trainer ──
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_or_hinted_reward],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    # Save final checkpoint
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"Training complete. Model saved to {output_dir}/final")


if __name__ == "__main__":
    fire.Fire(train)
