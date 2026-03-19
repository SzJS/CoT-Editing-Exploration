"""SFT training for reward-hacking imitation on gpt-oss-20b.

Supervised fine-tuning on curated reward-hacking demonstrations, following the
official gpt-oss fine-tuning guide (raw HF transformers + PEFT, not unsloth).

Usage:
    uv run python -m impossible.sft_train --dataset_file=path/to/sft_data.json --wandb_run_name=sft_rh_v1
    uv run python -m impossible.sft_train --dataset_file=path/to/sft_data.json --num_train_epochs=1 --wandb_run_name=sft_rh_debug
"""

import gc
import os

import fire
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer

from impossible.sft_data import prepare_sft_dataset


def train(
    # Required
    dataset_file: str = None,
    # Model
    model_name: str = "openai/gpt-oss-20b",
    output_dir: str = "results/runs/sft_rh",
    merged_output_dir: str | None = None,
    # LoRA
    lora_rank: int = 8,
    lora_alpha: int = 16,
    # Training
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 4096,
    # Harmony
    reasoning_effort: str | None = None,
    # W&B
    wandb_run_name: str | None = None,
    wandb_project: str = "cot-editing-exploration",
    # Options
    skip_merge: bool = False,
):
    """Run SFT on reward-hacking demonstrations.

    Args:
        dataset_file: Path to SFT dataset JSON (required).
        model_name: Base model to fine-tune.
        output_dir: Where to save LoRA adapter checkpoints.
        merged_output_dir: Where to save merged model. Default: {output_dir}/gpt-oss-20b-sft-merged.
            Must contain "gpt-oss" for GRPO compatibility.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        max_length: Maximum sequence length for training.
        reasoning_effort: Reasoning effort for Harmony system prompt ("low"/"medium"/"high").
        wandb_run_name: W&B run name for async review.
        wandb_project: W&B project name.
        skip_merge: Skip LoRA merge step after training.
    """
    if dataset_file is None:
        raise ValueError("--dataset_file is required")

    if merged_output_dir is None:
        merged_output_dir = os.path.join(output_dir, "gpt-oss-20b-sft-merged")

    # Set W&B project
    os.environ["WANDB_PROJECT"] = wandb_project

    # Load dataset
    dataset = prepare_sft_dataset(dataset_file, reasoning_effort=reasoning_effort)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with MXFP4 quantization (per official gpt-oss fine-tuning guide)
    print(f"Loading model {model_name} with MXFP4 quantization...")
    quantization_config = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )

    # Apply LoRA (per official gpt-oss fine-tuning guide):
    # - target_modules="all-linear" selects all linear layers as candidates
    # - target_parameters restricts to only MoE expert weight tensors within those layers
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        target_parameters=[
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Configure SFT training
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        weight_decay=0.1,
        adam_beta2=0.99,
        bf16=True,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        max_length=max_length,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        logging_steps=1,
        save_steps=50,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting SFT training: {len(dataset)} examples, {num_train_epochs} epochs")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"LoRA adapter saved to {output_dir}")

    if not skip_merge:
        # Merge LoRA weights into base model for standalone inference
        print(f"Merging LoRA adapter into base model...")
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype="auto",
            use_cache=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, output_dir)
        model = model.merge_and_unload()

        model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        print(f"Merged model saved to {merged_output_dir}")
    else:
        print("Skipping LoRA merge (--skip_merge)")


if __name__ == "__main__":
    fire.Fire(train)
