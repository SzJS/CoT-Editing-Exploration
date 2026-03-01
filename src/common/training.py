"""Shared GRPO training loop: unsloth + TRL."""

import os
import sys

from accelerate.utils import gather_object
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


class GRPOTrainerWithExtraCols(GRPOTrainer):
    """GRPOTrainer that logs extra dataset columns to the wandb completions table.

    Extra columns are injected into ``self._logs["rewards"]``, which TRL
    automatically spreads into the wandb table via ``**self._logs["rewards"]``.
    """

    def __init__(self, *args, extra_log_columns: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_log_columns = extra_log_columns or []

    def _generate_and_score_completions(self, inputs):
        extra_data = {}
        for col in self._extra_log_columns:
            if inputs and col in inputs[0]:
                extra_data[col] = [x[col] for x in inputs]

        result = super()._generate_and_score_completions(inputs)

        for col, values in extra_data.items():
            self._logs["rewards"][col].extend(gather_object(values))

        return result


def run_grpo(
    *,
    train_dataset,
    reward_fn,
    max_prompt_length: int,
    # Model
    model_name: str = "Qwen/Qwen3-4B",
    output_dir: str = "results/runs/grpo_rh",
    max_seq_length: int = 12288,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    load_in_4bit: bool = False,
    gpu_memory_utilization: float = 0.6,
    # GRPOConfig
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
    # Thinking mode
    disable_thinking: bool = False,
    # wandb
    wandb_project: str = "cot-editing-exploration",
    wandb_run_name: str | None = None,
    wandb_config_extra: dict | None = None,
    # Extra dataset columns to log in wandb completions table
    extra_log_columns: list[str] | None = None,
):
    """Run GRPO training with the given dataset and reward function.

    Loads a model with unsloth FastLanguageModel, applies LoRA, configures
    GRPOTrainer, patches the unsloth coef_1 padding bug, trains, and saves
    the final checkpoint.

    Args:
        train_dataset: HuggingFace Dataset formatted for TRL GRPOTrainer
            (must have a 'prompt' column with chat messages).
        reward_fn: Callable reward function compatible with TRL's GRPOTrainer.
        max_prompt_length: Maximum prompt token length for GRPOConfig.
        model_name: HuggingFace model identifier.
        output_dir: Directory for checkpoints and final model.
        max_seq_length: Maximum sequence length (prompt + completion).
        lora_rank: LoRA rank (r parameter).
        lora_alpha: LoRA alpha scaling factor.
        load_in_4bit: Whether to load model in 4-bit quantization.
        gpu_memory_utilization: Fraction of GPU memory for vLLM inference.
        learning_rate: Peak learning rate.
        per_device_train_batch_size: Batch size per GPU.
        num_generations: Number of completions to sample per prompt.
        gradient_accumulation_steps: Steps to accumulate before optimizer update.
        max_completion_length: Maximum tokens for generated completions.
        max_steps: Total training steps.
        beta: KL penalty coefficient for GRPO.
        temperature: Sampling temperature. None = auto per Qwen3 docs
            (think: 0.6, nothink: 0.7).
        top_p: Nucleus sampling threshold. None = auto per Qwen3 docs
            (think: 0.95, nothink: 0.8).
        top_k: Top-k sampling parameter.
        weight_decay: AdamW weight decay.
        adam_beta2: AdamW beta2.
        lr_scheduler_type: Learning rate schedule type (e.g. "cosine", "linear").
        warmup_steps: Number of warmup steps for the LR scheduler.
        save_steps: Save checkpoint every N steps.
        logging_steps: Log metrics every N steps.
        disable_thinking: If True, patch chat template to disable Qwen3 thinking mode.
        wandb_project: Weights & Biases project name.
        wandb_run_name: Optional W&B run name (auto-generated if None).
        wandb_config_extra: Extra key-value pairs to log to W&B config
            (e.g. hint type, eval order).
    """
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    # ── Resolve sampling defaults per Qwen3 official recommendations ──
    # https://huggingface.co/Qwen/Qwen3-4B#switching-between-thinking-and-non-thinking-mode
    if temperature is None:
        temperature = 0.7 if disable_thinking else 0.6
    if top_p is None:
        top_p = 0.8 if disable_thinking else 0.95
    print(f"Sampling: temperature={temperature}, top_p={top_p}, top_k={top_k}")

    # ── Model loading ──
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # ── Disable thinking mode (nothink baseline) ──
    # The official way is to pass enable_thinking=False to apply_chat_template(),
    # but TRL's GRPOTrainer._generate_single_turn() doesn't forward
    # chat_template_kwargs during generation (it creates a bare {"prompt": ...}
    # dict). So we patch the Jinja template to always take the nothink branch,
    # which produces identical output: an empty <think>\n\n</think>\n\n block
    # before the response.
    if disable_thinking:
        original_template = tokenizer.chat_template
        tokenizer.chat_template = original_template.replace(
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "{%- if true %}",
        )
        if tokenizer.chat_template == original_template:
            raise ValueError(
                "Failed to patch chat_template: target string not found. "
                "Is this a Qwen3 model with enable_thinking support?"
            )
        print("Thinking mode DISABLED (nothink baseline)")

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

    # ── Training config ──
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        optim="paged_adamw_8bit",
        weight_decay=weight_decay,
        adam_beta2=adam_beta2,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        beta=beta,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        report_to="wandb",
        log_completions=True,
        bf16=True,
        run_name=wandb_run_name,
        mask_truncated_completions=True,
        # Do NOT set use_vllm=True when using unsloth's fast_inference
    )

    # ── Trainer ──
    trainer = GRPOTrainerWithExtraCols(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        extra_log_columns=extra_log_columns,
    )

    # Log extra config to wandb (after trainer init creates the run)
    if wandb_config_extra:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(wandb_config_extra, allow_val_change=True)
        except ImportError:
            pass

    # ── Patch unsloth bug: coef_1 left-padding dimension mismatch ──
    # unsloth's grpo_accumulated_loss internally widens coef_1 by max_left_pad
    # (for prompt padding alignment), but compute_loss uses the original
    # completion_mask which doesn't have the padding columns → RuntimeError.
    # Fix: patch grpo_accumulated_loss to truncate coef_1 before returning.
    _trainer_module = type(trainer).__module__
    _mod = sys.modules.get(_trainer_module)
    if _mod is None:
        for name, mod in list(sys.modules.items()):
            if hasattr(mod, "grpo_accumulated_loss"):
                _mod = mod
                break
    if _mod and hasattr(_mod, "grpo_accumulated_loss"):
        _orig_gal = _mod.grpo_accumulated_loss

        def _patched_gal(*args, **kwargs):
            result = _orig_gal(*args, **kwargs)
            # coef_1 is only used for clip-ratio metrics, not loss/gradients
            loss, comp_len, mean_kl, delta, flat_is, coef_1 = result
            completion_mask = kwargs.get("completion_mask")
            if (completion_mask is not None and coef_1 is not None
                    and coef_1.dim() == 2
                    and coef_1.shape[1] > completion_mask.shape[1]):
                coef_1 = coef_1[:, -completion_mask.shape[1]:]
            return loss, comp_len, mean_kl, delta, flat_is, coef_1

        _mod.grpo_accumulated_loss = _patched_gal
        print("Applied unsloth coef_1 padding fix")
    else:
        print("WARNING: Could not find grpo_accumulated_loss to patch")

    trainer.train()

    # Save final checkpoint
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"Training complete. Model saved to {output_dir}/final")
