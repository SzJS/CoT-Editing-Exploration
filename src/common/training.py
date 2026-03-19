"""Shared GRPO training loop: unsloth + TRL."""

import os
import sys
from collections import defaultdict, deque

import torch
from accelerate.utils import gather_object
from unsloth import FastLanguageModel  # must import before trl (patches vLLM compat)
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template

from common.cot_editing import CotEditingStrategy, PrefillStrategy
from common.rewards import split_completion


class GRPOTrainerWithExtraCols(GRPOTrainer):
    """GRPOTrainer that logs extra columns to the wandb completions table.

    Supports two column sources:
    - **Dataset columns** (``extra_log_columns``): pulled from the input batch
      before reward evaluation. Injected into ``self._logs["rewards"]``.
    - **Reward metadata** (``_last_metadata``): set by reward functions after
      evaluation (e.g. 5-way classification ``category``). Stored separately
      in ``self._reward_metadata`` and merged into the wandb table only,
      avoiding TRL's ``print_prompt_completions_sample`` which applies
      ``:.2f`` formatting (incompatible with string values).
    """

    def __init__(self, *args, extra_log_columns: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_log_columns = extra_log_columns or []
        gen_bs = self.args.generation_batch_size
        self._reward_metadata = defaultdict(lambda: deque(maxlen=gen_bs))

    def _generate_and_score_completions(self, inputs):
        extra_data = {}
        for col in self._extra_log_columns:
            if inputs and col in inputs[0]:
                extra_data[col] = [x[col] for x in inputs]

        result = super()._generate_and_score_completions(inputs)

        # Inject extra dataset columns (numeric-safe, goes into rewards dict)
        for col, values in extra_data.items():
            self._logs["rewards"][col].extend(gather_object(values))

        # Collect reward-computed metadata (may contain strings like category).
        # Stored separately so print_prompt_completions_sample doesn't crash.
        for rf in self.reward_funcs:
            if hasattr(rf, "_last_metadata"):
                for col, values in rf._last_metadata.items():
                    self._reward_metadata[col].extend(gather_object(values))
                rf._last_metadata = {}

        # Split completions into reasoning/answer columns for wandb table.
        # Works for all formats: Harmony (full/stripped tokens), Qwen3 (think/prefill).
        # Note: self._logs["completion"] is already gathered by TRL's parent call,
        # so we extend directly without gather_object to avoid duplication.
        completions = self._logs.get("completion", [])
        if completions:
            reasoning_col = []
            answer_col = []
            for comp in completions:
                reasoning, answer = split_completion(comp)
                reasoning_col.append(reasoning)
                answer_col.append(answer)
            self._reward_metadata["reasoning"].extend(reasoning_col)
            self._reward_metadata["answer"].extend(answer_col)

        return result

    def log(self, logs, start_time=None):
        """Override to merge reward metadata into the wandb completions table.

        When ``_reward_metadata`` is non-empty, we suppress TRL's built-in
        completions logging (which can't handle string columns) and re-do it
        ourselves: rich-print with numeric-only rewards, then log the wandb
        table with both rewards and string metadata columns.
        """
        if not self._reward_metadata:
            super().log(logs, start_time)
            return

        # Temporarily suppress TRL's completions logging so we control it.
        orig_log_completions = self.log_completions
        self.log_completions = False
        super().log(logs, start_time)
        self.log_completions = orig_log_completions

        if not (self.accelerator.is_main_process and orig_log_completions):
            return

        # Rich-print with numeric rewards only (no string columns).
        try:
            from trl.trainer.utils import print_prompt_completions_sample
            from rich import get_console as _  # guard: rich available
            print_prompt_completions_sample(
                self._logs["prompt"],
                self._logs["completion"],
                self._logs["rewards"],
                self._logs["advantages"],
                self.state.global_step,
                self.num_completions_to_print,
            )
        except ImportError:
            pass

        # Wandb table with reward metadata columns merged in.
        if self.args.report_to and "wandb" in self.args.report_to:
            try:
                import wandb
                import pandas as pd
                if wandb.run is not None:
                    table = {
                        "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                        "prompt": self._logs["prompt"],
                        "completion": self._logs["completion"],
                        **self._logs["rewards"],
                        **self._reward_metadata,
                        "advantage": self._logs["advantages"],
                    }
                    df = pd.DataFrame(table)
                    if self.wandb_log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])
                    wandb.log({"completions": wandb.Table(dataframe=df)})
            except ImportError:
                pass


class GRPOTrainerWithCotEditing(GRPOTrainerWithExtraCols):
    """GRPOTrainer supporting CoT editing strategies during generation.

    - **Prefill**: modifies prompt text (appends after ``<think>\\n``) then generates
      normally. Single-phase, no post-generation editing.
    - **Two-phase** (insertion/resampling): generates normally, decides on edits,
      batch-regenerates continuations from edit points, assembles final completions.
    """

    def __init__(self, *args, cot_strategy: CotEditingStrategy | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cot_strategy = cot_strategy
        self._cot_edit_count = 0
        self._cot_total_count = 0
        gen_bs = self.args.generation_batch_size
        self._cot_original_completions = deque(maxlen=gen_bs)
        self._cot_prefill_texts = deque(maxlen=gen_bs)

    # ── Generation override ──

    def _generate_single_turn(self, prompts, images=None):
        if self._cot_strategy is None:
            return super()._generate_single_turn(prompts, images)

        if isinstance(self._cot_strategy, PrefillStrategy):
            return self._generate_with_prefill(prompts, images)

        if self._cot_strategy.needs_iterative:
            return self._generate_iterative(prompts, images)

        if self._cot_strategy.needs_two_phase:
            return self._generate_two_phase(prompts, images)

        return super()._generate_single_turn(prompts, images)

    def _generate_with_prefill(self, prompts, images=None):
        """Prefill path: modify prompt text before generation.

        Applies the chat template, inserts prefill text after ``<think>\\n``,
        then passes pre-templated strings to the parent's generation.
        Strings pass through ``maybe_apply_chat_template`` unchanged, so
        the prefill text is included in the prompt sent to vLLM.
        """
        # Apply chat template manually
        prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
            for p in prompts
        ]

        # Append prefill text after <think>\n
        edited = 0
        for i, pt in enumerate(prompts_text):
            modified, meta = self._cot_strategy.apply_to_prompt(pt)
            prompts_text[i] = modified
            if meta.get("cot_edited"):
                edited += 1
        self._cot_edit_count += edited
        self._cot_total_count += len(prompts_text)

        # Store prefill text for wandb completions table
        self._cot_prefill_texts.extend([self._cot_strategy.text] * len(prompts_text))

        # Pass pre-templated strings to parent — strings flow through
        # maybe_apply_chat_template unchanged (is_conversational=False)
        return super()._generate_single_turn(prompts_text, images)

    def _generate_two_phase(self, prompts, images=None):
        """Two-phase path: generate, decide on edits, batch-regenerate, assemble."""
        # Phase 1: normal generation
        prompt_ids, completion_ids, logprobs, fwd_kw = super()._generate_single_turn(
            prompts, images
        )

        # Decode completions and get chat-templated prompt text
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
            for p in prompts
        ]

        # Decide on each completion
        decisions = []
        regen_indices = []
        for i, (pt, ct) in enumerate(zip(prompts_text, completions_text)):
            d = self._cot_strategy.decide(pt, ct)
            d["original"] = ct
            decisions.append(d)
            if d["action"] != "none":
                regen_indices.append(i)

        self._cot_edit_count += len(regen_indices)
        self._cot_total_count += len(decisions)

        if not regen_indices:
            self._cot_original_completions.extend(["[UNCHANGED]"] * len(completions_text))
            return prompt_ids, completion_ids, logprobs, fwd_kw

        # Build extended prompts and compute per-prompt continuation budgets
        max_model_len = (self.max_prompt_length or 0) + self.max_completion_length
        extended_prompts = []
        per_prompt_max_tokens = []
        valid_regen_indices = []

        for i in regen_indices:
            prefix = decisions[i]["prefix"] + decisions[i].get("insert_text", "")
            extended = prompts_text[i] + prefix
            ext_tokens = len(self.processing_class.encode(extended, add_special_tokens=False))
            prefix_tokens = len(self.processing_class.encode(prefix, add_special_tokens=False))
            budget = max(64, self.max_completion_length - prefix_tokens)

            # Guard: skip edit if extended prompt + budget exceeds max_model_len
            if max_model_len and ext_tokens + budget > max_model_len:
                decisions[i]["action"] = "none"
                decisions[i]["original"] = completions_text[i]
                continue

            extended_prompts.append(extended)
            per_prompt_max_tokens.append(budget)
            valid_regen_indices.append(i)

        if not valid_regen_indices:
            self._cot_original_completions.extend(["[UNCHANGED]"] * len(completions_text))
            return prompt_ids, completion_ids, logprobs, fwd_kw

        # Batch-regenerate continuations (per-prompt max_tokens)
        continuations = self._generate_continuation_batch(
            extended_prompts, per_prompt_max_tokens
        )

        # Assemble final completions and re-tokenize
        for j, idx in enumerate(valid_regen_indices):
            final_text = self._cot_strategy.assemble(decisions[idx], continuations[j])
            new_tokens = self.processing_class.encode(final_text, add_special_tokens=False)
            # Truncate to budget (safety net)
            if len(new_tokens) > self.max_completion_length:
                new_tokens = new_tokens[: self.max_completion_length]
            completion_ids[idx] = new_tokens

        # Store original completions: [UNCHANGED] for unedited, full text for edited
        edited_set = set(valid_regen_indices)
        self._cot_original_completions.extend(
            ct if i in edited_set else "[UNCHANGED]"
            for i, ct in enumerate(completions_text)
        )

        # Force logprobs recomputation (text was edited)
        return prompt_ids, completion_ids, None, fwd_kw

    def _generate_iterative(self, prompts, images=None):
        """Iterative path: generate, detect keywords, redirect, loop until clean.

        Used by RedirectStrategy which needs multiple rounds of
        decide-regenerate until no keywords remain or max_iterations reached.
        """
        strategy = self._cot_strategy

        # Phase 1: normal generation
        prompt_ids, completion_ids, logprobs, fwd_kw = super()._generate_single_turn(
            prompts, images
        )

        # Decode completions and get chat-templated prompt text
        current_completions_text = list(self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        ))
        prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
            for p in prompts
        ]

        # Track which completions were ever edited
        ever_edited = set()
        original_texts = list(current_completions_text)

        # Iterative loop — only re-decide on active (previously matched) indices
        max_model_len = (self.max_prompt_length or 0) + self.max_completion_length
        active_indices = list(range(len(current_completions_text)))
        for iteration in range(strategy.max_iterations):
            # Decide only on active completions
            regen_indices = []
            decisions = [None] * len(current_completions_text)
            for i in active_indices:
                d = strategy.decide(prompts_text[i], current_completions_text[i])
                d["original"] = current_completions_text[i]
                decisions[i] = d
                if d["action"] != "none":
                    regen_indices.append(i)

            if not regen_indices:
                break

            # Build extended prompts and compute budgets
            extended_prompts = []
            per_prompt_max_tokens = []
            valid_regen_indices = []

            for i in regen_indices:
                prefix = decisions[i]["prefix"] + decisions[i].get("insert_text", "")
                extended = prompts_text[i] + prefix
                ext_tokens = len(self.processing_class.encode(extended, add_special_tokens=False))
                prefix_tokens = len(self.processing_class.encode(prefix, add_special_tokens=False))
                budget = max(64, self.max_completion_length - prefix_tokens)

                if max_model_len and ext_tokens + budget > max_model_len:
                    decisions[i]["action"] = "none"
                    decisions[i]["original"] = current_completions_text[i]
                    continue

                extended_prompts.append(extended)
                per_prompt_max_tokens.append(budget)
                valid_regen_indices.append(i)

            if not valid_regen_indices:
                break

            # Batch-regenerate continuations
            continuations = self._generate_continuation_batch(
                extended_prompts, per_prompt_max_tokens
            )

            # Assemble and update; narrow active set to regenerated indices
            for j, idx in enumerate(valid_regen_indices):
                current_completions_text[idx] = strategy.assemble(
                    decisions[idx], continuations[j]
                )
                ever_edited.add(idx)
            active_indices = list(valid_regen_indices)

        # Count unique completions edited (not per-iteration regenerations)
        self._cot_edit_count += len(ever_edited)
        self._cot_total_count += len(current_completions_text)

        # Re-tokenize any modified completions
        if ever_edited:
            for idx in ever_edited:
                new_tokens = self.processing_class.encode(
                    current_completions_text[idx], add_special_tokens=False
                )
                if len(new_tokens) > self.max_completion_length:
                    new_tokens = new_tokens[: self.max_completion_length]
                completion_ids[idx] = new_tokens

            # Store original completions for wandb
            self._cot_original_completions.extend(
                original_texts[i] if i in ever_edited else "[UNCHANGED]"
                for i in range(len(current_completions_text))
            )
            # Force logprobs recomputation
            return prompt_ids, completion_ids, None, fwd_kw

        self._cot_original_completions.extend(
            ["[UNCHANGED]"] * len(current_completions_text)
        )
        return prompt_ids, completion_ids, logprobs, fwd_kw

    def _generate_continuation_batch(
        self, extended_prompts: list[str], per_prompt_max_tokens: list[int]
    ) -> list[str]:
        """Generate continuations from extended prompts using vLLM colocate engine.

        Called during two-phase editing after phase 1 generation and decision.
        Uses the vLLM engine directly (unsloth forces colocate mode).

        Args:
            extended_prompts: Prompt + prefix + insert_text strings.
            per_prompt_max_tokens: Max new tokens for each prompt (avoids
                the longest prefix penalizing all regenerations).

        Returns list of decoded continuation strings.
        """
        from vllm import SamplingParams

        # Wake up vLLM if phase 1 put it to sleep
        if getattr(self.args, "vllm_enable_sleep_mode", False):
            torch.cuda.empty_cache()
            self.llm.wake_up()

        # Per-prompt SamplingParams so each gets its own max_tokens budget
        sampling_params_list = [
            SamplingParams(
                n=1,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if getattr(self, "min_p", None) is None else self.min_p,
                max_tokens=mt,
                repetition_penalty=getattr(self, "repetition_penalty", 1.0),
            )
            for mt in per_prompt_max_tokens
        ]

        lora_request = self.model.load_lora(
            "grpo_trainer_lora_model", load_tensors=True
        )
        outputs = self.llm.generate(
            extended_prompts,
            sampling_params=sampling_params_list,
            use_tqdm=False,
            lora_request=lora_request,
        )

        continuations = [output.outputs[0].text for output in outputs]

        # Put vLLM back to sleep if needed
        if getattr(self.args, "vllm_enable_sleep_mode", False):
            self.llm.sleep(level=1)

        return continuations

    # ── Metrics ──

    def log(self, logs, start_time=None):
        if self._cot_total_count > 0:
            logs["cot_editing/edited_ratio"] = self._cot_edit_count / self._cot_total_count
            logs["cot_editing/regen_count"] = self._cot_edit_count
        self._cot_edit_count = 0
        self._cot_total_count = 0

        # Inject original (pre-edit) completions into wandb table, but only
        # when the custom table path is already active (reward fns populated
        # _reward_metadata). Avoids accidentally activating the custom path.
        has_reward_meta = bool(self._reward_metadata)
        if self._cot_original_completions and has_reward_meta:
            self._reward_metadata["original_completion"].extend(self._cot_original_completions)
        self._cot_original_completions.clear()

        if self._cot_prefill_texts and has_reward_meta:
            self._reward_metadata["cot_prefill"].extend(self._cot_prefill_texts)
            # Prepend prefill text to reasoning column so it shows the full chain.
            # The last n entries in reasoning correspond to this batch's prefill texts.
            if "reasoning" in self._reward_metadata:
                reasoning_list = list(self._reward_metadata["reasoning"])
                prefill_list = list(self._cot_prefill_texts)
                n = min(len(prefill_list), len(reasoning_list))
                assert len(reasoning_list) >= n, (
                    f"reasoning/prefill deque size mismatch: {len(reasoning_list)} < {n}"
                )
                for i in range(n):
                    if prefill_list[i]:
                        reasoning_list[-(n - i)] = (
                            f"[PREFILL] {prefill_list[i]}\n\n{reasoning_list[-(n - i)]}"
                        )
                self._reward_metadata["reasoning"] = deque(
                    reasoning_list, maxlen=self._reward_metadata["reasoning"].maxlen
                )
        self._cot_prefill_texts.clear()

        super().log(logs, start_time)


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
    offload_embedding: bool = False,
    no_gradient_checkpointing: bool = False,
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
    # CoT editing strategy (None = no editing)
    cot_strategy: CotEditingStrategy | None = None,
    # Dataset ordering
    shuffle_dataset: bool = True,
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
        offload_embedding: Whether to offload embedding layer to CPU (for large
            vocab models like gpt-oss-120b with 201k vocab).
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
        extra_log_columns: Dataset columns to include in wandb completions table.
        cot_strategy: CoT editing strategy instance (None = no editing).
    """
    os.environ.setdefault("WANDB_PROJECT", wandb_project)

    is_gptoss = "gpt-oss" in model_name or "gpt_oss" in model_name

    # ── Resolve sampling defaults ──
    if is_gptoss:
        # OpenAI recommended: temperature=1.0, top_p=1.0, no top_k
        if temperature is None:
            temperature = 1.0
        if top_p is None:
            top_p = 1.0
        if top_k == 20:
            top_k = 0  # disable top_k for gpt-oss
    else:
        # Qwen3: https://huggingface.co/Qwen/Qwen3-4B#switching-between-thinking-and-non-thinking-mode
        if temperature is None:
            temperature = 0.7 if disable_thinking else 0.6
        if top_p is None:
            top_p = 0.8 if disable_thinking else 0.95
    print(f"Sampling: temperature={temperature}, top_p={top_p}, top_k={top_k}")

    # ── Model loading ──
    # Unsloth's vLLM colocate (fast_inference) extracts state dicts by traversing
    # the vLLM model's attributes (embed_tokens, self_attn, gate_up_proj, etc.).
    # gpt-oss uses entirely different names (embedding, attn, experts/router), so
    # colocate fails with AttributeError. Fall back to HF-native for all gpt-oss.
    use_fast_inference = not is_gptoss
    from_pretrained_kwargs = dict(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=use_fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if offload_embedding:
        from_pretrained_kwargs["offload_embedding"] = True
    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

    # ── Disable thinking mode (nothink baseline) ──
    # The official way is to pass enable_thinking=False to apply_chat_template(),
    # but TRL's GRPOTrainer._generate_single_turn() doesn't forward
    # chat_template_kwargs during generation (it creates a bare {"prompt": ...}
    # dict). So we patch the Jinja template to always take the nothink branch,
    # which produces identical output: an empty <think>\n\n</think>\n\n block
    # before the response.
    if disable_thinking and not is_gptoss:
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

    # gpt-oss MoE has fused gate_up_proj (not separate gate_proj/up_proj),
    # so the explicit list doesn't match. Use "all-linear" for gpt-oss.
    # For Qwen3, keep the explicit list per Unsloth's notebook.
    if is_gptoss:
        lora_target_modules = "all-linear"
    else:
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        use_gradient_checkpointing=False if no_gradient_checkpointing else "unsloth",
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
        shuffle_dataset=shuffle_dataset,
        gradient_checkpointing=not no_gradient_checkpointing,
        # When using unsloth's fast_inference, vLLM is colocated automatically.
        # When fast_inference=False (gpt-oss MoE), use TRL's native generation.
        use_vllm=False,
    )

    # ── Trainer ──
    trainer_cls = GRPOTrainerWithCotEditing if cot_strategy else GRPOTrainerWithExtraCols
    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        extra_log_columns=extra_log_columns,
    )
    if cot_strategy:
        trainer_kwargs["cot_strategy"] = cot_strategy
        print(f"CoT editing: strategy={cot_strategy.name}")
    trainer = trainer_cls(**trainer_kwargs)

    # Log extra config to wandb (after trainer init creates the run)
    wandb_config_extra = wandb_config_extra or {}
    if cot_strategy:
        wandb_config_extra["cot_strategy"] = cot_strategy.name
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
    # Strategy: find the function via compute_loss.__globals__ (most robust),
    # then fall back to sys.modules search.
    _gal_ref = None
    _gal_container = None  # dict where we need to write the patched version

    # Method 1: access via compute_loss method globals (same scope where it's called)
    _compute_loss = getattr(type(trainer), 'compute_loss', None)
    if _compute_loss and hasattr(_compute_loss, '__globals__'):
        _globals = _compute_loss.__globals__
        if 'grpo_accumulated_loss' in _globals:
            _gal_ref = _globals['grpo_accumulated_loss']
            _gal_container = _globals

    # Method 2: fall back to sys.modules search
    if _gal_ref is None:
        for name, mod in list(sys.modules.items()):
            if hasattr(mod, "grpo_accumulated_loss"):
                _gal_ref = mod.grpo_accumulated_loss
                _gal_container = mod.__dict__
                break

    if _gal_ref is not None:
        _orig_gal = _gal_ref

        def _patched_gal(*args, **kwargs):
            result = _orig_gal(*args, **kwargs)
            # Unsloth's chunked GRPO loss can return coef_1 and completion_mask
            # with mismatched dim 1 (padding/chunking artifacts). Align them.
            # Result layout: (loss, comp_len, mean_kl, delta, flat_is, coef_1[, mask])
            if len(result) >= 7:
                result = list(result)
                coef_1 = result[5]
                ret_mask = result[6]
                if (coef_1 is not None and hasattr(coef_1, 'dim') and coef_1.dim() == 2
                        and ret_mask is not None and hasattr(ret_mask, 'dim') and ret_mask.dim() == 2
                        and coef_1.shape[1] != ret_mask.shape[1]):
                    min_len = min(coef_1.shape[1], ret_mask.shape[1])
                    result[5] = coef_1[:, -min_len:]
                    result[6] = ret_mask[:, -min_len:]
                return tuple(result)
            elif len(result) >= 6:
                # Older unsloth without mask in return — align coef_1 to input mask
                completion_mask = kwargs.get("completion_mask")
                if completion_mask is not None:
                    result = list(result)
                    coef_1 = result[5]
                    if (coef_1 is not None and hasattr(coef_1, 'dim') and coef_1.dim() == 2
                            and coef_1.shape[1] != completion_mask.shape[1]):
                        min_len = min(coef_1.shape[1], completion_mask.shape[1])
                        result[5] = coef_1[:, -min_len:]
                    return tuple(result)
            return result

        _gal_container['grpo_accumulated_loss'] = _patched_gal
        print("Applied unsloth coef_1 padding fix")
    else:
        print("WARNING: Could not find grpo_accumulated_loss to patch")

    trainer.train()

    # Save final checkpoint
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"Training complete. Model saved to {output_dir}/final")
