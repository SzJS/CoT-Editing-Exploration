"""Merge SFT LoRA checkpoint into base model (bf16 output).

Loads base model with Mxfp4Config(dequantize=True) to match LoRA adapter parameter names,
merges LoRA weights, and saves the result in bf16.

Usage:
    uv run scripts/merge_sft_checkpoint.py \
        --base_model=openai/gpt-oss-20b \
        --adapter_dir=results/runs/sft_rh_v1/checkpoint-25 \
        --output_dir=results/runs/sft_rh_v1/checkpoint-25-merged
"""

import gc
import os

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Mxfp4Config


def merge(
    base_model: str = "openai/gpt-oss-20b",
    adapter_dir: str = "",
    output_dir: str = "",
):
    assert adapter_dir, "Must specify --adapter_dir"
    assert output_dir, "Must specify --output_dir"

    os.environ["HF_HOME"] = "/workspace/hf_cache"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_dir}")
    print(f"Output: {output_dir}")

    # Load base with Mxfp4Config(dequantize=True) to match adapter parameter names
    print("Loading base model with Mxfp4Config(dequantize=True)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Validate LoRA loaded
    lora_layers = [n for n, _ in model.named_parameters() if "lora_" in n]
    if not lora_layers:
        raise RuntimeError(
            "No LoRA parameters found after loading adapter — merge would produce unmodified base model"
        )
    print(f"  Found {len(lora_layers)} LoRA parameter tensors to merge")

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done!")


if __name__ == "__main__":
    fire.Fire(merge)
