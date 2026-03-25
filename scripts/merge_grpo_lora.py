"""Merge GRPO LoRA adapter into base model and requantize to MXFP4.

Usage:
    uv run scripts/merge_grpo_lora.py \
        --base_model=results/runs/sft_synthetic_v1/gpt-oss-20b-sft-mxfp4 \
        --adapter_dir=results/runs/some_rl_run/final \
        --output_dir=results/runs/some_rl_run \
        --dequantize_mxfp4
"""

import gc
import os

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from accelerate import infer_auto_device_map


def merge(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    dequantize_mxfp4: bool = False,
):
    """Merge LoRA adapter into base model and requantize to MXFP4.

    Args:
        dequantize_mxfp4: If True, load base_model as MXFP4 and dequantize to bf16.
            Use this when the base model is in MXFP4 format (HF-native key names)
            instead of BnB per-expert format.
    """
    merged_dir = os.path.join(output_dir, "merged")
    mxfp4_dir = os.path.join(output_dir, "mxfp4")

    print(f"Base model: {base_model}")
    print(f"Adapter: {adapter_dir}")
    print(f"Merged output: {merged_dir}")
    print(f"MXFP4 output: {mxfp4_dir}")
    print(f"Dequantize MXFP4: {dequantize_mxfp4}")

    # Step 1: Load base model in bf16
    print("\nLoading base model in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    load_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto",
    )
    if dequantize_mxfp4:
        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    # Step 2: Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Validate LoRA loaded
    lora_layers = [n for n, _ in model.named_parameters() if "lora_" in n]
    if not lora_layers:
        raise RuntimeError(
            "No LoRA parameters found after loading adapter — merge would produce unmodified base model"
        )
    print(f"  Found {len(lora_layers)} LoRA parameter tensors to merge")

    # Step 3: Merge and save
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")

    # Step 4: Requantize to MXFP4
    print("\nRe-quantizing to MXFP4...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    mxfp4_model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        quantization_config=Mxfp4Config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    os.makedirs(mxfp4_dir, exist_ok=True)
    mxfp4_model.save_pretrained(mxfp4_dir)
    tokenizer.save_pretrained(mxfp4_dir)
    print(f"MXFP4 model saved to {mxfp4_dir}")
    print("\nDone!")


if __name__ == "__main__":
    fire.Fire(merge)
