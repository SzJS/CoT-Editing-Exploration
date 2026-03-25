"""Re-quantize a bf16 merged model to MXFP4.

Usage:
    uv run scripts/requantize_mxfp4.py \
        --input=results/runs/sft_rh_v1/gpt-oss-20b-sft-merged \
        --output=results/runs/sft_rh_v1/gpt-oss-20b-sft-mxfp4
"""

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def requantize(
    input: str = "/workspace/results/runs/sft_rh_v1/gpt-oss-20b-sft-merged",
    output: str = "/workspace/results/runs/sft_rh_v1/gpt-oss-20b-sft-mxfp4",
):
    print(f"Loading bf16 model from {input} with MXFP4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        input,
        quantization_config=Mxfp4Config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Saving MXFP4 model to {output}...")
    model.save_pretrained(output)

    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(input)
    tokenizer.save_pretrained(output)

    print("Done!")


if __name__ == "__main__":
    fire.Fire(requantize)
