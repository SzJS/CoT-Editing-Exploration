"""Convert SFT merged model from HF native format to unsloth BnB 4-bit compatible format.

The SFT merge (via MXFP4+dequantize) produces a checkpoint with:
  - experts.gate_up_proj: fused tensor [num_experts, hidden_size, 2*expert_dim]
  - experts.down_proj: fused tensor [num_experts, expert_dim, hidden_size]
  - router.weight / router.bias: direct parameters

Unsloth's BnB 4-bit patch expects:
  - experts.gate_up_projs.{i}.weight: per-expert Linear [2*expert_dim, hidden_size]
  - experts.gate_up_projs.{i}.bias: per-expert Linear [2*expert_dim]
  - experts.down_projs.{i}.weight: per-expert Linear [hidden_size, expert_dim]
  - experts.down_projs.{i}.bias: per-expert Linear [hidden_size]
  - router.linear.weight / router.linear.bias

Usage:
    uv run scripts/convert_to_bnb4bit_format.py \
        --input=results/runs/sft_rh_v3/gpt-oss-20b-sft-merged \
        --output=results/runs/sft_rh_v3/gpt-oss-20b-sft-bnb4bit
"""

import json
import os
import shutil

import fire
import safetensors.torch
import torch


def convert(input: str, output: str):
    """Convert HF native gpt-oss checkpoint to unsloth BnB 4-bit compatible format."""

    os.makedirs(output, exist_ok=True)

    # Load the index
    index_path = os.path.join(input, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Figure out which shard files we need
    shard_files = sorted(set(weight_map.values()))
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Shard files: {shard_files}")
    print(f"Total keys: {len(weight_map)}")

    # Load all tensors
    all_tensors = {}
    for shard in shard_files:
        shard_path = os.path.join(input, shard)
        tensors = safetensors.torch.load_file(shard_path)
        all_tensors.update(tensors)
        print(f"  Loaded {shard}: {len(tensors)} tensors")

    # Convert
    new_tensors = {}
    converted_keys = set()

    for key, tensor in all_tensors.items():
        # Router: router.weight -> router.linear.weight
        if key.endswith(".mlp.router.weight"):
            new_key = key.replace(".mlp.router.weight", ".mlp.router.linear.weight")
            new_tensors[new_key] = tensor
            converted_keys.add(key)
            continue

        if key.endswith(".mlp.router.bias"):
            new_key = key.replace(".mlp.router.bias", ".mlp.router.linear.bias")
            new_tensors[new_key] = tensor
            converted_keys.add(key)
            continue

        # Experts: fused gate_up_proj [num_experts, hidden, 2*expert_dim] ->
        #   per-expert gate_up_projs.{i}.weight [2*expert_dim, hidden] (transposed for nn.Linear)
        if ".mlp.experts.gate_up_proj" == key.split("model.layers.")[-1].split(".", 1)[-1].rsplit(".", 0)[0] if False else key.endswith(".mlp.experts.gate_up_proj"):
            prefix = key[:-len("gate_up_proj")]
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                # nn.Linear stores weight as [out_features, in_features]
                # Fused tensor is [num_experts, hidden_size, 2*expert_dim]
                # So expert slice is [hidden_size, 2*expert_dim], need to transpose to [2*expert_dim, hidden_size]
                new_key = f"{prefix}gate_up_projs.{i}.weight"
                new_tensors[new_key] = tensor[i].T.contiguous()
            converted_keys.add(key)
            print(f"  Split {key}: [{tensor.shape}] -> {num_experts} per-expert weights")
            continue

        if key.endswith(".mlp.experts.gate_up_proj_bias"):
            prefix = key[:-len("gate_up_proj_bias")]
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                new_key = f"{prefix}gate_up_projs.{i}.bias"
                new_tensors[new_key] = tensor[i].contiguous()
            converted_keys.add(key)
            continue

        # Experts: fused down_proj [num_experts, expert_dim, hidden_size] ->
        #   per-expert down_projs.{i}.weight [hidden_size, expert_dim] (transposed for nn.Linear)
        if key.endswith(".mlp.experts.down_proj") and not key.endswith("_bias"):
            prefix = key[:-len("down_proj")]
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                # Fused tensor is [num_experts, expert_dim, hidden_size]
                # Need to transpose to [hidden_size, expert_dim] for nn.Linear
                new_key = f"{prefix}down_projs.{i}.weight"
                new_tensors[new_key] = tensor[i].T.contiguous()
            converted_keys.add(key)
            print(f"  Split {key}: [{tensor.shape}] -> {num_experts} per-expert weights")
            continue

        if key.endswith(".mlp.experts.down_proj_bias"):
            prefix = key[:-len("down_proj_bias")]
            num_experts = tensor.shape[0]
            for i in range(num_experts):
                new_key = f"{prefix}down_projs.{i}.bias"
                new_tensors[new_key] = tensor[i].contiguous()
            converted_keys.add(key)
            continue

        # Pass through everything else unchanged
        new_tensors[key] = tensor

    print(f"\nConverted {len(converted_keys)} keys")
    print(f"New total keys: {len(new_tensors)}")

    # Verify we have the expected structure for layer 0
    layer0_mlp = [k for k in sorted(new_tensors) if "layers.0.mlp" in k]
    print(f"\nLayer 0 MLP keys ({len(layer0_mlp)}):")
    for k in layer0_mlp:
        print(f"  {k}: {new_tensors[k].shape}")

    # Save as a single safetensors file (model is ~40GB bf16, fits in memory)
    output_file = os.path.join(output, "model.safetensors")
    print(f"\nSaving to {output_file}...")
    safetensors.torch.save_file(new_tensors, output_file)

    # Create index
    new_weight_map = {k: "model.safetensors" for k in new_tensors}
    new_index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in new_tensors.values())},
        "weight_map": new_weight_map,
    }
    index_output = os.path.join(output, "model.safetensors.index.json")
    with open(index_output, "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=True)

    # Copy config and tokenizer files
    for fname in os.listdir(input):
        if fname.endswith(".json") and fname != "model.safetensors.index.json":
            shutil.copy2(os.path.join(input, fname), os.path.join(output, fname))
        elif fname.endswith(".jinja") or fname == "tokenizer.model" or fname == "merges.txt" or fname == "vocab.json":
            shutil.copy2(os.path.join(input, fname), os.path.join(output, fname))

    # Remove any quantization_config from config.json (it's bf16, BnB will quantize on load)
    config_path = os.path.join(output, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    if "quantization_config" in config:
        del config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Removed quantization_config from config.json")

    print("\nDone!")


if __name__ == "__main__":
    fire.Fire(convert)
