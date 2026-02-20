#!/usr/bin/env bash
set -euo pipefail

# Runpod H100 SXM setup script
# Usage: bash setup.sh

echo "=== CoT Editing Exploration - Runpod Setup ==="

# Parallel compilation jobs (H100 SXM has many cores)
export MAX_JOBS=8

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/0.6.6/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Initialize git submodules (datasets)
echo "Initializing submodules..."
git submodule update --init --recursive

# Install project dependencies
echo "Installing dependencies with uv..."
uv sync

# wandb setup
echo ""
echo "=== wandb setup ==="
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Set WANDB_API_KEY env var or run: uv run wandb login"
else
    echo "WANDB_API_KEY detected"
fi

echo ""
echo "=== Setup complete ==="
echo "Run training with: uv run python -m cot_editing.train"
