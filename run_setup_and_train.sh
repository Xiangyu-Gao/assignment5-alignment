#!/usr/bin/env bash
set -euo pipefail

# Install uv

curl -LsSf https://astral.sh/uv/install.sh | sh

# Add $HOME/.local/bin to PATH

source $HOME/.local/bin/env

# ── Configuration ──────────────────────────────────────────────
REPO_URL="https://github.com/Xiangyu-Gao/assignment5-alignment.git"
CLONE_DIR="assignment5-alignment"

# ── 1. Clone the repository ───────────────────────────────────
if [ -d "$CLONE_DIR" ]; then
    echo "Directory '$CLONE_DIR' already exists, skipping clone."
else
    echo "Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$CLONE_DIR"
fi

cd "$CLONE_DIR"

# ── 2. Install dependencies (flash-attn requires two-pass install) ──
echo "Installing packages (pass 1: excluding flash-attn) ..."
uv sync --no-install-package flash-attn

echo "Installing packages (pass 2: including flash-attn) ..."
uv sync

# ── 3. Download the model ─────────────────────────────────────
echo "Downloading Qwen2.5-Math-1.5B model ..."
uv run python scripts/download_model.py \
    --repo-id Qwen/Qwen2.5-Math-1.5B \
    --save-dir ./data/models/Qwen2.5-Math-1.5B \
    --method snapshot \
    --no-symlinks \
    --verify

# ── 4. Set wandb credentials ──────────────────────────────────
export WANDB_API_KEY="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"
export WANDB_ENTITY="xygaoece"

# ── 5. Run GRPO training ──────────────────────────────────────
echo "Starting GRPO training ..."
uv run python train_grpo.py