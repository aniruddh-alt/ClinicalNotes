#!/bin/bash
# RunPod Setup Script
# Sets up the project, downloads data, and runs all training configs.
#
# Usage:
#   bash scripts/setup_runpod.sh
#
# Prerequisites:
#   - RunPod instance with GPU (A40/A100 recommended)
#   - HuggingFace token with access to:
#       - meta-llama/Meta-Llama-3.1-8B-Instruct (gated model)
#       - aniruddhr04/mimic-iv-bhc-oncology (private dataset)
#   - Git SSH key or HTTPS credentials configured

set -euo pipefail

WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/summarization"
REPO_URL="https://github.com/aniruddh-alt/ClinicalNotes.git"
BRANCH="feature/finetuning"

echo "============================================"
echo "  RunPod Project Setup"
echo "============================================"
echo ""

# ─── Step 1: Clone the repository ───────────────────────────────────
echo "[1/6] Cloning repository..."
if [ -d "${PROJECT_DIR}" ]; then
    echo "  Project directory already exists. Pulling latest changes..."
    cd "${PROJECT_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    cd "${WORKSPACE}"
    git clone -b "${BRANCH}" "${REPO_URL}" summarization
    cd "${PROJECT_DIR}"
fi
echo "  Done."
echo ""

# ─── Step 2: Install uv and dependencies ────────────────────────────
echo "[2/6] Installing uv and project dependencies..."
if ! command -v uv &> /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install project with all extras
cd "${PROJECT_DIR}"
uv sync --all-extras
echo "  Done."
echo ""

# ─── Step 3: Install flash-attn + liger-kernel ──────────────────────
echo "[3/6] Installing flash-attn and liger-kernel..."
uv pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "  Warning: flash-attn installation failed."
    echo "  Llama config uses flash_attention_2 - training may fall back to sdpa."
}
uv pip install liger-kernel || {
    echo "  Warning: liger-kernel installation failed."
    echo "  Llama and Qwen configs use enable_liger_kernel."
}
echo "  Done."
echo ""

# ─── Step 4: HuggingFace login ──────────────────────────────────────
echo "[4/6] Hugging Face authentication..."
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  Using HF_TOKEN environment variable..."
    huggingface-cli login --token "${HF_TOKEN}"
else
    echo "  No HF_TOKEN found. Starting interactive login..."
    echo "  You need access to:"
    echo "    - meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "    - aniruddhr04/mimic-iv-bhc-oncology"
    huggingface-cli login
fi
echo "  Done."
echo ""

# ─── Step 5: Download datasets ──────────────────────────────────────
echo "[5/6] Downloading datasets from Hugging Face..."
cd "${PROJECT_DIR}"
uv run python scripts/download_data_hf.py
echo ""

# Verify datasets exist
echo "  Verifying dataset files..."
for split in train valid test; do
    file="${PROJECT_DIR}/MIMIC-IV-BHC/oncology-splits/${split}.jsonl"
    if [ -f "${file}" ]; then
        lines=$(wc -l < "${file}")
        size=$(du -h "${file}" | cut -f1)
        echo "    ${split}.jsonl: ${lines} examples (${size})"
    else
        echo "    ERROR: ${file} not found!"
        exit 1
    fi
done
echo "  Done."
echo ""

# ─── Step 6: Create output directories ──────────────────────────────
echo "[6/6] Creating output directories..."
mkdir -p "${PROJECT_DIR}/outputs/model-llama3_1-8b"
mkdir -p "${PROJECT_DIR}/outputs/model-qwen2_5-7b"
mkdir -p "${PROJECT_DIR}/outputs/model-phi-mini"
echo "  Done."
echo ""

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Training configs:"
echo ""
echo "  1. Llama 3.1 8B (QLoRA + Flash Attention 2):"
echo "     uv run oumi train -c configs/oumi/train_sft_llama3_1_8b.yaml"
echo ""
echo "  2. Qwen 2.5 7B (QLoRA):"
echo "     uv run oumi train -c configs/oumi/train_sft_qwen2_5_7b.yaml"
echo ""
echo "  3. Phi-3 Mini (LoRA):"
echo "     uv run oumi train -c configs/oumi/train_sft_phi_mini.yaml"
echo ""
echo "Run all sequentially:"
echo "  bash scripts/run_all_training.sh"
echo ""
