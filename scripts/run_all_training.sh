#!/bin/bash
# Run all training configs sequentially on RunPod
#
# Usage:
#   bash scripts/run_all_training.sh
#
# Run a specific model only:
#   bash scripts/run_all_training.sh llama
#   bash scripts/run_all_training.sh qwen
#   bash scripts/run_all_training.sh phi

set -euo pipefail

PROJECT_DIR="/workspace/summarization"
cd "${PROJECT_DIR}"

run_training() {
    local name="$1"
    local config="$2"

    echo ""
    echo "============================================"
    echo "  Training: ${name}"
    echo "  Config:   ${config}"
    echo "============================================"
    echo ""

    start_time=$(date +%s)
    uv run oumi train -c "${config}"
    end_time=$(date +%s)

    elapsed=$(( end_time - start_time ))
    hours=$(( elapsed / 3600 ))
    minutes=$(( (elapsed % 3600) / 60 ))
    seconds=$(( elapsed % 60 ))
    echo ""
    echo "  Completed: ${name} in ${hours}h ${minutes}m ${seconds}s"
    echo ""
}

MODEL_FILTER="${1:-all}"

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "llama" ]; then
    run_training "Llama 3.1 8B (QLoRA)" "configs/oumi/train_sft_llama3_1_8b.yaml"
fi

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "qwen" ]; then
    run_training "Qwen 2.5 7B (QLoRA)" "configs/oumi/train_sft_qwen2_5_7b.yaml"
fi

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "phi" ]; then
    run_training "Phi-3 Mini (LoRA)" "configs/oumi/train_sft_phi_mini.yaml"
fi

echo ""
echo "============================================"
echo "  All training jobs complete!"
echo "============================================"
