#!/bin/bash
# Run ROUGE evaluation for fine-tuned models.
#
# The custom eval function must be imported before oumi runs, so we
# use python -c to import it first, then call oumi evaluate.
#
# Usage:
#   bash scripts/run_eval.sh llama
#   bash scripts/run_eval.sh qwen
#   bash scripts/run_eval.sh all

set -euo pipefail

PROJECT_DIR="/workspace/summarization"
cd "${PROJECT_DIR}"

# Ensure rouge-score is installed
uv pip install rouge-score 2>/dev/null || true

run_eval() {
    local name="$1"
    local config="$2"

    echo ""
    echo "============================================"
    echo "  Evaluating: ${name}"
    echo "  Config:     ${config}"
    echo "============================================"
    echo ""

    start_time=$(date +%s)

    # Import the custom eval function, then run oumi evaluate
    uv run python -c "
import clinicalnotes_1.eval_rouge  # registers rouge_summarization
from oumi.evaluate import evaluate
from oumi.core.configs import EvaluationConfig
config = EvaluationConfig.from_yaml('${config}')
evaluate(config)
"

    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    minutes=$(( elapsed / 60 ))
    seconds=$(( elapsed % 60 ))
    echo ""
    echo "  Completed: ${name} in ${minutes}m ${seconds}s"
    echo ""
}

MODEL_FILTER="${1:-all}"

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "llama" ]; then
    run_eval "Llama 3.1 8B" "configs/oumi/eval_llama3_1_8b.yaml"
fi

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "qwen" ]; then
    run_eval "Qwen 2.5 7B" "configs/oumi/eval_qwen2_5_7b.yaml"
fi

if [ "${MODEL_FILTER}" = "all" ] || [ "${MODEL_FILTER}" = "phi" ]; then
    run_eval "Phi-3 Mini" "configs/oumi/eval_phi_mini.yaml"
fi

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Results saved to outputs/eval-*/"
echo "============================================"
