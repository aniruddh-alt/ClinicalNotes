#!/bin/bash
# Smoke test: catches runtime errors BEFORE committing to full training runs.
#
# Checks:
#   1. GPU / CUDA availability
#   2. Dataset files exist and are valid JSONL
#   3. Python imports (oumi, torch, bitsandbytes, flash-attn)
#   4. HuggingFace model access (gated model auth)
#   5. 1-step training dry run for each config (catches real config/model errors)
#
# Usage:
#   bash scripts/smoke_test.sh

set -euo pipefail

PROJECT_DIR="/workspace/summarization"
cd "${PROJECT_DIR}"

PASS=0
FAIL=0
WARNINGS=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  WARN: $1"; WARNINGS=$((WARNINGS + 1)); }

echo "============================================"
echo "  Smoke Test"
echo "============================================"
echo ""

# ─── 1. GPU / CUDA ──────────────────────────────────────────────────
echo "[1/5] Checking GPU / CUDA..."
if uv run python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" 2>/dev/null; then
    pass "CUDA available"
else
    fail "CUDA not available - training will not work"
fi
echo ""

# ─── 2. Dataset files ───────────────────────────────────────────────
echo "[2/5] Checking dataset files..."
all_data_ok=true
for split in train valid test; do
    file="${PROJECT_DIR}/MIMIC-IV-BHC/oncology-splits/${split}.jsonl"
    if [ -f "${file}" ]; then
        lines=$(wc -l < "${file}")
        # Check first line is valid JSON
        if uv run python -c "import json; json.loads(open('${file}').readline())" 2>/dev/null; then
            pass "${split}.jsonl exists (${lines} lines, valid JSONL)"
        else
            fail "${split}.jsonl exists but is not valid JSONL"
            all_data_ok=false
        fi
    else
        fail "${split}.jsonl not found at ${file}"
        all_data_ok=false
    fi
done

# Check that train.jsonl has the expected SFT format (messages field)
if [ "${all_data_ok}" = true ]; then
    if uv run python -c "
import json
with open('${PROJECT_DIR}/MIMIC-IV-BHC/oncology-splits/train.jsonl') as f:
    row = json.loads(f.readline())
assert 'messages' in row, f'Missing messages field. Keys: {list(row.keys())}'
assert isinstance(row['messages'], list), 'messages is not a list'
assert len(row['messages']) >= 2, f'Expected >= 2 messages, got {len(row[\"messages\"])}'
print(f'  Format: {len(row[\"messages\"])} messages per example (roles: {[m[\"role\"] for m in row[\"messages\"]]})')
" 2>&1; then
        pass "SFT format (messages field) is correct"
    else
        fail "train.jsonl does not have expected SFT format"
    fi
fi
echo ""

# ─── 3. Python imports ──────────────────────────────────────────────
echo "[3/5] Checking Python dependencies..."
for pkg in "oumi" "torch" "transformers" "peft" "trl" "datasets"; do
    if uv run python -c "import ${pkg}" 2>/dev/null; then
        pass "${pkg}"
    else
        fail "${pkg} not importable"
    fi
done

# bitsandbytes (needed for QLoRA)
if uv run python -c "import bitsandbytes" 2>/dev/null; then
    pass "bitsandbytes (QLoRA)"
else
    fail "bitsandbytes not importable - QLoRA configs will fail"
fi

# flash-attn (needed for Llama config)
if uv run python -c "import flash_attn" 2>/dev/null; then
    pass "flash-attn"
else
    warn "flash-attn not installed - Llama config may fall back to sdpa"
fi

# liger-kernel (needed for Llama + Qwen configs)
if uv run python -c "import liger_kernel" 2>/dev/null; then
    pass "liger-kernel"
else
    fail "liger-kernel not importable - Llama and Qwen configs will fail (pip install liger-kernel)"
fi
echo ""

# ─── 4. HuggingFace model access ────────────────────────────────────
echo "[4/5] Checking HuggingFace authentication..."
if uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
user = api.whoami()
print(f'  Logged in as: {user[\"name\"]}')
" 2>/dev/null; then
    pass "HuggingFace authentication"
else
    fail "Not logged into HuggingFace - run: huggingface-cli login"
fi

# Check access to gated Llama model
if uv run python -c "
from huggingface_hub import model_info
info = model_info('meta-llama/Meta-Llama-3.1-8B-Instruct')
print(f'  Llama 3.1 8B access: OK')
" 2>/dev/null; then
    pass "Llama 3.1 8B model access"
else
    fail "No access to meta-llama/Meta-Llama-3.1-8B-Instruct (request access on HF)"
fi
echo ""

# ─── 5. 1-step training dry run ─────────────────────────────────────
echo "[5/5] Running 1-step training dry run for each config..."
echo "  (This downloads model weights on first run - may take a few minutes)"
echo ""

CONFIGS=(
    "Llama 3.1 8B:configs/oumi/train_sft_llama3_1_8b.yaml"
    "Qwen 2.5 7B:configs/oumi/train_sft_qwen2_5_7b.yaml"
    "Phi-3 Mini:configs/oumi/train_sft_phi_mini.yaml"
)

for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    config="${entry##*:}"
    echo "  Testing: ${name} (${config})..."

    # Override to run only 1 step with minimal resources
    smoke_log="/tmp/smoke_test_${name// /_}.log"
    if uv run oumi train -c "${config}" \
        --training.max_steps=1 \
        --training.save_steps=0 \
        --training.logging_steps=1 \
        --training.num_train_epochs=1 \
        --training.output_dir="/tmp/smoke_test_${name// /_}" \
        > "${smoke_log}" 2>&1; then
        pass "${name} - training starts successfully"
    else
        fail "${name} - training failed to start"
        echo "  --- Error log (last 30 lines) ---"
        tail -30 "${smoke_log}"
        echo "  --- End error log ---"
    fi
    echo ""

    # Clean up smoke test output
    rm -rf "/tmp/smoke_test_${name// /_}"
done

# ─── Summary ────────────────────────────────────────────────────────
echo "============================================"
echo "  Smoke Test Results"
echo "============================================"
echo "  PASS:     ${PASS}"
echo "  FAIL:     ${FAIL}"
echo "  WARNINGS: ${WARNINGS}"
echo ""

if [ "${FAIL}" -gt 0 ]; then
    echo "  Some checks FAILED. Fix the issues above before running training."
    exit 1
else
    echo "  All checks passed! Safe to run training."
    echo ""
    echo "  Next step:"
    echo "    bash scripts/run_all_training.sh"
fi
