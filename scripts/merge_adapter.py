#!/usr/bin/env python3
"""Merge a LoRA/QLoRA adapter into the base model for fast inference.

Usage:
    python scripts/merge_adapter.py \
        --base meta-llama/Meta-Llama-3.1-8B-Instruct \
        --adapter /workspace/summarization/outputs/model-llama3_1-8b \
        --output /workspace/summarization/outputs/model-llama3_1-8b-merged

    python scripts/merge_adapter.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --adapter /workspace/summarization/outputs/model-qwen2_5-7b \
        --output /workspace/summarization/outputs/model-qwen2_5-7b-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to save GPU memory
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    merged.save_pretrained(args.output)

    # Save tokenizer too
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print("Done!")


if __name__ == "__main__":
    main()
