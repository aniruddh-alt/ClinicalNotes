#!/usr/bin/env python3
"""Evaluate fine-tuned summarization models with ROUGE, BLEU, and BERTScore.

Downloads adapters from HuggingFace, merges into base models, runs inference
via vLLM for fast batch generation, and computes metrics mirroring the
MIMIC-IV-BHC paper's evaluation protocol.

Usage:
    # Evaluate all three models on 100 test samples
    python scripts/evaluate_models.py --num-samples 100

    # Evaluate a single model
    python scripts/evaluate_models.py --models llama --num-samples 20

    # Evaluate with all test samples
    python scripts/evaluate_models.py --num-samples 0

    # Fall back to native transformers if vLLM isn't available
    python scripts/evaluate_models.py --engine native --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from bert_score import score as compute_bert_score
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Model registry ──────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    base_model: str
    adapter_repo: str
    output_key: str


MODELS: dict[str, ModelSpec] = {
    "llama": ModelSpec(
        name="Llama 3.1 8B",
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        adapter_repo="aniruddhr04/clinicalnotes-llama3-8b-lora",
        output_key="llama3_1_8b",
    ),
    "qwen": ModelSpec(
        name="Qwen 2.5 7B",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        adapter_repo="aniruddhr04/clinicalnotes-qwen2-7b-lora",
        output_key="qwen2_5_7b",
    ),
    "phi": ModelSpec(
        name="Phi-3 Mini",
        base_model="microsoft/Phi-3-mini-4k-instruct",
        adapter_repo="aniruddhr04/clinicalnotes-phi3-mini-lora",
        output_key="phi3_mini",
    ),
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_test_data(
    path: str, num_samples: int = 0
) -> list[dict[str, str]]:
    """Load test JSONL and return list of {input, reference} dicts."""
    examples = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            messages = record.get("messages", [])
            user_input = ""
            reference = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_input = msg["content"]
                elif msg["role"] == "assistant":
                    reference = msg["content"]
            if user_input and reference:
                examples.append({"input": user_input, "reference": reference})

    if num_samples > 0:
        examples = examples[:num_samples]

    print(f"Loaded {len(examples)} test examples")
    return examples


# ── Model loading (merge adapter into base) ─────────────────────────────────

def merge_model(spec: ModelSpec, cache_dir: str) -> Path:
    """Download adapter from HF, merge into base, save to disk. Returns path."""
    merged_path = Path(cache_dir) / f"model-{spec.output_key}-merged"

    if merged_path.exists() and any(merged_path.glob("*.safetensors")):
        print(f"  Using cached merged model at {merged_path}")
        return merged_path

    print(f"  Downloading base model: {spec.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        spec.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Downloading adapter: {spec.adapter_repo}")
    model = PeftModel.from_pretrained(base_model, spec.adapter_repo)

    print("  Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_path))

    tokenizer = AutoTokenizer.from_pretrained(
        spec.adapter_repo, trust_remote_code=True
    )
    tokenizer.save_pretrained(str(merged_path))

    del model, base_model
    torch.cuda.empty_cache()

    return merged_path


# ── Inference engines ────────────────────────────────────────────────────────

def generate_vllm(
    model_path: Path,
    examples: list[dict[str, str]],
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate summaries using vLLM (fast, batched, paged attention)."""
    from vllm import LLM, SamplingParams

    print("  Loading model with vLLM...")
    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts using chat template
    prompts = []
    for ex in examples:
        messages = [{"role": "user", "content": ex["input"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,  # Greedy for reproducibility
    )

    print(f"  Generating {len(prompts)} summaries with vLLM...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    generated = [output.outputs[0].text.strip() for output in outputs]
    print(f"  vLLM inference complete: {len(generated)} examples in {elapsed:.1f}s "
          f"({elapsed/len(generated):.1f}s/example)")

    # Free GPU memory
    del llm
    torch.cuda.empty_cache()

    return generated


def generate_native(
    model_path: Path,
    examples: list[dict[str, str]],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> list[str]:
    """Generate summaries using native transformers (fallback)."""
    print("  Loading model with transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    generated = []
    total = len(examples)
    start = time.time()

    for i in range(0, total, batch_size):
        batch = examples[i : i + batch_size]
        batch_inputs = []

        for ex in batch:
            messages = [{"role": "user", "content": ex["input"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_inputs.append(text)

        encodings = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3584,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = encodings["input_ids"][j].shape[0]
            new_tokens = output[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated.append(text.strip())

        done = min(i + batch_size, total)
        elapsed = time.time() - start
        rate = elapsed / done if done > 0 else 0
        eta = rate * (total - done)
        print(f"  Generated {done}/{total} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", end="\r")

    elapsed = time.time() - start
    print(f"\n  Native inference complete: {total} examples in {elapsed:.1f}s "
          f"({elapsed/total:.1f}s/example)")

    del model
    torch.cuda.empty_cache()

    return generated


# ── Metrics ──────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    """Compute bootstrap 95% CI for the mean."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_bootstrap)
    )
    alpha = (1 - ci) / 2
    return means[int(alpha * n_bootstrap)], means[int((1 - alpha) * n_bootstrap)]


def compute_all_metrics(
    references: list[str], generated: list[str]
) -> tuple[dict, dict]:
    """Compute ROUGE, BLEU, and BERTScore with CIs."""
    print("  Computing ROUGE...")
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge1_f, rouge2_f, rougeL_f = [], [], []
    for ref, gen in zip(references, generated):
        r = scorer.score(ref, gen)
        rouge1_f.append(r["rouge1"].fmeasure)
        rouge2_f.append(r["rouge2"].fmeasure)
        rougeL_f.append(r["rougeL"].fmeasure)

    print("  Computing BLEU...")
    bleu = BLEU(effective_order=True)
    bleu_scores = [
        bleu.sentence_score(gen, [ref]).score / 100.0
        for ref, gen in zip(references, generated)
    ]

    print("  Computing BERTScore (may take a minute)...")
    _, _, bert_f1 = compute_bert_score(
        generated, references, lang="en", verbose=False, batch_size=32
    )
    bert_f1 = bert_f1.tolist()

    # Aggregate with CIs
    metrics: dict[str, dict] = {}
    per_example: dict[str, list[float]] = {}
    for name, vals in [
        ("rouge1_f", rouge1_f),
        ("rouge2_f", rouge2_f),
        ("rougeL_f", rougeL_f),
        ("bleu", bleu_scores),
        ("bertscore_f1", bert_f1),
    ]:
        arr = np.array(vals)
        ci_lo, ci_hi = bootstrap_ci(vals)
        metrics[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95": [ci_lo, ci_hi],
        }
        per_example[name] = vals

    return metrics, per_example


# ── Main ─────────────────────────────────────────────────────────────────────

def print_results_table(all_results: dict[str, dict]):
    """Print a comparison table of all models."""
    print(f"\n{'='*80}")
    print("  RESULTS SUMMARY")
    print(f"{'='*80}")

    header = f"{'Metric':<20}"
    for model_name in all_results:
        header += f"  {model_name:<22}"
    print(header)
    print("-" * 80)

    for metric in ["rouge1_f", "rouge2_f", "rougeL_f", "bleu", "bertscore_f1"]:
        row = f"{metric:<20}"
        for model_name, result in all_results.items():
            m = result["metrics"][metric]
            row += f"  {m['mean']:.4f} ± {m['std']:.4f}     "
        print(row)

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned summarization models"
    )
    parser.add_argument(
        "--test-data",
        default="/workspace/summarization/MIMIC-IV-BHC/oncology-splits/test.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test samples (0 = all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/summarization/outputs/eval-comparison",
        help="Directory to save results",
    )
    parser.add_argument(
        "--engine",
        choices=["vllm", "native"],
        default="vllm",
        help="Inference engine: vllm (fast) or native (fallback)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for native inference (ignored for vllm)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per summary",
    )
    parser.add_argument(
        "--cache-dir",
        default="/workspace/summarization/outputs",
        help="Directory for cached merged models",
    )
    args = parser.parse_args()

    # Resolve model list
    if "all" in args.models:
        model_keys = list(MODELS.keys())
    else:
        model_keys = args.models

    # Load test data
    examples = load_test_data(args.test_data, args.num_samples)
    references = [ex["reference"] for ex in examples]

    # Evaluate each model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for key in model_keys:
        spec = MODELS[key]
        print(f"\n{'#'*60}")
        print(f"  EVALUATING: {spec.name}")
        print(f"{'#'*60}")

        # Merge model (cached if already done)
        merged_path = merge_model(spec, args.cache_dir)

        # Generate summaries
        if args.engine == "vllm":
            generated = generate_vllm(
                merged_path, examples,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            generated = generate_native(
                merged_path, examples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )

        # Compute metrics
        metrics, per_example = compute_all_metrics(references, generated)

        # Save per-example results
        per_example_path = output_dir / f"{spec.output_key}_per_example.jsonl"
        with per_example_path.open("w") as f:
            for i in range(len(generated)):
                record = {
                    "reference": references[i],
                    "generated": generated[i],
                    **{k: v[i] for k, v in per_example.items()},
                }
                f.write(json.dumps(record) + "\n")

        all_results[spec.name] = {"metrics": metrics}

        print(f"\n  {spec.name} results:")
        for metric, vals in metrics.items():
            print(f"    {metric}: {vals['mean']:.4f} ± {vals['std']:.4f} "
                  f"(95% CI: [{vals['ci95'][0]:.4f}, {vals['ci95'][1]:.4f}])")

    # Save aggregate results
    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    if len(all_results) > 1:
        print_results_table(all_results)


if __name__ == "__main__":
    main()
