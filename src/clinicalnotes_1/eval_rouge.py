"""Custom summarization evaluation function for Oumi.

Computes ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and BERTScore on generated summaries
vs reference summaries, mirroring the MIMIC-IV-BHC paper's evaluation protocol.

Reports mean, std, and bootstrap 95% CIs for each metric. Also saves per-example
results to a JSONL file for downstream analysis.

Usage:
    bash scripts/run_eval.sh llama
    bash scripts/run_eval.sh all
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from bert_score import score as bert_score
from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation, Message, Role
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def _load_conversations(
    dataset_path: str, num_samples: int | None
) -> tuple[list[str], list[Conversation]]:
    """Load JSONL file and split into references and input conversations."""
    references: list[str] = []
    input_conversations: list[Conversation] = []

    with Path(dataset_path).open() as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break

            record = json.loads(line)
            messages = record.get("messages", [])

            reference = ""
            input_messages: list[Message] = []
            for msg in messages:
                role_str = msg.get("role", "")
                content = msg.get("content", "")
                if role_str == "assistant":
                    reference = content
                else:
                    input_messages.append(Message(role=Role.USER, content=content))

            references.append(reference)
            input_conversations.append(Conversation(messages=input_messages))

    return references, input_conversations


def _bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0

    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return lo, hi


def _compute_rouge(
    references: list[str], generated: list[str]
) -> dict[str, list[float]]:
    """Compute per-example ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores: dict[str, list[float]] = {
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
    }
    for ref, gen in zip(references, generated):
        result = scorer.score(ref, gen)
        scores["rouge1_f"].append(result["rouge1"].fmeasure)
        scores["rouge2_f"].append(result["rouge2"].fmeasure)
        scores["rougeL_f"].append(result["rougeL"].fmeasure)
    return scores


def _compute_bleu(references: list[str], generated: list[str]) -> list[float]:
    """Compute per-example sentence-level BLEU scores."""
    bleu = BLEU(effective_order=True)
    scores: list[float] = []
    for ref, gen in zip(references, generated):
        result = bleu.sentence_score(gen, [ref])
        scores.append(result.score / 100.0)  # Normalize to 0-1
    return scores


def _compute_bertscore(
    references: list[str], generated: list[str]
) -> list[float]:
    """Compute per-example BERTScore F1."""
    _, _, f1 = bert_score(
        generated,
        references,
        lang="en",
        verbose=True,
        batch_size=16,
    )
    return f1.tolist()


@register_evaluation_function("summarization_metrics")
def summarization_metrics(
    task_params: EvaluationTaskParams,
    config: EvaluationConfig,
    inference_engine: BaseInferenceEngine,
    dataset_path: str = "",
) -> dict[str, Any]:
    """Full summarization evaluation: ROUGE, BLEU, and BERTScore.

    Mirrors the MIMIC-IV-BHC paper's quantitative evaluation protocol.
    Reports mean, std, and 95% bootstrap CIs for each metric.
    Saves per-example results to output_dir/per_example_results.jsonl.
    """
    if not dataset_path:
        raise ValueError(
            "eval_kwargs.dataset_path must be set to the path of the test JSONL file."
        )

    # Load data
    references, input_conversations = _load_conversations(
        dataset_path, task_params.num_samples
    )

    print(f"Running inference on {len(input_conversations)} examples...")
    output_conversations = inference_engine.infer(input_conversations)

    # Extract generated text
    generated: list[str] = []
    valid_refs: list[str] = []
    num_skipped = 0

    for ref, conv in zip(references, output_conversations):
        gen = conv.last_message().content or ""
        if not ref or not gen:
            num_skipped += 1
            continue
        generated.append(gen)
        valid_refs.append(ref)

    print(f"Computing ROUGE scores on {len(generated)} examples...")
    rouge_scores = _compute_rouge(valid_refs, generated)

    print("Computing BLEU scores...")
    bleu_scores = _compute_bleu(valid_refs, generated)

    print("Computing BERTScore (this may take a minute)...")
    bertscore_f1 = _compute_bertscore(valid_refs, generated)

    # Aggregate all per-example scores
    all_scores: dict[str, list[float]] = {
        **rouge_scores,
        "bleu": bleu_scores,
        "bertscore_f1": bertscore_f1,
    }

    # Compute summary statistics: mean, std, 95% CI
    metrics: dict[str, Any] = {}
    for key, vals in all_scores.items():
        arr = np.array(vals)
        ci_lo, ci_hi = _bootstrap_ci(vals)
        metrics[f"{key}_mean"] = float(arr.mean())
        metrics[f"{key}_std"] = float(arr.std())
        metrics[f"{key}_ci95_lo"] = ci_lo
        metrics[f"{key}_ci95_hi"] = ci_hi

    metrics["num_evaluated"] = float(len(generated))
    metrics["num_skipped"] = float(num_skipped)

    # Save per-example results for downstream analysis
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_example_path = output_dir / "per_example_results.jsonl"
    with per_example_path.open("w") as f:
        for i in range(len(generated)):
            record = {
                "reference": valid_refs[i],
                "generated": generated[i],
                "rouge1_f": rouge_scores["rouge1_f"][i],
                "rouge2_f": rouge_scores["rouge2_f"][i],
                "rougeL_f": rouge_scores["rougeL_f"][i],
                "bleu": bleu_scores[i],
                "bertscore_f1": bertscore_f1[i],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Per-example results saved to {per_example_path}")

    return metrics
