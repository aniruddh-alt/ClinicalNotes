"""Custom ROUGE evaluation function for Oumi.

Registers a custom evaluation that runs inference on the test set
and computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores against
reference summaries.

Usage:
    oumi evaluate -c configs/oumi/eval_llama3_1_8b.yaml
"""

from __future__ import annotations

from typing import Any

from oumi.builders import build_dataset
from oumi.core.configs import EvaluationConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation, Role
from rouge_score import rouge_scorer


@register_evaluation_function("rouge_summarization")
def rouge_summarization(
    task_params: EvaluationTaskParams,
    config: EvaluationConfig,
    inference_engine: BaseInferenceEngine,
    dataset_path: str = "",
) -> dict[str, Any]:
    """Evaluate summarization quality using ROUGE scores.

    The dataset is loaded from the path specified in eval_kwargs.dataset_path.
    It expects text_sft format with user/assistant message pairs where the
    assistant message is the reference summary. We strip it before inference,
    generate a new summary, and compare with ROUGE.
    """
    if not dataset_path:
        raise ValueError(
            "eval_kwargs.dataset_path must be set to the path of the test JSONL file."
        )

    dataset = build_dataset(
        dataset_name="text_sft",
        tokenizer=None,
        dataset_path=dataset_path,
    )

    num_samples = task_params.num_samples
    if num_samples is None:
        num_samples = len(dataset)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Extract reference summaries and build input-only conversations
    references: list[str] = []
    input_conversations: list[Conversation] = []

    for i in range(num_samples):
        conversation = dataset.conversation(i)

        # Find the reference assistant message and keep only user messages
        reference = ""
        input_messages = []
        for message in conversation.messages:
            if message.role == Role.ASSISTANT:
                reference = message.content or ""
            else:
                input_messages.append(message)

        references.append(reference)
        input_conversations.append(
            Conversation(messages=input_messages, metadata=conversation.metadata)
        )

    # Run inference — model generates new assistant messages
    output_conversations = inference_engine.infer(input_conversations)

    # Score generated vs reference summaries
    scores: dict[str, list[float]] = {
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
        "rouge1_p": [],
        "rouge2_p": [],
        "rougeL_p": [],
        "rouge1_r": [],
        "rouge2_r": [],
        "rougeL_r": [],
    }

    num_skipped = 0
    for reference, conversation in zip(references, output_conversations):
        generated = conversation.last_message().content or ""

        if not reference or not generated:
            num_skipped += 1
            continue

        result = scorer.score(reference, generated)

        scores["rouge1_f"].append(result["rouge1"].fmeasure)
        scores["rouge2_f"].append(result["rouge2"].fmeasure)
        scores["rougeL_f"].append(result["rougeL"].fmeasure)
        scores["rouge1_p"].append(result["rouge1"].precision)
        scores["rouge2_p"].append(result["rouge2"].precision)
        scores["rougeL_p"].append(result["rougeL"].precision)
        scores["rouge1_r"].append(result["rouge1"].recall)
        scores["rouge2_r"].append(result["rouge2"].recall)
        scores["rougeL_r"].append(result["rougeL"].recall)

    # Compute averages
    metrics: dict[str, float] = {}
    for key, vals in scores.items():
        metrics[key] = sum(vals) / len(vals) if vals else 0.0

    metrics["num_evaluated"] = float(len(scores["rouge1_f"]))
    metrics["num_skipped"] = float(num_skipped)

    return metrics
