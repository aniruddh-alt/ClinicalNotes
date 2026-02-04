from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert medical scribe. Summarize the following clinical notes "
    "into a Brief Hospital Course (BHC) paragraph primarily for other physicians. "
    "Use professional medical terminology."
)


@dataclass(frozen=True)
class BhcRow:
    note_id: str
    input: str
    target: str


def _row_to_messages(row: BhcRow) -> list[dict[str, str]]:
    """Convert a BhcRow to the chat messages format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row.input.strip()},
        {"role": "assistant", "content": row.target.strip()},
    ]


def _messages_to_text(messages: list[dict[str, str]], join_style: str) -> str:
    """Create a text field from messages for SFT training."""
    if join_style == "plain":
        return "\n\n".join(message["content"].strip() for message in messages)
    if join_style == "chatml":
        chunks = [
            f"<|{message['role']}|>\n{message['content'].strip()}"
            for message in messages
        ]
        return "\n".join(chunks) + "\n<|end|>"
    raise ValueError(f"Unsupported join_style: {join_style}")


def convert_csv_to_jsonl_sft(
    csv_path: Path,
    out_path: Path,
    limit: int | None = None,
    join_style: str = "plain",
) -> None:
    """Convert the PhysioNet MIMIC-IV-BHC CSV into JSONL with a `messages` field.

    Output format:
    ```json
    {
      "messages": [
        {
          "role": "system",
          "content": "You are an expert medical scribe. Summarize the following clinical notes into a Brief Hospital Course (BHC) paragraph primarily for other physicians. Use professional medical terminology."
        },
        {
          "role": "user",
          "content": "<SEX> F <SERVICE> SURGERY <ALLERGIES> Patient recorded as having No Known Allergies... [REST OF INPUT COLUMN]"
        },
        {
          "role": "assistant",
          "content": "The patient was admitted to the Acute Care Surgery service on ___ and was transferred... [TARGET COLUMN]"
        }
      ]
    }
    ```
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        csv_path.open("r", encoding="utf-8", newline="") as f_in,
        out_path.open("w", encoding="utf-8") as f_out,
    ):
        reader = csv.DictReader(f_in)
        required = {"note_id", "input", "target"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Unexpected CSV header. Expected at least {sorted(required)}; got {reader.fieldnames}"
            )

        for idx, raw in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            row = BhcRow(
                note_id=str(raw["note_id"]),
                input=str(raw["input"]),
                target=str(raw["target"]),
            )
            messages = _row_to_messages(row)
            ex = {"messages": messages, "text": _messages_to_text(messages, join_style)}
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_csv_to_jsonl_sft(
        Path("MIMIC-IV-BHC/mimic-iv-bhc.csv"), Path("MIMIC-IV-BHC/mimic-iv-bhc.jsonl")
    )
