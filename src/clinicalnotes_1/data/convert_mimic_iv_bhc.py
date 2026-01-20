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


def convert_csv_to_jsonl_sft(
    csv_path: Path,
    out_path: Path,
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

        for raw in reader:
            row = BhcRow(
                note_id=str(raw["note_id"]),
                input=str(raw["input"]),
                target=str(raw["target"]),
            )
            ex = {
                "messages": _row_to_messages(row),
            }
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_csv_to_jsonl_sft(
        Path("MIMIC-IV-BHC/mimic-iv-bhc.csv"), Path("MIMIC-IV-BHC/mimic-iv-bhc.jsonl")
    )
