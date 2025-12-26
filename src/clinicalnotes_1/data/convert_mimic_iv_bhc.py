from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BhcRow:
    note_id: str
    input: str
    target: str


JoinStyle = Literal["plain", "chatml"]


def _row_to_text(row: BhcRow, *, join_style: JoinStyle) -> str:
    prompt = row.input.strip()
    response = row.target.strip()
    if join_style == "plain":
        return f"Instruction:\n{prompt}\n\nResponse:\n{response}\n"
    if join_style == "chatml":
        # Conservative, no dependency on tokenizer chat templates.
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
    raise ValueError(f"Unknown join_style: {join_style}")


def convert_csv_to_jsonl_sft(
    *,
    csv_path: Path,
    out_path: Path,
    limit: int | None = None,
    join_style: JoinStyle = "plain",
) -> None:
    """Convert the PhysioNet MIMIC-IV-BHC CSV into JSONL with a `text` field.

    Oumi's `text_sft` dataset class expects a JSONL where each example contains `text`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
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
                "id": row.note_id,
                "text": _row_to_text(row, join_style=join_style),
                "meta": {"source": "physionet/mimic-iv-bhc"},
            }
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_written += 1
            if limit is not None and n_written >= limit:
                break


