from __future__ import annotations

import json
from pathlib import Path

from clinicalnotes_1.data.convert_mimic_iv_bhc import convert_csv_to_jsonl_sft


def test_convert_writes_jsonl(tmp_path: Path) -> None:
    csv_text = (
        "note_id,input,target,input_tokens,target_tokens\n"
        "n1,hello,world,1,1\n"
        "n2,foo,bar,1,1\n"
    )
    csv_path = tmp_path / "x.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    out_path = tmp_path / "out.jsonl"
    convert_csv_to_jsonl_sft(csv_path=csv_path, out_path=out_path, limit=1)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["id"] == "n1"
    assert "Instruction" in obj["text"]


