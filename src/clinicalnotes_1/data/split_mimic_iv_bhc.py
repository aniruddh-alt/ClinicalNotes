from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from clinicalnotes_1.data.convert_mimic_iv_bhc import BhcRow, _row_to_messages

DEFAULT_SEED = 42


def _load_rows(csv_path: Path) -> list[BhcRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        required = {"input", "target"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Unexpected CSV header. Expected at least {sorted(required)}; got {reader.fieldnames}"
            )

        rows: list[BhcRow] = []
        for idx, raw in enumerate(reader):
            note_id = str(raw.get("note_id") or f"row-{idx}")
            rows.append(
                BhcRow(
                    note_id=note_id,
                    input=str(raw["input"]),
                    target=str(raw["target"]),
                )
            )
        return rows


def _split_rows(
    rows: list[BhcRow], train_ratio: float, val_ratio: float
) -> tuple[list[BhcRow], list[BhcRow], list[BhcRow]]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")
    if not rows:
        raise ValueError("No rows found in CSV.")

    total = len(rows)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_rows = rows[:train_count]
    val_rows = rows[train_count : train_count + val_count]
    test_rows = rows[train_count + val_count :]
    return train_rows, val_rows, test_rows


def _write_jsonl(rows: list[BhcRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            ex = {
                "note_id": row.note_id,
                "messages": _row_to_messages(row),
            }
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")


def split_csv_to_jsonl_sft(
    csv_path: Path,
    out_dir: Path,
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[Path, Path, Path]:
    rows = _load_rows(csv_path)
    rng = random.Random(seed)
    rng.shuffle(rows)

    train_rows, val_rows, test_rows = _split_rows(rows, train_ratio, val_ratio)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "valid.jsonl"
    test_path = out_dir / "test.jsonl"

    _write_jsonl(train_rows, train_path)
    _write_jsonl(val_rows, val_path)
    _write_jsonl(test_rows, test_path)

    return train_path, val_path, test_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="split-mimic-iv-bhc",
        description="Split MIMIC-IV-BHC CSV into train/valid/test JSONL files.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("MIMIC-IV-BHC/mimic-iv-bhc.csv"),
        help="Path to mimic-iv-bhc.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("MIMIC-IV-BHC/splits"),
        help="Output directory for train/valid/test JSONL files.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed.")
    p.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    train_path, val_path, test_path = split_csv_to_jsonl_sft(
        csv_path=args.csv,
        out_dir=args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    total = sum(1 for _ in train_path.open("r", encoding="utf-8")) + sum(
        1 for _ in val_path.open("r", encoding="utf-8")
    ) + sum(1 for _ in test_path.open("r", encoding="utf-8"))
    print(
        "Wrote splits:",
        f"train={train_path}",
        f"valid={val_path}",
        f"test={test_path}",
        f"total={total}",
    )


if __name__ == "__main__":
    main()
