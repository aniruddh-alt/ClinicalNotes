from __future__ import annotations

import argparse
from pathlib import Path

from clinicalnotes_1.data.convert_mimic_iv_bhc import convert_csv_to_jsonl_sft


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clinicalnotes", description="ClinicalNotes-1 utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert-bhc", help="Convert MIMIC-IV-BHC CSV to Oumi-compatible JSONL (SFT)")
    c.add_argument(
        "--csv",
        type=Path,
        default=Path("physionet.org/files/labelled-notes-hospital-course/1.2.0/mimic-iv-bhc.csv"),
        help="Path to mimic-iv-bhc.csv",
    )
    c.add_argument("--out", type=Path, default=Path("data/mimic-iv-bhc/sft.jsonl"))
    c.add_argument("--limit", type=int, default=0, help="If >0, write only first N examples")
    c.add_argument(
        "--join-style",
        choices=["chatml", "plain"],
        default="plain",
        help="How to build the `text` field used for SFT",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "convert-bhc":
        convert_csv_to_jsonl_sft(
            csv_path=args.csv,
            out_path=args.out,
            limit=args.limit if args.limit > 0 else None,
            join_style=args.join_style,
        )
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


