from __future__ import annotations

import argparse
from pathlib import Path

from clinicalnotes_1.data.convert_mimic_iv_bhc import convert_csv_to_jsonl_sft
from clinicalnotes_1.data.database import load_csv_to_sqlite


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clinicalnotes", description="ClinicalNotes-1 utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Dataset Conversion
    c = sub.add_parser("convert-bhc", help="Convert MIMIC-IV-BHC CSV to Oumi-compatible JSONL (SFT)")
    c.add_argument(
        "--csv",
        type=Path,
        default=Path("MIMIC-IV-BHC/mimic-iv-bhc.csv"),
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

    # Database loading
    db = sub.add_parser("load-db", help="Load MIMIC-IV-BHC CSV into SQLite database")
    db.add_argument(
        "--csv",
        type=Path,
        default=Path("MIMIC-IV-BHC/mimic-iv-bhc.csv"),
        help="Path to mimic-iv-bhc.csv",
    )
    db.add_argument(
        "--db",
        type=Path,
        default=Path("data/notes.db"),
        help="Path to the output SQLite database",
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
    elif args.cmd == "load-db":
        load_csv_to_sqlite(
            csv_path=args.csv,
            db_path=args.db,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
