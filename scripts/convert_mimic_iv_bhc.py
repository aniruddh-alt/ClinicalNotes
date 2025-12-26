from __future__ import annotations

from pathlib import Path

from clinicalnotes_1.data.convert_mimic_iv_bhc import convert_csv_to_jsonl_sft


def main() -> None:
    convert_csv_to_jsonl_sft(
        csv_path=Path("physionet.org/files/labelled-notes-hospital-course/1.2.0/mimic-iv-bhc.csv"),
        out_path=Path("data/mimic-iv-bhc/sft.jsonl"),
    )


if __name__ == "__main__":
    main()


