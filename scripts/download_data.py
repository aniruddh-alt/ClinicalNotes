#!/usr/bin/env python3
"""Download datasets for training on GPU providers.

This script downloads the MIMIC-IV-BHC datasets using DVC.
Run this script in your training environment before starting finetuning.

Usage:
    python scripts/download_data.py

Requirements:
    - DVC must be installed: pip install dvc[s3]
    - DVC remote must be configured (if using remote storage)
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a shell command and check for errors."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def main() -> None:
    """Download datasets using DVC."""
    repo_root = Path(__file__).parent.parent

    print("Downloading datasets with DVC...")
    print("This may take a while depending on your connection speed.\n")

    # Pull data from DVC remote
    print("Running: dvc pull")
    run_command(["dvc", "pull"], cwd=repo_root)

    print("\n✓ Dataset download complete!")
    print("\nDatasets available at:")
    print("  - MIMIC-IV-BHC/mimic-iv-bhc.csv")
    print("  - MIMIC-IV-BHC/oncology-splits/train.jsonl")
    print("  - MIMIC-IV-BHC/oncology-splits/valid.jsonl")
    print("  - MIMIC-IV-BHC/oncology-splits/test.jsonl")


if __name__ == "__main__":
    main()
