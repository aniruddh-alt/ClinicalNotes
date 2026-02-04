#!/usr/bin/env python3
"""Download datasets from private Hugging Face dataset for training on GPU providers.

This script downloads the MIMIC-IV-BHC oncology splits from your private HF dataset.
Run this script in your training environment before starting finetuning.

Usage:
    # First, login to Hugging Face
    huggingface-cli login

    # Then run this script
    python scripts/download_data_hf.py

Requirements:
    - huggingface-hub: pip install huggingface-hub
    - datasets: pip install datasets
"""

import sys
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    """Download datasets from Hugging Face."""
    repo_name = "aniruddhr04/mimic-iv-bhc-oncology"
    output_dir = Path("MIMIC-IV-BHC/oncology-splits")

    print(f"Downloading datasets from Hugging Face: {repo_name}")
    print("This may take a while depending on your connection speed.\n")

    try:
        # Load dataset from private HF repo
        print("Loading dataset...")
        dataset = load_dataset(repo_name)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save splits as JSONL files
        for split_name in ["train", "valid", "test"]:
            if split_name in dataset:
                output_file = output_dir / f"{split_name}.jsonl"
                print(f"Saving {split_name} split to {output_file}...")

                # Save as JSONL
                dataset[split_name].to_json(output_file, orient="records", lines=True)

                print(f"  ✓ {split_name}.jsonl saved ({len(dataset[split_name])} examples)")

        print("\n✓ Dataset download complete!")
        print("\nDatasets available at:")
        print(f"  - {output_dir / 'train.jsonl'}")
        print(f"  - {output_dir / 'valid.jsonl'}")
        print(f"  - {output_dir / 'test.jsonl'}")

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nMake sure you:")
        print("  1. Have run 'huggingface-cli login'")
        print("  2. Have access to the private dataset")
        print(f"  3. Can access: https://huggingface.co/datasets/{repo_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
