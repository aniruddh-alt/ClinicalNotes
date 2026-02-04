#!/usr/bin/env python3
"""Upload MIMIC-IV-BHC oncology splits to a private Hugging Face dataset.

IMPORTANT: This script uploads to a PRIVATE dataset to comply with PhysioNet
data use agreements. The dataset will only be accessible to you.

Usage:
    # First, login to Hugging Face
    huggingface-cli login

    # Then run this script
    python scripts/upload_to_hf.py --repo-name your-username/mimic-iv-bhc-oncology

Requirements:
    - huggingface-hub: pip install huggingface-hub
    - You must have a Hugging Face account
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def upload_dataset(repo_name: str, data_dir: Path) -> None:
    """Upload dataset to a private Hugging Face repository.

    Args:
        repo_name: HF repo name (format: username/dataset-name)
        data_dir: Local directory containing the dataset files
    """
    api = HfApi()

    # Create private repository
    print(f"Creating private dataset repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=True,  # CRITICAL: Keep this private!
            exist_ok=True,  # Don't error if repo exists
        )
        print(f"✓ Repository ready: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"Repository already exists: {repo_name}")
        print(f"Continuing with upload...")

    # Upload files
    print(f"\nUploading files from {data_dir}...")

    files_to_upload = [
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
    ]

    for filename in files_to_upload:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=repo_name,
                repo_type="dataset",
            )
            print(f"  ✓ {filename} uploaded")
        else:
            print(f"  ⚠ Warning: {filename} not found, skipping")

    # Create README with data card
    readme_content = f"""---
license: other
task_categories:
- text-generation
- summarization
language:
- en
tags:
- medical
- clinical-notes
- mimic-iv
pretty_name: MIMIC-IV-BHC Oncology Splits
---

# MIMIC-IV-BHC Oncology Notes Dataset

**IMPORTANT: This is a PRIVATE dataset containing credentialed PhysioNet data.**

## Dataset Description

This dataset contains train/validation/test splits of oncology clinical notes from MIMIC-IV-BHC.

- **Train**: Clinical notes for model training
- **Valid**: Validation set for hyperparameter tuning
- **Test**: Held-out test set for final evaluation

## Data Format

Each split is in JSONL format with the following structure:

```json
{{"messages": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
]}}
```

## License

This dataset is derived from MIMIC-IV-BHC and is subject to the PhysioNet Credentialed Health Data License 1.5.0.

**License URL**: https://physionet.org/content/mimic-iv-bhc/view-license/1.0/

## Access Requirements

This dataset is derived from MIMIC-IV-BHC, which requires:
1. PhysioNet credentialed access
2. Completion of CITI training
3. Signed data use agreement

**Do NOT share this dataset publicly or with unauthorized users.**

## Citation

```bibtex
@article{{johnson2023mimic,
  title={{MIMIC-IV}},
  author={{Johnson, Alistair EW and Bulgarelli, Lucas and Shen, Lu and others}},
  journal={{PhysioNet}},
  year={{2023}}
}}
```

## Usage

```python
from datasets import load_dataset

# Only works if you have access to this private repo
dataset = load_dataset("{repo_name}")
```
"""

    readme_path = Path("/tmp/README.md")
    readme_path.write_text(readme_content)

    print("\nUploading README...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
    )
    print("  ✓ README.md uploaded")

    print(f"\n✓ Upload complete!")
    print(f"\nDataset URL: https://huggingface.co/datasets/{repo_name}")
    print(f"\nTo download on GPU provider:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{repo_name}')")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload MIMIC-IV-BHC oncology splits to private HF dataset"
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="HF repository name (format: username/dataset-name)",
    )
    parser.add_argument(
        "--data-dir",
        default="MIMIC-IV-BHC/oncology-splits",
        help="Directory containing train/valid/test.jsonl files",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run this script from the repository root.")
        return

    upload_dataset(args.repo_name, data_dir)


if __name__ == "__main__":
    main()
