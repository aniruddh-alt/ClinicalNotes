#!/bin/bash
# Download datasets for training on GPU providers
#
# This script downloads the MIMIC-IV-BHC datasets using DVC.
# Run this script in your training environment before starting finetuning.
#
# Usage:
#   bash scripts/download_data.sh
#
# Requirements:
#   - DVC must be installed: pip install dvc[s3]
#   - DVC remote must be configured (if using remote storage)

set -e

echo "Downloading datasets with DVC..."
echo "This may take a while depending on your connection speed."
echo ""

# Navigate to repo root
cd "$(dirname "$0")/.."

# Pull data from DVC remote
echo "Running: dvc pull"
dvc pull

echo ""
echo "✓ Dataset download complete!"
echo ""
echo "Datasets available at:"
echo "  - MIMIC-IV-BHC/mimic-iv-bhc.csv"
echo "  - MIMIC-IV-BHC/oncology-splits/train.jsonl"
echo "  - MIMIC-IV-BHC/oncology-splits/valid.jsonl"
echo "  - MIMIC-IV-BHC/oncology-splits/test.jsonl"
