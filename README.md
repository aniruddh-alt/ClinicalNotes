# Patient Centric Clinical Notes Summarization for LLMs

## Motivation

This project provides an end-to-end pipeline for clinical notes summarization using the MIMIC-IV-BHC dataset.

## Project Layout

- **`src/clinicalnotes_1/`**: Library code (strict typing).
  - **`data/`**: Data processing and SQLite management.
- **`configs/oumi/`**: Oumi YAML configs for analyze/train/eval.
- **`tests/`**: Pytest tests.
- **`scripts/`**: Utility scripts for data downloading and setup.
- **`MIMIC-IV-BHC/`**: Raw dataset files (managed by DVC, not committed to git).

## Dataset Management

This project uses **DVC (Data Version Control)** to manage large datasets. Dataset files are **not stored in git** - only lightweight `.dvc` pointer files are committed.

### Why DVC?

- **No bloated git repos**: Datasets (2.5GB CSV, 205MB splits) stay out of git history
- **Version control for data**: Track dataset changes like code
- **Easy sharing**: Works with S3, GCS, or local storage
- **Perfect for GPU providers**: Download datasets on-demand in training environments

### Dataset Files (managed by DVC)

- `MIMIC-IV-BHC/mimic-iv-bhc.csv` (2.5GB) - Raw clinical notes
- `MIMIC-IV-BHC/oncology-splits/` (205MB) - Train/valid/test JSONL splits

## Quickstart

### 1. Setup Environment
Create/activate environment (using `uv`):

```bash
uv sync --all-extras
```

### 2. Download Datasets (via DVC)

**On your local machine** (if you have the original data):
```bash
# Datasets are already tracked by DVC
# The .dvc files are in git, data files are in .gitignore
```

**On GPU provider / new machine**:
```bash
# Install DVC
pip install dvc[s3]

# Download datasets
bash scripts/download_data.sh
# or
python scripts/download_data.py
```

**Note**: You'll need to configure DVC remote storage first (see "Setting up DVC Remote" below).

### 3. Load Data into SQLite
Load the 270K clinical notes into a SQLite database for fast querying:

```bash
uv run clinicalnotes load-db
```
*This creates `data/notes.db`.*

### 4. Convert to Training Format
Convert the MIMIC-IV-BHC CSV → JSONL SFT dataset for Oumi:

```bash
uv run clinicalnotes convert-bhc --out data/mimic-iv-bhc/sft.jsonl
```

### 5. Split for Training/Validation/Test
Create a seeded 70/15/15 split in JSONL format:

```bash
uv run python -m clinicalnotes_1.data.split_mimic_iv_bhc \
  --csv MIMIC-IV-BHC/mimic-iv-bhc.csv \
  --out-dir MIMIC-IV-BHC/splits \
  --seed 42
```

### 6. Pipeline with Oumi
Analyze the dataset:
```bash
uv run oumi analyze -c configs/oumi/analyze.yaml
```

Run training:
```bash
uv run oumi train -c configs/oumi/train_sft_llama3_1_8b.yaml
uv run oumi train -c configs/oumi/train_sft_qwen2_5_7b.yaml
uv run oumi train -c configs/oumi/train_sft_phi_mini.yaml
```

### Training on GPU Providers (Runpod, Lambda Labs, etc.)

**Setup workflow:**

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd summarization
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[data]"
   ```

3. Download datasets using DVC:
   ```bash
   bash scripts/download_data.sh
   ```

4. Verify datasets are available:
   ```bash
   ls -lh MIMIC-IV-BHC/oncology-splits/
   ```

5. Start training:
   ```bash
   oumi train -c configs/oumi/train_sft_llama3_1_8b.yaml
   ```

**Note**: If you use a different filesystem path, update the dataset paths in `configs/oumi/*.yaml`.

Run evaluation:
```bash
uv run oumi evaluate -c configs/oumi/eval.yaml
```

## Setting up DVC Remote Storage

To share datasets across machines and team members, configure a DVC remote. Choose one option:

### Option 1: Amazon S3 (Recommended for production)

```bash
# Configure S3 remote
dvc remote add -d myremote s3://my-bucket/clinicalnotes-data

# Set AWS credentials (if needed)
dvc remote modify myremote access_key_id <YOUR_ACCESS_KEY>
dvc remote modify myremote secret_access_key <YOUR_SECRET_KEY>

# Push datasets to S3
dvc push
```

### Option 2: Google Cloud Storage

```bash
# Install GCS support
pip install dvc[gs]

# Configure GCS remote
dvc remote add -d myremote gs://my-bucket/clinicalnotes-data

# Push datasets
dvc push
```

### Option 3: Local/SSH (for team sharing)

```bash
# Configure SSH remote
dvc remote add -d myremote ssh://user@server/path/to/dvc-storage

# Push datasets
dvc push
```

### Option 4: Hugging Face Hub (Great for ML projects)

```bash
# Install HF support
pip install huggingface-hub

# Login to HF
huggingface-cli login

# Upload manually using HF Hub Python API
# See scripts/upload_to_hf.py for an example
```

**Important**: After configuring a remote, commit the changes:
```bash
git add .dvc/config
git commit -m "Configure DVC remote storage"
```

## Development

- **Format/lint**:
  ```bash
  uv run ruff format .
  uv run ruff check . --fix
  ```

- **Type check**:
  ```bash
  uv run ty check
  uv run mypy .
  ```

- **Tests**:
  ```bash
  uv run pytest
  ```
