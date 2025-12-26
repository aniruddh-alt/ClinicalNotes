# Patient Centric Clinical Notes Summarization for LLMs

## Motivation

This project provides an end-to-end pipeline for clinical notes summarization using the MIMIC-IV-BHC dataset.

## Project Layout

- **`src/clinicalnotes_1/`**: Library code (strict typing).
  - **`data/`**: Data processing and SQLite management.
- **`configs/oumi/`**: Oumi YAML configs for analyze/train/eval.
- **`tests/`**: Pytest tests.
- **`MIMIC-IV-BHC/`**: Raw dataset files (CSV, license).

## Quickstart

### 1. Setup Environment
Create/activate environment (using `uv`):

```bash
uv sync --all-extras
```

### 2. Load Data into SQLite
Load the 270K clinical notes into a SQLite database for fast querying:

```bash
uv run clinicalnotes load-db
```
*This creates `data/notes.db`.*

### 3. Convert to Training Format
Convert the MIMIC-IV-BHC CSV → JSONL SFT dataset for Oumi:

```bash
uv run clinicalnotes convert-bhc --out data/mimic-iv-bhc/sft.jsonl
```

### 4. Pipeline with Oumi
Analyze the dataset:
```bash
uv run oumi analyze -c configs/oumi/analyze.yaml
```

Run training:
```bash
uv run oumi train -c configs/oumi/train_sft.yaml
```

Run evaluation:
```bash
uv run oumi evaluate -c configs/oumi/eval.yaml
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
