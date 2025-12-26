# Patient Centric Clinical Notes Summarization for LLMs

## Motivation

This project aims to provide a end-to-end pipeline for 

## Project layout (best-practice `src/` layout)

- **`src/clinicalnotes_1/`**: library code (strict typing).
- **`scripts/`**: runnable utilities (thin wrappers).
- **`configs/oumi/`**: Oumi YAML configs for analyze/train/eval.
- **`tests/`**: pytest tests.
- **`physionet.org/`**: mirrored dataset files (as provided in this repo).

## Quickstart

Create/activate env (using `uv`):

```bash
uv sync --all-extras
```

Convert the included MIMIC-IV-BHC CSV → JSONL SFT dataset:

```bash
uv run clinicalnotes convert-bhc --out data/mimic-iv-bhc/sft.jsonl
```

Analyze the dataset with Oumi:

```bash
uv run oumi analyze -c configs/oumi/analyze.yaml
```

Run a tiny demo SFT training (change settings before real runs):

```bash
uv run oumi train -c configs/oumi/train_sft.yaml
```

Run an example eval:

```bash
uv run oumi evaluate -c configs/oumi/eval.yaml
```

## Tooling

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

- **Pre-commit** (optional but recommended):

```bash
uv run pre-commit install
```