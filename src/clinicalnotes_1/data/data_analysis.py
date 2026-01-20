import random
from pathlib import Path

import pandas as pd


def visualize_data(num_samples: int = 3):
    """Prints the data to understand inputs and outputs"""
    data: Path = Path("MIMIC-IV-BHC/mimic-iv-bhc.csv")
    df = pd.read_csv(data)

    print("\n" + "=" * 100)
    print(f"SAMPLE RECORDS (showing {num_samples} random examples)")
    print("=" * 100)

    # Get random samples
    sample_indices = random.sample(range(len(df)), min(num_samples, len(df)))

    for idx, row_idx in enumerate(sample_indices, 1):
        row = df.iloc[row_idx]
        print(f"\n{'-' * 100}")
        print(f"SAMPLE {idx} (Row {row_idx})")
        print(f"{'-' * 100}")
        print(f"Note ID: {row['note_id']}")
        print(f"Input tokens: {row['input_tokens']}")
        print(f"Target tokens: {row['target_tokens']}")

        print(f"\nINPUT (first 500 chars):")
        print("-" * 50)
        print(row["input"][:500])
        if len(row["input"]) > 500:
            print(f"... ({len(row['input']) - 500} more characters)")

        print(f"\nTARGET (first 500 chars):")
        print("-" * 50)
        print(row["target"][:500])
        if len(row["target"]) > 500:
            print(f"... ({len(row['target']) - 500} more characters)")


if __name__ == "__main__":
    visualize_data()
