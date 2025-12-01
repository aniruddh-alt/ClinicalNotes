"""
Script to explore MIMIC-IV discharge notes dataset
and extract clinical notes with their summaries
"""

import pandas as pd
import gzip

# Path to the discharge notes
DISCHARGE_PATH = "physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"

# Load the discharge notes
print("Loading discharge notes...")
df = pd.read_csv(DISCHARGE_PATH, compression='gzip')

print(f"\n{'='*60}")
print("DATASET OVERVIEW")
print(f"{'='*60}")
print(f"Total number of notes: {len(df):,}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nNote types: {df['note_type'].unique()}")

# Show first few rows (without full text)
print(f"\n{'='*60}")
print("SAMPLE ROWS (metadata)")
print(f"{'='*60}")
print(df[['note_id', 'subject_id', 'hadm_id', 'note_type', 'charttime']].head(10))

# Analyze a single note structure
print(f"\n{'='*60}")
print("SAMPLE DISCHARGE NOTE STRUCTURE")
print(f"{'='*60}")
sample_note = df.iloc[0]['text']
print(f"Note ID: {df.iloc[0]['note_id']}")
print(f"Subject ID: {df.iloc[0]['subject_id']}")
print(f"Note length: {len(sample_note):,} characters")
print(f"\n--- FULL NOTE ---\n")
print(sample_note)

# Find common section headers in notes
print(f"\n{'='*60}")
print("COMMON SECTION HEADERS IN NOTES")
print(f"{'='*60}")

# Check a sample of notes for section headers
section_keywords = [
    "Brief Hospital Course",
    "Discharge Summary",
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Medications on Admission",
    "Discharge Medications"
]

print("\nChecking section prevalence in first 100 notes...")
sample_df = df.head(100)
for keyword in section_keywords:
    count = sample_df['text'].str.contains(keyword, case=False).sum()
    print(f"  '{keyword}': {count}/100 notes")

