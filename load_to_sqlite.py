"""
Load MIMIC-IV discharge notes into SQLite database for faster querying
"""

import sqlite3
import pandas as pd
import gzip
import os

DB_PATH = "discharge_notes.db"
DISCHARGE_PATH = "physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"

# Remove existing database
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Removed existing {DB_PATH}")

# Create SQLite connection
conn = sqlite3.connect(DB_PATH)
print(f"Created SQLite database: {DB_PATH}")

# Load data in chunks for memory efficiency
print(f"\nLoading {DISCHARGE_PATH}...")
print("This may take a few minutes...")

chunk_size = 10000
total_rows = 0

for i, chunk in enumerate(pd.read_csv(DISCHARGE_PATH, compression='gzip', chunksize=chunk_size)):
    # Write to SQLite
    chunk.to_sql('discharge_notes', conn, if_exists='append', index=False)
    total_rows += len(chunk)
    print(f"  Loaded {total_rows:,} rows...", end='\r')

print(f"\n\nTotal rows loaded: {total_rows:,}")

# Create indexes for faster queries
print("\nCreating indexes...")
cursor = conn.cursor()
cursor.execute("CREATE INDEX idx_note_id ON discharge_notes(note_id)")
cursor.execute("CREATE INDEX idx_subject_id ON discharge_notes(subject_id)")
cursor.execute("CREATE INDEX idx_hadm_id ON discharge_notes(hadm_id)")
conn.commit()

# Show table info
print("\nTable structure:")
cursor.execute("PRAGMA table_info(discharge_notes)")
for col in cursor.fetchall():
    print(f"  {col[1]}: {col[2]}")

# Quick sample query
print("\nSample data (first 3 notes metadata):")
cursor.execute("SELECT note_id, subject_id, hadm_id, note_type, LENGTH(text) as text_length FROM discharge_notes LIMIT 3")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()
print(f"\n✅ Database ready! Query with: sqlite3 {DB_PATH}")

