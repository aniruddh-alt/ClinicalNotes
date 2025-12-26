from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Generator

from tqdm import tqdm


def load_csv_to_sqlite(csv_path: Path, db_path: Path) -> None:
    """Load the MIMIC-IV-BHC CSV into a SQLite database.

    Args:
        csv_path: Path to the mimic-iv-bhc.csv file.
        db_path: Path where the SQLite database will be created.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with index
    cursor.execute("DROP TABLE IF EXISTS notes")
    cursor.execute(
        """
        CREATE TABLE notes (
            note_id TEXT PRIMARY KEY,
            input TEXT NOT NULL,
            target TEXT NOT NULL,
            input_tokens INTEGER,
            target_tokens INTEGER
        )
        """
    )
    cursor.execute("CREATE INDEX idx_note_id ON notes(note_id)")

    # Read CSV and insert in chunks
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        
        # Get row count for progress bar if possible
        # Since we know it's ~270K, we can use a hardcoded estimate or count if needed.
        # For simplicity, we just use a progress bar without a total if count is slow.
        # But user previously ran wc -l so we know it's 270034.
        pbar = tqdm(desc="Importing notes", unit="rows", total=270034)
        
        chunk_size = 5000
        chunk: list[tuple[Any, ...]] = []
        
        for row in reader:
            chunk.append((
                row["note_id"],
                row["input"],
                row["target"],
                int(row["input_tokens"]) if row.get("input_tokens") else None,
                int(row["target_tokens"]) if row.get("target_tokens") else None
            ))
            
            if len(chunk) >= chunk_size:
                cursor.executemany(
                    "INSERT INTO notes VALUES (?, ?, ?, ?, ?)", chunk
                )
                conn.commit()
                pbar.update(len(chunk))
                chunk = []
        
        if chunk:
            cursor.executemany(
                "INSERT INTO notes VALUES (?, ?, ?, ?, ?)", chunk
            )
            conn.commit()
            pbar.update(len(chunk))
            
        pbar.close()

    conn.close()


def get_note_by_id(db_path: Path, note_id: str) -> dict[str, Any] | None:
    """Retrieve a specific note by its ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,))
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def get_notes_paginated(
    db_path: Path, limit: int = 100, offset: int = 0
) -> list[dict[str, Any]]:
    """Retrieve a page of notes from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM notes LIMIT ? OFFSET ?", (limit, offset)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def search_notes(db_path: Path, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Basic search for notes containing a query string in input or target."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    like_query = f"%{query}%"
    cursor.execute(
        "SELECT * FROM notes WHERE input LIKE ? OR target LIKE ? LIMIT ?",
        (like_query, like_query, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def stream_notes(db_path: Path, chunk_size: int = 1000) -> Generator[dict[str, Any], None, None]:
    """Generator to stream all notes from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM notes")
    while True:
        rows = cursor.fetchmany(chunk_size)
        if not rows:
            break
        for row in rows:
            yield dict(row)
            
    conn.close()

