"""
Query clinical notes and extract summaries from SQLite database
"""

import sqlite3
import re

DB_PATH = "discharge_notes.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def extract_section(text, section_name):
    """Extract a specific section from the discharge note"""
    # Pattern to find section and capture until next section header or end
    pattern = rf'{section_name}[:\s]*\n(.*?)(?=\n[A-Z][A-Za-z\s]+:\s*\n|\n___|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_brief_hospital_course(text):
    """Extract the Brief Hospital Course (main summary) from the note"""
    return extract_section(text, "Brief Hospital Course")

def extract_discharge_diagnosis(text):
    """Extract discharge diagnosis"""
    return extract_section(text, "Discharge Diagnosis")

def extract_chief_complaint(text):
    """Extract chief complaint"""
    return extract_section(text, "Chief Complaint")

def get_note_with_summary(note_id):
    """Get a single note with its extracted summary"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM discharge_notes WHERE note_id = ?", (note_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        note = {
            'note_id': row[0],
            'subject_id': row[1],
            'hadm_id': row[2],
            'note_type': row[3],
            'note_seq': row[4],
            'charttime': row[5],
            'storetime': row[6],
            'full_text': row[7],
            'chief_complaint': extract_chief_complaint(row[7]),
            'brief_hospital_course': extract_brief_hospital_course(row[7]),
            'discharge_diagnosis': extract_discharge_diagnosis(row[7])
        }
        return note
    return None

def get_sample_notes(limit=5):
    """Get sample notes with their summaries"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT note_id, subject_id, hadm_id, text FROM discharge_notes LIMIT {limit}")
    rows = cursor.fetchall()
    conn.close()
    
    notes = []
    for row in rows:
        note = {
            'note_id': row[0],
            'subject_id': row[1],
            'hadm_id': row[2],
            'full_text': row[3],
            'chief_complaint': extract_chief_complaint(row[3]),
            'brief_hospital_course': extract_brief_hospital_course(row[3]),
            'discharge_diagnosis': extract_discharge_diagnosis(row[3])
        }
        notes.append(note)
    return notes

def search_notes_by_diagnosis(keyword, limit=10):
    """Search notes by keyword in discharge diagnosis"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT note_id, subject_id, hadm_id, text FROM discharge_notes WHERE text LIKE ? LIMIT ?",
        (f'%{keyword}%', limit)
    )
    rows = cursor.fetchall()
    conn.close()
    
    notes = []
    for row in rows:
        note = {
            'note_id': row[0],
            'subject_id': row[1], 
            'hadm_id': row[2],
            'chief_complaint': extract_chief_complaint(row[3]),
            'brief_hospital_course': extract_brief_hospital_course(row[3]),
            'discharge_diagnosis': extract_discharge_diagnosis(row[3])
        }
        notes.append(note)
    return notes

# Demo: Show sample notes with summaries
if __name__ == "__main__":
    print("="*80)
    print("MIMIC-IV DISCHARGE NOTES - SAMPLE WITH SUMMARIES")
    print("="*80)
    
    # Get database stats
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM discharge_notes")
    total = cursor.fetchone()[0]
    print(f"\nTotal notes in database: {total:,}")
    conn.close()
    
    # Show 3 sample notes
    print("\n" + "="*80)
    print("SAMPLE NOTES WITH EXTRACTED SUMMARIES")
    print("="*80)
    
    notes = get_sample_notes(3)
    
    for i, note in enumerate(notes, 1):
        print(f"\n{'─'*80}")
        print(f"NOTE {i}: {note['note_id']}")
        print(f"Subject ID: {note['subject_id']} | Admission ID: {note['hadm_id']}")
        print(f"{'─'*80}")
        
        print(f"\n📋 CHIEF COMPLAINT:")
        print(note['chief_complaint'] or "Not found")
        
        print(f"\n📊 DISCHARGE DIAGNOSIS:")
        diagnosis = note['discharge_diagnosis']
        if diagnosis:
            # Truncate if too long
            print(diagnosis[:500] + "..." if len(diagnosis) > 500 else diagnosis)
        else:
            print("Not found")
        
        print(f"\n📝 BRIEF HOSPITAL COURSE (SUMMARY):")
        summary = note['brief_hospital_course']
        if summary:
            # Truncate if too long for display
            print(summary[:1000] + "..." if len(summary) > 1000 else summary)
        else:
            print("Not found")
        
        print(f"\n📄 Full note length: {len(note['full_text']):,} characters")

