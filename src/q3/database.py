from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    version TEXT NOT NULL,
    version_date TEXT NOT NULL,
    effective_from TEXT,
    effective_to TEXT,
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS document_relations (
    id INTEGER PRIMARY KEY,
    source_document_id INTEGER NOT NULL REFERENCES documents(id),
    relation_type TEXT NOT NULL,
    target_document_id INTEGER REFERENCES documents(id),
    target_value TEXT,
    evidence_chunk_id TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS retrieval_runs (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    final_answer TEXT
);
CREATE TABLE IF NOT EXISTS retrieval_results (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES retrieval_runs(run_id),
    chunk_id TEXT NOT NULL,
    bm25_score REAL,
    vector_score REAL,
    jev_score REAL,
    rank INTEGER,
    selected INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_results_run ON retrieval_results(run_id);
"""


def create_database(path: str | Path) -> None:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    connection.commit()
    connection.close()
