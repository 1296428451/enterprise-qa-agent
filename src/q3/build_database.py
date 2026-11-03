from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.database import create_database


def build_database(documents_path: str | Path, database_path: str | Path) -> None:
    records = json.loads(Path(documents_path).read_text(encoding="utf-8"))
    create_database(database_path)
    connection = sqlite3.connect(database_path)
    for record in records:
        content_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT OR REPLACE INTO documents(document_id,title,document_type,version,version_date,content,content_hash) VALUES(?,?,?,?,?,?,?)",
            (record["document_id"], record["document_id"], record["metadata"].get("type", "faq"), record["version"], record["version_date"], record["text"], content_hash),
        )
        row = connection.execute("SELECT id FROM documents WHERE document_id=?", (record["document_id"],)).fetchone()
        connection.execute(
            "INSERT OR REPLACE INTO document_chunks(chunk_id,document_id,chunk_index,text) VALUES(?,?,?,?)",
            (record["chunk_id"], row[0], 0, record["text"]),
        )
    connection.commit()
    connection.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    build_database(root / "data" / "q3_documents.json", root / "data" / "knowledge.db")
    print(root / "data" / "knowledge.db")
