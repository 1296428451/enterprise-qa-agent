from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

if __package__:
    from .database import create_database
else:  # direct: python src/q3/manage.py
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.q3.database import create_database

REQUIRED = ("document_id", "title", "document_type", "version", "version_date", "content")


def _chunks(record: dict[str, Any]) -> list[str]:
    chunks = record.get("chunks")
    if chunks is None:
        return [record["content"]]
    if not isinstance(chunks, list) or not chunks or any(
        not isinstance(item, (str, dict)) for item in chunks
    ):
        raise ValueError("chunks must be a non-empty array of strings or objects")
    result = []
    for item in chunks:
        text = item if isinstance(item, str) else item.get("text", item.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each chunk must contain non-empty text")
        result.append(text)
    return result


def _validate(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("each document must be an object")
    missing = [field for field in REQUIRED if not isinstance(record.get(field), str) or not record[field].strip()]
    if missing:
        raise ValueError("missing or empty fields: " + ", ".join(missing))
    chunks = _chunks(record)
    if "chunks" in record and "".join(chunks) != record["content"]:
        raise ValueError("content does not match concatenated chunks")
    return record


def import_documents(input_path: str | Path, database_path: str | Path, rebuild_index: bool = False) -> dict[str, int]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = payload.get("documents") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("input must contain a documents array")
    create_database(database_path)
    connection = sqlite3.connect(database_path)
    inserted = skipped = 0
    try:
        for raw in records:
            record = _validate(raw)
            content = record["content"]
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            existing = connection.execute(
                "SELECT id, content_hash FROM documents WHERE document_id=? AND version=?",
                (record["document_id"], record["version"]),
            ).fetchone()
            if existing:
                if existing[1] != content_hash:
                    raise ValueError(f"document/version already exists with different content: {record['document_id']}@{record['version']}")
                skipped += 1
                continue
            cursor = connection.execute(
                "INSERT INTO documents(document_id,title,document_type,version,version_date,content,content_hash) VALUES(?,?,?,?,?,?,?)",
                (record["document_id"], record["title"], record["document_type"], record["version"], record["version_date"], content, content_hash),
            )
            document_row_id = cursor.lastrowid
            for index, text in enumerate(_chunks(record)):
                chunk_id = f"{record['document_id']}@{record['version']}:{index}"
                connection.execute(
                    "INSERT INTO document_chunks(chunk_id,document_id,chunk_index,text) VALUES(?,?,?,?)",
                    (chunk_id, document_row_id, index, text),
                )
            inserted += 1
        if rebuild_index:
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_text ON document_chunks(text)")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {"inserted": inserted, "skipped": skipped, "total": len(records),
            "index_rebuilt": bool(rebuild_index), "index_type": "sqlite_text" if rebuild_index else None}


def list_versions(database_path: str | Path, document_id: str | None = None) -> list[dict[str, Any]]:
    create_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        if document_id:
            rows = connection.execute(
                "SELECT document_id, title, document_type, version, version_date, content_hash "
                "FROM documents WHERE document_id=? ORDER BY version_date, version",
                (document_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT document_id, title, document_type, version, version_date, content_hash "
                "FROM documents ORDER BY document_id, version_date, version"
            ).fetchall()
        return [dict(zip(("document_id", "title", "document_type", "version", "version_date", "content_hash"), row)) for row in rows]
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage enterprise knowledge documents")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--db", default="data/knowledge.db", type=Path)
    parser.add_argument("--list", action="store_true", dest="list_versions", help="列出文档版本")
    parser.add_argument("--document-id", help="仅列出指定文档的版本")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args(argv)
    if args.list_versions:
        print(json.dumps(list_versions(args.db, args.document_id), ensure_ascii=False, indent=2))
        return 0
    if not args.input:
        parser.error("--input is required unless --list is used")
    try:
        print(json.dumps(import_documents(args.input, args.db, args.rebuild_index), ensure_ascii=False))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
