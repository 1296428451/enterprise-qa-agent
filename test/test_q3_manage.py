import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.manage import import_documents, list_versions


class KnowledgeImportTests(unittest.TestCase):
    def record(self, version="1", content="报销期限为30天"):
        return {"document_id": "DNEW", "title": "新制度", "document_type": "policy", "version": version,
                "version_date": "2025-01-01" if version == "1" else "2026-01-01", "content": content,
                "chunks": [{"text": content}]}

    def test_idempotent_versions_and_sqlite_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "docs.json"; db = root / "knowledge.db"
            source.write_text(json.dumps({"documents": [self.record()]}, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(import_documents(source, db)["inserted"], 1)
            self.assertEqual(import_documents(source, db)["skipped"], 1)
            source.write_text(json.dumps({"documents": [self.record("2", "报销期限为60天")]}, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(import_documents(source, db)["inserted"], 1)
            self.assertEqual(len(list_versions(db, "DNEW")), 2)
            result = EnterpriseQA(load_chunks(db)).answer("报销期限")
            self.assertEqual(result.status, "version_conflict")
            self.assertTrue(result.citations)

    def test_conflicting_hash_and_mixed_batch_roll_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "docs.json"; db = root / "knowledge.db"
            source.write_text(json.dumps({"documents": [self.record()]}, ensure_ascii=False), encoding="utf-8")
            import_documents(source, db)
            source.write_text(json.dumps({"documents": [self.record(content="changed"), self.record("3")]}, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                import_documents(source, db)
            connection = sqlite3.connect(db)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)
            connection.close()

    def test_direct_manage_script_and_rebuild_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "docs.json"; db = root / "knowledge.db"
            source.write_text(json.dumps({"documents": [self.record()]}, ensure_ascii=False), encoding="utf-8")
            completed = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "src/q3/manage.py"),
                                        "--input", str(source), "--db", str(db), "--rebuild-index"],
                                       capture_output=True, text=True, check=True)
            output = json.loads(completed.stdout)
            self.assertTrue(output["index_rebuilt"])
            self.assertEqual(output["index_type"], "sqlite_text")
            self.assertTrue(EnterpriseQA(load_chunks(db)).answer("报销期限").citations)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "docs.json"; db = root / "knowledge.db"
            source.write_text(json.dumps({"documents": [{"document_id": "bad"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                import_documents(source, db)

    def test_content_chunks_mismatch_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "docs.json"; db = root / "knowledge.db"
            invalid = self.record(); invalid["content"] = "正文"; invalid["chunks"] = [{"text": "不一致"}]
            source.write_text(json.dumps({"documents": [invalid]}, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                import_documents(source, db)
            connection = sqlite3.connect(db)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 0)
            connection.close()


if __name__ == "__main__":
    unittest.main()
