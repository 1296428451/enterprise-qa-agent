import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.database import create_database
from src.q3.llm import MockLLM


class EnterpriseQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks(Path(__file__).resolve().parents[1] / "data" / "q3_documents.json")

    def test_sqlite_schema_is_created(self):
        import tempfile
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.db"
            create_database(path)
            connection = sqlite3.connect(path)
            names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            connection.close()
            self.assertIn("documents", names)
            self.assertIn("document_chunks", names)
            self.assertIn("retrieval_results", names)

    def test_mock_llm_adapter_is_deterministic(self):
        planner = MockLLM()
        plan = planner.plan("报销期限多久？")
        self.assertEqual(plan.queries, ["报销期限多久？"])
        self.assertTrue(planner.verify("报销期限多久？", "30天", [{"document_id": "D01"}])["verified"])

    def test_conflicting_policies_are_not_flattened(self):
        result = EnterpriseQA(self.chunks).answer("普通费用报销期限多久？")
        self.assertEqual(result.status, "conflict_detected")
        self.assertEqual({item["document_id"] for item in result.citations}, {"D01", "D02"})
        self.assertEqual(result.verification["action"], "ask_user")

    def test_specific_question_returns_citation(self):
        result = EnterpriseQA(self.chunks).answer("差旅报销需要什么材料？")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.citations[0]["document_id"], "D03")
        self.assertTrue(result.verification["verified"])

    def test_out_of_scope_returns_insufficient_evidence(self):
        result = EnterpriseQA(self.chunks).answer("公司是否提供住房补贴？")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertFalse(result.verification["verified"])

    def test_unrelated_employee_benefits_query_has_insufficient_evidence(self):
        result = EnterpriseQA(self.chunks).answer("员工福利积分如何兑换？")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, [])

        result = EnterpriseQA(self.chunks, max_evidence=1).answer("审批进度在哪里查看？")
        self.assertLessEqual(len(result.citations), 1)


if __name__ == "__main__":
    unittest.main()
