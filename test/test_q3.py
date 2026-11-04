import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.database import create_database
from src.q3.evaluate import evaluate as evaluate_q3
from src.q3.llm import MockLLM, OpenAICompatibleLLM
from src.q3.models import ScoredChunk
from src.q3.provider_reranker import ProviderReranker


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

    def test_openai_compatible_llm_parses_plan_and_verification(self):
        llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
        responses = [
            {"queries": ["报销期限", "费用提交日期"], "intent": "查询报销截止时间"},
            {"verified": True, "unsupported_claims": [], "action": "answer"},
        ]
        with patch.object(llm, "_complete_json", side_effect=responses) as complete:
            plan = llm.plan("报销期限多久？", [{"role": "user", "content": "报销"}])
            verified = llm.verify("报销期限多久？", "30天", [{"document_id": "D01", "quote": "30天内"}])
        self.assertEqual(plan.queries, ["报销期限", "费用提交日期"])
        self.assertTrue(verified["verified"])
        self.assertEqual(complete.call_count, 2)

    def test_openai_compatible_llm_rejects_malformed_plan(self):
        llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
        with patch.object(llm, "_complete_json", return_value={"queries": [], "intent": "x"}):
            with self.assertRaises(ValueError):
                llm.plan("报销期限多久？")
        with patch.object(llm, "_complete_json", return_value={
            "verified": True, "unsupported_claims": ["unsupported"], "action": "answer"
        }):
            with self.assertRaises(ValueError):
                llm.verify("报销期限多久？", "answer", [])

    def test_openai_compatible_llm_sends_auth_and_parses_http_json(self):
        llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "secret-value")
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"queries":["报销"],"intent":"lookup"}'}}]}).encode())
        with patch("src.q3.llm.urlopen", return_value=response) as open_url:
            plan = llm.plan("报销期限？")
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-value")
        self.assertEqual(plan.queries, ["报销"])

    def test_provider_reranker_orders_candidates_by_provider_score(self):
        items = [ScoredChunk(self.chunks[0], bm25_score=0.4), ScoredChunk(self.chunks[1], bm25_score=0.8)]
        body = {"data": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9}]}
        response = io.BytesIO(json.dumps(body).encode())
        reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "secret-value")
        with patch("src.q3.provider_reranker.urlopen", return_value=response):
            ranked = reranker.score("报销期限", items)
        self.assertEqual(ranked[0].chunk.chunk_id, self.chunks[1].chunk_id)
        self.assertEqual(ranked[0].rerank_score, 0.9)

    def test_provider_reranker_rejects_missing_scores(self):
        item = ScoredChunk(self.chunks[0])
        reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "test-key")
        with patch("src.q3.provider_reranker.urlopen", return_value=io.BytesIO(b'{"data":[]}')):
            with self.assertRaises(ValueError):
                reranker.score("报销", [item])

    def test_provider_reranker_builds_v1_endpoint(self):
        reranker = ProviderReranker("https://example.invalid/v1", "test-model", "test-key")
        self.assertEqual(reranker.base_url, "https://example.invalid/v1/rerank")

    def test_openai_verifier_can_veto_answer_and_preserve_claims(self):
        class VetoLLM(MockLLM):
            def verify(self, question, answer, citations):
                return {"verified": False, "unsupported_claims": ["缺少适用部门"], "action": "ask_user"}
        result = EnterpriseQA(self.chunks, llm=VetoLLM()).answer("差旅报销需要什么材料？")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIsNone(result.answer)
        self.assertEqual(result.gaps, ["缺少适用部门"])
        self.assertTrue(result.citations)

    def test_query_plan_must_contain_non_empty_queries(self):
        class InvalidPlanner(MockLLM):
            def plan(self, question, history=None):
                return type("Plan", (), {"queries": [], "intent": "invalid"})()
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            EnterpriseQA(self.chunks, llm=InvalidPlanner()).answer("差旅报销")

    def test_planner_queries_are_merged_before_reranking(self):
        class MultiQueryLLM(MockLLM):
            def plan(self, question, history=None):
                return type("Plan", (), {"queries": [question, "差旅 报销 发票 行程单 审批记录"], "intent": "multi-query"})()
        result = EnterpriseQA(self.chunks, llm=MultiQueryLLM()).answer("需要发票吗？")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.citations[0]["document_id"], "D03")

    def test_evaluation_reports_retrieval_answer_and_abstention_metrics(self):
        report = evaluate_q3(
            Path(__file__).resolve().parents[1] / "data" / "q3_eval_cases.json",
            Path(__file__).resolve().parents[1] / "data" / "knowledge.db",
        )
        self.assertEqual(report["total"], 10)
        self.assertEqual(report["query_retrieval_accuracy"], 0.9)
        self.assertEqual(report["answer_exact_match_accuracy"], 1.0)
        self.assertEqual(report["answerable_cases"], 5)
        self.assertEqual(report["correct_abstention_accuracy"], 1.0)
        self.assertEqual(report["non_answerable_cases"], 5)
        housing = next(case for case in report["cases"] if case["case_id"] == "q3-04")
        self.assertEqual(housing["document_precision"], 1.0)
        self.assertEqual(housing["document_recall"], 1.0)

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
