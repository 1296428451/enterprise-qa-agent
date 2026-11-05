import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.database import create_database
from src.q3.evaluate import evaluate as evaluate_q3
from src.q3.llm import MockLLM, OpenAICompatibleLLM
from src.q3.models import ScoredChunk
from src.q3.provider_reranker import ProviderReranker


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(Path(__file__).resolve().parents[1] / "data" / "q3_documents.json")


def test_sqlite_schema_is_created():
    import tempfile
    import sqlite3
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "knowledge.db"
        create_database(path)
        connection = sqlite3.connect(path)
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        connection.close()
        assert "documents" in names
        assert "document_chunks" in names
        assert "retrieval_results" in names


def test_mock_llm_adapter_is_deterministic():
    planner = MockLLM()
    plan = planner.plan("报销期限多久？")
    assert plan.queries == ["报销期限多久？"]
    assert planner.verify("报销期限多久？", "30天", [{"document_id": "D01"}])["verified"]


def test_openai_compatible_llm_parses_plan_and_verification():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    responses = [
        {"queries": ["报销期限", "费用提交日期"], "intent": "查询报销截止时间"},
        {"verified": True, "unsupported_claims": [], "action": "answer"},
    ]
    with patch.object(llm, "_complete_json", side_effect=responses) as complete:
        plan = llm.plan("报销期限多久？", [{"role": "user", "content": "报销"}])
        verified = llm.verify("报销期限多久？", "30天", [{"document_id": "D01", "quote": "30天内"}])
    assert plan.queries == ["报销期限", "费用提交日期"]
    assert verified["verified"]
    assert complete.call_count == 2


def test_openai_compatible_llm_parses_and_validates_react_actions():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    with patch.object(llm, "_complete_json", return_value={
        "action": "search", "query": "差旅报销材料", "reason": "改写查询"
    }):
        decision = llm.decide_action("需要什么材料？", [])
    assert (decision.action, decision.query, decision.reason) == ("search", "差旅报销材料", "改写查询")

    for payload, message in (
        ({"action": "search"}, "requires a non-empty query"),
        ({"action": "search", "query": "  "}, "requires a non-empty query"),
        ({"action": "stop", "reason": "结束"}, "action must be search, answer, ask_user, or insufficient_evidence"),
        ({"action": "unknown"}, "action must be search, answer, ask_user, or insufficient_evidence"),
    ):
        with patch.object(llm, "_complete_json", return_value=payload):
            with pytest.raises(ValueError, match=message):
                llm.decide_action("问题", [])


def test_openai_compatible_llm_rejects_malformed_plan():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    with patch.object(llm, "_complete_json", return_value={"queries": [], "intent": "x"}):
        with pytest.raises(ValueError):
            llm.plan("报销期限多久？")
    with patch.object(llm, "_complete_json", return_value={
        "verified": True, "unsupported_claims": ["unsupported"], "action": "answer"
    }):
        with pytest.raises(ValueError):
            llm.verify("报销期限多久？", "answer", [])


def test_openai_compatible_llm_sends_auth_and_parses_http_json():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "secret-value")
    response = io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"queries":["报销"],"intent":"lookup"}'}}]}).encode())
    with patch("src.q3.llm.urlopen", return_value=response) as open_url:
        plan = llm.plan("报销期限？")
    request = open_url.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer secret-value"
    assert plan.queries == ["报销"]


def test_provider_reranker_orders_candidates_by_provider_score(chunks):
    items = [ScoredChunk(chunks[0], bm25_score=0.4), ScoredChunk(chunks[1], bm25_score=0.8)]
    body = {"data": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9}]}
    response = io.BytesIO(json.dumps(body).encode())
    reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "secret-value")
    with patch("src.q3.provider_reranker.urlopen", return_value=response):
        ranked = reranker.score("报销期限", items)
    assert ranked[0].chunk.chunk_id == chunks[1].chunk_id
    assert ranked[0].rerank_score == 0.9


def test_provider_reranker_supports_openai_chat_endpoint(chunks):
    items = [ScoredChunk(chunks[0]), ScoredChunk(chunks[1])]
    body = {"choices": [{"message": {"content": '{"scores":[0.1,0.8]}'}}]}
    reranker = ProviderReranker("https://example.invalid/v1/chat/completions", "test-model", "secret-value")
    with patch("src.q3.provider_reranker.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
        ranked = reranker.score("报销期限", items)
    assert reranker.endpoint_type == "chat"
    assert ranked[0].chunk.chunk_id == chunks[1].chunk_id


def test_provider_reranker_rejects_malformed_chat_scores(chunks):
    reranker = ProviderReranker("https://example.invalid/v1/chat/completions", "test-model", "test-key")
    body = {"choices": [{"message": {"content": '{"scores":[true]}'}}]}
    with patch("src.q3.provider_reranker.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
        with pytest.raises(ValueError, match="scores"):
            reranker.score("报销", [ScoredChunk(chunks[0])])


    item = ScoredChunk(chunks[0])
    reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "test-key")
    with patch("src.q3.provider_reranker.urlopen", return_value=io.BytesIO(b'{"data":[]}')):
        with pytest.raises(ValueError):
            reranker.score("报销", [item])


def test_provider_reranker_builds_v1_endpoint():
    reranker = ProviderReranker("https://example.invalid/v1", "test-model", "test-key")
    assert reranker.base_url == "https://example.invalid/v1/rerank"


@pytest.mark.parametrize("citation_indices", [None, [], [0], [2], [1, 1], [True], ["1"]])
def test_invalid_or_missing_citation_indices_abstain_without_fallback(chunks, citation_indices):
    class InvalidCitationLLM(MockLLM):
        def answer_from_evidence(self, question, evidence_input):
            return {"answer": "未被有效引用的答案", "citation_indices": citation_indices}

        def verify(self, question, answer, citations):
            raise AssertionError("verification must not accept invalid citations")

    result = EnterpriseQA(chunks, llm=InvalidCitationLLM()).answer("差旅报销需要什么材料？")
    assert result.status == "insufficient_evidence"
    assert result.answer is None
    assert result.citations == []
    assert result.gaps == ["答案引用索引缺失、重复或无效"]
    assert result.verification["verified"] is False


def test_openai_verifier_can_veto_answer_and_preserve_claims(chunks):
    class VetoLLM(MockLLM):
        def verify(self, question, answer, citations):
            return {"verified": False, "unsupported_claims": ["缺少适用部门"], "action": "ask_user"}
    result = EnterpriseQA(chunks, llm=VetoLLM()).answer("差旅报销需要什么材料？")
    assert result.status == "insufficient_evidence"
    assert result.answer is None
    assert result.gaps == ["缺少适用部门"]
    assert result.citations


def test_query_plan_must_contain_non_empty_queries(chunks):
    class InvalidPlanner(MockLLM):
        def plan(self, question, history=None):
            return type("Plan", (), {"queries": [], "intent": "invalid"})()
    with pytest.raises(ValueError, match="non-empty list"):
        EnterpriseQA(chunks, llm=InvalidPlanner()).answer("差旅报销")


def test_planner_queries_are_merged_before_reranking(chunks):
    class MultiQueryLLM(MockLLM):
        def plan(self, question, history=None):
            return type("Plan", (), {"queries": [question, "差旅 报销 发票 行程单 审批记录"], "intent": "multi-query"})()
    result = EnterpriseQA(chunks, llm=MultiQueryLLM()).answer("需要发票吗？")
    assert result.status == "completed"
    assert result.citations[0]["document_id"] == "D03"


def test_evaluation_reports_retrieval_answer_and_abstention_metrics():
    report = evaluate_q3(
        Path(__file__).resolve().parents[1] / "data" / "q3_eval_cases.json",
        Path(__file__).resolve().parents[1] / "data" / "knowledge.db",
    )
    assert report["total"] == 10
    assert report["query_retrieval_accuracy"] == 0.9
    assert report["answer_exact_match_accuracy"] == 1.0
    assert report["answerable_cases"] == 5
    assert report["correct_abstention_accuracy"] == 1.0
    assert report["non_answerable_cases"] == 5
    housing = next(case for case in report["cases"] if case["case_id"] == "q3-04")
    assert housing["document_precision"] == 1.0
    assert housing["document_recall"] == 1.0


def test_conflicting_policies_are_not_flattened(chunks):
    result = EnterpriseQA(chunks).answer("普通费用报销期限多久？")
    assert result.status == "conflict_detected"
    assert {item["document_id"] for item in result.citations} == {"D01", "D02"}
    assert result.verification["action"] == "ask_user"


def test_specific_question_returns_citation(chunks):
    result = EnterpriseQA(chunks).answer("差旅报销需要什么材料？")
    assert result.status == "completed"
    assert result.citations[0]["document_id"] == "D03"
    assert result.verification["verified"]


def test_react_rewrites_query_after_empty_search(chunks):
    class RewriteLLM(MockLLM):
        def __init__(self):
            self.decisions = [("search", "不存在的关键词"), ("search", "差旅报销需要什么材料？"), ("answer", None)]
        def decide_action(self, question, observations, history=None):
            action, query = self.decisions[min(len(observations), len(self.decisions) - 1)]
            return type("Decision", (), {"action": action, "query": query, "reason": "测试"})()
    result = EnterpriseQA(chunks, llm=RewriteLLM()).answer("材料要求是什么？")
    assert result.status == "completed"
    assert any(item.get("action") == "search" and item.get("query") == "不存在的关键词" for item in result.trace)
    assert any(item.get("action") == "search" and item.get("query") == "差旅报销需要什么材料？" for item in result.trace)


def test_react_provider_decisions_rewrite_empty_search_then_answer(chunks):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    decisions = [
        {"action": "search", "query": "不存在的关键词", "reason": "首次检索"},
        {"action": "search", "query": "差旅报销需要什么材料？", "reason": "改写查询"},
        {"action": "answer", "reason": "证据充分"},
    ]
    observations_seen = []

    def complete_json(instruction, payload):
        if instruction.startswith("Choose exactly one action:"):
            observations_seen.append(json.loads(json.dumps(payload["observations"])))
            return decisions.pop(0)
        if instruction.startswith("Generate retrieval queries"):
            return {"queries": [payload["question"]], "intent": "查询制度"}
        if instruction.startswith("Answer using only numbered evidence"):
            return {"answer": "需要发票和行程单", "citation_indices": [1]}
        if instruction.startswith("Check whether every claim"):
            return {"verified": True, "unsupported_claims": [], "action": "answer"}
        raise AssertionError(f"unexpected LLM instruction: {instruction}")

    with patch.object(llm, "_complete_json", side_effect=complete_json) as complete:
        result = EnterpriseQA(chunks, llm=llm).answer("材料要求是什么？")

    assert result.status == "completed"
    assert {citation["document_id"] for citation in result.citations} == {"D03"}
    search_steps = [step for step in result.trace if step.get("action") == "search"]
    assert [step["query"] for step in search_steps] == ["不存在的关键词", "差旅报销需要什么材料？"]
    assert observations_seen == [
        [],
        [{"query": "不存在的关键词", "evidence_count": 0, "conflict": False,
          "status": "insufficient_evidence", "rewritten_query": True}],
        [{"query": "不存在的关键词", "evidence_count": 0, "conflict": False,
          "status": "insufficient_evidence", "rewritten_query": True},
         {"query": "差旅报销需要什么材料？", "evidence_count": len(result.citations), "conflict": False,
          "status": "completed"}],
    ]
    assert complete.call_count == 7


def test_react_first_action_must_be_search(chunks):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    for action in ("answer", "ask_user", "insufficient_evidence"):
        with patch.object(llm, "_complete_json", return_value={"action": action, "reason": "首步动作"}):
            with pytest.raises(ValueError, match="first ReAct action must be search"):
                EnterpriseQA(chunks, llm=llm).answer("测试问题")


def test_react_max_steps_caps_search_loop(chunks):
    class EndlessLLM(MockLLM):
        def decide_action(self, question, observations, history=None):
            return type("Decision", (), {"action": "search", "query": "无关问题", "reason": "继续"})()
    result = EnterpriseQA(chunks, llm=EndlessLLM(), max_steps=5).answer("无关问题")
    assert result.status == "insufficient_evidence"
    assert len([item for item in result.trace if item.get("step")]) == 5


def test_react_out_of_scope_stays_insufficient(chunks):
    result = EnterpriseQA(chunks).answer("公司是否提供住房补贴？")
    assert result.status == "insufficient_evidence"
    assert not result.verification["verified"]


def test_unrelated_employee_benefits_query_has_insufficient_evidence(chunks):
    result = EnterpriseQA(chunks).answer("员工福利积分如何兑换？")
    assert result.status == "insufficient_evidence"
    assert result.citations == []

    result = EnterpriseQA(chunks, max_evidence=1).answer("审批进度在哪里查看？")
    assert len(result.citations) <= 1
