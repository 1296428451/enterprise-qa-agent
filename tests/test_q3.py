import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
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


def test_rerank_top_limits_candidates_entering_extraction(chunks):
    """config.retrieval.rerank_top_k must actually reach the pipeline."""
    from src.q3.query_aware import ExtractionResult

    seen = {}

    class RecordingExtractor:
        def extract(self, question, candidates):
            seen["count"] = len(candidates)
            return ExtractionResult(question=question, evidence=[], candidates_in=len(candidates),
                                    candidates_with_evidence=0, total_source_chars=0,
                                    total_extracted_chars=0, compression_note="test",
                                    token_budget=20000, candidates_skipped_budget=0)

    # A broad query matches several chunks; the cap must truncate them.
    qa = EnterpriseQA(chunks, extractor=RecordingExtractor(), rerank_top=1)
    result = qa.answer("报销 期限 材料 审批")
    assert seen["count"] == 1
    assert result.trace[0]["step"] == 1
    assert result.trace[0]["tool_parameters"]["rerank_top"] == 1


def _both_clause_reranker(chunks):
    by_id = {chunk.document_id: chunk for chunk in chunks}

    class BothReranker:
        def score(self, query, items):
            # Force both the general and the department-specific clause into evidence,
            # which is what a provider reranker does and what triggers the conflict path.
            return [ScoredChunk(by_id["D02"], 1.0, 0.9, 1.0),
                    ScoredChunk(by_id["D01"], 0.9, 0.8, 0.9)]

    return BothReranker()


def test_scope_specific_clause_outranks_general_clause(chunks):
    qa = EnterpriseQA(chunks, reranker=_both_clause_reranker(chunks))
    result = qa.answer("研发部门有财务要报销，需要多长时间才可以？")
    assert result.status == "completed"
    assert [citation["document_id"] for citation in result.citations] == ["D02"]
    assert "15 天" in (result.answer or "")
    actions = [step.get("action") for step in result.trace]
    # The heuristic still flags a possible conflict, but the answer call decides.
    assert "detect_conflict" in actions
    assert "answer_from_evidence" in actions


def test_general_and_specific_clause_still_conflict_without_scope_mention(chunks):
    qa = EnterpriseQA(chunks, reranker=_both_clause_reranker(chunks))
    result = qa.answer("报销期限多久？")
    assert result.status == "conflict_detected"
    assert {citation["document_id"] for citation in result.citations} == {"D01", "D02"}
    # The conflict comes from the answer call, not from an early rule return.
    calls = [step for step in result.trace if step.get("action") == "answer_from_evidence"]
    assert calls and "conflicting" in calls[-1]["result"]


def test_answer_call_resolves_scope_and_reports_real_conflicts():
    from src.q3.llm import MockLLM, parse_evidence

    evidence = """Question: 研发部门有财务要报销，需要多长时间才可以？

Evidence:
[1] (D01) 普通费用报销
    Quote: 普通费用报销应在费用发生后 30 天内提交。
    Scope: 所有正式员工
[2] (D02) 差旅报销
    Quote: 普通费用报销应在费用发生后 15 天内提交。
    Scope: 研发部门"""
    parsed = parse_evidence(evidence)
    assert [entry["source_id"] for entry in parsed] == ["D01", "D02"]
    assert parsed[1]["scope"] == "研发部门"

    scoped = MockLLM().answer_from_evidence("研发部门有财务要报销，需要多长时间才可以？", evidence)
    assert scoped["conflict"] is False
    assert scoped["citation_indices"] == [2]
    assert "15 天" in scoped["answer"]

    ambiguous = MockLLM().answer_from_evidence("报销期限多久？", evidence)
    assert ambiguous["conflict"] is True
    assert ambiguous["answer"] == "" and ambiguous["citation_indices"] == []


class _ScriptedAnswerLLM(MockLLM):
    """Reuse the deterministic planner/verifier, script only the answer payload."""

    def __init__(self, payload):
        self.payload = payload

    def answer_from_evidence(self, question, evidence_input):
        return dict(self.payload)


def test_declined_answer_still_returns_a_summary_with_gap(chunks):
    """A model that declines must not leave the caller with a null answer."""
    gap = "制度未规定审批与到账周期"
    llm = _ScriptedAnswerLLM({"answer": "", "citation_indices": [], "conflict": False, "gap": gap})
    result = EnterpriseQA(chunks, llm=llm, reranker=_both_clause_reranker(chunks)).answer(
        "研发部门有财务要报销，需要多长时间才可以？")
    assert result.status == "completed"
    assert result.answer and "15 天" in result.answer      # closest clause is quoted
    assert gap in result.answer                             # the missing part is stated
    assert result.citations                                 # still traceable
    assert result.gaps == [gap]
    results = [step.get("result") or "" for step in result.trace if step.get("action") == "answer_from_evidence"]
    assert any("summarised the closest evidence" in text for text in results)


def test_partial_answer_keeps_model_gap(chunks):
    gap = "制度未规定审批与到账周期"
    llm = _ScriptedAnswerLLM({"answer": "研发部门需在费用发生后 15 天内提交。",
                              "citation_indices": [1], "conflict": False, "gap": gap})
    result = EnterpriseQA(chunks, llm=llm, reranker=_both_clause_reranker(chunks)).answer(
        "研发部门有财务要报销，需要多长时间才可以？")
    assert result.status == "completed"
    assert result.answer and "15 天" in result.answer
    assert result.gaps == [gap]


def _sse_content(content):
    delta = json.dumps({"choices": [{"delta": {"content": content}}]})
    return httpx.Response(200, content=f"data: {delta}\n\ndata: [DONE]\n\n".encode())


def _sse_transport(content):
    return httpx.MockTransport(lambda request: _sse_content(content))


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
    async def complete(instruction, payload):
        return responses.pop(0)
    with patch.object(llm, "_complete_json_async", side_effect=complete) as mocked:
        plan = llm.plan("报销期限多久？", [{"role": "user", "content": "报销"}])
        verified = llm.verify("报销期限多久？", "30天", [{"document_id": "D01", "quote": "30天内"}])
    assert plan.queries == ["报销期限", "费用提交日期"]
    assert verified["verified"]
    assert mocked.call_count == 2


def test_openai_compatible_llm_parses_and_validates_react_actions():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    with patch.object(llm, "_complete_json_async", return_value={
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
        with patch.object(llm, "_complete_json_async", return_value=payload):
            with pytest.raises(ValueError, match=message):
                llm.decide_action("问题", [])


def test_openai_compatible_llm_rejects_malformed_plan():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    with patch.object(llm, "_complete_json_async", return_value={"queries": [], "intent": "x"}):
        with pytest.raises(ValueError):
            llm.plan("报销期限多久？")
    with patch.object(llm, "_complete_json_async", return_value={
        "verified": True, "unsupported_claims": ["unsupported"], "action": "answer"
    }):
        with pytest.raises(ValueError):
            llm.verify("报销期限多久？", "answer", [])


def test_openai_compatible_llm_sends_auth_and_parses_http_json():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _sse_content('{"queries":["报销"],"intent":"lookup"}')

    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "secret-value",
                              transport=httpx.MockTransport(handler))
    plan = llm.plan("报销期限？")
    assert seen["url"] == "https://example.invalid/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-value"
    assert plan.queries == ["报销"]


def test_provider_reranker_orders_candidates_by_provider_score(chunks):
    items = [ScoredChunk(chunks[0], bm25_score=0.4), ScoredChunk(chunks[1], bm25_score=0.8)]
    body = {"data": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9}]}
    reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "secret-value",
                                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    ranked = reranker.score("报销期限", items)
    assert ranked[0].chunk.chunk_id == chunks[1].chunk_id
    assert ranked[0].rerank_score == 0.9


def test_provider_reranker_supports_results_response_and_normalizes_native_endpoint(chunks):
    items = [ScoredChunk(chunks[0]), ScoredChunk(chunks[1])]
    body = {"results": [
        {"index": 1, "relevance_score": 0.9, "document": {"text": "kept by provider"}},
        {"index": 0, "relevance_score": 0.2},
    ]}
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "test-key",
                                transport=httpx.MockTransport(handler))
    ranked = reranker.score("报销期限", items)
    assert ranked[0].chunk.chunk_id == chunks[1].chunk_id
    assert ranked[0].rerank_score == 0.9
    assert seen["url"] == "https://example.invalid/v1/rerank"
    assert seen["body"]["top_n"] == 2 and seen["body"]["return_documents"] is True


def test_provider_reranker_supports_openai_chat_endpoint(chunks):
    items = [ScoredChunk(chunks[0]), ScoredChunk(chunks[1])]
    reranker = ProviderReranker("https://example.invalid/v1/chat/completions", "test-model", "secret-value",
                                transport=_sse_transport('{"scores":[0.1,0.8]}'))
    ranked = reranker.score("报销期限", items)
    assert reranker.endpoint_type == "chat"
    assert ranked[0].chunk.chunk_id == chunks[1].chunk_id


def test_provider_reranker_rejects_malformed_chat_scores(chunks):
    reranker = ProviderReranker("https://example.invalid/v1/chat/completions", "test-model", "secret-value",
                                transport=_sse_transport('{"scores":[true]}'))
    with pytest.raises(ValueError, match="scores"):
        reranker.score("报销", [ScoredChunk(chunks[0])])

    item = ScoredChunk(chunks[0])
    reranker = ProviderReranker("https://example.invalid/v1/rerank", "test-model", "test-key",
                                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []})))
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
    assert result.gaps == ["答案的引用编号缺失、越界或重复"]
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
    # Conflict is an early-return branch; it must still expose the ReAct
    # decision step so the run record stays auditable.
    react_steps = [item for item in result.trace if "step" in item]
    assert react_steps and react_steps[0]["action"] == "search"
    assert react_steps[0]["result_summary"] == "conflict_detected"
    assert "next_action_reason" in react_steps[0]


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

    async def complete(instruction, payload):
        if instruction.startswith("Choose exactly one action:"):
            observations_seen.append(json.loads(json.dumps(payload["observations"])))
            return decisions.pop(0)
        if instruction.startswith("Generate retrieval queries"):
            return {"queries": [payload["question"]], "intent": "查询制度"}
        if instruction.startswith("Answer the question using only the numbered evidence"):
            return {"answer": "需要发票和行程单", "citation_indices": [1], "conflict": False}
        if instruction.startswith("Check whether every claim"):
            return {"verified": True, "unsupported_claims": [], "action": "answer"}
        raise AssertionError(f"unexpected LLM instruction: {instruction}")

    with patch.object(llm, "_complete_json_async", side_effect=complete) as complete_mock:
        result = EnterpriseQA(chunks, llm=llm).answer("材料要求是什么？")

    assert result.status == "completed"
    assert {citation["document_id"] for citation in result.citations} == {"D03"}
    search_steps = [step for step in result.trace if step.get("action") == "search"]
    assert [step["query"] for step in search_steps] == ["不存在的关键词", "差旅报销需要什么材料？"]
    # The observation now carries the verified evidence back to the decision
    # model, so the decision step can see the answer it just produced.
    assert observations_seen[0] == []
    assert observations_seen[1] == [
        {"query": "不存在的关键词", "evidence_count": 0, "conflict": False,
         "status": "insufficient_evidence", "verified": False, "rewritten_query": True},
    ]
    assert len(observations_seen[2]) == 2
    second = observations_seen[2][1]
    assert second["query"] == "差旅报销需要什么材料？"
    assert second["status"] == "completed"
    assert second["conflict"] is False
    assert second["evidence_count"] == len(result.citations)
    assert second["verified"] is True
    assert second["answer_preview"] == "需要发票和行程单"
    assert second["evidence_quotes"] and all(isinstance(q, str) and q for q in second["evidence_quotes"])
    assert complete_mock.call_count == 7


def test_react_first_action_must_be_search(chunks):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "test-model", "test-key")
    for action in ("answer", "ask_user", "insufficient_evidence"):
        with patch.object(llm, "_complete_json_async", return_value={"action": action, "reason": "首步动作"}):
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
