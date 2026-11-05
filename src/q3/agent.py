from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .async_provider import run_sync
from .llm import MockLLM
from .models import Answer, Chunk
from .query_aware import MockQueryAwareExtractor
from .retrieval import EvidenceSelector, MockJevReranker, detect_conflict, hybrid_candidates, merge_candidates


class EnterpriseQA:
    def __init__(self, chunks: list[Chunk], candidate_limit: int = 200, max_evidence: int = 8,
                 semantic_retriever=None, reranker=None, llm=None, extractor=None, max_steps: int = 5):
        self.chunks = chunks
        self.candidate_limit = candidate_limit
        self.max_steps = max_steps
        self.semantic_retriever = semantic_retriever
        self.selector = EvidenceSelector(maximum=max_evidence)
        self.reranker = reranker or MockJevReranker()
        self.llm = llm or MockLLM()
        self.extractor = extractor or MockQueryAwareExtractor()

    async def _answer_once_async(self, question: str, history: list[dict] | None = None) -> Answer:
        plan = await self.llm.aplan(question, history) if hasattr(self.llm, "aplan") else await self.llm.plan_async(question, history) if hasattr(self.llm, "plan_async") else self.llm.plan(question, history)
        if not plan.queries or not all(isinstance(query, str) and query.strip() for query in plan.queries):
            raise ValueError("planner must return a non-empty list of query strings")
        if not isinstance(plan.intent, str):
            raise ValueError("planner intent must be a string")
        trace = [{"action": "plan_queries", "result": "queries planned", "queries": plan.queries, "intent": plan.intent}]
        groups = [hybrid_candidates(query, self.chunks, self.candidate_limit, self.semantic_retriever)
                  for query in plan.queries]
        candidates = merge_candidates(groups, self.candidate_limit)
        trace.append({"action": "hybrid_retrieve", "result": f"{len(candidates)} candidates"})
        ranked = await self.reranker.ascore(question, candidates) if hasattr(self.reranker, "ascore") else await self.reranker.score_async(question, candidates) if hasattr(self.reranker, "score_async") else self.reranker.score(question, candidates)
        top_k = ranked[:20]
        trace.append({"action": "rerank", "result": f"{len(ranked)} ranked, top {len(top_k)} selected"})
        extraction = await self.extractor.aextract(question, top_k) if hasattr(self.extractor, "aextract") else await self.extractor.extract_async(question, top_k) if hasattr(self.extractor, "extract_async") else self.extractor.extract(question, top_k)
        trace.append({"action": "query_aware_extract",
                      "result": f"{extraction.candidates_with_evidence}/{extraction.candidates_in} candidates have evidence",
                      "compression_ratio": round(extraction.compression_ratio, 3), "note": extraction.compression_note})
        evidence = self.selector.select(ranked)
        citations = self.selector.citations(evidence)
        trace.append({"action": "select_evidence", "result": f"{len(evidence)} evidence chunks"})
        if not evidence:
            return Answer("insufficient_evidence", None, [], ["没有检索到相关资料"], trace,
                          {"candidate_count": 0, "selected_count": 0}, {"verified": False})
        extracted_items = [top_k[e.index - 1] for e in extraction.evidence
                           if e.relevance != "none" and e.quote and 1 <= e.index <= len(top_k)]
        conflict = detect_conflict(extracted_items or evidence)
        versions_by_document = {}
        for item in (extracted_items or evidence):
            versions_by_document.setdefault(item.chunk.document_id, set()).add(item.chunk.version)
        version_conflict = any(len(versions) > 1 for versions in versions_by_document.values())
        if version_conflict:
            trace.append({"action": "verify_answer", "result": "multiple document versions; ask user"})
            return Answer("version_conflict", None, citations, ["同一文档检索到多个版本，无法自动选择"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence)},
                          {"verified": False, "conflict_detected": True, "version_conflict": True, "action": "ask_user"})
        if conflict:
            trace.append({"action": "verify_answer", "result": "conflict detected; ask user"})
            return Answer("conflict_detected", None, citations, ["检索到的制度内容存在冲突"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence)},
                          {"verified": False, "conflict_detected": True, "action": "ask_user"})
        answer_input = extraction.to_answer_input()
        extracted_citations = [
            {"document_id": e.source_id, "chunk_id": e.chunk_id,
             "version": next((c.chunk.version for c in top_k if c.chunk.chunk_id == e.chunk_id), ""),
             "quote": e.quote, "key_info": e.key_info}
            for e in extraction.evidence if e.relevance != "none" and e.quote
        ]
        if not extracted_citations:
            trace.append({"action": "answer_from_evidence", "result": "no relevant evidence extracted"})
            return Answer("insufficient_evidence", None, [], ["抽取后未发现与问题相关的证据"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence),
                           "extracted_evidence_count": 0, "answer_input": answer_input}, {"verified": False})
        generated = await self.llm.aanswer_from_evidence(question, answer_input) if hasattr(self.llm, "aanswer_from_evidence") else await self.llm.answer_from_evidence_async(question, answer_input) if hasattr(self.llm, "answer_from_evidence_async") else self.llm.answer_from_evidence(question, answer_input)
        answer_text = generated.get("answer", "")
        cited_indices = generated.get("citation_indices")
        if (not isinstance(cited_indices, list) or not cited_indices
                or any(type(index) is not int or index < 1 or index > len(extracted_citations) for index in cited_indices)
                or len(set(cited_indices)) != len(cited_indices)):
            trace.append({"action": "answer_from_evidence", "result": "invalid citation indices"})
            return Answer("insufficient_evidence", None, [], ["答案引用索引缺失、重复或无效"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence),
                           "extracted_evidence_count": extraction.candidates_with_evidence,
                           "answer_input": answer_input}, {"verified": False, "action": "ask_user"})
        valid_citations = [extracted_citations[index - 1] for index in cited_indices]
        trace.append({"action": "answer_from_evidence", "result": f"generated answer with {len(valid_citations)} citations"})
        verification = await self.llm.averify(question, answer_text, valid_citations) if hasattr(self.llm, "averify") else await self.llm.verify_async(question, answer_text, valid_citations) if hasattr(self.llm, "verify_async") else self.llm.verify(question, answer_text, valid_citations)
        trace.append({"action": "verify_answer", "result": "passed" if verification["verified"] else "unsupported claims"})
        status = "completed" if verification["verified"] and answer_text else "insufficient_evidence"
        gaps = [] if status == "completed" else verification.get("unsupported_claims", []) or ["答案未通过引用核验"]
        return Answer(status, answer_text if status == "completed" else None, valid_citations, gaps, trace,
                      {"candidate_count": len(candidates), "selected_count": len(evidence),
                       "extracted_evidence_count": extraction.candidates_with_evidence,
                       "compression_ratio": round(extraction.compression_ratio, 3), "answer_input": answer_input}, verification)

    def _answer_once(self, question: str, history: list[dict] | None = None) -> Answer:
        return run_sync(self._answer_once_async(question, history))

    async def answer_async(self, question: str, history: list[dict] | None = None) -> Answer:
        observations = []
        trace = []
        current_query = question
        result = None
        for step in range(self.max_steps):
            decision = await self.llm.adecide_action(question, observations, history) if hasattr(self.llm, "adecide_action") else await self.llm.decide_action_async(question, observations, history) if hasattr(self.llm, "decide_action_async") else self.llm.decide_action(question, observations, history)
            trace.append({"step": step + 1, "action": decision.action, "query": decision.query,
                          "tool_parameters": {"candidate_limit": self.candidate_limit, "rerank_top": 20},
                          "result_summary": "", "next_action_reason": decision.reason})
            if step == 0 and decision.action != "search":
                raise ValueError("the first ReAct action must be search")
            if decision.action == "search":
                current_query = decision.query or current_query
                result = await self._answer_once_async(current_query, history)
                observation = {"query": current_query, "evidence_count": len(result.citations),
                              "conflict": result.status in {"conflict_detected", "version_conflict"}, "status": result.status}
                if result.status in {"insufficient_evidence", "conflict_detected", "version_conflict"}:
                    observation["rewritten_query"] = current_query != question
                observations.append(observation)
                trace[-1]["result_summary"] = result.status
                if result.status in {"conflict_detected", "version_conflict"}:
                    return result
                continue
            if decision.action in {"ask_user", "insufficient_evidence", "stop"}:
                if result is None:
                    return Answer("insufficient_evidence", None, [], [decision.reason or "证据不足"], trace,
                                  {"candidate_count": 0, "selected_count": 0}, {"verified": False, "action": "ask_user"})
                result.trace = trace + result.trace
                if decision.action == "ask_user":
                    result.verification["action"] = "ask_user"
                if decision.action in {"insufficient_evidence", "stop"} and result.status == "completed":
                    result.status, result.answer = "insufficient_evidence", None
                return result
            if decision.action == "answer" and result is not None:
                result.trace = trace + result.trace
                return result
            raise ValueError(f"unsupported action: {decision.action}")
        if result is None:
            return Answer("insufficient_evidence", None, [], ["达到最大推理步数"], trace,
                          {"candidate_count": 0, "selected_count": 0}, {"verified": False, "action": "ask_user"})
        result.trace = trace + result.trace
        if result.status == "completed":
            result.status, result.answer = "insufficient_evidence", None
        result.gaps = result.gaps or ["达到最大推理步数"]
        result.verification["action"] = "ask_user"
        return result

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        return run_sync(self.answer_async(question, history))


def load_chunks(path: str | Path) -> list[Chunk]:
    source = Path(path)
    if source.suffix == ".db":
        connection = sqlite3.connect(source)
        rows = connection.execute(
            "SELECT c.chunk_id, d.document_id, d.version, d.version_date, c.text "
            "FROM document_chunks c JOIN documents d ON d.id=c.document_id ORDER BY c.id"
        ).fetchall()
        connection.close()
        return [Chunk(chunk_id=row[0], document_id=row[1], version=row[2], version_date=row[3], text=row[4]) for row in rows]
    records = json.loads(source.read_text(encoding="utf-8"))
    return [Chunk(**record) for record in records]
