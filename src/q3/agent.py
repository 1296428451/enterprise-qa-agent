from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .llm import MockLLM
from .models import Answer, Chunk
from .query_aware import MockQueryAwareExtractor
from .retrieval import EvidenceSelector, MockJevReranker, detect_conflict, hybrid_candidates, merge_candidates


class EnterpriseQA:
    def __init__(self, chunks: list[Chunk], candidate_limit: int = 50, max_evidence: int = 8,
                 semantic_retriever=None, reranker=None, llm=None, extractor=None):
        self.chunks = chunks
        self.candidate_limit = candidate_limit
        self.semantic_retriever = semantic_retriever
        self.selector = EvidenceSelector(maximum=max_evidence)
        self.reranker = reranker or MockJevReranker()
        self.llm = llm or MockLLM()
        self.extractor = extractor or MockQueryAwareExtractor()

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        plan = self.llm.plan(question, history)
        if not plan.queries or not all(isinstance(query, str) and query.strip() for query in plan.queries):
            raise ValueError("planner must return a non-empty list of query strings")
        if not isinstance(plan.intent, str):
            raise ValueError("planner intent must be a string")
        trace = [{"action": "plan_queries", "result": "queries planned", "queries": plan.queries, "intent": plan.intent}]
        groups = [hybrid_candidates(query, self.chunks, self.candidate_limit, self.semantic_retriever)
                  for query in plan.queries]
        candidates = merge_candidates(groups, self.candidate_limit)
        trace.append({"action": "hybrid_retrieve", "result": f"{len(candidates)} candidates"})
        ranked = self.reranker.score(question, candidates)
        top_k = ranked[:20]
        trace.append({"action": "rerank", "result": f"{len(ranked)} ranked, top {len(top_k)} selected"})
        extraction = self.extractor.extract(question, top_k)
        trace.append({
            "action": "query_aware_extract",
            "result": f"{extraction.candidates_with_evidence}/{extraction.candidates_in} candidates have evidence",
            "compression_ratio": round(extraction.compression_ratio, 3),
            "note": extraction.compression_note,
        })
        evidence = self.selector.select(ranked)
        citations = self.selector.citations(evidence)
        trace.append({"action": "select_evidence", "result": f"{len(evidence)} evidence chunks"})
        if not evidence:
            return Answer("insufficient_evidence", None, [], ["没有检索到相关资料"], trace,
                          {"candidate_count": 0, "selected_count": 0}, {"verified": False})
        conflict = detect_conflict(evidence)
        versions_by_document = {}
        for item in evidence:
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
        answer = evidence[0].chunk.text
        verification = self.llm.verify(question, answer, citations)
        trace.append({"action": "verify_answer", "result": "passed" if verification["verified"] else "unsupported claims"})
        status = "completed" if verification["verified"] else "insufficient_evidence"
        gaps = [] if verification["verified"] else verification["unsupported_claims"] or ["答案未通过引用核验"]
        return Answer(status, answer if verification["verified"] else None, citations, gaps, trace,
                      {"candidate_count": len(candidates), "selected_count": len(evidence),
                       "extracted_evidence_count": extraction.candidates_with_evidence,
                       "compression_ratio": round(extraction.compression_ratio, 3),
                       "answer_input": answer_input}, verification)


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
