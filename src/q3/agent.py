from __future__ import annotations

import json
from pathlib import Path

from .models import Answer, Chunk
from .retrieval import EvidenceSelector, MockJevReranker, detect_conflict, hybrid_candidates


class EnterpriseQA:
    def __init__(self, chunks: list[Chunk], candidate_limit: int = 50, max_evidence: int = 8):
        self.chunks = chunks
        self.candidate_limit = candidate_limit
        self.selector = EvidenceSelector(maximum=max_evidence)
        self.reranker = MockJevReranker()

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        trace = [{"action": "plan_queries", "result": "deterministic query plan", "query": question}]
        candidates = hybrid_candidates(question, self.chunks, self.candidate_limit)
        trace.append({"action": "hybrid_retrieve", "result": f"{len(candidates)} candidates"})
        ranked = self.reranker.score(question, candidates)
        trace.append({"action": "jev_rerank", "result": f"{len(ranked)} ranked candidates"})
        evidence = self.selector.select(ranked)
        citations = self.selector.citations(evidence)
        trace.append({"action": "select_evidence", "result": f"{len(evidence)} evidence chunks"})
        if not evidence:
            return Answer("insufficient_evidence", None, [], ["没有检索到相关资料"], trace,
                          {"candidate_count": 0, "selected_count": 0}, {"verified": False})
        conflict = detect_conflict(evidence)
        if conflict:
            trace.append({"action": "verify_answer", "result": "conflict detected; ask user"})
            return Answer("conflict_detected", None, citations, ["检索到的制度内容存在冲突"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence)},
                          {"verified": False, "conflict_detected": True, "action": "ask_user"})
        answer = evidence[0].chunk.text
        trace.append({"action": "verify_answer", "result": "passed"})
        return Answer("completed", answer, citations, [], trace,
                      {"candidate_count": len(candidates), "selected_count": len(evidence)},
                      {"verified": True, "conflict_detected": False})


def load_chunks(path: str | Path) -> list[Chunk]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**record) for record in records]
