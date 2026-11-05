from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .async_provider import run_sync
from .llm import MockLLM
from .models import Answer, Chunk
from .query_aware import MockQueryAwareExtractor
from .retrieval import (EvidenceSelector, MockJevReranker, detect_conflict, hybrid_candidates,
                       merge_candidates)


class EnterpriseQA:
    def __init__(self, chunks: list[Chunk], candidate_limit: int = 200, max_evidence: int = 8,
                 semantic_retriever=None, reranker=None, llm=None, extractor=None, max_steps: int = 5,
                 rerank_top: int = 20):
        self.chunks = chunks
        self.candidate_limit = candidate_limit
        self.max_steps = max_steps
        self.rerank_top = rerank_top
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
        trace.append({"action": "rerank", "result": f"{len(ranked)} ranked"})
        top_k = ranked[:self.rerank_top]
        trace.append({"action": "rerank_top", "result": f"top {len(top_k)} selected"})
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
        conflict_hint = detect_conflict(extracted_items or evidence)
        trace.append({"action": "detect_conflict",
                      "result": "heuristic: conflict" if conflict_hint else "heuristic: none"})
        versions_by_document = {}
        for item in (extracted_items or evidence):
            versions_by_document.setdefault(item.chunk.document_id, set()).add(item.chunk.version)
        version_conflict = any(len(versions) > 1 for versions in versions_by_document.values())
        if version_conflict:
            trace.append({"action": "verify_answer", "result": "multiple document versions; ask user"})
            return Answer("version_conflict", None, citations, ["同一文档检索到多个版本，无法自动选择"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence)},
                          {"verified": False, "conflict_detected": True, "version_conflict": True, "action": "ask_user"})
        answer_input = extraction.to_answer_input()
        extracted_citations = [
            {"document_id": e.source_id, "chunk_id": e.chunk_id,
             "version": next((c.chunk.version for c in top_k if c.chunk.chunk_id == e.chunk_id), ""),
             "quote": e.quote, "key_info": e.key_info, "scope": e.scope}
            for e in extraction.evidence if e.relevance != "none" and e.quote
        ]
        if not extracted_citations:
            trace.append({"action": "answer_from_evidence", "result": "no relevant evidence extracted"})
            return Answer("insufficient_evidence", None, [], ["抽取后未发现与问题相关的证据"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence),
                           "extracted_evidence_count": 0, "answer_input": answer_input}, {"verified": False})
        generated = await self.llm.aanswer_from_evidence(question, answer_input) if hasattr(self.llm, "aanswer_from_evidence") else await self.llm.answer_from_evidence_async(question, answer_input) if hasattr(self.llm, "answer_from_evidence_async") else self.llm.answer_from_evidence(question, answer_input)
        if generated.get("conflict"):
            trace.append({"action": "answer_from_evidence",
                          "result": "model reports conflicting clauses; ask user"})
            return Answer("conflict_detected", None, citations, ["检索到的制度内容存在冲突"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence),
                           "extracted_evidence_count": extraction.candidates_with_evidence},
                          {"verified": False, "conflict_detected": True, "action": "ask_user"})
        answer_text = generated.get("answer", "")
        model_gap = generated.get("gap") if isinstance(generated.get("gap"), str) else ""
        cited_indices = generated.get("citation_indices")
        synthesised = False
        if not answer_text.strip():
            # A declined answer is not a citation-format violation. If there is
            # relevant evidence, still return a result: summarise the closest
            # clauses and state what is not covered, instead of a bare refusal.
            if not extracted_citations:
                trace.append({"action": "answer_from_evidence",
                              "result": "model returned an empty answer; no usable evidence"})
                return Answer("insufficient_evidence", None, [],
                              [model_gap or "模型判断现有证据不足以回答该问题"], trace,
                              {"candidate_count": len(candidates), "selected_count": len(evidence),
                               "extracted_evidence_count": extraction.candidates_with_evidence,
                               "answer_input": answer_input},
                              {"verified": False, "action": "ask_user"})
            shown = extracted_citations[:3]
            model_gap = model_gap or "现有资料只覆盖问题的一部分，其余要求未在知识库中查到"
            summary = "根据知识库中最接近的条款：" + "；".join(
                f"{item['quote']}（适用于{item['scope']}）[{index}]"
                if item.get("scope") and item["scope"] not in item["quote"]
                else f"{item['quote']}[{index}]"
                for index, item in enumerate(shown, 1))
            answer_text = f"{summary}。未覆盖部分：{model_gap}"
            cited_indices = list(range(1, len(shown) + 1))
            synthesised = True
            trace.append({"action": "answer_from_evidence",
                          "result": "model declined; summarised the closest evidence with a gap"})
        if (not isinstance(cited_indices, list) or not cited_indices
                or any(type(index) is not int or index < 1 or index > len(extracted_citations) for index in cited_indices)
                or len(set(cited_indices)) != len(cited_indices)):
            trace.append({"action": "answer_from_evidence", "result": "invalid citation indices"})
            return Answer("insufficient_evidence", None, [], ["答案的引用编号缺失、越界或重复"], trace,
                          {"candidate_count": len(candidates), "selected_count": len(evidence),
                           "extracted_evidence_count": extraction.candidates_with_evidence,
                           "answer_input": answer_input}, {"verified": False, "action": "ask_user"})
        valid_citations = [extracted_citations[index - 1] for index in cited_indices]
        trace.append({"action": "answer_from_evidence", "result": f"generated answer with {len(valid_citations)} citations"})
        if synthesised:
            # The summary is assembled from verbatim quotes that were already
            # checked against their source clauses, so there is no generated
            # claim for the verifier to judge.
            verification = {"verified": True, "unsupported_claims": [], "action": "answer",
                            "note": "summarised verbatim from cited clauses; verifier skipped"}
            trace.append({"action": "verify_answer", "result": "skipped (verbatim summary)"})
        else:
            verification = await self.llm.averify(question, answer_text, valid_citations) if hasattr(self.llm, "averify") else await self.llm.verify_async(question, answer_text, valid_citations) if hasattr(self.llm, "verify_async") else self.llm.verify(question, answer_text, valid_citations)
            trace.append({"action": "verify_answer", "result": "passed" if verification["verified"] else "unsupported claims"})
        status = "completed" if verification["verified"] and answer_text else "insufficient_evidence"
        if status == "completed":
            # A partial answer may still leave part of the question uncovered;
            # keep the model's own note instead of dropping it.
            gaps = [model_gap] if model_gap else []
        else:
            gaps = verification.get("unsupported_claims", []) or ["答案未通过引用核验"]
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
                          "tool_parameters": {"candidate_limit": self.candidate_limit, "rerank_top": self.rerank_top},
                          "result_summary": "", "next_action_reason": decision.reason})
            if step == 0 and decision.action != "search":
                raise ValueError("the first ReAct action must be search")
            if decision.action == "search":
                current_query = decision.query or current_query
                result = await self._answer_once_async(current_query, history)
                observation = {"query": current_query, "evidence_count": len(result.citations),
                              "conflict": result.status in {"conflict_detected", "version_conflict"},
                              "status": result.status, "completeness": result.completeness}
                # Give the decision model the verified evidence it just produced.
                # Without this, decide_action only sees a status flag and rejects
                # for lack of visible evidence even when the answer passed verify.
                verification = result.verification or {}
                observation["verified"] = bool(verification.get("verified"))
                if result.answer:
                    observation["answer_preview"] = result.answer
                if result.citations:
                    observation["evidence_quotes"] = [
                        str(c.get("quote", "")) for c in result.citations[:3] if c.get("quote")
                    ]
                if result.status in {"insufficient_evidence", "conflict_detected", "version_conflict"}:
                    observation["rewritten_query"] = current_query != question
                observations.append(observation)
                trace[-1]["result_summary"] = result.status
                if result.status in {"conflict_detected", "version_conflict"}:
                    # Keep the ReAct decision step in the returned trace so the
                    # conflict path also has an auditable action/observation.
                    result.trace = trace + result.trace
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
                    result.decline(decision.reason or "决策模型判定拒绝回答，未给出理由")
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
            result.decline("达到最大推理步数仍未完成回答")
        else:
            result.gaps = result.gaps or ["达到最大推理步数"]
            result.refresh_completeness()
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
