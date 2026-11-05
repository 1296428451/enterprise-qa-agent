from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from .async_provider import (DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_RETRIES, chat_url,
                             parse_chat_json, post_chat_stream, run_sync)
from .models import ScoredChunk
from .retrieval import scope_phrase

QUERY_AWARE_PROMPT = """\
You are an evidence extraction system for enterprise knowledge QA.
Given a user question and candidate documents, extract only the text spans \
that directly help answer the question.

## Rules
1. Extract only spans that directly answer or provide facts needed to answer.
2. Remove boilerplate, unrelated clauses, and tangential content.
3. Preserve the exact wording of extracted spans (extractive, not abstractive).
4. If a candidate has no relevant content, set relevance to "none" and quote to "".
5. Never answer the question yourself. Only extract evidence.
6. Output numbered evidence items with source IDs and concise key info.

## Example
Question: 报销期限多久？
Candidate 1 (D01): 普通费用报销应在费用发生后 30 天内提交。适用于所有正式员工。\
年会筹备工作由行政部负责。
Candidate 2 (D02): 普通费用报销应在费用发生后 15 天内提交。适用于研发部门。

Expected output:
{"evidence":[{"index":1,"source_id":"D01","relevance":"high",\
"quote":"普通费用报销应在费用发生后 30 天内提交。","key_info":"报销期限 30 天"},\
{"index":2,"source_id":"D02","relevance":"high",\
"quote":"普通费用报销应在费用发生后 15 天内提交。","key_info":"报销期限 15 天"}],\
"compression_note":"移除年会筹备和适用范围描述，保留期限条款"}

## Output format (JSON only)
{"evidence":[{"index":1,"source_id":"D01","relevance":"high|medium|none",\
"quote":"exact span","key_info":"key fact, <=20 words"}],\
"compression_note":"one sentence describing what was removed"}
"""

_SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；\n]*")


@dataclass(frozen=True)
class ExtractedEvidence:
    index: int
    source_id: str
    chunk_id: str
    relevance: str
    quote: str
    key_info: str
    scope: str = ""


@dataclass
class ExtractionResult:
    question: str
    evidence: list[ExtractedEvidence]
    candidates_in: int
    candidates_with_evidence: int
    total_source_chars: int
    total_extracted_chars: int
    compression_note: str
    token_budget: int = 20000
    candidates_skipped_budget: int = 0

    @property
    def compression_ratio(self) -> float:
        if not self.total_source_chars:
            return 0.0
        return self.total_extracted_chars / self.total_source_chars

    def to_answer_input(self) -> str:
        parts = [f"Question: {self.question}", "", "Evidence:"]
        for item in self.evidence:
            if item.relevance == "none":
                continue
            parts.append(f"[{item.index}] ({item.source_id}) {item.key_info}")
            parts.append(f"    Quote: {item.quote}")
            if item.scope:
                # The applicability phrase decides which clause governs a
                # scoped question, so it must travel with the evidence.
                parts.append(f"    Scope: {item.scope}")
        return "\n".join(parts)


def estimate_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_chars = len(text) - cjk
    return cjk + max(ascii_chars // 4, 0)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def _overlap_score(sentence: str, question_tokens: set[str]) -> float:
    from .retrieval import tokenize
    tokens = set(tokenize(sentence))
    if not tokens or not question_tokens:
        return 0.0
    return len(tokens & question_tokens) / len(question_tokens)


class MockQueryAwareExtractor:
    """Deterministic offline extractor: picks sentences with highest token overlap."""

    def extract(self, question: str, candidates: list[ScoredChunk]) -> ExtractionResult:
        from .retrieval import tokenize
        question_tokens = set(tokenize(question))
        evidence: list[ExtractedEvidence] = []
        total_src = total_ext = 0
        with_ev = 0
        for rank, item in enumerate(candidates, 1):
            text = item.chunk.text
            total_src += len(text)
            sentences = split_sentences(text)
            scored = [(s, _overlap_score(s, question_tokens)) for s in sentences]
            kept = [s for s, score in scored if score >= 0.15]
            quote = " ".join(kept)
            relevance = "high" if quote and any(sc >= 0.5 for _, sc in scored) else (
                "medium" if quote else "none"
            )
            key_info = kept[0] if kept else ""
            total_ext += len(quote)
            if quote:
                with_ev += 1
            evidence.append(ExtractedEvidence(
                index=rank,
                source_id=item.chunk.document_id,
                chunk_id=item.chunk.chunk_id,
                relevance=relevance,
                quote=quote,
                key_info=key_info[:80],
                scope=scope_phrase(item.chunk.text),
            ))
        return ExtractionResult(
            question=question,
            evidence=evidence,
            candidates_in=len(candidates),
            candidates_with_evidence=with_ev,
            total_source_chars=total_src,
            total_extracted_chars=total_ext,
            compression_note="mock: removed sentences with low question-token overlap",
        )


class OpenAICompatibleQueryAwareExtractor:
    def __init__(self, base_url: str, model: str, api_key: str, max_tokens: int = 20000,
                 timeout: float = 120.0, transport: httpx.AsyncBaseTransport | None = None,
                 client: httpx.AsyncClient | None = None, verify: bool | str = True,
                 idle_timeout: float = DEFAULT_IDLE_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.chat_url = chat_url(self.base_url)
        self.timeout_config = httpx.Timeout(timeout)
        self.transport = transport
        self.client = client
        self.verify_ssl = verify
        self.idle_timeout = idle_timeout
        self.max_retries = max_retries

    async def _call_async(self, prompt_payload: dict) -> dict:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": QUERY_AWARE_PROMPT},
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
        }
        result = await post_chat_stream(self.chat_url, self.api_key, body, self.timeout_config,
                                        self.idle_timeout, self.max_retries, self.transport,
                                        self.client, self.verify_ssl)
        return parse_chat_json(result, "Query-aware response must contain choices[0].message.content JSON object")

    def _call(self, prompt_payload: dict) -> dict:
        return run_sync(self._call_async(prompt_payload))

    async def extract_async(self, question: str, candidates: list[ScoredChunk]) -> ExtractionResult:
        payload_candidates = []
        total_src = 0
        skipped_budget = 0
        used_tokens = estimate_tokens(QUERY_AWARE_PROMPT) + estimate_tokens(question)
        selected_candidates = []
        for rank, item in enumerate(candidates, 1):
            text = item.chunk.text
            candidate_tokens = estimate_tokens(json.dumps({"index": rank, "source_id": item.chunk.document_id, "text": text}, ensure_ascii=False))
            if used_tokens + candidate_tokens > self.max_tokens:
                skipped_budget += 1
                continue
            used_tokens += candidate_tokens
            total_src += len(text)
            selected_candidates.append((rank, item))
            payload_candidates.append({
                "index": rank,
                "source_id": item.chunk.document_id,
                "text": text,
            })
        raw = await self._call_async({"question": question, "candidates": payload_candidates})
        items = raw.get("evidence")
        if not isinstance(items, list):
            raise ValueError("Query-aware extraction evidence must be a list")
        candidate_by_rank = dict(selected_candidates)
        evidence: list[ExtractedEvidence] = []
        total_ext = 0
        seen: set[int] = set()
        for entry in items:
            idx = int(entry["index"])
            if idx not in candidate_by_rank or idx in seen:
                raise ValueError(f"Invalid or duplicate evidence index: {idx}")
            seen.add(idx)
            candidate = candidate_by_rank[idx]
            relevance = entry.get("relevance", "none")
            if relevance not in {"high", "medium", "none"}:
                raise ValueError(f"Invalid relevance: {relevance}")
            quote = str(entry.get("quote", ""))
            if quote and quote not in candidate.chunk.text:
                raise ValueError(f"Extracted quote is not an exact span of candidate {idx}")
            key_info = str(entry.get("key_info", ""))
            total_ext += len(quote)
            evidence.append(ExtractedEvidence(
                index=idx,
                source_id=candidate.chunk.document_id,
                chunk_id=candidate.chunk.chunk_id,
                relevance=relevance,
                quote=quote,
                key_info=key_info[:120],
                scope=scope_phrase(candidate.chunk.text),
            ))
        evidence.sort(key=lambda e: e.index)
        return ExtractionResult(
            question=question,
            evidence=evidence,
            candidates_in=len(candidates),
            candidates_with_evidence=sum(1 for e in evidence if e.quote),
            total_source_chars=total_src,
            total_extracted_chars=total_ext,
            compression_note=str(raw.get("compression_note", "")),
            token_budget=self.max_tokens,
            candidates_skipped_budget=skipped_budget,
        )

    async def aextract(self, question: str, candidates: list[ScoredChunk]) -> ExtractionResult:
        return await self.extract_async(question, candidates)

    def extract(self, question: str, candidates: list[ScoredChunk]) -> ExtractionResult:
        return run_sync(self.extract_async(question, candidates))

    @classmethod
    def from_environment(cls, max_tokens: int = 20000):
        import os
        base_url = os.environ.get("QUERY_AWARE_BASE_URL") or os.environ["OPENAI_BASE_URL"]
        model = os.environ.get("QUERY_AWARE_MODEL") or os.environ["OPENAI_MODEL"]
        api_key = os.environ.get("QUERY_AWARE_API_KEY") or os.environ["OPENAI_API_KEY"]
        return cls(base_url, model, api_key, max_tokens=max_tokens)
