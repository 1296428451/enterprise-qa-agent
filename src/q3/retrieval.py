from __future__ import annotations

import math
import re
from collections import Counter

from .models import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
_STOPWORDS = set("的了和是我你他她它在有无要什么是否公司提供可以普通需要应该最晚吗？员工" )


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequency = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(freq.values()) for freq in self.term_frequency]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for freq in self.term_frequency:
            self.document_frequency.update(freq.keys())

    def search(self, query: str, limit: int = 50) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        total = len(self.chunks)
        scored: list[ScoredChunk] = []
        for index, chunk in enumerate(self.chunks):
            score = 0.0
            matched_terms = 0
            for term in query_terms:
                tf = self.term_frequency[index].get(term, 0)
                if not tf:
                    continue
                matched_terms += 1
                df = self.document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(self.avg_length, 1))
                score += idf * tf * (self.k1 + 1) / denominator
            if score > 0 and (matched_terms >= 2 or len(query_terms) <= 1):
                scored.append(ScoredChunk(chunk=chunk, bm25_score=score))
        return sorted(scored, key=lambda item: item.bm25_score, reverse=True)[:limit]


def merge_candidates(groups: list[list[ScoredChunk]], limit: int = 50) -> list[ScoredChunk]:
    merged: dict[str, ScoredChunk] = {}
    for group in groups:
        for item in group:
            current = merged.get(item.chunk.chunk_id)
            if current is None or item.bm25_score > current.bm25_score:
                merged[item.chunk.chunk_id] = item
    return sorted(merged.values(), key=lambda item: item.bm25_score, reverse=True)[:limit]


def normalize_scores(items: list[ScoredChunk]) -> list[ScoredChunk]:
    if not items:
        return []
    maximum = max(item.bm25_score for item in items) or 1.0
    return [ScoredChunk(item.chunk, item.bm25_score / maximum, item.vector_score, item.rerank_score) for item in items]


class MockSemanticRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def search(self, query: str, limit: int = 50) -> list[ScoredChunk]:
        query_terms = set(tokenize(query))
        results = []
        for chunk in self.chunks:
            terms = set(tokenize(chunk.text))
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            matched_terms = len(query_terms & terms)
            minimum_overlap = 0.5 if len(query_terms) > 1 else 0.25
            if matched_terms >= 2 and overlap >= minimum_overlap:
                results.append(ScoredChunk(chunk=chunk, vector_score=overlap))
        return sorted(results, key=lambda item: item.vector_score, reverse=True)[:limit]


class MockJevReranker:
    def score(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        scored = []
        for item in items:
            score = 0.55 * item.bm25_score + 0.45 * item.vector_score
            scored.append(ScoredChunk(item.chunk, item.bm25_score, item.vector_score, score))
        return sorted(scored, key=lambda item: item.rerank_score, reverse=True)


class EvidenceSelector:
    def __init__(self, minimum_score: float = 0.2, maximum: int = 8):
        self.minimum_score = minimum_score
        self.maximum = maximum

    def select(self, items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not items:
            return []
        top_score = items[0].rerank_score
        cutoff = max(self.minimum_score, top_score * 0.35)
        selected = [item for item in items if item.rerank_score >= cutoff][:self.maximum]
        return selected

    @staticmethod
    def citations(items: list[ScoredChunk]) -> list[dict[str, str]]:
        return [{"document_id": item.chunk.document_id, "chunk_id": item.chunk.chunk_id,
                 "version": item.chunk.version, "quote": item.chunk.text} for item in items]


def detect_conflict(items: list[ScoredChunk]) -> bool:
    by_key: dict[str, set[str]] = {}
    for item in items:
        key = " ".join(sorted(set(tokenize(item.chunk.text)) & {"天", "期限", "报销", "时间"}))
        if key:
            by_key.setdefault(key, set()).add(item.chunk.text)
    return any(len(values) > 1 for values in by_key.values())


def hybrid_candidates(query: str, chunks: list[Chunk], limit: int = 50, semantic_retriever=None) -> list[ScoredChunk]:
    bm25 = normalize_scores(BM25(chunks).search(query, limit))
    semantic = (semantic_retriever or MockSemanticRetriever(chunks)).search(query, limit)
    by_id: dict[str, ScoredChunk] = {}
    for item in bm25 + semantic:
        current = by_id.get(item.chunk.chunk_id)
        bm25_score = max(current.bm25_score if current else 0.0, item.bm25_score)
        vector_score = max(current.vector_score if current else 0.0, item.vector_score)
        by_id[item.chunk.chunk_id] = ScoredChunk(item.chunk, bm25_score, vector_score)
    return [item for item in by_id.values()
            if 0.55 * item.bm25_score + 0.45 * item.vector_score >= 0.2][:limit]


__all__ = ["BM25", "Chunk", "EvidenceSelector", "MockJevReranker", "detect_conflict", "hybrid_candidates"]
