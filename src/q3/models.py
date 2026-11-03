from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    version: str
    version_date: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class Answer:
    status: str
    answer: str | None
    citations: list[dict[str, Any]]
    gaps: list[str]
    trace: list[dict[str, Any]]
    retrieval: dict[str, Any]
    verification: dict[str, Any]
