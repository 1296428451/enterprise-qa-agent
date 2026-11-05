from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

from .async_provider import DEFAULT_TIMEOUT, chat_url, parse_chat_json, post_json, run_sync
from .models import ScoredChunk


@dataclass
class ProviderReranker:
    base_url: str
    model: str
    api_key: str
    timeout: float = 60.0
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.endpoint_type = "chat"
        elif self.base_url.endswith("/rerank"):
            self.endpoint_type = "rerank"
        else:
            self.endpoint_type = "rerank"
            if self.base_url.endswith("/v1"):
                self.base_url += "/rerank"
        self.timeout_config = httpx.Timeout(self.timeout)

    async def _chat_scores_async(self, query: str, items: list[ScoredChunk]) -> list[float]:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": 'Return JSON only: {"scores":[number per document in input order]}.'},
                {"role": "user", "content": json.dumps({
                    "query": query,
                    "documents": [item.chunk.text for item in items],
                }, ensure_ascii=False)},
            ],
        }
        response = await post_json(chat_url(self.base_url), self.api_key, body, self.timeout_config, self.transport)
        parsed = parse_chat_json(response, "chat reranker response must contain choices[0].message.content JSON")
        scores = parsed.get("scores")
        if not isinstance(scores, list) or len(scores) != len(items):
            raise ValueError("chat reranker response must contain one score per candidate")
        if not all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores):
            raise ValueError("chat reranker scores must be numbers")
        return [float(score) for score in scores]

    async def score_async(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not items:
            return []
        if self.endpoint_type == "chat":
            scores = await self._chat_scores_async(query, items)
        else:
            body = {
                "model": self.model,
                "query": query,
                "documents": [item.chunk.text for item in items],
                "top_n": len(items),
                "return_documents": True,
            }
            response = await post_json(self.base_url, self.api_key, body, self.timeout_config, self.transport)
            data = response.get("results", response.get("data"))
            if not isinstance(data, list) or len(data) != len(items):
                raise ValueError("reranker response must include one result per candidate")
            scores: list[float | None] = [None] * len(items)
            for entry in data:
                if (not isinstance(entry, dict) or type(entry.get("index")) is not int
                        or not isinstance(entry.get("relevance_score"), (int, float))
                        or isinstance(entry.get("relevance_score"), bool)):
                    raise ValueError("reranker entries require integer index and numeric relevance_score")
                index = entry["index"]
                if index < 0 or index >= len(items) or scores[index] is not None:
                    raise ValueError(f"reranker returned invalid or duplicate document index: {index}")
                scores[index] = float(entry["relevance_score"])
            if any(score is None for score in scores):
                raise ValueError("reranker response omitted a document index")
        ranked = [ScoredChunk(item.chunk, item.bm25_score, item.vector_score, scores[index])
                  for index, item in enumerate(items)]
        return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)

    def score(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        return run_sync(self.score_async(query, items))

    @classmethod
    def from_environment(cls):
        return cls(os.environ["RERANK_BASE_URL"], os.environ["RERANK_MODEL"], os.environ["RERANK_API_KEY"])
