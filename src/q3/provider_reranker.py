from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen

from .models import ScoredChunk


@dataclass
class ProviderReranker:
    base_url: str
    model: str
    api_key: str

    def __post_init__(self):
        if self.base_url.endswith("/v1"):
            self.base_url += "/rerank"

    def score(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not items:
            return []
        payload = json.dumps({"model": self.model, "query": query, "documents": [item.chunk.text for item in items]}).encode("utf-8")
        request = Request(self.base_url, data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        data = body.get("data")
        if not isinstance(data, list):
            raise ValueError("reranker response must contain a data array")
        if len(data) != len(items):
            raise ValueError("reranker response must include one score per candidate")
        scores = [None] * len(items)
        for entry in data:
            index = int(entry["index"])
            if index < 0 or index >= len(items) or scores[index] is not None:
                raise ValueError(f"reranker returned invalid or duplicate document index: {index}")
            scores[index] = float(entry["relevance_score"])
        ranked = [ScoredChunk(item.chunk, item.bm25_score, item.vector_score, scores[index])
                  for index, item in enumerate(items)]
        return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)

    @classmethod
    def from_environment(cls):
        return cls(os.environ["RERANK_BASE_URL"], os.environ["RERANK_MODEL"], os.environ["RERANK_API_KEY"])
