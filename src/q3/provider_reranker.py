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

    def score(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        payload = json.dumps({"model": self.model, "query": query, "documents": [item.chunk.text for item in items]}).encode("utf-8")
        request = Request(self.base_url, data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        scores = {int(item["index"]): float(item["relevance_score"]) for item in body["data"]}
        return sorted([ScoredChunk(item.chunk, item.bm25_score, item.vector_score, scores[index]) for index, item in enumerate(items)], key=lambda item: item.rerank_score, reverse=True)

    @classmethod
    def from_environment(cls):
        return cls(os.environ["RERANK_BASE_URL"], os.environ["RERANK_MODEL"], os.environ["RERANK_API_KEY"])
