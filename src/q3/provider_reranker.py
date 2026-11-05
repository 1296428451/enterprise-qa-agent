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
        """Resolve explicit endpoint forms without silently changing provider APIs.

        `/rerank` uses the native reranker schema. `/chat/completions` uses an
        OpenAI-compatible chat endpoint and expects JSON scores in the message.
        A bare `/v1` remains the historical native-rerank default.
        """
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.endpoint_type = "chat"
        elif self.base_url.endswith("/rerank"):
            self.endpoint_type = "rerank"
        else:
            self.endpoint_type = "rerank"
            if self.base_url.endswith("/v1"):
                self.base_url += "/rerank"

    def _chat_scores(self, query: str, items: list[ScoredChunk]) -> list[float]:
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Return JSON only: {\\\"scores\\\":[number per document in input order]}."},
                {"role": "user", "content": json.dumps({
                    "query": query,
                    "documents": [item.chunk.text for item in items],
                }, ensure_ascii=False)},
            ],
        }).encode("utf-8")
        request = Request(self.base_url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        })
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("chat reranker response must contain choices[0].message.content JSON") from exc
        scores = parsed.get("scores") if isinstance(parsed, dict) else parsed
        if not isinstance(scores, list) or len(scores) != len(items):
            raise ValueError("chat reranker response must contain one score per candidate")
        if not all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores):
            raise ValueError("chat reranker scores must be numbers")
        return [float(score) for score in scores]

    def score(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not items:
            return []
        if self.endpoint_type == "chat":
            scores = self._chat_scores(query, items)
        else:
            payload = json.dumps({"model": self.model, "query": query, "documents": [item.chunk.text for item in items]}).encode("utf-8")
            request = Request(self.base_url, data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            data = body.get("results", body.get("data"))
            if not isinstance(data, list):
                raise ValueError("reranker response must contain a results or data array")
            if len(data) != len(items):
                raise ValueError("reranker response must include one score per candidate")
            scores = [None] * len(items)
            for entry in data:
                if not isinstance(entry, dict) or not isinstance(entry.get("index"), int) or isinstance(entry.get("index"), bool) or not isinstance(entry.get("relevance_score"), (int, float)) or isinstance(entry.get("relevance_score"), bool):
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

    @classmethod
    def from_environment(cls):
        return cls(os.environ["RERANK_BASE_URL"], os.environ["RERANK_MODEL"], os.environ["RERANK_API_KEY"])
