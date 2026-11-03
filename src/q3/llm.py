from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass
class QueryPlan:
    queries: list[str]
    intent: str


class MockLLM:
    def plan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        return QueryPlan([question], question)

    def verify(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        return {"verified": bool(answer and citations), "unsupported_claims": [], "action": "answer" if answer and citations else "ask_user"}


class OpenAICompatibleLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ["OPENAI_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["OPENAI_MODEL"]
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]

    def _complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
        request = Request(self.base_url + "/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode())
        return payload["choices"][0]["message"]["content"]

    def plan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        prompt = "Return JSON with keys queries (array of strings) and intent. Do not answer.\\nQuestion: " + question
        payload = json.loads(self._complete(prompt))
        return QueryPlan(payload["queries"], payload["intent"])

    def verify(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        prompt = "Return JSON with verified, unsupported_claims, action. Check whether the answer is supported by citations.\\n" + json.dumps({"question": question, "answer": answer, "citations": citations}, ensure_ascii=False)
        return json.loads(self._complete(prompt))
