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
        verified = bool(answer and citations)
        return {"verified": verified, "unsupported_claims": [], "action": "answer" if verified else "ask_user"}


class OpenAICompatibleLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.environ["OPENAI_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["OPENAI_MODEL"]
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.chat_completions_url = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"

    def _complete_json(self, instruction: str, payload: dict) -> dict:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }).encode("utf-8")
        request = Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
        )
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object")
        return parsed

    def plan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        result = self._complete_json(
            "Generate retrieval queries for the user's question. Do not answer. Return JSON with string-array queries and string intent.",
            {"question": question, "history": history or []},
        )
        queries, intent = result["queries"], result["intent"]
        if not isinstance(queries, list) or not queries or not all(isinstance(item, str) and item.strip() for item in queries):
            raise ValueError("LLM planner queries must be a non-empty array of strings")
        if not isinstance(intent, str):
            raise ValueError("LLM planner intent must be a string")
        return QueryPlan(queries, intent)

    def verify(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        result = self._complete_json(
            "Check whether every claim in answer is supported by the quoted citations. Return JSON with boolean verified, string-array unsupported_claims, and action equal to answer or ask_user.",
            {"question": question, "answer": answer, "citations": citations},
        )
        if not isinstance(result.get("verified"), bool):
            raise ValueError("LLM verifier verified must be a boolean")
        claims = result.get("unsupported_claims")
        if not isinstance(claims, list) or not all(isinstance(item, str) for item in claims):
            raise ValueError("LLM verifier unsupported_claims must be an array of strings")
        if result.get("action") not in {"answer", "ask_user"}:
            raise ValueError("LLM verifier action must be answer or ask_user")
        if result["verified"] and (claims or result["action"] != "answer"):
            raise ValueError("LLM verifier returned contradictory approval fields")
        return result
