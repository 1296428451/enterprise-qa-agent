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

    def answer_from_evidence(self, question: str, evidence_input: str) -> dict:
        lines = [line.strip() for line in evidence_input.splitlines() if line.strip().startswith("Quote:")]
        quote = lines[0].replace("Quote:", "", 1).strip() if lines else ""
        return {"answer": quote, "citation_indices": [1] if quote else []}

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
            self.chat_completions_url,
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

    def answer_from_evidence(self, question: str, evidence_input: str) -> dict:
        result = self._complete_json(
            "Answer using only numbered evidence. Cite factual claims with [index]. If evidence is insufficient or conflicting, return an empty answer. Return JSON with string answer and integer-array citation_indices.",
            {"question": question, "evidence": evidence_input},
        )
        answer = result.get("answer")
        citations = result.get("citation_indices")
        if not isinstance(answer, str) or not isinstance(citations, list) or not all(isinstance(i, int) for i in citations):
            raise ValueError("LLM answer must contain answer and citation_indices")
        return result

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
