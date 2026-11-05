from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass
class QueryPlan:
    queries: list[str]
    intent: str


@dataclass
class ActionDecision:
    action: str
    query: str | None = None
    reason: str = ""


class MockLLM:
    def plan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        return QueryPlan([question], question)

    def decide_action(self, question: str, observations: list[dict], history: list[dict] | None = None) -> ActionDecision:
        if not observations:
            return ActionDecision("search", question, "开始检索")
        latest = observations[-1]
        if latest.get("conflict"):
            return ActionDecision("ask_user", reason="发现冲突证据")
        if latest.get("evidence_count", 0):
            return ActionDecision("answer", reason="已有可用证据")
        if latest.get("rewritten_query"):
            return ActionDecision("insufficient_evidence", reason="改写查询后仍无证据")
        return ActionDecision("search", question + " 相关制度", "首次检索无证据，改写查询")

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

    def decide_action(self, question: str, observations: list[dict], history: list[dict] | None = None) -> ActionDecision:
        result = self._complete_json(
            "Choose exactly one action: search, answer, ask_user, or insufficient_evidence. For search provide a rewritten query. Return strict JSON with action, optional query, and reason.",
            {"question": question, "observations": observations, "history": history or []},
        )
        action = result.get("action")
        if action not in {"search", "answer", "ask_user", "insufficient_evidence"}:
            raise ValueError("LLM action must be search, answer, ask_user, or insufficient_evidence")
        query = result.get("query")
        if action == "search" and (not isinstance(query, str) or not query.strip()):
            raise ValueError("LLM search action requires a non-empty query")
        if query is not None and not isinstance(query, str):
            raise ValueError("LLM action query must be a string")
        reason = result.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError("LLM action reason must be a string")
        return ActionDecision(action, query, reason)

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
