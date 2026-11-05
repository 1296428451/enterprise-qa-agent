from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

from .async_provider import (DEFAULT_IDLE_TIMEOUT, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, chat_url,
                             parse_chat_json, post_chat_stream, run_sync)


@dataclass
class QueryPlan:
    queries: list[str]
    intent: str


@dataclass
class ActionDecision:
    action: str
    query: str | None = None
    reason: str = ""


def parse_evidence(text: str) -> list[dict]:
    """Parse the numbered evidence block produced by ExtractionResult.to_answer_input."""
    entries: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        header = re.match(r"^\[(\d+)\]\s*\(([^)]+)\)", line)
        if header:
            if current:
                entries.append(current)
            current = {"index": int(header.group(1)), "source_id": header.group(2),
                       "quote": "", "scope": ""}
        elif current is not None and line.startswith("Quote:"):
            current["quote"] = line[len("Quote:"):].strip()
        elif current is not None and line.startswith("Scope:"):
            current["scope"] = line[len("Scope:"):].strip()
    if current:
        entries.append(current)
    return entries


class MockLLM:
    def plan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        return QueryPlan([question], question)

    async def plan_async(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        return self.plan(question, history)

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

    async def decide_action_async(self, question: str, observations: list[dict], history: list[dict] | None = None) -> ActionDecision:
        return self.decide_action(question, observations, history)

    def answer_from_evidence(self, question: str, evidence_input: str) -> dict:
        """Deterministic stand-in for the single answer call.

        Mirrors what the provider is asked to do: prefer a clause whose scope the
        question names, and report a conflict when the surviving clauses disagree.
        """
        from .retrieval import tokenize

        entries = [entry for entry in parse_evidence(evidence_input) if entry["quote"]]
        if not entries:
            return {"answer": "", "citation_indices": [], "conflict": False}
        question_tokens = set(tokenize(question))
        governing = []
        for entry in entries:
            scope = entry["scope"]
            if not scope or any(token in scope for token in ("所有", "全体", "全部", "全公司")):
                continue
            scope_tokens = set(tokenize(scope))
            if scope_tokens and len(scope_tokens & question_tokens) / len(scope_tokens) >= 0.6:
                governing.append(entry)
        if len({entry["scope"] for entry in governing}) == 1:
            chosen = governing[0]
            return {"answer": chosen["quote"], "citation_indices": [chosen["index"]], "conflict": False}
        values = {match.group(0) for entry in entries
                  for match in [re.search(r"\d+", entry["quote"])] if match}
        if len(entries) >= 2 and len(values) > 1:
            return {"answer": "", "citation_indices": [], "conflict": True}
        chosen = entries[0]
        return {"answer": chosen["quote"], "citation_indices": [chosen["index"]],
                "conflict": False, "gap": ""}

    async def answer_from_evidence_async(self, question: str, evidence_input: str) -> dict:
        return self.answer_from_evidence(question, evidence_input)

    def verify(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        verified = bool(answer and citations)
        return {"verified": verified, "unsupported_claims": [], "action": "answer" if verified else "ask_user"}

    async def verify_async(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        return self.verify(question, answer, citations)


class OpenAICompatibleLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 response_fixture: dict | None = None, timeout: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None, client: httpx.AsyncClient | None = None,
                 verify: bool | str = True, idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.base_url = (base_url or os.environ["OPENAI_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["OPENAI_MODEL"]
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.response_fixture = response_fixture
        self.chat_completions_url = chat_url(self.base_url)
        self.timeout = httpx.Timeout(timeout)
        self.transport = transport
        self.client = client
        self.verify_ssl = verify
        self.idle_timeout = idle_timeout
        self.max_retries = max_retries

    async def _complete_json_async(self, instruction: str, payload: dict) -> dict:
        patched_async = self.__dict__.get("_complete_json_async")
        if patched_async is not None:
            result = patched_async(instruction, payload)
            if hasattr(result, "__await__"):
                result = await result
            return result
        patched_sync = self.__dict__.get("_complete_json")
        if patched_sync is not None:
            return patched_sync(instruction, payload)
        if self.response_fixture is not None:
            result = self.response_fixture
        else:
            body = {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            }
            result = await post_chat_stream(self.chat_completions_url, self.api_key, body, self.timeout,
                                            self.idle_timeout, self.max_retries, self.transport,
                                            self.client, self.verify_ssl)
        return parse_chat_json(result, "LLM response must contain choices[0].message.content JSON object")

    async def _acomplete_json(self, instruction: str, payload: dict) -> dict:
        return await self._complete_json_async(instruction, payload)

    def _complete_json(self, instruction: str, payload: dict) -> dict:
        return run_sync(self._complete_json_async(instruction, payload))

    async def decide_action_async(self, question: str, observations: list[dict], history: list[dict] | None = None) -> ActionDecision:
        result = await self._complete_json_async(
            "Choose exactly one action: search, answer, ask_user, or insufficient_evidence. IMPORTANT: if observations is empty, you MUST return action=search with a rewritten query. Never answer from general knowledge. When the latest observation has status=completed, verified=true, a non-empty answer_preview and evidence_quotes from the knowledge base, choose action=answer to return that verified answer; do not choose insufficient_evidence merely because the full policy text is not quoted in the observation. An observation with completeness=partial still carries a usable answer whose uncovered parts are listed in gaps, so treat it like completeness=complete and choose action=answer. Return strict JSON with action, optional query, and reason.",
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

    def decide_action(self, question: str, observations: list[dict], history: list[dict] | None = None) -> ActionDecision:
        return run_sync(self.decide_action_async(question, observations, history))

    async def plan_async(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        result = await self._complete_json_async(
            "Generate retrieval queries for the user's question. Do not answer. Return JSON with string-array queries and string intent.",
            {"question": question, "history": history or []},
        )
        queries, intent = result["queries"], result["intent"]
        if not isinstance(queries, list) or not queries or not all(isinstance(item, str) and item.strip() for item in queries):
            raise ValueError("LLM planner queries must be a non-empty array of strings")
        if not isinstance(intent, str):
            raise ValueError("LLM planner intent must be a string")
        return QueryPlan(queries, intent)

    async def aplan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        return await self.plan_async(question, history)

    def plan(self, question: str, history: list[dict] | None = None) -> QueryPlan:
        return run_sync(self.plan_async(question, history))

    async def adecide_action(self, question: str, observations: list[dict], history: list[dict] | None = None) -> ActionDecision:
        return await self.decide_action_async(question, observations, history)

    async def aanswer_from_evidence(self, question: str, evidence_input: str) -> dict:
        return await self.answer_from_evidence_async(question, evidence_input)

    async def answer_from_evidence_async(self, question: str, evidence_input: str) -> dict:
        result = await self._complete_json_async(
            "Answer the question using only the numbered evidence. Cite every factual claim with "
            "[index]. Each item carries a Scope line when the clause declares one: when the "
            "question names a scope, the matching scoped clause governs and a general clause that "
            "would otherwise disagree is not a conflict. If the question mixes something the "
            "evidence covers with something it does not, answer the covered part with citations and "
            "put the missing part in gap. If the clauses still contradict each other for this "
            "question, do not pick a side: return an empty answer, an empty citation_indices array "
            "and conflict=true. Whenever any evidence is relevant you MUST return a non-empty "
            "answer: state what the evidence does support with citations, even if that is only part "
            "of what was asked, and put everything it does not cover into gap. Only return an empty "
            "answer when no evidence is relevant at all. Return strict JSON with string answer, "
            "integer-array citation_indices, boolean conflict and string gap.",
            {"question": question, "evidence": evidence_input},
        )
        answer = result.get("answer")
        citations = result.get("citation_indices")
        conflict = result.get("conflict", False)
        if not isinstance(answer, str) or not isinstance(citations, list) or not all(type(i) is int for i in citations):
            raise ValueError("LLM answer must contain answer and citation_indices")
        if not isinstance(conflict, bool):
            raise ValueError("LLM answer conflict must be a boolean")
        if not isinstance(result.get("gap", ""), str):
            raise ValueError("LLM answer gap must be a string")
        return result

    def answer_from_evidence(self, question: str, evidence_input: str) -> dict:
        return run_sync(self.answer_from_evidence_async(question, evidence_input))

    async def verify_async(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        result = await self._complete_json_async(
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
            # Models sometimes approve while still listing gaps or asking the
            # user. Treat that as "not verified" instead of aborting the run:
            # the safe outcome is abstention, not a traceback in the CLI.
            result["verified"] = False
            result["unsupported_claims"] = claims or [
                "verifier approved the answer while also requesting confirmation"
            ]
            result["action"] = "ask_user"
        return result

    async def averify(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        return await self.verify_async(question, answer, citations)

    def verify(self, question: str, answer: str | None, citations: list[dict]) -> dict:
        return run_sync(self.verify_async(question, answer, citations))
