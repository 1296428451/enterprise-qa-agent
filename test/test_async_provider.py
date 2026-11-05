import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.async_provider import post_json
from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.llm import OpenAICompatibleLLM
from src.q3.models import ScoredChunk
from src.q3.provider_reranker import ProviderReranker
from src.q3.query_aware import OpenAICompatibleQueryAwareExtractor


def test_post_json_parses_and_closes_transport():
    class Transport(httpx.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            return httpx.Response(200, json={"ok": True})

        async def aclose(self):
            self.closed = True

    transport = Transport()
    result = asyncio.run(post_json("https://example.invalid/v1", "secret", {"x": 1}, transport=transport))
    assert result == {"ok": True}
    assert transport.closed


def test_post_json_http_error_and_timeout_propagate():
    error_transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "down"}))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(post_json("https://example.invalid/v1", "secret", {}, transport=error_transport))

    def timeout_handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(post_json("https://example.invalid/v1", "secret", {}, timeout=httpx.Timeout(0.01), transport=httpx.MockTransport(timeout_handler)))


def test_async_llm_and_reranker_schemas():
    llm_body = {"choices": [{"message": {"content": '{"action":"search","query":"报销","reason":"改写"}'}}]}
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "model", "secret",
                             transport=httpx.MockTransport(lambda request: httpx.Response(200, json=llm_body)))
    decision = asyncio.run(llm.decide_action_async("问题", []))
    assert decision.query == "报销"

    items = [ScoredChunk(type("Chunk", (), {"text": "报销", "chunk_id": "1", "document_id": "D1", "version": "v1"})())]
    rerank_body = {"results": [{"index": 0, "relevance_score": 0.7}]}
    reranker = ProviderReranker("https://example.invalid/v1/rerank", "model", "secret",
                                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=rerank_body)))
    assert asyncio.run(reranker.score_async("报销", items))[0].rerank_score == 0.7


def test_async_react_mock_path():
    chunks = load_chunks("C:/Users/ASUS/Desktop/小影笔试/agent-code/data/q3_documents.json")
    result = asyncio.run(EnterpriseQA(chunks).answer_async("差旅报销需要什么材料？"))
    assert result.status == "completed"
    actions = [item.get("action") for item in result.trace]
    assert "search" in actions
    assert "answer" in actions
