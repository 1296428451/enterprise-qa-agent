import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.async_provider import ProviderRequestError, post_chat_stream, post_json
from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.llm import OpenAICompatibleLLM
from src.q3.models import Chunk, ScoredChunk
from src.q3.provider_reranker import ProviderReranker


def _event(text):
    return ("data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n").encode()


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, parts):
        self.parts = parts

    async def __aiter__(self):
        for part in self.parts:
            if isinstance(part, Exception):
                raise part
            yield part

    async def aclose(self):
        pass


def _transport(streams, seen=None):
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        if seen is not None:
            seen.append(request)
        stream = streams.pop(0)
        return httpx.Response(200, stream=stream)

    return httpx.MockTransport(handler), calls


def test_sse_aggregates_json_deltas_and_done():
    seen = []
    content = json.dumps({"queries": ["报销"], "intent": "期限"}, ensure_ascii=False)
    raw = _event(content[:10]) + _event(content[10:]) + b"data: [DONE]\n\n"
    transport, calls = _transport([BytesStream([raw])], seen)
    llm = OpenAICompatibleLLM("https://mock.invalid/v1", "model", "fake-key", transport=transport)
    result = asyncio.run(llm._acomplete_json("instruction", {}))
    assert result == {"queries": ["报销"], "intent": "期限"}
    assert calls["count"] == 1
    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://mock.invalid/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer fake-key"
    assert json.loads(req.content)["stream"] is True


def test_sse_idle_timeout_discards_partial_and_retries_once():
    seen = []
    good = json.dumps({"ok": True})
    first = BytesStream([_event('{"partial":'), httpx.ReadTimeout("idle")])
    second = BytesStream([_event(good), b"data: [DONE]\n\n"])
    transport, calls = _transport([first, second], seen)
    response = asyncio.run(post_chat_stream(
        "https://mock.invalid/chat/completions", "fake", {"model": "m"},
        idle_timeout=1, max_retries=1, transport=transport,
    ))
    assert response["choices"][0]["message"]["content"] == {"ok": True}
    assert calls["count"] == 2


def test_sse_retries_exactly_three_times_then_fails():
    streams = [BytesStream([_event('{"p":'), httpx.ReadTimeout("idle")]) for _ in range(4)]
    transport, calls = _transport(streams)
    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(post_chat_stream(
            "https://mock.invalid/chat/completions", "fake", {}, idle_timeout=1, max_retries=3,
            transport=transport,
        ))
    assert error.value.category == "transport"
    assert calls["count"] == 4


def test_sse_http_errors_and_schema_errors_do_not_retry():
    calls = {"count": 0}

    def error_handler(request):
        calls["count"] += 1
        return httpx.Response(429, json={"error": "rate limited"})

    with pytest.raises(ProviderRequestError) as error:
        asyncio.run(post_chat_stream("https://mock.invalid/chat/completions", "fake", {},
                                     max_retries=3, transport=httpx.MockTransport(error_handler)))
    assert error.value.category == "http" and error.value.status_code == 429
    assert calls["count"] == 1


def test_post_json_owns_closes_but_does_not_close_injected_client():
    closed = {"count": 0}

    class TrackTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json={"ok": True})

        async def aclose(self):
            closed["count"] += 1

    async def run():
        owned = TrackTransport()
        assert await post_json("https://mock.invalid", "fake", {}, transport=owned) == {"ok": True}
        assert closed["count"] == 1
        external_transport = TrackTransport()
        async with httpx.AsyncClient(transport=external_transport, verify=True) as client:
            assert await post_json("https://mock.invalid", "fake", {}, client=client) == {"ok": True}
            assert closed["count"] == 1
        assert closed["count"] == 2

    asyncio.run(run())


def test_async_rerank_then_qa_answer_verify_order_and_loop_await():
    chunks = load_chunks("C:/Users/ASUS/Desktop/小影笔试/agent-code/data/q3_documents.json")
    order = []

    class OrderedLLM:
        async def adecide_action(self, question, observations, history=None):
            order.append("decision")
            action = "answer" if observations and observations[-1].get("status") == "completed" else "search"
            return type("Decision", (), {"action": action, "query": question, "reason": "test"})()

        async def aplan(self, question, history=None):
            return type("Plan", (), {"queries": [question], "intent": "test"})()

        async def aanswer_from_evidence(self, question, evidence):
            order.append("answer")
            return {"answer": "需要发票和行程单", "citation_indices": [1]}

        async def averify(self, question, answer, citations):
            order.append("verify")
            return {"verified": True, "unsupported_claims": [], "action": "answer"}

    class OrderedReranker:
        async def ascore(self, query, items):
            order.append("rerank")
            return [ScoredChunk(item.chunk, item.bm25_score, item.vector_score, 1.0) for item in items]

    class OrderedExtractor:
        async def aextract(self, question, items):
            order.append("extract")
            from src.q3.query_aware import MockQueryAwareExtractor
            return MockQueryAwareExtractor().extract(question, items)

    async def run_inside_loop():
        qa = EnterpriseQA(chunks, llm=OrderedLLM(), reranker=OrderedReranker(), extractor=OrderedExtractor())
        return await qa.answer_async("差旅报销需要什么材料？")

    result = asyncio.run(run_inside_loop())
    assert result.status == "completed"
    assert order[:5] == ["decision", "rerank", "extract", "answer", "verify"]
    assert order[-1] == "decision"


def test_post_chat_stream_preserves_injected_client_and_cancellation():
    async def run():
        stream_transport, _ = _transport([BytesStream([_event('{"ok":true}'), b"data: [DONE]\n\n"])])
        injected = httpx.AsyncClient(transport=stream_transport, verify=True)
        result = await post_chat_stream("https://mock.invalid/chat/completions", "fake", {}, client=injected)
        assert result["choices"][0]["message"]["content"] == {"ok": True}
        assert not injected.is_closed
        await injected.aclose()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _event('{"partial":')
                await asyncio.Event().wait()
            async def aclose(self):
                pass

        transport, _ = _transport([BlockingStream()])
        async with httpx.AsyncClient(transport=transport, verify=True) as client:
            task = asyncio.create_task(post_chat_stream("https://mock.invalid/chat/completions", "fake", {},
                                                        idle_timeout=5, client=client))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_tls_verify_false_is_rejected():
    with pytest.raises(ValueError, match="TLS verification"):
        asyncio.run(post_json("https://mock.invalid", "fake", {}, verify=False,
                              transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))))
