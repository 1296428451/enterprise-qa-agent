from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(60.0)
DEFAULT_IDLE_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
TRANSIENT_ERRORS = (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout)


class ProviderRequestError(RuntimeError):
    def __init__(self, category: str, message: str, status_code: int | None = None):
        self.category = category
        self.status_code = status_code
        super().__init__(message)


async def post_json(url: str, api_key: str, payload: dict, timeout: httpx.Timeout = DEFAULT_TIMEOUT,
                    transport: httpx.AsyncBaseTransport | None = None,
                    client: httpx.AsyncClient | None = None, verify: bool | str = True) -> dict:
    if verify is False:
        raise ValueError("TLS verification cannot be disabled")
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout, verify=verify, transport=transport)
    try:
        response = await active_client.post(url, json=payload, headers={"Authorization": "Bearer " + api_key})
        response.raise_for_status()
        try:
            result = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderRequestError("json", "provider returned invalid JSON", response.status_code) from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderRequestError("http", "provider returned an HTTP error", exc.response.status_code) from exc
    except httpx.RequestError as exc:
        raise ProviderRequestError("transport", type(exc).__name__) from exc
    finally:
        if owns_client:
            await active_client.aclose()
    if not isinstance(result, dict):
        raise ProviderRequestError("schema", "provider response must be a JSON object")
    return result


async def post_chat_stream(url: str, api_key: str, payload: dict, timeout: httpx.Timeout = DEFAULT_TIMEOUT,
                           idle_timeout: float = DEFAULT_IDLE_TIMEOUT, max_retries: int = DEFAULT_MAX_RETRIES,
                           transport: httpx.AsyncBaseTransport | None = None,
                           client: httpx.AsyncClient | None = None, verify: bool | str = True) -> dict:
    if verify is False:
        raise ValueError("TLS verification cannot be disabled")
    request_payload = {**payload, "stream": True}
    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=timeout, verify=verify, transport=transport)
    try:
        for attempt in range(max_retries + 1):
            fragments: list[str] = []
            try:
                async with active_client.stream("POST", url, json=request_payload,
                                                headers={"Authorization": "Bearer " + api_key}) as response:
                    response.raise_for_status()
                    async for line in _iter_sse_lines(response, idle_timeout):
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            return _parse_streamed_chat("".join(fragments), response.status_code)
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise ProviderRequestError("json", "provider returned invalid SSE JSON", response.status_code) from exc
                        choices = event.get("choices") if isinstance(event, dict) else None
                        if not isinstance(choices, list):
                            raise ProviderRequestError("schema", "stream event must contain choices", response.status_code)
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        if not isinstance(delta, dict):
                            raise ProviderRequestError("schema", "stream delta must be an object", response.status_code)
                        content = delta.get("content", "")
                        if content is not None and not isinstance(content, str):
                            raise ProviderRequestError("schema", "stream content must be a string", response.status_code)
                        fragments.append(content or "")
                raise ProviderRequestError("schema", "stream ended before [DONE]")
            except asyncio.CancelledError:
                raise
            except ProviderRequestError as exc:
                if exc.category != "idle_timeout" or attempt >= max_retries:
                    raise
            except TRANSIENT_ERRORS as exc:
                if attempt >= max_retries:
                    raise ProviderRequestError("transport", type(exc).__name__) from exc
            except httpx.HTTPStatusError as exc:
                raise ProviderRequestError("http", "provider returned an HTTP error", exc.response.status_code) from exc
            except httpx.RequestError as exc:
                raise ProviderRequestError("transport", type(exc).__name__) from exc
        raise AssertionError("unreachable")
    finally:
        if owns_client:
            await active_client.aclose()


async def _iter_sse_lines(response: httpx.Response, idle_timeout: float):
    iterator = response.aiter_lines().__aiter__()
    while True:
        try:
            line = await asyncio.wait_for(iterator.__anext__(), timeout=idle_timeout)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise ProviderRequestError("idle_timeout", "provider stream idle timeout") from exc
        yield line


def _parse_streamed_chat(content: str, status_code: int) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProviderRequestError("json", "provider stream content is not valid JSON", status_code) from exc
    if not isinstance(parsed, dict):
        raise ProviderRequestError("schema", "provider stream content must be a JSON object", status_code)
    return {"choices": [{"message": {"content": parsed}}]}


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def parse_chat_json(body: dict, error_message: str) -> dict:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ProviderRequestError("schema", error_message) from exc
    if not isinstance(parsed, dict):
        raise ProviderRequestError("schema", error_message)
    return parsed


def run_sync(awaitable: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("async API required inside a running event loop")
