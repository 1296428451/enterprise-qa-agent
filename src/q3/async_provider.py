from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Any

import httpx


DEFAULT_TIMEOUT = httpx.Timeout(60.0)


async def post_json(url: str, api_key: str, payload: dict, timeout: httpx.Timeout = DEFAULT_TIMEOUT,
                    transport: httpx.AsyncBaseTransport | None = None) -> dict:
    async with httpx.AsyncClient(timeout=timeout, verify=True, transport=transport) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Authorization": "Bearer " + api_key},
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise ValueError("provider response must be a JSON object")
    return result


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def parse_chat_json(body: dict, error_message: str) -> dict:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(error_message) from exc
    if not isinstance(parsed, dict):
        raise ValueError(error_message)
    return parsed


def run_sync(awaitable: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("async API required inside a running event loop")
