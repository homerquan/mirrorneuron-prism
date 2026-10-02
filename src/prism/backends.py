"""Direct OpenAI-compatible HTTP adapter. No hidden retries or caller tools."""

import asyncio
import json
import time

import anyio
import httpx

from .contracts import parse_json
from .errors import PrismError


def provider_error(status):
    if status in {401, 403}:
        return PrismError(
            "backend authentication failed", "upstream_authentication", 502
        )
    if status == 429:
        return PrismError("backend rate limit exceeded", "upstream_rate_limit", 429)
    return PrismError("backend rejected or failed the request", "upstream_error", 502)


def validate_completion(data):
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("choices"), list)
        or len(data["choices"]) != 1
    ):
        raise PrismError(
            "backend returned an invalid choice list", "invalid_backend_output", 502
        )
    choice = data["choices"][0]
    if (
        not isinstance(choice, dict)
        or type(choice.get("index")) is not int
        or choice.get("index") != 0
    ):
        raise PrismError(
            "backend returned an invalid choice", "invalid_backend_output", 502
        )
    message = choice.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or choice.get("finish_reason")
        not in {"stop", "length", "tool_calls", "content_filter"}
    ):
        raise PrismError(
            "backend returned an invalid assistant message",
            "invalid_backend_output",
            502,
        )
    content = message.get("content")
    calls = message.get("tool_calls")
    if choice["finish_reason"] == "tool_calls" and not calls:
        raise PrismError("missing backend tool calls", "invalid_backend_output", 502)
    if content is not None and not isinstance(content, str):
        raise PrismError(
            "backend returned non-text content", "invalid_backend_output", 502
        )
    if calls is not None:
        ids = set()
        if not isinstance(calls, list) or not calls:
            raise PrismError(
                "backend returned invalid tool calls", "invalid_backend_output", 502
            )
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not call["id"]
                or call["id"] in ids
                or call.get("type") != "function"
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not isinstance(call["function"].get("arguments"), str)
            ):
                raise PrismError(
                    "backend returned invalid tool calls", "invalid_backend_output", 502
                )
            ids.add(call["id"])
        if choice["finish_reason"] != "tool_calls":
            raise PrismError(
                "tool call finish reason mismatch", "invalid_backend_output", 502
            )
    return data


class OpenAIBackend:
    def __init__(self, models, client=None):
        self.models = models
        self.client = client or httpx.AsyncClient(
            timeout=None, follow_redirects=False, trust_env=False
        )
        self.owned_client = client is None
        self.semaphores = {
            key: asyncio.Semaphore(m.concurrency) for key, m in models.items()
        }

    async def close(self):
        if self.owned_client:
            await self.client.aclose()

    def request(self, model, messages, parameters, output, stream=False):
        payload = {
            "model": model.name,
            "messages": messages,
            **parameters,
            "stream": stream,
        }
        # Preserve the requested upstream limit spelling on the direct route.
        if not ({"max_tokens", "max_completion_tokens"} & payload.keys()):
            payload["max_completion_tokens"] = output
        if stream:
            payload["stream_options"] = {"include_usage": True}
        headers = {"content-type": "application/json"}
        key = model.credential()
        if key:
            headers["authorization"] = f"Bearer {key}"
        return self.client.build_request(
            "POST",
            model.base_url.rstrip("/") + "/chat/completions",
            json=payload,
            headers=headers,
        )

    async def complete(self, model, messages, parameters, output, ledger, reservation):
        started = time.monotonic()
        usage = None
        status = "failed"
        try:
            async with self.semaphores[model.id]:
                await ledger.start(reservation)
                request = self.request(model, messages, parameters, output)
                async with asyncio.timeout(ledger.remaining_seconds()):
                    response = await self.client.send(request, stream=True)
                    try:
                        if response.status_code != 200:
                            raise provider_error(response.status_code)
                        chunks = bytearray()
                        async for chunk in response.aiter_bytes():
                            chunks.extend(chunk)
                            if len(chunks) > output * 32 + 65536:
                                raise PrismError(
                                    "backend result exceeded artifact size cap",
                                    "invalid_backend_output",
                                    502,
                                )
                        data = validate_completion(parse_json(chunks))
                        usage = data.get("usage")
                        status = "complete"
                        return data
                    finally:
                        with anyio.move_on_after(2, shield=True):
                            await response.aclose()
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except TimeoutError as exc:
            raise PrismError(
                "backend/request deadline exceeded", "deadline_exceeded", 504
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise PrismError(
                "backend connection or protocol failure", "upstream_error", 502
            ) from exc
        finally:
            with anyio.CancelScope(shield=True):
                await ledger.finish(
                    reservation, usage, status, (time.monotonic() - started) * 1000
                )

    async def stream(self, model, messages, parameters, output, ledger, reservation):
        started = time.monotonic()
        usage = None
        status = "failed"
        try:
            async with self.semaphores[model.id]:
                await ledger.start(reservation)
                async with asyncio.timeout(ledger.remaining_seconds()):
                    response = await self.client.send(
                        self.request(model, messages, parameters, output, True),
                        stream=True,
                    )
                    try:
                        if response.status_code != 200:
                            raise provider_error(response.status_code)
                        # Bound each event before parsing; do not buffer the entire stream.
                        pending = bytearray()
                        finished_choice = False
                        total_bytes = 0
                        async for chunk in response.aiter_bytes():
                            pending.extend(chunk)
                            if len(pending) > output * 32 + 65536:
                                raise PrismError(
                                    "upstream SSE event too large",
                                    "invalid_backend_output",
                                    502,
                                )
                            while b"\n" in pending:
                                line, _, remaining = pending.partition(b"\n")
                                pending = bytearray(remaining)
                                line = line.rstrip(b"\r")
                                if not line.startswith(b"data:"):
                                    continue
                                event = line[5:].strip()
                                if event == b"[DONE]":
                                    if not finished_choice:
                                        raise PrismError(
                                            "upstream stream ended without finish reason",
                                            "incomplete_stream",
                                            502,
                                        )
                                    status = "complete"
                                    return
                                data = parse_json(event)
                                choices = (
                                    data.get("choices")
                                    if isinstance(data, dict)
                                    else None
                                )
                                if not isinstance(choices, list) or len(choices) > 1:
                                    raise PrismError(
                                        "invalid upstream SSE choices",
                                        "invalid_backend_output",
                                        502,
                                    )
                                if data.get("usage") is not None:
                                    usage = data["usage"]
                                if choices:
                                    choice = choices[0]
                                    if (
                                        not isinstance(choice, dict)
                                        or choice.get("index") != 0
                                        or not isinstance(choice.get("delta"), dict)
                                    ):
                                        raise PrismError(
                                            "invalid upstream delta",
                                            "invalid_backend_output",
                                            502,
                                        )
                                    if finished_choice:
                                        raise PrismError(
                                            "upstream content after finish",
                                            "invalid_backend_output",
                                            502,
                                        )
                                    delta = choice["delta"]
                                    if "role" in delta and delta["role"] != "assistant":
                                        raise PrismError(
                                            "invalid upstream role delta",
                                            "invalid_backend_output",
                                            502,
                                        )
                                    if (
                                        "content" in delta
                                        and delta["content"] is not None
                                        and not isinstance(delta["content"], str)
                                    ):
                                        raise PrismError(
                                            "invalid upstream text delta",
                                            "invalid_backend_output",
                                            502,
                                        )
                                    total_bytes += len(json.dumps(delta).encode())
                                    if total_bytes > output * 32 + 65536:
                                        raise PrismError(
                                            "upstream stream exceeded output cap",
                                            "invalid_backend_output",
                                            502,
                                        )
                                    finish = choice.get("finish_reason")
                                    if finish is not None:
                                        if finish not in {
                                            "stop",
                                            "length",
                                            "tool_calls",
                                            "content_filter",
                                        }:
                                            raise PrismError(
                                                "invalid upstream finish reason",
                                                "invalid_backend_output",
                                                502,
                                            )
                                        finished_choice = True
                                yield data
                        raise PrismError(
                            "upstream stream interrupted", "incomplete_stream", 502
                        )
                    finally:
                        with anyio.move_on_after(2, shield=True):
                            await response.aclose()
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except TimeoutError as exc:
            raise PrismError(
                "backend/request deadline exceeded", "deadline_exceeded", 504
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise PrismError(
                "backend connection or protocol failure", "upstream_error", 502
            ) from exc
        finally:
            with anyio.CancelScope(shield=True):
                await ledger.finish(
                    reservation, usage, status, (time.monotonic() - started) * 1000
                )
