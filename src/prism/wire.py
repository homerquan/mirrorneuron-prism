"""Bound SDK HTTP responses and preserve strict OpenAI-compatible wire checks."""

import contextvars
import json

import httpx

from .contracts import parse_json
from .errors import PrismError

CALL = contextvars.ContextVar("prism_sdk_call")


class CheckedStream(httpx.AsyncByteStream):
    def __init__(self, response, cap, sse, call):
        self.response, self.cap, self.sse = response, cap, sse
        self.call = call

    async def __aiter__(self):
        pending = bytearray()
        finished = False
        done = False
        total = 0
        async for chunk in self.response.aiter_bytes():
            if self.sse:
                pending.extend(chunk)
                if len(pending) > self.cap:
                    raise PrismError(
                        "upstream SSE event too large", "invalid_backend_output", 502
                    )
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    if not line.startswith(b"data:"):
                        continue
                    event = line[5:].strip()
                    if event == b"[DONE]":
                        if not finished:
                            raise PrismError(
                                "upstream stream ended without finish reason",
                                "incomplete_stream",
                                502,
                            )
                        done = True
                        continue
                    data = parse_json(event)
                    if isinstance(data, dict) and data.get("usage") is not None:
                        self.call["wire_usage"] = data["usage"]
                    choices = data.get("choices") if isinstance(data, dict) else None
                    if done or not isinstance(choices, list) or len(choices) > 1:
                        raise PrismError(
                            "invalid upstream SSE choices",
                            "invalid_backend_output",
                            502,
                        )
                    if choices:
                        choice = choices[0]
                        delta = (
                            choice.get("delta") if isinstance(choice, dict) else None
                        )
                        if (
                            finished
                            or choice.get("index") != 0
                            or not isinstance(delta, dict)
                        ):
                            raise PrismError(
                                "invalid upstream delta", "invalid_backend_output", 502
                            )
                        if delta.get("role", "assistant") != "assistant" or (
                            delta.get("content") is not None
                            and not isinstance(delta["content"], str)
                        ):
                            raise PrismError(
                                "invalid upstream role/text delta",
                                "invalid_backend_output",
                                502,
                            )
                        total += len(json.dumps(delta).encode())
                        if total > self.cap:
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
                            finished = True
            else:
                total += len(chunk)
                if total > self.cap:
                    raise PrismError(
                        "backend result exceeded artifact size cap",
                        "invalid_backend_output",
                        502,
                    )
            yield chunk
        if self.sse and not done:
            raise PrismError(
                "upstream stream ended without DONE", "incomplete_stream", 502
            )

    async def aclose(self):
        await self.response.aclose()


class GuardedTransport(httpx.AsyncBaseTransport):
    def __init__(self, client, compatible=False):
        self.client, self.compatible = client, compatible

    async def handle_async_request(self, request):
        from .backends import provider_error, response_error, validate_completion

        call = CALL.get()
        if self.compatible:
            call["wire_usage"] = None
            # LiteLLM maps developer to system using its static catalog. A
            # compatible server receives the operator/caller's exact messages.
            payload = parse_json(request.content)
            payload["messages"] = call["messages"]
            payload["model"] = call["name"]
            payload["stream"] = call["stream"]
            request = httpx.Request(
                request.method,
                request.url,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k != "content-length"
                    and not (
                        k == "authorization"
                        and call["unauthenticated"]
                        and v == "Bearer prism-no-key"
                    )
                },
                json=payload,
                extensions=request.extensions,
            )
        response = await self.client.send(request, stream=True)
        response.extensions["prism_wire"] = call
        call["responses"].append(response)
        try:
            if response.status_code >= 400:
                raise await response_error(response)
            streaming = call["stream"]
            checked = httpx.Response(
                response.status_code,
                # CheckedStream consumes aiter_bytes(), which already decodes
                # gzip/br. Retaining content-encoding would decode it twice.
                headers={
                    k: v
                    for k, v in response.headers.items()
                    if k not in {"content-encoding", "content-length"}
                },
                stream=CheckedStream(
                    response,
                    call["output"] * 32 + 65536,
                    self.compatible and streaming,
                    call,
                ),
                request=request,
            )
            if not streaming:
                data = await checked.aread()
                parsed = parse_json(data)
                if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                    # OpenRouter can return a provider error in an HTTP 200 body.
                    status = parsed["error"].get("code", 502)
                    if type(status) is not int or not 400 <= status <= 599:
                        status = 502
                    error = provider_error(status, data)
                    error.upstream_status = status
                    raise error
                if self.compatible:
                    validate_completion(parsed)
                if self.compatible or isinstance(parsed, dict) and "choices" in parsed:
                    call["wire_usage"] = parsed.get("usage")
            return checked
        except BaseException:
            await response.aclose()
            raise
