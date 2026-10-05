"""LiteLLM SDK adapter with bounded calls, strict errors, and no hidden retries."""

import asyncio
import copy
import json
import os
import time

import anyio
import httpx

from .errors import PrismError


def provider_error(status, body=b""):
    # Classify a bounded error body, but never expose its contents or credentials.
    text = body.decode("utf-8", errors="ignore").lower()
    if any(
        marker in text
        for marker in (
            "context size has been exceeded",
            "context_length_exceeded",
            "exceeds the available context",
            "exceed the available context",
            "maximum context length",
        )
    ):
        return PrismError(
            "backend runtime context limit exceeded",
            "upstream_context_length_exceeded",
            502,
        )
    if status in {401, 403}:
        return PrismError(
            "backend authentication failed", "upstream_authentication", 502
        )
    if status == 429:
        return PrismError("backend rate limit exceeded", "upstream_rate_limit", 429)
    if status in {400, 404, 422} and any(
        marker in text
        for marker in (
            "not supported",
            "does not support",
            "unsupported",
            "no endpoints found that support",
        )
    ):
        return PrismError(
            "backend rejects requested feature", "upstream_unsupported_feature", 422
        )
    return PrismError(
        f"backend rejected or failed the request (HTTP {status})", "upstream_error", 502
    )


async def response_error(response):
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk[: 8192 - len(body)])
        if len(body) >= 8192:
            break
    error = provider_error(response.status_code, body)
    error.upstream_status = response.status_code
    return error


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


def sdk_error(exc):
    """Recover guarded errors through SDK exception wrappers without leaking text."""
    pending, seen = [exc], set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, PrismError):
            return error
        pending.extend(
            e
            for e in (
                getattr(error, "__cause__", None),
                getattr(error, "__context__", None),
                getattr(error, "original_exception", None),
            )
            if isinstance(e, BaseException)
        )
    if type(exc).__name__ == "UnsupportedParamsError":
        return PrismError(
            "backend adapter rejects requested feature",
            "upstream_unsupported_feature",
            422,
        )
    if "Timeout" in type(exc).__name__:
        return PrismError("backend/request deadline exceeded", "deadline_exceeded", 504)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        error = provider_error(status)
        error.upstream_status = status
        return error
    return PrismError("backend connection or protocol failure", "upstream_error", 502)


class LiteLLMBackend:
    def __init__(self, models, client=None):
        self.models = models
        self.client = client or httpx.AsyncClient(
            timeout=None, follow_redirects=False, trust_env=False
        )
        self.owned_client = client is None
        from .wire import GuardedTransport

        self.compatible_http = httpx.AsyncClient(
            transport=GuardedTransport(self.client, True), timeout=None
        )
        self.native_http = httpx.AsyncClient(
            transport=GuardedTransport(self.client), timeout=None
        )
        self.native_handler = None
        self.cost_tracker = None
        groups = {}
        for model in models.values():
            key = self.engine_key(model)
            groups[key] = min(groups.get(key, model.concurrency), model.concurrency)
        shared = {key: asyncio.Semaphore(limit) for key, limit in groups.items()}
        self.semaphores = {ref: shared[self.engine_key(m)] for ref, m in models.items()}
        self.rate_locks = {key: asyncio.Lock() for key in groups}
        self.next_start = {}
        self.rate_limits = {}
        for m in models.values():
            if m.rate_limit_rpm:
                key = self.engine_key(m)
                self.rate_limits[key] = min(
                    self.rate_limits.get(key, m.rate_limit_rpm), m.rate_limit_rpm
                )

    @staticmethod
    def engine_key(model):
        provider = model.provider or (
            "openai" if model.base_url else model.name.split("/", 1)[0]
        )
        name = model.name
        if not (provider == "openai" and model.base_url) and name.startswith(
            provider + "/"
        ):
            name = name[len(provider) + 1 :]
        return (provider, (model.base_url or "").rstrip("/"), name)

    async def close(self):
        await self.compatible_http.aclose()
        await self.native_http.aclose()
        if self.owned_client:
            await self.client.aclose()

    async def pace(self, model):
        key = self.engine_key(model)
        if key not in self.rate_limits:
            return
        async with self.rate_locks[key]:
            delay = max(0, self.next_start.get(key, 0) - time.monotonic())
            if delay:
                await asyncio.sleep(delay)
            self.next_start[key] = time.monotonic() + 60 / self.rate_limits[key]

    async def call(
        self, model, messages, parameters, output, timeout, stream=False, responses=None
    ):
        # Lazy import keeps CLI/config tooling independent of SDK import work.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        os.environ.setdefault("LITELLM_LOG", "ERROR")
        import litellm

        from .wire import CALL

        litellm.suppress_debug_info = True
        kwargs = model.call_parameters(parameters, output)
        provider = model.provider or ("openai" if model.base_url else None)
        compatible = provider == "openai" and model.base_url is not None
        credential = model.credential()
        resolved_name = model.name
        if not compatible:
            resolved_name, provider, _, _ = litellm.get_llm_provider(
                model=model.name, custom_llm_provider=provider, api_base=model.base_url
            )
        if provider == "openai":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=credential
                or (
                    "prism-no-key"
                    if compatible
                    else os.environ.get("OPENAI_API_KEY", "")
                ),
                base_url=model.base_url
                or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                http_client=self.compatible_http,
                max_retries=0,
            )
        elif provider == "azure":
            from openai import AsyncAzureOpenAI

            client = AsyncAzureOpenAI(
                api_key=credential or os.environ.get("AZURE_API_KEY"),
                azure_endpoint=model.base_url or os.environ.get("AZURE_API_BASE"),
                api_version=model.api_version or os.environ.get("AZURE_API_VERSION"),
                azure_ad_token=model.provider_options.get("azure_ad_token"),
                http_client=self.compatible_http,
                max_retries=0,
            )
        else:
            if self.native_handler is None:
                from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

                handler = AsyncHTTPHandler()
                await handler.client.aclose()
                handler.client = self.native_http
                self.native_handler = handler
            client = self.native_handler
        kwargs.update(
            model=model.name if compatible else resolved_name,
            messages=copy.deepcopy(messages),
            stream=stream,
            api_key=credential,
            client=client,
            timeout=min(timeout, model.timeout_seconds),
            num_retries=0,
            max_retries=0,
            drop_params=False,
            caching=False,
            **{"no-log": True},
        )
        if provider:
            kwargs["custom_llm_provider"] = provider
        if model.base_url:
            kwargs["api_base"] = model.base_url
        if model.api_version:
            kwargs["api_version"] = model.api_version
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        if compatible:
            # Let compatible endpoints decide parameter support, not the catalog.
            kwargs["allowed_openai_params"] = list(
                model.call_parameters(parameters, output)
            )
        kwargs.update(model.provider_options)
        token = CALL.set(
            {
                "messages": copy.deepcopy(messages),
                "name": model.name if compatible else resolved_name,
                "output": output,
                "stream": stream,
                "responses": responses if responses is not None else [],
                "unauthenticated": compatible and not credential,
            }
        )
        try:
            result = await litellm.acompletion(**kwargs)
            wire = CALL.get()
            if not stream and "wire_usage" in wire:
                result = (
                    result.model_dump(exclude_none=True)
                    if hasattr(result, "model_dump")
                    else result
                )
                result["usage"] = wire["wire_usage"]
            return result
        finally:
            CALL.reset(token)

    async def complete(self, model, messages, parameters, output, ledger, reservation):
        started, usage, status, diagnostics = time.monotonic(), None, "failed", {}
        responses = []
        try:
            async with asyncio.timeout(
                min(ledger.remaining_seconds(), model.timeout_seconds)
            ):
                async with self.semaphores[model.id]:
                    await self.pace(model)
                    await ledger.start(reservation)
                    result = await self.call(
                        model,
                        messages,
                        parameters,
                        output,
                        ledger.remaining_seconds(),
                        responses=responses,
                    )
                    data = (
                        result.model_dump(exclude_none=True)
                        if hasattr(result, "model_dump")
                        else result
                    )
                    usage = data.get("usage") if isinstance(data, dict) else None
                    if len(json.dumps(data).encode()) > output * 32 + 65536:
                        raise PrismError(
                            "backend result exceeded artifact size cap",
                            "invalid_backend_output",
                            502,
                        )
                    data = validate_completion(data)
                    diagnostics["finish_reason"] = data["choices"][0]["finish_reason"]
                    status = "complete"
                    return data
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            error = sdk_error(exc)
            if isinstance(exc, TimeoutError):
                error = PrismError(
                    "backend/request deadline exceeded", "deadline_exceeded", 504
                )
            diagnostics.update(
                error_code=error.code,
                upstream_status=getattr(error, "upstream_status", None),
            )
            raise error from exc
        finally:
            with anyio.CancelScope(shield=True):
                await close_responses(responses)
                try:
                    await ledger.finish(
                        reservation,
                        usage,
                        status,
                        (time.monotonic() - started) * 1000,
                        diagnostics=diagnostics,
                    )
                finally:
                    if self.cost_tracker:
                        self.cost_tracker.record_call(model, usage, reservation)

    async def stream(self, model, messages, parameters, output, ledger, reservation):
        started, usage, status, diagnostics = time.monotonic(), None, "failed", {}
        events = None
        responses = []
        try:
            async with asyncio.timeout(
                min(ledger.remaining_seconds(), model.timeout_seconds)
            ):
                async with self.semaphores[model.id]:
                    await self.pace(model)
                    await ledger.start(reservation)
                    events = await self.call(
                        model,
                        messages,
                        parameters,
                        output,
                        ledger.remaining_seconds(),
                        True,
                        responses,
                    )
                    finished, total = False, 0
                    async for event in events:
                        data = (
                            event.model_dump(exclude_none=True)
                            if hasattr(event, "model_dump")
                            else event
                        )
                        choices = (
                            data.get("choices") if isinstance(data, dict) else None
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
                            delta = choice.get("delta")
                            # SDKs may attach usage to an empty terminal choice.
                            if finished:
                                if delta or choice.get("finish_reason") is not None:
                                    raise PrismError(
                                        "upstream content after finish",
                                        "invalid_backend_output",
                                        502,
                                    )
                                data["choices"] = []
                            else:
                                if choice.get("index") != 0 or not isinstance(
                                    delta, dict
                                ):
                                    raise PrismError(
                                        "invalid upstream delta",
                                        "invalid_backend_output",
                                        502,
                                    )
                                if delta.get("role", "assistant") != "assistant" or (
                                    delta.get("content") is not None
                                    and not isinstance(delta["content"], str)
                                ):
                                    raise PrismError(
                                        "invalid upstream text delta",
                                        "invalid_backend_output",
                                        502,
                                    )
                                total += len(json.dumps(delta).encode())
                                if total > output * 32 + 65536:
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
                                    diagnostics["finish_reason"] = finish
                        yield data
                    if not finished:
                        raise PrismError(
                            "upstream stream ended without finish reason",
                            "incomplete_stream",
                            502,
                        )
                    status = "complete"
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except Exception as exc:
            error = sdk_error(exc)
            if isinstance(exc, TimeoutError):
                error = PrismError(
                    "backend/request deadline exceeded", "deadline_exceeded", 504
                )
            diagnostics.update(
                error_code=error.code,
                upstream_status=getattr(error, "upstream_status", None),
            )
            raise error from exc
        finally:
            with anyio.CancelScope(shield=True):
                if events is not None:
                    with anyio.move_on_after(2, shield=True):
                        await events.aclose()
                for response in responses:
                    wire = response.extensions.get("prism_wire", {})
                    if "wire_usage" in wire:
                        usage = wire["wire_usage"]
                await close_responses(responses)
                try:
                    await ledger.finish(
                        reservation,
                        usage,
                        status,
                        (time.monotonic() - started) * 1000,
                        diagnostics=diagnostics,
                    )
                finally:
                    if self.cost_tracker:
                        self.cost_tracker.record_call(model, usage, reservation)


# Backwards-compatible import for integrations that inject an HTTP test client.
OpenAIBackend = LiteLLMBackend


async def close_responses(responses):
    for response in responses:
        with anyio.move_on_after(2, shield=True):
            await response.aclose()
            # httpx marks is_closed before awaiting stream cleanup; cancellation
            # can interrupt that await inside the SDK. Finish the stream close.
            await response.stream.aclose()
        response.extensions.pop("prism_wire", None)
