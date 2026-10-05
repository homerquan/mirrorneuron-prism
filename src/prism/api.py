"""Thin authenticated ASGI Chat Completions service."""

import asyncio
import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .backends import OpenAIBackend
from .capacity import CapacityEvaluator, ProfileCapacityEvaluator
from .config import load_config
from .contracts import parse_json, validate_request
from .decision import LayaDecision
from .engine import ExecutionEngine
from .errors import PrismError
from .telemetry import TraceStore


async def connected(request, coroutine):
    """Cancel preparation and all descendants on an actual ASGI disconnect event."""
    task = asyncio.create_task(coroutine)

    async def disconnect():
        while True:
            event = await request.receive()
            if event["type"] == "http.disconnect":
                return

    watcher = asyncio.create_task(disconnect())
    try:
        done, _ = await asyncio.wait(
            {task, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if watcher in done and task not in done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise PrismError("client disconnected", "client_disconnected", 499)
        return await task
    finally:
        for pending in (task, watcher):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


def create_app(
    config, *, models=None, backend=None, decision_agent=None, no_auth=False
):
    if models is None:
        config, models = load_config(config)
    transport = backend or OpenAIBackend(models)
    decision = LayaDecision(config.decision, decision_agent)
    engine = ExecutionEngine(config, models, transport, decision)
    traces = TraceStore(config.server.trace_capacity, config.server.trace_ttl_seconds)
    admitted = 0
    capacity = CapacityEvaluator(models, transport, config.capacity)
    profile_capacity = ProfileCapacityEvaluator(engine, config.capacity)
    profile_capacity.slots = capacity.slots

    @asynccontextmanager
    async def lifespan(app):
        if not no_auth and not os.environ.get(config.server.api_key_env):
            raise ValueError("Prism API key environment variable is unset")
        try:
            await asyncio.to_thread(decision.prepare)
            yield
        finally:
            decision.close()
            await transport.close()

    app = FastAPI(
        title="Prism",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.engine = engine
    app.state.traces = traces
    app.state.capacity = capacity
    app.state.no_auth = no_auth

    @app.exception_handler(PrismError)
    async def public_error(request, error):
        headers = (
            {"x-request-id": request.state.request_id}
            if hasattr(request.state, "request_id")
            else {}
        )
        return JSONResponse(error.body(), status_code=error.status, headers=headers)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        return JSONResponse(
            PrismError("internal execution failure", "internal_error", 500).body(),
            status_code=500,
        )

    def authenticate(request):
        if no_auth:
            # One public trace namespace; callers cannot select owners via headers.
            return "anonymous"
        configured = os.environ.get(config.server.api_key_env)
        if not configured:
            raise PrismError(
                "service authentication is not configured", "configuration_error", 503
            )
        authorization = request.headers.get("authorization", "")
        expected = "Bearer " + configured
        if not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise PrismError("invalid API key", "invalid_api_key", 401)
        return hashlib.sha256(authorization.encode()).hexdigest()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        authenticate(request)
        return {
            "object": "list",
            "data": [
                {"id": alias, "object": "model", "created": 0, "owned_by": "prism"}
                for alias in config.profiles
            ],
        }

    @app.get("/v1/prism/traces/{request_id}")
    async def show_trace(request: Request, request_id: str):
        owner = authenticate(request)
        trace = traces.get(owner, request_id)
        if trace is None:
            raise PrismError("trace not found or expired", "trace_not_found", 404)
        return trace

    @app.get("/capacity")
    @app.post("/capacity")
    async def show_capacity(
        request: Request, model: str | None = None, refresh: bool = False
    ):
        nonlocal admitted
        authenticate(request)
        if admitted >= config.server.max_concurrent_requests:
            raise PrismError(
                "service admission capacity exceeded", "admission_limit", 429
            )
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > 1024:
                raise PrismError(
                    "capacity request body exceeds 1024 bytes", "payload_too_large", 413
                )
        if payload:
            try:
                body = parse_json(payload)
            except (ValueError, UnicodeDecodeError) as exc:
                raise PrismError("malformed JSON request", "invalid_json") from exc
            if (
                request.method != "POST"
                or not isinstance(body, dict)
                or set(body) - {"model", "refresh"}
            ):
                raise PrismError(
                    "capacity body accepts only model and refresh", "invalid_request"
                )
            if "model" in body:
                if (
                    not isinstance(body["model"], str)
                    or not body["model"]
                    or "model" in request.query_params
                ):
                    raise PrismError(
                        "model must be a single nonempty configured alias",
                        "invalid_request",
                    )
                model = body["model"]
            if "refresh" in body:
                if (
                    type(body["refresh"]) is not bool
                    or "refresh" in request.query_params
                ):
                    raise PrismError(
                        "refresh must be a single boolean", "invalid_request"
                    )
                refresh = body["refresh"]
        profile = None
        if model is None:
            ids = list(models)
        elif model in config.profiles:
            profile = config.profiles[model]
            ids = list(
                dict.fromkeys(
                    ref
                    for ref in (
                        profile.direct,
                        profile.worker,
                        profile.worker_fallback,
                        profile.synthesizer,
                        profile.structured_output_model,
                        profile.verifier,
                        profile.evidence_compaction.model
                        if profile.evidence_compaction
                        else None,
                        profile.evidence_reduction.model
                        if profile.evidence_reduction
                        else None,
                        *(
                            profile.optimization.model_ids
                            if profile.optimization
                            else []
                        ),
                    )
                    if ref
                )
            )
        elif model in models:
            ids = [model]
        else:
            raise PrismError(
                "unknown configured model combination or profile",
                "model_not_found",
                404,
            )
        admitted += 1
        try:
            measured = await connected(request, capacity.query(ids, refresh))
            result = {
                "object": "capacity",
                "source": "live_evaluation",
                "data": measured,
            }
            if profile:
                observation = (
                    await connected(request, profile_capacity.query([model], refresh))
                )[0]
                result["profile"] = {
                    "model": model,
                    "aggregation": "end_to_end_profile",
                    "supported": {
                        feature: value["supported"]
                        for feature, value in observation["capabilities"].items()
                    },
                    "evaluation": observation,
                }
            return JSONResponse(result, headers={"cache-control": "no-store"})
        finally:
            admitted -= 1

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        nonlocal admitted
        owner = authenticate(request)
        if admitted >= config.server.max_concurrent_requests:
            raise PrismError(
                "service admission capacity exceeded", "admission_limit", 429
            )
        admitted += 1
        streaming_owns_admission = False
        execution = None
        try:
            payload = bytearray()
            async for chunk in request.stream():
                payload.extend(chunk)
                if len(payload) > config.server.max_payload_bytes:
                    raise PrismError(
                        "HTTP payload exceeds admission limit", "payload_too_large", 413
                    )
            try:
                body = validate_request(parse_json(payload))
            except (ValueError, UnicodeDecodeError) as exc:
                raise PrismError("malformed JSON request", "invalid_json") from exc
            execution = engine.prepare(body)
            request.state.request_id = execution["trace"]["request_id"]
            await connected(request, engine.select_policy(execution))
            headers = {
                "x-request-id": request.state.request_id,
                "x-prism-accounting": "prism-utf8-v1",
                "x-prism-policy": execution["strategy"],
                "x-prism-coverage": "focused"
                if execution["strategy"] == "retrieve_read"
                else "visual_observations"
                if execution["strategy"] == "vision_synthesis"
                else "text_observations"
                if execution["strategy"] == "text_synthesis"
                else "full",
            }
            if body.get("stream"):
                buffered = (
                    execution["strategy"] != "direct"
                    or body.get("response_format", {}).get("type", "text") != "text"
                )
                if buffered:
                    result = await connected(request, engine.execute(execution))
                    events = engine.buffered_stream(execution, result)
                    headers["x-prism-stream-mode"] = "buffered"
                else:
                    events = engine.direct_stream(execution)
                    headers["x-prism-stream-mode"] = "forwarded"
                # Open/validate upstream BEFORE committing the HTTP success response.
                try:
                    first = await connected(request, anext(events))
                except BaseException:
                    await events.aclose()
                    raise

                async def sse():
                    nonlocal admitted
                    execution["trace"]["delivery_status"] = "streaming"
                    try:
                        yield "data: " + json.dumps(first) + "\n\n"
                        async for event in events:
                            yield "data: " + json.dumps(event) + "\n\n"
                        yield "data: [DONE]\n\n"
                        execution["trace"]["delivery_status"] = "complete"
                    except PrismError as exc:
                        # Post-commit failure cannot become a replacement answer.
                        yield "data: " + json.dumps(exc.body()) + "\n\n"
                    finally:
                        if execution["trace"]["delivery_status"] != "complete":
                            execution["trace"]["delivery_status"] = "interrupted"
                        with anyio.CancelScope(shield=True):
                            try:
                                await events.aclose()
                            finally:
                                engine.finalize_trace(
                                    execution, execution["trace"]["stop_reason"]
                                )
                                traces.put(owner, execution["trace"])
                                admitted -= 1

                streaming_owns_admission = True
                headers.update({"cache-control": "no-cache", "x-accel-buffering": "no"})
                return StreamingResponse(
                    sse(), media_type="text/event-stream", headers=headers
                )
            result = await connected(request, engine.execute(execution))
            return JSONResponse(result, headers=headers)
        finally:
            if not streaming_owns_admission:
                admitted -= 1
                if execution is not None:
                    traces.put(owner, execution["trace"])

    return app
