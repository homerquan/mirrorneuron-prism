import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from prism.api import create_app
from prism.backends import OpenAIBackend
from prism.config import PrismConfig, RawModel


def completion(content="hello", finish="stop", **message):
    return {
        "id": "upstream-id",
        "object": "chat.completion",
        "created": 1,
        "model": "physical-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **message},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    }


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "test-secret")
    small = RawModel(
        id="small",
        name="physical-small",
        base_url="http://worker/v1",
        context_window=4096,
        max_output_tokens=512,
        safety_margin=64,
    )
    large = RawModel(
        id="large",
        name="physical-large",
        base_url="http://synth/v1",
        context_window=32768,
        max_output_tokens=512,
        safety_margin=64,
    )
    config = PrismConfig(
        profiles={
            "prism": {
                "direct": "small",
                "worker": "small",
                "synthesizer": "large",
                "public_max_output_tokens": 256,
                "worker_output_tokens": 128,
                "partition_bytes": 600,
            },
            "prism-direct": {
                "direct": "small",
                "strategy": "direct",
                "public_max_output_tokens": 256,
                "worker_output_tokens": 128,
            },
        }
    )
    return config, {"small": small, "large": large}


def app_with(harness, handler):
    config, models = harness
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(config, models=models, backend=OpenAIBackend(models, client))
    return app, client


@pytest.mark.asyncio
async def test_sdk_direct_alias_usage_auth_and_trace(harness):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    app, backend_client = app_with(harness, handler)
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        sdk = AsyncOpenAI(
            base_url="http://prism/v1", api_key="test-secret", http_client=client
        )
        assert (await sdk.models.list()).data[0].id == "prism"
        raw = await sdk.chat.completions.with_raw_response.create(
            model="prism",
            messages=[
                {"role": "developer", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=100,
        )
        result = raw.parse()
        assert result.model == "prism"
        assert result.choices[0].message.content == "hello"
        assert (
            result.usage.completion_tokens == 5
        )  # Logical UTF-8 bytes, not provider bill.
        assert calls[0]["model"] == "physical-small"
        assert calls[0]["messages"][0]["role"] == "developer"
        assert calls[0]["max_tokens"] == 100
        assert "max_completion_tokens" not in calls[0]
        trace = (
            await client.get(
                "/v1/prism/traces/" + raw.headers["x-request-id"],
                headers={"authorization": "Bearer test-secret"},
            )
        ).json()
        assert (
            trace["execution_usage"]["calls"][0]["provider_usage"]["completion_tokens"]
            == 10
        )
        assert "Hello" not in json.dumps(trace)
        assert (await client.get("/v1/models")).status_code == 401


@pytest.mark.asyncio
async def test_large_source_preserved_coverage_and_context_bounds(harness):
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if request.url.host == "worker":
            work = json.loads(payload["messages"][-1]["content"])
            return httpx.Response(
                200,
                json=completion(
                    json.dumps(
                        {
                            "status": "complete",
                            "records": [
                                {"quote": work["source"], "fact": "source fact"}
                            ],
                            "needs": [],
                        }
                    )
                ),
            )
        return httpx.Response(200, json=completion("source-backed answer"))

    app, client_backend = app_with(harness, handler)
    source = "policy α has an exception\n" * 180
    body = {
        "model": "prism",
        "messages": [
            {
                "role": "user",
                "content": "Summarize these policies.\n<prism-source>"
                + source
                + "</prism-source>\nAnswer in Spanish.",
            }
        ],
    }
    async with (
        client_backend,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client,
    ):
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        trace = (
            await client.get("/v1/prism/traces/" + response.headers["x-request-id"])
        ).json()
        coverage = trace["coverage"]
        assert set(coverage["required_partitions"]) == set(
            coverage["validated_partitions"]
        )
        assert len(coverage["required_partitions"]) > 1
        workers = [c for c in calls if c["model"] == "physical-small"]
        submitted = "".join(
            json.loads(c["messages"][-1]["content"])["source"] for c in workers
        )
        assert submitted == source
        assert all("Answer in Spanish" in json.dumps(c["messages"]) for c in workers)
        assert "Answer in Spanish" in calls[-1]["messages"][0]["content"]
        assert trace["strategy"] == "evidence_map"
        assert trace["execution_usage"]["consumed_or_conservatively_charged"][
            "calls"
        ] == len(calls)
        from prism.contracts import check_context

        for call in calls:
            model = harness[1][
                "small" if call["model"] == "physical-small" else "large"
            ]
            check_context(
                call["messages"],
                {
                    "temperature": call.get("temperature", 0),
                    "response_format": call.get("response_format", {}),
                },
                call["max_completion_tokens"],
                model,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["bad_quote", "truncated", "needs", "invalid_json"])
async def test_incomplete_evidence_never_synthesized(harness, failure):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        work = json.loads(json.loads(request.content)["messages"][-1]["content"])
        text = json.dumps(
            {
                "status": "complete",
                "records": [
                    {
                        "quote": "fabricated quote"
                        if failure == "bad_quote"
                        else work["source"],
                        "fact": "a fact",
                    }
                ],
                "needs": ["a definition"] if failure == "needs" else [],
            }
        )
        return httpx.Response(
            200,
            json=completion(
                "{bad" if failure == "invalid_json" else text,
                "length" if failure == "truncated" else "stop",
            ),
        )

    app, backend_client = app_with(harness, handler)
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-secret"},
            json={
                "model": "prism",
                "messages": [
                    {
                        "role": "user",
                        "content": "Summarize. <prism-source>"
                        + "evidence text\n" * 600
                        + "</prism-source>",
                    }
                ],
            },
        )
        assert response.status_code == 502, response.text
        assert "synth" not in calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_patch,code",
    [
        ({"n": 2}, "unsupported_feature"),
        ({"made_up": True}, "unsupported_parameter"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {}}],
                    }
                ]
            },
            "unsupported_feature",
        ),
        ({"max_tokens": 5, "max_completion_tokens": 10}, "invalid_request"),
        ({"model": "unknown"}, "model_not_found"),
        ({"response_format": None}, "invalid_request"),
        ({"response_format": {"type": []}}, "invalid_request"),
        ({"reasoning_effort": []}, "invalid_request"),
        ({"messages": [{"role": [], "content": "hi"}]}, "invalid_request"),
        ({"parallel_tool_calls": 1}, "invalid_request"),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "x",
                        "schema": {"$ref": "https://invalid.example/schema"},
                    },
                }
            },
            "invalid_request",
        ),
    ],
)
async def test_compatibility_gate_before_dispatch(harness, body_patch, code):
    calls = []
    app, backend_client = app_with(harness, lambda req: calls.append(req))
    body = {
        "model": "prism",
        "messages": [{"role": "user", "content": "Hi"}],
        **body_patch,
    }
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"authorization": "Bearer test-secret"},
        )
        assert response.json()["error"]["code"] == code
        assert not calls


@pytest.mark.asyncio
async def test_tools_forwarded_and_never_executed(harness):
    calls = []
    tool = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "send_email", "arguments": '{"to":"a"}'},
    }

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json=completion(None, "tool_calls", tool_calls=[tool])
        )

    app, backend_client = app_with(harness, handler)
    body = {
        "model": "prism",
        "messages": [{"role": "user", "content": "Send email"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "send_email", "parameters": {"type": "object"}},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "send_email"}},
        "parallel_tool_calls": False,
    }
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["tool_calls"] == [tool]
        assert calls[0]["tool_choice"] == body["tool_choice"]
        assert calls[0]["parallel_tool_calls"] is False
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_stream_sdk_stable_ids_finish_and_usage(harness):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["stream_options"]["include_usage"] is True
        events = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 10}},
        ]
        return httpx.Response(
            200,
            content="".join("data: " + json.dumps(e) + "\n\n" for e in events)
            + "data: [DONE]\n\n",
        )

    app, backend_client = app_with(harness, handler)
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        sdk = AsyncOpenAI(
            base_url="http://prism/v1", api_key="test-secret", http_client=client
        )
        stream = await sdk.chat.completions.create(
            model="prism",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [c async for c in stream]
        assert len({c.id for c in chunks}) == 1
        assert all(c.model == "prism" for c in chunks)
        assert chunks[1].choices[0].delta.content == "hello"
        assert chunks[2].choices[0].finish_reason == "stop"
        assert not chunks[-1].choices and chunks[-1].usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_structured_stream_buffered_and_invalid_schema_rejected(harness):
    def handler(request):
        assert not json.loads(request.content)["stream"]
        return httpx.Response(200, json=completion('{"ok":true}'))

    app, backend_client = app_with(harness, handler)
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client,
    ):
        body = {
            "model": "prism",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        response = await client.post("/v1/chat/completions", json=body)
        assert (
            response.status_code == 200
            and response.headers["x-prism-stream-mode"] == "buffered"
        )
        assert "[DONE]" in response.text
        body["response_format"]["json_schema"]["schema"]["properties"]["ok"] = {
            "type": "string"
        }
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_backend_output"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_auth_errors_are_sanitized_before_commit(harness, stream):
    app, backend_client = app_with(
        harness, lambda req: httpx.Response(401, json={"secret": "upstream-secret"})
    )
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-secret"},
            json={
                "model": "prism",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": stream,
            },
        )
        assert response.status_code == 502
        assert "upstream-secret" not in response.text


@pytest.mark.asyncio
async def test_no_silent_truncation_ambiguous_or_exhaustive(harness):
    app, backend_client = app_with(
        harness, lambda req: pytest.fail("must not call backend")
    )
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client,
    ):
        for content, code in (
            ("x" * 10000, "ambiguous_source_boundary"),
            (
                "Count every event. <prism-source>"
                + "event\n" * 2000
                + "</prism-source>",
                "unsupported_coverage_contract",
            ),
        ):
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "prism",
                    "messages": [{"role": "user", "content": content}],
                },
            )
            assert response.json()["error"]["code"] == code


@pytest.mark.asyncio
async def test_payload_admission_and_budget_before_backend(harness):
    config, models = harness
    config.server.max_payload_bytes = 1024
    app, backend_client = app_with(
        harness, lambda req: pytest.fail("must not call backend")
    )
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client,
    ):
        assert (
            await client.post("/v1/chat/completions", content=b"x" * 1025)
        ).status_code == 413
        config.profiles["prism"].limits.max_output_tokens = 1
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "prism", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 413


@pytest.mark.asyncio
async def test_deadline_cancels_started_work_and_keeps_accounting(harness):
    async def handler(request):
        await asyncio.sleep(10)
        return httpx.Response(200, json=completion())

    harness[0].profiles["prism"].limits.deadline_seconds = 0.02
    app, backend_client = app_with(harness, handler)
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "prism", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 504
        trace = (
            await client.get("/v1/prism/traces/" + response.headers["x-request-id"])
        ).json()
        physical = trace["execution_usage"]
        assert physical["consumed_or_conservatively_charged"]["calls"] == 1
        assert physical["calls"][0]["usage_status"] == "unknown"


@pytest.mark.asyncio
async def test_disconnect_cancels_child_through_asgi_request_path(harness):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app, backend_client = app_with(harness, handler)
    queue = asyncio.Queue()
    await queue.put(
        {
            "type": "http.request",
            "body": json.dumps(
                {"model": "prism", "messages": [{"role": "user", "content": "Hi"}]}
            ).encode(),
            "more_body": False,
        }
    )
    sent = []

    async def send(event):
        sent.append(event)

    async with backend_client:
        task = asyncio.create_task(
            app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/v1/chat/completions",
                    "raw_path": b"/v1/chat/completions",
                    "query_string": b"",
                    "root_path": "",
                    "headers": [(b"authorization", b"Bearer test-secret")],
                    "server": ("prism", 80),
                    "client": ("client", 1),
                },
                queue.get,
                send,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        await queue.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=1)
        assert cancelled.is_set()
        assert sent[0]["status"] == 499


@pytest.mark.asyncio
async def test_stream_failure_after_commit_has_error_and_no_success_marker(harness):
    def handler(request):
        return httpx.Response(
            200,
            content='data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n',
        )

    app, backend_client = app_with(harness, handler)
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "prism",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "partial" in response.text and "incomplete_stream" in response.text
        assert "[DONE]" not in response.text


@pytest.mark.asyncio
async def test_stream_tool_deltas_indices_and_named_choice(harness):
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"key":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"value"}'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    app, backend_client = app_with(
        harness,
        lambda req: httpx.Response(
            200,
            content="".join("data: " + json.dumps(e) + "\n\n" for e in events)
            + "data: [DONE]\n\n",
        ),
    )
    async with (
        backend_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        sdk = AsyncOpenAI(
            base_url="http://prism/v1", api_key="test-secret", http_client=client
        )
        stream = await sdk.chat.completions.create(
            model="prism",
            messages=[{"role": "user", "content": "Look up key"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
            parallel_tool_calls=False,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [c async for c in stream]
        assert chunks[0].choices[0].delta.tool_calls[0].id == "call_a"
        arguments = "".join(
            c.choices[0].delta.tool_calls[0].function.arguments
            for c in chunks
            if c.choices and c.choices[0].delta.tool_calls
        )
        assert json.loads(arguments) == {"key": "value"}
        assert chunks[-2].choices[0].finish_reason == "tool_calls"
        assert chunks[-1].usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_stream_disconnect_closes_backend_and_releases_admission(harness):
    import hashlib

    closed = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
            await asyncio.sleep(10)

        async def aclose(self):
            await asyncio.sleep(0.01)
            closed.set()

    harness[0].server.max_concurrent_requests = 1
    app, backend_client = app_with(
        harness, lambda req: httpx.Response(200, stream=Stream())
    )
    queue = asyncio.Queue()
    await queue.put(
        {
            "type": "http.request",
            "body": json.dumps(
                {
                    "model": "prism",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                }
            ).encode(),
            "more_body": False,
        }
    )
    sent = []

    async def send(event):
        sent.append(event)
        if event["type"] == "http.response.body" and b"partial" in event.get(
            "body", b""
        ):
            await queue.put({"type": "http.disconnect"})

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", b"Bearer test-secret")],
        "server": ("prism", 80),
        "client": ("client", 1),
    }
    async with backend_client:
        await asyncio.wait_for(app(scope, queue.get, send), timeout=1)
        assert closed.is_set()
        request_id = dict(sent[0]["headers"])[b"x-request-id"].decode()
        owner = hashlib.sha256(b"Bearer test-secret").hexdigest()
        trace = app.state.traces.get(owner, request_id)
        assert trace["delivery_status"] == "interrupted"
        assert (
            trace["execution_usage"]["consumed_or_conservatively_charged"]["calls"] == 1
        )
        # A subsequent request must enter dispatch rather than hit a leaked admission slot.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://prism",
            headers={"authorization": "Bearer test-secret"},
        ) as client:
            result = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "prism",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "n": 2,
                },
            )
            assert result.json()["error"]["code"] == "unsupported_feature"
