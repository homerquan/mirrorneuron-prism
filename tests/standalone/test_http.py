"""Opt-in socket conformance: actual Uvicorn, OpenAI SDK, and HTTP backend adapter."""

import json
import socket
import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

from prism.api import create_app
from prism.config import PrismConfig, RawModel


@contextmanager
def serve(app):
    sock = socket.create_server(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "server failed to start"
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.integration
def test_sdk_over_real_http(monkeypatch):
    from fastapi import Request

    monkeypatch.setenv("PRISM_API_KEY", "socket-test")
    upstream = FastAPI()
    calls = []

    @upstream.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        calls.append(body)
        if body["stream"]:
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "prism"},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            ]
            return StreamingResponse(
                iter(
                    ["data: " + json.dumps(e) + "\n\n" for e in events]
                    + ["data: [DONE]\n\n"]
                ),
                media_type="text/event-stream",
            )
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "prism"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    with serve(upstream) as upstream_url:
        model = RawModel(id="physical", name="physical", base_url=upstream_url + "/v1")
        config = PrismConfig(profiles={"prism": {"direct": "physical"}})
        app = create_app(config, models={"physical": model})
        with (
            serve(app) as prism_url,
            OpenAI(
                base_url=prism_url + "/v1", api_key="socket-test", max_retries=0
            ) as client,
        ):
            messages = [
                {"role": "developer", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
            ]
            response = client.chat.completions.create(model="prism", messages=messages)
            assert (
                response.model == "prism"
                and response.choices[0].message.content == "prism"
            )
            stream = client.chat.completions.create(
                model="prism",
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = list(stream)
            assert chunks[0].choices[0].delta.content == "prism"
            assert chunks[-1].usage.completion_tokens == 5
            assert len({c.id for c in chunks}) == 1
            assert calls[0]["messages"] == messages
            assert calls[0]["model"] == "physical"
