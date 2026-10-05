import gzip
import json
from importlib.resources import files
from pathlib import Path

import httpx
import pytest

from prism.api import create_app
from prism.backends import LiteLLMBackend
from prism.capacity import ProfileCapacityEvaluator, image_challenge
from prism.cli import main
from prism.config import CapacityConfig, Limits, PrismConfig, RawModel
from prism.contracts import has_images
from prism.errors import PrismError
from prism.profiles import load_profile
from prism.runtime import Ledger

from .test_litellm_capacity import completion, decode_colors
from .test_policies import Agent


def setup(handler, *, no_auth=True, **profile):
    models = {
        "vision": RawModel(
            id="vision",
            name="physical-vision",
            base_url="http://fixture/v1",
            capabilities={"text", "image"},
        ),
        "text": RawModel(
            id="text",
            name="physical-text",
            base_url="http://fixture/v1",
            capabilities={"text", "json_object", "json_schema"},
        ),
    }
    config = PrismConfig(
        profiles={
            "mix": {
                "direct": "text",
                "worker": "vision",
                "synthesizer": "text",
                "strategy": "auto",
                "allowed_policies": ["direct", "vision_synthesis"],
                **profile,
            }
        }
    )
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        config,
        models=models,
        backend=LiteLLMBackend(models, upstream),
        decision_agent=Agent(),
        no_auth=no_auth,
    )
    return app, upstream


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_image_handoff_real_sdk_json_output_and_buffered_sse(stream, monkeypatch):
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    messages, colors = image_challenge()
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["model"] == "physical-vision":
            assert has_images(body["messages"])
            # The pixel challenge has only one original message plus the stage instruction.
            return httpx.Response(
                200, json=completion(",".join(decode_colors(body["messages"])))
            )
        assert not has_images(body["messages"])
        assert "data:image" not in json.dumps(body)
        packet = json.loads(body["messages"][-1]["content"])
        assert packet["prism_visual_observations"] == ",".join(colors)
        assert body["response_format"]["type"] == "json_object"
        return httpx.Response(200, json=completion(json.dumps({"colors": colors})))

    app, upstream = setup(handler)
    async with (
        upstream,
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        assert (await client.get("/v1/models")).status_code == 200
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mix",
                "messages": messages,
                "stream": stream,
                "response_format": {"type": "json_object"},
                "max_tokens": 128,
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-prism-policy"] == "vision_synthesis"
        assert response.headers["x-prism-coverage"] == "visual_observations"
        if stream:
            assert response.headers["x-prism-stream-mode"] == "buffered"
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            content = "".join(
                c.get("delta", {}).get("content", "")
                for e in events
                for c in e.get("choices", [])
            )
            assert response.text.rstrip().endswith("data: [DONE]")
        else:
            content = response.json()["choices"][0]["message"]["content"]
        assert json.loads(content) == {"colors": colors}
        client.headers["authorization"] = "Bearer ignored-in-no-auth-mode"
        trace = (
            await client.get("/v1/prism/traces/" + response.headers["x-request-id"])
        ).json()
        assert len(trace["execution_usage"]["calls"]) == len(calls) == 2
        assert "data:image" not in json.dumps(trace)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["truncated", "oversized", "tools", "empty"])
async def test_invalid_vision_observations_abort_before_synthesis(failure):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        data = completion(
            "x" * 1000
            if failure == "oversized"
            else ""
            if failure == "empty"
            else "red,green,blue,yellow",
            "length" if failure == "truncated" else "stop",
        )
        if failure == "tools":
            data["choices"][0]["finish_reason"] = "tool_calls"
            data["choices"][0]["message"]["tool_calls"] = [
                {
                    "id": "tool",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ]
        return httpx.Response(200, json=data)

    app, upstream = setup(handler, intermediate_max_bytes=512)
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "mix", "messages": image_challenge()[0]},
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_intermediate_output"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_vision_budgets_and_direct_only_features_reject_before_dispatch():
    calls = []
    app, upstream = setup(lambda request: calls.append(request))
    engine = app.state.engine
    body = {"model": "mix", "messages": image_challenge()[0]}
    for field in (
        {"seed": 7},
        {"tools": [{"type": "function", "function": {"name": "f"}}]},
    ):
        with pytest.raises(PrismError, match="capability"):
            engine.prepare({**body, **field})
    engine.config.profiles["mix"].limits.max_calls = 1
    with pytest.raises(PrismError):
        engine.prepare(body)
    assert not calls
    await upstream.aclose()


@pytest.mark.asyncio
async def test_profile_capacity_grades_actual_handoff_not_individual_models():
    def handler(request):
        body = json.loads(request.content)
        if body["model"] == "physical-vision":
            return httpx.Response(
                200, json=completion(",".join(decode_colors(body["messages"])))
            )
        # An individually vision-capable worker cannot prove that synthesis works.
        return httpx.Response(200, json=completion("wrong answer"))

    app, upstream = setup(handler)
    async with upstream:
        result = await ProfileCapacityEvaluator(
            app.state.engine, CapacityConfig()
        ).evaluate("mix")
    assert result["scope"] == "end_to_end_profile"
    assert result["capabilities"]["image"]["supported"] is False
    assert result["capabilities"]["image"]["status"] == "failed"


@pytest.mark.asyncio
async def test_default_auth_startup_and_routes_still_require_key(monkeypatch):
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    app, upstream = setup(
        lambda request: httpx.Response(200, json=completion()), no_auth=False
    )
    async with upstream:
        with pytest.raises(ValueError, match="API key"):
            async with app.router.lifespan_context(app):
                pytest.fail("missing key must fail startup")
        monkeypatch.setenv("PRISM_API_KEY", "secret")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client:
            for path in ("/v1/models", "/capacity", "/v1/prism/traces/id"):
                assert (await client.get(path)).status_code == 401
            assert (
                await client.post("/v1/chat/completions", json={})
            ).status_code == 401


def test_packaged_openrouter_preset_and_explicit_serve_flag(tmp_path, monkeypatch):
    import uvicorn

    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    assert main(["init", "--preset", "openrouter", "--out-dir", str(tmp_path)]) == 0
    path = str(tmp_path / "profiles/prism-vision-llm.json")
    config, models = load_profile(path)
    assert all(
        m.name.startswith("openrouter/") and m.name.endswith(":free")
        for m in models.values()
    )
    assert all(
        m.input_cost_per_million == m.output_cost_per_million == 0
        for m in models.values()
    )
    assert config.profiles["prism-vision-llm"].worker == "nemotron-nano-reasoning"
    assert main(["start", "--profile", path]) == 2
    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs))
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-upstream-key")
    assert main(["start", "--profile", path, "--no-auth"]) == 0
    assert calls[0][0].state.no_auth is True
    root = Path(__file__).resolve().parents[2]
    resources = files("prism").joinpath("resources", "openrouter")
    bundled = json.loads(resources.joinpath("prism.json").read_text())
    for alias, profile in bundled["profiles"].items():
        name = "prism-openrouter" if alias == "prism" else alias
        standalone = json.loads((root / "profiles" / f"{name}.json").read_text())
        assert standalone["profile"] == profile
    assert json.loads(resources.joinpath("models.json").read_text()) == json.loads(
        (root / "models/openrouter-nemotron-mix.json").read_text()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("embedded_error", [False, True])
async def test_compressed_responses_and_http_200_provider_errors(
    provider, embedded_error
):
    model = RawModel(
        id="m",
        name="physical",
        provider=provider,
        base_url="http://fixture/v1",
        api_key="dummy",
    )
    data = (
        {"error": {"code": 429, "message": "private provider details"}}
        if embedded_error
        else completion()
    )

    def handler(request):
        return httpx.Response(
            200,
            content=gzip.compress(json.dumps(data).encode()),
            headers={"content-encoding": "gzip", "content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        backend = LiteLLMBackend({"m": model}, upstream)
        try:
            ledger = Ledger(Limits())
            reservation = await ledger.reserve("call", model, 1000, 128)
            call = backend.complete(
                model, [{"role": "user", "content": "OK"}], {}, 128, ledger, reservation
            )
            if embedded_error:
                with pytest.raises(PrismError) as error:
                    await call
                assert error.value.code == "upstream_rate_limit"
                assert "private" not in str(error.value)
            else:
                assert (await call)["choices"][0]["message"]["content"] == "OK"
        finally:
            await backend.close()
