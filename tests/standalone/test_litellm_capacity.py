import asyncio
import base64
import json
import re
import struct
import zlib

import httpx
import pytest

from prism.api import create_app
from prism.backends import LiteLLMBackend
from prism.capacity import CapacityEvaluator, image_challenge
from prism.config import CapacityConfig, Limits, PrismConfig, RawModel, load_config
from prism.contracts import check_context, validate_request
from prism.errors import PrismError
from prism.runtime import Ledger


def completion(text="OK", finish="stop", **extra):
    return {
        "id": "upstream",
        "object": "chat.completion",
        "created": 1,
        "model": "physical",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
        **extra,
    }


def opencode(provider="vendor", npm="@ai-sdk/openai-compatible", **definition):
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider: {
                "npm": npm,
                "options": {"baseURL": "http://backend/v1", "apiKeyEnv": "VENDOR_KEY"},
                "models": {
                    "alias": {
                        "name": "Display name",
                        "model": "vendor/physical",
                        "tools": False,
                        "extra_body": {
                            "max_tokens": 256,
                            "temperature": 0,
                            "top_k": 64,
                        },
                        **definition,
                    }
                },
            }
        },
    }


def test_directory_registry_auto_aliases_and_parameters(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps(opencode("nvidia")))
    (folder / "b.json").write_text(
        json.dumps(
            opencode(
                "docker",
                timeout_seconds=600,
                rate_limit_rpm=20,
                prism={"context_window": 8192},
            )
        )
    )
    (tmp_path / "prism.json").write_text(json.dumps({"models_file": "models"}))
    config, models = load_config(tmp_path / "prism.json")
    assert set(models) == {"nvidia/alias", "docker/alias"}
    assert set(config.profiles) == set(models)
    model = models["docker/alias"]
    assert model.name == "vendor/physical" and model.provider == "openai"
    assert model.parameters == {"max_tokens": 256, "temperature": 0, "top_k": 64}
    assert model.context_window == 8192 and model.max_output_tokens == 256
    assert model.timeout_seconds == 600 and model.rate_limit_rpm == 20
    assert "tools" not in model.capabilities
    assert config.profiles[model.id].strategy == "direct"
    assert config.profiles[model.id].public_max_output_tokens == 256
    (folder / "duplicate.json").write_text((folder / "a.json").read_text())
    with pytest.raises(ValueError, match="unique"):
        load_config(tmp_path / "prism.json")


@pytest.mark.parametrize(
    ("npm", "provider"),
    [
        ("@ai-sdk/anthropic", "anthropic"),
        ("@ai-sdk/google", "gemini"),
        ("@ai-sdk/azure", "azure"),
        ("@ai-sdk/amazon-bedrock", "bedrock"),
        ("@ai-sdk/google-vertex", "vertex_ai"),
    ],
)
def test_native_opencode_transport(tmp_path, npm, provider):
    (tmp_path / "models.json").write_text(json.dumps(opencode(npm=npm)))
    (tmp_path / "prism.json").write_text("{}")
    _, models = load_config(tmp_path / "prism.json")
    assert models["vendor/alias"].provider == provider


@pytest.mark.parametrize(
    "parameters",
    [
        {"mock_response": "fake"},
        {"num_retries": 3},
        {"api_key": "secret"},
        {"extra_body": {"model": "other"}},
        {"extra_body": {"max_tokens": 99999}},
        {"max_tokens": 32, "max_completion_tokens": 64},
        {"max_tokens": False},
    ],
)
def test_model_defaults_cannot_override_execution(parameters):
    with pytest.raises(ValueError):
        RawModel(id="m", name="m", base_url="http://backend/v1", parameters=parameters)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "name"),
    [
        ("openai", "gpt-6-astra"),
        ("openai", "gpt-6-luna"),
        ("anthropic", "claude-opus-5-5"),
        ("anthropic", "claude-haiku-4-5"),
        ("gemini", "gemini-3.5-flash-lite"),
        ("gemini", "gemini-3.8-flash"),
        ("gemini", "gemini-3.1-pro-preview"),
        ("anthropic", "claude-sonnet-4-5"),
        ("gemini", "gemini-2.5-flash"),
        ("azure", "deployment"),
        ("openrouter", "openai/gpt-4o"),
        ("deepseek", "deepseek-chat"),
        ("groq", "llama-3.3-70b-versatile"),
    ],
)
async def test_real_sdk_native_vendor_protocols(provider, name):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request, body))
        if provider == "anthropic":
            assert request.url.path.endswith("/messages")
            assert body["max_tokens"] == 128
            data = {
                "id": "a",
                "type": "message",
                "role": "assistant",
                "model": name,
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 40, "output_tokens": 10},
            }
        elif provider == "gemini":
            assert request.url.path.endswith(":generateContent")
            assert (
                body["generationConfig"].get(
                    "maxOutputTokens", body["generationConfig"].get("max_output_tokens")
                )
                == 128
            )
            data = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "OK"}]},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 40,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 50,
                },
            }
        else:
            assert request.url.path.endswith("/chat/completions")
            if provider == "azure":
                assert "/deployments/deployment/" in request.url.path
                assert request.url.params["api-version"] == "2024-10-21"
            data = completion(service_tier="on_demand")
        return httpx.Response(200, json=data)

    model = RawModel(
        id="m",
        name=name,
        provider=provider,
        base_url="http://vendor/v1",
        api_key="not-real",
        api_version="2024-10-21" if provider == "azure" else None,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = LiteLLMBackend({"m": model}, client)
        try:
            ledger = Ledger(Limits())
            reservation = await ledger.reserve("call", model, 1000, 128)
            result = await backend.complete(
                model,
                [{"role": "user", "content": "Reply OK"}],
                {},
                128,
                ledger,
                reservation,
            )
            assert result["choices"][0]["message"]["content"] == "OK"
            assert ledger.usage[0]["provider_usage"]["completion_tokens"] == 10
            assert len(calls) == 1
        finally:
            await backend.close()


@pytest.mark.asyncio
async def test_native_prefix_sdk_dispatch_and_no_implicit_retries(monkeypatch):
    import litellm

    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        return completion()

    monkeypatch.setattr(litellm, "acompletion", fake)
    model = RawModel(id="native", name="anthropic/claude-sonnet-4-5", api_key="hidden")
    backend = LiteLLMBackend({"native": model})
    try:
        ledger = Ledger(Limits())
        reservation = await ledger.reserve("x", model, 1000, 128)
        await backend.complete(
            model, [{"role": "user", "content": "hi"}], {}, 128, ledger, reservation
        )
        call = calls[0]
        assert call["custom_llm_provider"] == "anthropic"
        assert call["model"] == "claude-sonnet-4-5"
        assert (
            call["drop_params"] is False
            and call["num_retries"] == call["max_retries"] == 0
        )
        assert call["no-log"] is True and call["caching"] is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_compatible_defaults_unknown_wire_fields_and_limit_precedence():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion())

    model = RawModel(
        id="m",
        name="openai/physical",
        base_url="http://backend/v1",
        parameters={
            "temperature": 0.7,
            "max_tokens": 256,
            "reasoning": {"enabled": True},
            "extra_body": {"custom_vendor_mode": "fast"},
        },
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = LiteLLMBackend({"m": model}, client)
        try:
            ledger = Ledger(Limits())
            reservation = await ledger.reserve("x", model, 1000, 64)
            await backend.complete(
                model,
                [{"role": "developer", "content": "original"}],
                {"max_completion_tokens": 64, "temperature": 0},
                64,
                ledger,
                reservation,
            )
            body = calls[0]
            assert body["model"] == "openai/physical"
            assert body["messages"][0]["role"] == "developer"
            assert body["custom_vendor_mode"] == "fast"
            assert body["reasoning"] == {"enabled": True}
            assert body["temperature"] == 0 and body["max_completion_tokens"] == 64
            assert "max_tokens" not in body
        finally:
            await backend.close()


def decode_colors(messages):
    url = messages[0]["content"][1]["image_url"]["url"]
    png = base64.b64decode(url.split(",")[1])
    pos, blocks = 8, []
    while pos < len(png):
        size = struct.unpack(">I", png[pos : pos + 4])[0]
        kind = png[pos + 4 : pos + 8]
        if kind == b"IDAT":
            blocks.append(png[pos + 8 : pos + 8 + size])
        pos += size + 12
    raw = zlib.decompress(b"".join(blocks))
    palette = {
        (230, 0, 0): "red",
        (0, 160, 0): "green",
        (0, 0, 230): "blue",
        (255, 230, 0): "yellow",
        (140, 0, 180): "purple",
        (255, 130, 0): "orange",
        (0, 0, 0): "black",
        (255, 255, 255): "white",
    }
    result = []
    for x, y in [(32, 32), (96, 32), (32, 96), (96, 96)]:
        offset = y * (128 * 3 + 1) + 1 + x * 3
        result.append(palette[tuple(raw[offset : offset + 3])])
    return result


class ProbeBackend:
    def __init__(self, fail=None, error=None, finish="stop"):
        self.calls = []
        self.fail, self.error, self.finish = fail, error, finish

    async def close(self):
        pass

    async def complete(self, model, messages, parameters, output, ledger, reservation):
        await ledger.start(reservation)
        await asyncio.sleep(0.001)
        self.calls.append((model.id, messages, parameters, output))
        if self.error:
            await ledger.finish(reservation, status="failed")
            raise self.error
        fmt = parameters.get("response_format", {}).get("type")
        text = messages[0]["content"]
        if fmt == "json_schema":
            nonce = parameters["response_format"]["json_schema"]["schema"][
                "properties"
            ]["tag"]["enum"][0]
            content = json.dumps(
                {"tag": nonce, "count": "7" if self.fail == "schema" else 7}
            )
        elif fmt == "json_object":
            nonce = re.search(r'tag="([a-f0-9]+)"', text)[1]
            content = json.dumps({"tag": nonce, "count": 7})
        elif isinstance(text, list):
            content = (
                "wrong" if self.fail == "image" else ",".join(decode_colors(messages))
            )
        else:
            a, b, c, d, e = map(int, re.findall(r"\d+", text)[1:6])
            successors = dict(re.findall(r"([A-D]) must run before ([A-D])", text))
            first = next(iter(set(successors) - set(successors.values())))
            sequence = [first]
            while sequence[-1] in successors:
                sequence.append(successors[sequence[-1]])
            order = "".join(sequence)
            content = f"inventory={a + b * c - d + e};order={order[::-1] if self.fail == 'reasoning' else order}"
        data = completion(content, self.finish)
        await ledger.finish(reservation, data["usage"])
        return data


def evaluator(backend, **settings):
    model = RawModel(
        id="m", name="m", base_url="http://backend/v1", capabilities={"text"}
    )
    return CapacityEvaluator({"m": model}, backend, CapacityConfig(**settings))


@pytest.mark.asyncio
async def test_measured_capacity_ignores_declarations_and_scores_pixels_and_reasoning():
    backend = ProbeBackend()
    probe = evaluator(backend)
    result = (await probe.query(["m"]))[0]
    assert len(backend.calls) == 4
    assert all(v["supported"] is True for v in result["capabilities"].values())
    assert result["capabilities"]["reasoning"]["passed"] == 2
    assert result["capabilities"]["reasoning"]["total"] == 2
    assert result["capabilities"]["reasoning"]["reasoning_metadata_observed"] is False
    assert "messages" not in json.dumps(result)
    for _ in range(4):
        messages, colors = image_challenge()
        assert decode_colors(messages) == colors
        assert ",".join(colors) not in messages[0]["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fail", "feature", "passed"),
    [
        ("schema", "json_schema", 0),
        ("image", "image", 0),
        ("reasoning", "reasoning", 1),
    ],
)
async def test_probe_rejects_plausible_but_incorrect_answers(fail, feature, passed):
    result = (await evaluator(ProbeBackend(fail=fail)).query(["m"]))[0]
    observation = result["capabilities"][feature]
    assert observation["supported"] is False and observation["passed"] == passed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        PrismError("sanitized", "upstream_authentication", 502),
        PrismError("sanitized", "deadline_exceeded", 504),
        PrismError("sanitized", "upstream_rate_limit", 429),
    ],
)
async def test_unavailable_is_unknown_and_secret_free(error):
    result = (await evaluator(ProbeBackend(error=error)).query(["m"]))[0]
    assert all(
        v["supported"] is None and v["error_code"] == error.code
        for v in result["capabilities"].values()
    )
    assert "sanitized" not in json.dumps(result)


@pytest.mark.asyncio
async def test_truncated_probe_is_inconclusive():
    result = (await evaluator(ProbeBackend(finish="length")).query(["m"]))[0]
    assert all(
        v["supported"] is None and v["status"] == "inconclusive"
        for v in result["capabilities"].values()
    )


@pytest.mark.asyncio
async def test_cache_expiry_refresh_configuration_credentials_and_concurrent_dedup(
    monkeypatch,
):
    backend = ProbeBackend()
    probe = evaluator(backend)
    first, second = await asyncio.gather(
        probe.evaluate("m", True), probe.evaluate("m", True)
    )
    assert (
        len(backend.calls) == 4
        and first["cached"] is False
        and second["cached"] is True
    )
    assert (await probe.evaluate("m"))["cached"] is True
    await probe.evaluate("m", True)
    assert len(backend.calls) == 8
    probe.models["m"].parameters["temperature"] = 0.2
    await probe.evaluate("m")
    assert len(backend.calls) == 12
    probe.models["m"].api_key_env = "ROTATING_KEY"
    monkeypatch.setenv("ROTATING_KEY", "first")
    await probe.evaluate("m")
    monkeypatch.setenv("ROTATING_KEY", "second")
    result = await probe.evaluate("m")
    assert len(backend.calls) == 20 and "second" not in json.dumps(result)
    probe.cache["m"] = (probe.cache["m"][0], 0, probe.cache["m"][2])
    await probe.evaluate("m")
    assert len(backend.calls) == 24


@pytest.mark.asyncio
async def test_capacity_api_auth_unknown_profile_query_and_post_refresh(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "test-secret")
    backend = ProbeBackend()
    probe = evaluator(backend)
    config = PrismConfig(profiles={"public": {"direct": "m", "strategy": "direct"}})
    app = create_app(config, models=probe.models, backend=backend)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://prism"
    ) as client:
        assert (await client.get("/capacity")).status_code == 401
        client.headers["authorization"] = "Bearer test-secret"
        assert (
            await client.get("/capacity", params={"model": "no-such-model"})
        ).status_code == 404
        assert not backend.calls
        response = await client.get("/capacity", params={"model": "public"})
        assert (
            response.status_code == 200
            and response.headers["cache-control"] == "no-store"
        )
        assert response.json()["profile"]["supported"] == {
            "json_object": False,
            "json_schema": False,
            "image": False,
            "reasoning": True,
        }
        assert response.json()["profile"]["aggregation"] == "end_to_end_profile"
        assert response.json()["data"][0]["capabilities"]["image"]["supported"] is True
        assert (await client.get("/capacity", params={"model": "m"})).json()["data"][0][
            "cached"
        ] is True
        assert (
            await client.post("/capacity", params={"model": "m", "refresh": True})
        ).json()["data"][0]["cached"] is False
        assert len(backend.calls) == 9
        assert (
            await client.get("/capacity", params={"refresh": "invalid"})
        ).status_code == 422


@pytest.mark.asyncio
async def test_image_direct_uses_sdk_and_cannot_be_partitioned(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "test-secret")
    messages, _ = image_challenge()
    body = validate_request({"model": "public", "messages": messages})
    model = RawModel(id="m", name="physical", base_url="http://backend/v1")
    assert check_context(messages, {}, 256, model) > model.image_token_reserve
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion("four colors"))

    config = PrismConfig(profiles={"public": {"direct": "m"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        backend = LiteLLMBackend({"m": model}, upstream)
        app = create_app(config, models={"m": model}, backend=backend)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://prism",
                headers={"authorization": "Bearer test-secret"},
            ) as client:
                response = await client.post("/v1/chat/completions", json=body)
                assert response.status_code == 200
                assert response.headers["x-prism-policy"] == "direct"
                assert calls[0]["messages"] == messages
        finally:
            await backend.close()
    config.profiles["public"].allowed_policies = ["evidence_map"]
    app = create_app(config, models={"m": model}, backend=ProbeBackend())
    with pytest.raises(PrismError):
        app.state.engine.prepare(body)


@pytest.mark.parametrize(
    ("content", "expected_score"),
    [
        ("inventory=84;order=CBDA", 2),
        ("Here is the final answer: inventory = 84; order = C, B, D, A.", 2),
        ("inventory=84;order=CBD A", 2),
        ("inventory=26;order=CBDA", 1),
        ("inventory=84;order=ABCD", 1),
        ("no final answers", 0),
    ],
)
def test_reasoning_grade_accepts_harmless_formatting_and_scores_each_task(
    content, expected_score
):
    from prism.capacity import grade_reasoning

    assert grade_reasoning(content, [84, "CBDA"]) == (expected_score, 2)


@pytest.mark.asyncio
async def test_sdk_missing_usage_remains_unknown_and_complete_cancellation_closes_body():
    entered, closed = asyncio.Event(), asyncio.Event()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            yield b"{"
            await asyncio.sleep(10)

        async def aclose(self):
            await asyncio.sleep(0.001)
            closed.set()

    model = RawModel(id="m", name="physical", base_url="http://backend/v1")
    missing = completion()
    del missing["usage"]
    mode = "missing"

    def handler(request):
        return (
            httpx.Response(200, json=missing)
            if mode == "missing"
            else httpx.Response(200, stream=SlowBody())
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        backend = LiteLLMBackend({"m": model}, http)
        try:
            ledger = Ledger(Limits())
            reservation = await ledger.reserve("no-usage", model, 1000, 128)
            result = await backend.complete(
                model, [{"role": "user", "content": "hi"}], {}, 128, ledger, reservation
            )
            assert result["usage"] is None
            assert ledger.usage[0]["provider_usage"] is None
            mode = "slow"
            reservation = await ledger.reserve("cancel", model, 1000, 128)
            task = asyncio.create_task(
                backend.complete(
                    model,
                    [{"role": "user", "content": "hi"}],
                    {},
                    128,
                    ledger,
                    reservation,
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
            assert not ledger.snapshot()["outstanding_nodes"]
        finally:
            await backend.close()


@pytest.mark.asyncio
async def test_capacity_json_post_is_bounded_and_validated(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "test-secret")
    backend = ProbeBackend()
    probe = evaluator(backend)
    app = create_app(
        PrismConfig(profiles={"public": {"direct": "m"}}),
        models=probe.models,
        backend=backend,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://prism",
        headers={"authorization": "Bearer test-secret"},
    ) as client:
        response = await client.post("/capacity", json={"model": "m", "refresh": True})
        assert response.status_code == 200 and len(backend.calls) == 4
        for body in [
            {"refresh": "true"},
            {"model": []},
            {"api_base": "http://other"},
            [],
        ]:
            assert (await client.post("/capacity", json=body)).status_code == 400
        assert (await client.post("/capacity", content="[")).json()["error"][
            "code"
        ] == "invalid_json"
        assert (
            await client.post("/capacity", params={"model": "m"}, json={"model": "m"})
        ).status_code == 400
        assert (await client.post("/capacity", content="x" * 1025)).status_code == 413
        assert len(backend.calls) == 4


@pytest.mark.parametrize("limit", [256, 16384])
def test_auto_alias_honors_configured_output_default(tmp_path, limit):
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "m",
                        "name": "physical",
                        "base_url": "http://backend/v1",
                        "max_output_tokens": limit,
                        "parameters": {"max_tokens": limit},
                    }
                ]
            }
        )
    )
    (tmp_path / "prism.json").write_text("{}")
    config, _ = load_config(tmp_path / "prism.json")
    assert config.profiles["m"].public_max_output_tokens == limit


def test_equivalent_transports_share_the_physical_engine_key():
    legacy = RawModel(id="legacy", name="physical", base_url="http://backend/v1")
    explicit = legacy.model_copy(update={"id": "explicit", "provider": "openai"})
    assert LiteLLMBackend.engine_key(legacy) == LiteLLMBackend.engine_key(explicit)
    native = RawModel(
        id="native",
        name="anthropic/claude-sonnet-4-5",
        provider="anthropic",
        base_url="https://vendor",
    )
    unprefixed = native.model_copy(update={"name": "claude-sonnet-4-5"})
    assert LiteLLMBackend.engine_key(native) == LiteLLMBackend.engine_key(unprefixed)


@pytest.mark.asyncio
async def test_model_timeout_is_enforced_even_when_ledger_allows_longer():
    async def slow(request):
        await asyncio.sleep(10)
        return httpx.Response(200, json=completion())

    model = RawModel(
        id="m", name="physical", base_url="http://backend/v1", timeout_seconds=0.02
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as http:
        backend = LiteLLMBackend({"m": model}, http)
        try:
            ledger = Ledger(Limits(deadline_seconds=60))
            reservation = await ledger.reserve("timeout", model, 1000, 128)
            with pytest.raises(PrismError) as error:
                await asyncio.wait_for(
                    backend.complete(
                        model,
                        [{"role": "user", "content": "hi"}],
                        {},
                        128,
                        ledger,
                        reservation,
                    ),
                    1,
                )
            assert error.value.code == "deadline_exceeded"
            assert not ledger.snapshot()["outstanding_nodes"]
        finally:
            await backend.close()
