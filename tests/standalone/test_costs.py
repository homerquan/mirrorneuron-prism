import io
import json

import httpx
import pytest
from rich.console import Console

from prism.api import create_app
from prism.backends import LiteLLMBackend
from prism.cli_ui import Output
from prism.config import Limits, PrismConfig, RawModel
from prism.costs import CostTracker, usage_cost
from prism.runtime import Ledger

from .test_proxy import completion


def priced_model(id="small", input_price="$1/m", output_price="$5/m"):
    return RawModel(
        id=id,
        name=f"physical-{id}",
        base_url="http://fixture/v1",
        input_cost_per_million=input_price,
        output_cost_per_million=output_price,
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("$5/m", 5),
        ("$0.25/1M", 0.25),
        ("5 / million", 5),
        (0, 0),
        (2.5, 2.5),
        (None, None),
    ],
)
def test_price_shorthand(value, expected):
    assert priced_model(output_price=value).output_cost_per_million == expected


@pytest.mark.parametrize(
    "value", ["$5/k", "$-1/m", "free", "$NaN/m", True, -1, float("inf")]
)
def test_invalid_prices_rejected(value):
    with pytest.raises(ValueError):
        priced_model(output_price=value)


def test_reported_cost_includes_both_input_and_output():
    model = priced_model()
    assert (
        usage_cost(model, {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
        == 6
    )
    assert usage_cost(model, {"prompt_tokens": 200_000, "completion_tokens": 0}) == 0.2
    assert usage_cost(model, {"prompt_tokens": 0, "completion_tokens": 200_000}) == 1
    assert usage_cost(model, {"prompt_tokens": True, "completion_tokens": 2}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_api_costs_accumulate_provider_usage_and_require_auth(
    stream, monkeypatch
):
    monkeypatch.setenv("PRISM_API_KEY", "cost-client-secret")
    model = priced_model()
    config = PrismConfig(profiles={"test": {"direct": "small", "strategy": "direct"}})

    def handler(request):
        if not stream:
            return httpx.Response(200, json=completion())
        events = [
            {
                "id": "upstream-id",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "upstream-id",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": "upstream-id",
                "choices": [],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 10,
                    "total_tokens": 50,
                },
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(e) + "\n\n" for e in events)
            + "data: [DONE]\n\n",
        )

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reports = []
    app = create_app(
        config,
        models={"small": model},
        backend=LiteLLMBackend({"small": model}, upstream),
        cost_reporter=reports.append,
    )
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        assert (await client.get("/v1/prism/costs")).status_code == 401
        headers = {"Authorization": "Bearer cost-client-secret"}
        for _ in range(2):
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Reply hello"}],
                    "stream": stream,
                },
            )
            assert response.status_code == 200, response.text
        stats = (await client.get("/v1/prism/costs", headers=headers)).json()
    assert stats["physical_calls"] == stats["requests"] == 2
    assert stats["reported_input_tokens"] == 80
    assert stats["reported_output_tokens"] == 20
    assert stats["pricing_mode"] == "configured"
    assert stats["models"][0]["reported_input_tokens"] == 80
    assert stats["models"][0]["reported_output_tokens"] == 20
    assert stats["models"][0]["physical_calls"] == 2
    assert stats["last_request"]["strategy"] == "direct"
    assert stats["total_cost_usd"] == pytest.approx(2 * (40 + 10 * 5) / 1_000_000)
    assert stats["estimated_saved_usd"] == 0
    assert stats["estimated_saved_percent"] == 0
    assert len(reports) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cheap_input, expected_sign", [(1, 1), (500, -1)])
async def test_mixing_savings_sum_all_stages_and_can_be_negative(
    cheap_input, expected_sign, monkeypatch
):
    worker = priced_model("worker", cheap_input, 5)
    final = priced_model("final", 10, 50)
    config = PrismConfig(
        profiles={
            "mix": {
                "direct": "final",
                "worker": "worker",
                "synthesizer": "final",
                "strategy": "text_synthesis",
                "allowed_policies": ["direct", "text_synthesis"],
                "public_max_output_tokens": 512,
                "worker_output_tokens": 512,
            }
        }
    )
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=completion("bounded worker notes"))
        )
    )
    models = {"worker": worker, "final": final}
    app = create_app(
        config, models=models, backend=LiteLLMBackend(models, upstream), no_auth=True
    )
    monkeypatch.setattr("prism.costs.estimate_prompt_tokens", lambda body: 1000)
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mix",
                "messages": [
                    {
                        "role": "user",
                        "content": 'Summarize <prism-source id="policy">Policy facts</prism-source>',
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        stats = (await client.get("/v1/prism/costs")).json()
    actual = (40 * cheap_input + 10 * 5 + 40 * 10 + 10 * 50) / 1_000_000
    baseline = (1000 * 10 + 10 * 50) / 1_000_000
    assert stats["total_cost_usd"] == pytest.approx(actual)
    assert stats["estimated_saved_usd"] == pytest.approx(baseline - actual)
    assert stats["estimated_saved_percent"] == pytest.approx(
        (baseline - actual) / baseline * 100
    )
    assert stats["estimated_saved_usd"] * expected_sign > 0


@pytest.mark.asyncio
async def test_missing_usage_prices_and_started_failures_are_never_free():
    priced = priced_model()
    unknown = priced_model("unknown", None, None)
    tracker = CostTracker({"small": priced, "unknown": unknown})
    ledger = Ledger(Limits())
    for model, usage in [
        (priced, None),
        (unknown, {"prompt_tokens": 40, "completion_tokens": 10}),
    ]:
        reservation = await ledger.reserve(model.id, model, 100, 50)
        await ledger.start(reservation)
        tracker.record_call(model, usage, reservation)
    stats = tracker.snapshot()
    assert stats["total_cost_usd"] is None
    assert stats["unpriced_calls"] == stats["unreported_usage_calls"] == 1
    assert stats["unknown_usage_upper_estimate_usd"] == pytest.approx(
        (100 + 50 * 5) / 1_000_000
    )
    assert stats["estimated_saved_percent"] is None
    assert CostTracker({}).snapshot()["physical_calls"] == 0


def test_cost_output_is_readable_and_negative_savings_are_visible():
    stream = io.StringIO()
    console = Console(file=stream, color_system=None)
    output = Output("table", errors=console)
    tracker = CostTracker({})
    tracker.compared_requests = 1
    tracker.compared_cost = 0.02
    tracker.baseline_cost = 0.01
    output.costs(tracker.snapshot())
    assert "$-0.010000 (-100.0%)" in stream.getvalue()
    assert "Token cost since start" in stream.getvalue()


@pytest.mark.asyncio
async def test_failed_upstream_calls_and_capacity_probes_are_in_lifetime_spend():
    model = priced_model()
    config = PrismConfig(profiles={"test": {"direct": "small", "strategy": "direct"}})
    failing = False

    def handler(request):
        return (
            httpx.Response(500) if failing else httpx.Response(200, json=completion())
        )

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        config,
        models={"small": model},
        backend=LiteLLMBackend({"small": model}, upstream),
        no_auth=True,
    )
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        assert (await client.get("/capacity?model=small")).status_code == 200
        before = (await client.get("/v1/prism/costs")).json()
        assert before["physical_calls"] == 4
        assert before["total_cost_usd"] == pytest.approx(4 * (40 + 10 * 5) / 1_000_000)
        failing = True
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 502
        stats = (await client.get("/v1/prism/costs")).json()
    assert stats["physical_calls"] == 5
    assert stats["unreported_usage_calls"] == 1
    assert stats["total_cost_usd"] is None
    assert stats["known_cost_usd"] == before["total_cost_usd"]
    assert stats["excluded_requests"] == 1
    assert stats["estimated_saved_usd"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude", ["images", "missing-tokenizer", "baseline-context"])
async def test_uncomparable_mixed_requests_keep_spend_but_do_not_claim_savings(
    exclude, monkeypatch
):
    worker = priced_model("worker")
    final = priced_model("final", 10, 50)
    baseline = priced_model("baseline", 10, 50)
    if exclude == "baseline-context":
        baseline = baseline.model_copy(
            update={"context_window": 1024, "max_output_tokens": 512}
        )
    config = PrismConfig(
        profiles={
            "mix": {
                "direct": "final",
                "worker": "worker",
                "synthesizer": "final",
                "cost_baseline_model": "baseline",
                "strategy": "vision_synthesis"
                if exclude == "images"
                else "text_synthesis",
                "allowed_policies": ["direct", "vision_synthesis", "text_synthesis"],
                "public_max_output_tokens": 256,
                "worker_output_tokens": 256,
            }
        }
    )
    models = {m.id: m for m in [worker, final, baseline]}
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=completion("bounded notes"))
        )
    )
    app = create_app(
        config, models=models, backend=LiteLLMBackend(models, upstream), no_auth=True
    )
    if exclude == "missing-tokenizer":
        monkeypatch.setattr("prism.costs.estimate_prompt_tokens", lambda body: None)
    if exclude == "images":
        from prism.capacity import image_challenge

        messages = image_challenge()[0]
    else:
        messages = [
            {
                "role": "user",
                "content": 'Summarize <prism-source id="data">'
                + "policy details " * 200
                + "</prism-source>",
            }
        ]
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions", json={"model": "mix", "messages": messages}
        )
        assert response.status_code == 200, response.text
        stats = (await client.get("/v1/prism/costs")).json()
    assert stats["total_cost_usd"] > 0
    assert stats["compared_requests"] == 0
    assert stats["excluded_requests"] == 1
    assert stats["estimated_saved_usd"] is None
