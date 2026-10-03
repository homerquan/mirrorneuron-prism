"""Cost/power routing, per-call controls, and bounded draft/review execution."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from prism.api import create_app
from prism.backends import OpenAIBackend
from prism.cli import main
from prism.config import OptimizationConfig, PrismConfig, RawModel, load_config
from prism.contracts import check_context, context_controls
from prism.errors import PrismError
from prism.planning import review_messages, synthesis_messages

from .test_http import serve
from .test_policies import Agent
from .test_proxy import completion


class PhysicalBackend:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def respond(self, request):
        body = json.loads(request.content)
        self.calls.append(body)
        if body["stream"]:
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "final answer"},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 10},
                },
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n",
            )
        last = body["messages"][-1]["content"]
        if last.startswith("Prepare a draft"):
            if self.failure == "draft_length":
                return httpx.Response(200, json=completion("draft", finish="length"))
            if self.failure == "draft_bytes":
                return httpx.Response(200, json=completion("é" * 512))
            if self.failure == "draft_escaping":
                return httpx.Response(200, json=completion("\x00" * 90))
            if self.failure == "draft_refusal":
                return httpx.Response(200, json=completion(None, refusal="refused"))
            if self.failure == "draft_tools":
                return httpx.Response(
                    200,
                    json=completion(
                        None,
                        finish="tool_calls",
                        tool_calls=[
                            {
                                "id": "internal",
                                "type": "function",
                                "function": {"name": "forbidden", "arguments": "{}"},
                            }
                        ],
                    ),
                )
            result = completion("draft answer")
        elif last.startswith("{"):
            packet = json.loads(last)
            if "prism_review" in packet:
                result = completion(
                    '{"answer":"final answer"}'
                    if body.get("response_format")
                    else "final answer"
                )
            elif "prism_compaction" in packet:
                result = completion(
                    json.dumps(
                        {
                            "summary": "Source facts and their qualifications.",
                            "unresolved": [],
                            "record_ids": [],
                        }
                    )
                )
            elif "prism_draft" in packet:
                content = '{"issues":[],"suggestions":["clarify"]}'
                if self.failure == "review_json":
                    content = "not JSON"
                elif self.failure == "review_fields":
                    content = '{"issues":"bad","suggestions":[]}'
                elif self.failure == "review_extra":
                    content = '{"issues":[],"suggestions":[],"extra":true}'
                elif self.failure == "review_bytes":
                    content = json.dumps({"issues": ["x" * 512], "suggestions": []})
                if self.failure == "review_http":
                    return httpx.Response(503)
                result = completion(
                    content,
                    finish="length" if self.failure == "review_length" else "stop",
                )
            elif "records" in packet:
                result = completion(
                    json.dumps(
                        {
                            "supported": True,
                            "issues": [],
                            "checked_record_ids": [
                                record["record_id"] for record in packet["records"]
                            ],
                        }
                    )
                )
            elif "partition_id" in packet or "partitions" in packet:

                def artifact(part):
                    return {
                        "status": "complete",
                        "records": [{"quote": part["source"], "fact": "source fact"}],
                        "needs": [],
                    }

                result = completion(
                    json.dumps(
                        {
                            "partitions": [
                                {"partition_id": part["partition_id"], **artifact(part)}
                                for part in packet["partitions"]
                            ]
                        }
                        if "partitions" in packet
                        else artifact(packet)
                    )
                )
            else:
                result = completion(
                    '{"answer":"final answer"}'
                    if body.get("response_format")
                    else "final answer"
                )
        else:
            result = completion("final answer")
        return httpx.Response(200, json=result)


def setup(
    monkeypatch, *, agent=None, failure=None, profiles=None, models=None, **profile
):
    monkeypatch.setenv("PRISM_API_KEY", "optimization-test")
    if models is None:
        models = {
            ref: RawModel(
                id=ref,
                name="physical-" + ref,
                base_url="http://fixture/v1",
                power_rating=rating,
                input_cost_per_million=price,
                output_cost_per_million=price,
                safety_margin=64,
            )
            for ref, rating, price in (
                ("cheap", 3, 0.1),
                ("mid", 6, 0.35),
                ("strong", 9, 1),
            )
        }
    default = {
        "direct": "cheap",
        "worker": "cheap",
        "synthesizer": "strong",
        "public_max_output_tokens": 512,
        "worker_output_tokens": 128,
        "intermediate_max_bytes": 512,
        "partition_bytes": 1800,
        "allowed_policies": ["direct", "draft_review"],
        "optimization": {"model_ids": list(models), "enforce_stage_power_order": False},
        **profile,
    }
    config = PrismConfig(profiles=profiles or {"prism": default})
    physical = PhysicalBackend(failure)
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(physical.respond))
    app = create_app(
        config,
        models=models,
        backend=OpenAIBackend(models, upstream),
        decision_agent=agent or Agent("abstain"),
    )
    return app, upstream, physical, config, models


def body(priority=None, **fields):
    request = {
        "model": "prism",
        "messages": [{"role": "user", "content": "Explain a design tradeoff."}],
        **fields,
    }
    if priority is not None:
        request["context_management"] = [{"prism_cost_priority": priority}]
    return request


async def send(app, request):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://prism",
        headers={"authorization": "Bearer optimization-test"},
    ) as client:
        response = await client.post("/v1/chat/completions", json=request)
        trace = None
        if "x-request-id" in response.headers:
            trace = (
                await client.get("/v1/prism/traces/" + response.headers["x-request-id"])
            ).json()
        return response, trace


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [None],
        [{}],
        [{"extra": 1}],
        [{"prism_cost_priority": 0.2}, {"prism_cost_priority": 0.3}],
    ],
)
def test_malformed_controls(value):
    with pytest.raises(PrismError) as error:
        context_controls(value)
    assert error.value.param == "context_management"


@pytest.mark.parametrize(
    "key,values",
    [
        (
            "prism_cost_priority",
            [-0.1, 1.1, True, "0.8", None, float("nan"), float("inf"), 10**400],
        ),
        ("prism_max_cost_usd", [0, -1, True, "1", None, 10**400]),
        ("prism_max_calls", [0, -1, True, 1.5, "2", None]),
        ("prism_model_ids", [[], ["cheap", "cheap"], [1], [""], "cheap", None]),
        ("prism_allowed_policies", [[], ["direct", "direct"], ["unknown"], [1], None]),
    ],
)
def test_invalid_control_values(key, values):
    for value in values:
        with pytest.raises(PrismError):
            context_controls([{key: value}])


@pytest.mark.parametrize("rating", [0, 11, True, 2.5, "3"])
def test_invalid_model_rating(rating):
    with pytest.raises(ValidationError):
        RawModel(id="x", name="x", base_url="http://fixture/v1", power_rating=rating)


@pytest.mark.parametrize("priority", [-1, 2, True, "0.5", float("nan")])
def test_invalid_profile_priority(priority):
    with pytest.raises(ValidationError):
        OptimizationConfig(model_ids=["cheap"], default_cost_priority=priority)


@pytest.mark.parametrize(
    "missing",
    [
        "power_rating",
        "input_cost_per_million",
        "output_cost_per_million",
        "unknown_model",
        "duplicate_model",
    ],
)
@pytest.mark.asyncio
async def test_registry_validation_and_legacy_compatibility(
    monkeypatch, tmp_path, missing
):
    _, upstream, _, config, models = setup(monkeypatch)
    data = {"models": [model.model_dump(mode="json") for model in models.values()]}
    if missing == "unknown_model":
        config.profiles["prism"].optimization.model_ids.append("missing")
    elif missing == "duplicate_model":
        config.profiles["prism"].optimization.model_ids.append("cheap")
    else:
        data["models"][0][missing] = None
    (tmp_path / "models.json").write_text(json.dumps(data))
    (tmp_path / "prism.json").write_text(config.model_dump_json())
    with pytest.raises(ValueError):
        load_config(tmp_path / "prism.json")
    config.profiles["prism"].optimization = None
    (tmp_path / "prism.json").write_text(config.model_dump_json())
    assert (
        load_config(tmp_path / "prism.json")[0].profiles["prism"].optimization is None
    )
    await upstream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "priority,expected",
    [(0, "strong"), (0.2, "strong"), (0.5, "mid"), (0.8, "cheap"), (1, "cheap")],
)
async def test_direct_cost_power_endpoints_and_intermediate_weights(
    monkeypatch, priority, expected
):
    app, upstream, physical, config, _ = setup(monkeypatch, allowed_policies=["direct"])
    before = config.profiles["prism"].model_dump_json()
    async with upstream:
        response, trace = await send(app, body(priority))
    assert response.status_code == 200, response.text
    assert physical.calls[0]["model"] == "physical-" + expected
    assert trace["stage_models"] == {"direct": expected}
    assert (
        trace["optimization"]["effective_controls"]["prism_cost_priority"] == priority
    )
    assert "context_management" not in physical.calls[0]
    assert config.profiles["prism"].model_dump_json() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("price", [0, 0.1])
async def test_equal_cost_power_and_stable_model_ties(monkeypatch, price):
    app, upstream, physical, _, models = setup(monkeypatch, allowed_policies=["direct"])
    for model in models.values():
        model.input_cost_per_million = model.output_cost_per_million = price
    models["mid"].power_rating = models["strong"].power_rating
    async with upstream:
        response, trace = await send(app, body(1))
    assert response.status_code == 200
    assert physical.calls[0]["model"] == "physical-mid"
    assert trace["optimization"]["selected"]["normalized_cost"] == 0


@pytest.mark.asyncio
async def test_request_subsets_limits_defaults_and_no_shared_mutation(monkeypatch):
    app, upstream, physical, config, _ = setup(
        monkeypatch,
        optimization={
            "model_ids": ["cheap", "mid", "strong"],
            "default_cost_priority": 1,
        },
        limits={"max_calls": 8, "max_cost_usd": 1},
    )
    before = config.profiles["prism"].model_dump_json()
    request = body(
        context_management=[
            {"prism_cost_priority": 0},
            {"prism_model_ids": ["mid"]},
            {"prism_allowed_policies": ["direct"]},
            {"prism_max_calls": 1, "prism_max_cost_usd": 0.1},
        ]
    )
    async with upstream:
        response, trace = await send(app, request)
        second, second_trace = await send(app, body(context_management=[]))
    assert response.status_code == second.status_code == 200
    assert [call["model"] for call in physical.calls] == [
        "physical-mid",
        "physical-cheap",
    ]
    assert trace["eligible_policies"] == ["direct"]
    assert trace["optimization"]["effective_controls"]["prism_max_calls"] == 1
    assert second_trace["optimization"]["effective_controls"]["prism_max_calls"] == 8
    assert config.profiles["prism"].model_dump_json() == before
    assert response.json()["usage"] == second.json()["usage"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [
        {"prism_model_ids": ["outsider"]},
        {"prism_allowed_policies": ["retrieve_read"]},
        {"prism_max_calls": 9},
        {"prism_max_cost_usd": 2},
        {"type": "compaction", "compact_threshold": 200000},
    ],
)
async def test_rejected_overrides_dispatch_nothing(monkeypatch, control):
    app, upstream, physical, _, _ = setup(
        monkeypatch, limits={"max_calls": 8, "max_cost_usd": 1}
    )
    async with upstream:
        response, _ = await send(app, body(context_management=[control]))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "context_management"
    assert not physical.calls


@pytest.mark.asyncio
async def test_legacy_profiles_reject_controls_but_allow_empty_array(monkeypatch):
    app, upstream, physical, _, _ = setup(
        monkeypatch, optimization=None, allowed_policies=["direct"]
    )
    async with upstream:
        invalid, _ = await send(app, body(0.8))
        valid, trace = await send(app, body(context_management=[]))
    assert invalid.status_code == 400 and valid.status_code == 200
    assert "optimization" not in trace and len(physical.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent,priority,expected",
    [
        (Agent("draft_review"), 0, "draft_review"),
        (Agent("draft_review"), 1, "direct"),
        (Agent("draft_review", probability=0.1), 0, "direct"),
        (Agent("draft_review", truncated=True), 0, "direct"),
        (Agent("unknown"), 0, "direct"),
        (Agent("abstain"), 0, "direct"),
    ],
)
async def test_laya_bonus_changes_selection_but_obeys_cost_and_confidence(
    monkeypatch, agent, priority, expected
):
    app, upstream, physical, _, _ = setup(monkeypatch, agent=agent)
    async with upstream:
        response, trace = await send(app, body(priority))
    assert response.status_code == 200, response.text
    assert trace["strategy"] == expected
    assert len(physical.calls) == (3 if expected == "draft_review" else 1)
    state, questions = agent.calls[-1]
    assert state["prism_cost_priority"] == priority
    assert all(
        candidate["stage_models"] and candidate["power_ratings"]
        for candidate in state["candidate_plans"]
    )
    assert "task-fit" in questions["strategy"]["instructions"]
    assert trace["optimization"]["selected"]["task_fit"] == (
        1 if expected == "draft_review" else 0
    )


@pytest.mark.asyncio
async def test_shadow_and_unavailable_decision_use_ranking(monkeypatch):
    app, upstream, _, config, _ = setup(monkeypatch, agent=Agent("draft_review"))
    config.decision.mode = "shadow"
    async with upstream:
        response, trace = await send(app, body(0))
        assert (
            trace["strategy"] == "direct"
            and trace["decision"]["reason"] == "shadow_mode"
        )
        app.state.engine.decision.agent = None
        second, second_trace = await send(app, body(0))
    assert response.status_code == second.status_code == 200
    assert second_trace["decision"]["reason"] == "model_not_prepared"
    assert second_trace["strategy"] == "direct"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_mixed_models_final_json_and_buffered_delivery(monkeypatch, stream):
    app, upstream, physical, _, models = setup(monkeypatch, strategy="draft_review")
    models["strong"].capabilities = {"text", "json_schema"}
    models["cheap"].capabilities = models["mid"].capabilities = {"text", "json_object"}
    request = body(
        0,
        stream=stream,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    )
    async with upstream:
        response, trace = await send(app, request)
    assert response.status_code == 200, response.text
    assert [call["model"] for call in physical.calls] == [
        "physical-strong",
        "physical-mid",
        "physical-strong",
    ]
    assert trace["optimization"]["selected"]["model_power"] == pytest.approx(8.25)
    for call in physical.calls:
        assert call["messages"][: len(request["messages"])] == request["messages"]
        assert "context_management" not in call
    assert trace["execution_usage"]["consumed_or_conservatively_charged"]["calls"] == 3
    if stream:
        assert response.headers["x-prism-stream-mode"] == "buffered"
        assert "draft answer" not in response.text and "clarify" not in response.text
        assert "[DONE]" in response.text
    else:
        assert (
            json.loads(response.json()["choices"][0]["message"]["content"])["answer"]
            == "final answer"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,count",
    [
        ("draft_length", 1),
        ("draft_bytes", 1),
        ("draft_escaping", 1),
        ("draft_refusal", 1),
        ("draft_tools", 1),
        ("review_json", 2),
        ("review_fields", 2),
        ("review_extra", 2),
        ("review_bytes", 2),
        ("review_length", 2),
        ("review_http", 2),
    ],
)
async def test_failed_intermediate_never_synthesizes(monkeypatch, failure, count):
    app, upstream, physical, _, _ = setup(
        monkeypatch, failure=failure, strategy="draft_review"
    )
    async with upstream:
        response, trace = await send(app, body(0))
    assert response.status_code == 502
    assert len(physical.calls) == count
    assert not trace["execution_usage"]["outstanding_nodes"]
    assert (
        trace["execution_usage"]["consumed_or_conservatively_charged"]["calls"] == count
    )


@pytest.mark.asyncio
async def test_whole_graph_reserved_before_drafting_and_runtime_context_bounds(
    monkeypatch,
):
    app, upstream, physical, _, models = setup(monkeypatch, strategy="draft_review")
    engine = app.state.engine
    execution = engine.prepare(body(0))
    backend = engine.backend
    original = backend.complete
    seen = []

    async def complete_checked(
        model, messages, parameters, output, ledger, reservation
    ):
        seen.append(reservation.node_id)
        assert set(ledger.reservations) == {"draft", "review", "synthesis"}
        assert (
            check_context(messages, parameters, output, model)
            <= reservation.input_tokens
        )
        return await original(model, messages, parameters, output, ledger, reservation)

    monkeypatch.setattr(backend, "complete", complete_checked)
    async with upstream:
        result = await engine.execute(execution)
    assert result["choices"][0]["message"]["content"] == "final answer"
    assert seen == ["draft", "review", "synthesis"] and len(physical.calls) == 3
    # Escaping and Unicode also stay inside the preflight envelope at the byte cap.
    cap = execution["profile"].intermediate_max_bytes
    draft = "\x00" * ((cap - 2) // 6)
    review = {"issues": ["é" * 100], "suggestions": []}
    compiled = execution["policy_plans"]["draft_review"]
    assert (
        check_context(
            review_messages(execution, draft),
            {"temperature": 0, "response_format": {"type": "json_object"}},
            128,
            models[execution["profile"].verifier],
        )
        <= compiled["reservations"][1][1]
    )
    assert (
        check_context(
            synthesis_messages(execution, draft, review),
            execution["parameters"],
            512,
            compiled["final"],
        )
        <= compiled["reservations"][2][1]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control", [{"prism_max_calls": 2}, {"prism_max_cost_usd": 1e-12}]
)
async def test_infeasible_whole_graph_dispatches_nothing(monkeypatch, control):
    app, upstream, physical, _, _ = setup(monkeypatch, strategy="draft_review")
    async with upstream:
        response, _ = await send(app, body(context_management=[control]))
    assert response.status_code == 413
    assert not physical.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"seed": 42},
        {"logprobs": True},
        {"reasoning_effort": "low"},
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Continue"},
            ]
        },
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ]
        },
    ],
)
async def test_direct_only_requests_still_select_models(monkeypatch, extra):
    app, upstream, physical, _, models = setup(monkeypatch, agent=Agent("draft_review"))
    for model in models.values():
        model.capabilities.update({"seed", "logprobs", "reasoning_effort"})
    async with upstream:
        response, trace = await send(app, body(0, **extra))
    assert response.status_code == 200, response.text
    assert trace["strategy"] == "direct" and trace["stage_models"] == {
        "direct": "strong"
    }
    assert len(physical.calls) == 1


@pytest.mark.asyncio
async def test_forwarded_sse_uses_selected_backend(monkeypatch):
    app, upstream, physical, _, _ = setup(monkeypatch, allowed_policies=["direct"])
    async with upstream:
        response, trace = await send(
            app, body(0, stream=True, stream_options={"include_usage": True})
        )
    assert response.status_code == 200
    assert response.headers["x-prism-stream-mode"] == "forwarded"
    assert (
        physical.calls[0]["model"] == "physical-strong" and physical.calls[0]["stream"]
    )
    assert "[DONE]" in response.text
    assert trace["execution_usage"]["calls"][0]["model_id"] == "strong"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy", ["evidence_map", "batched_map", "verified_map", "retrieve_read"]
)
async def test_source_pipelines_choose_stage_models_and_preserve_coverage(
    monkeypatch, policy
):
    app, upstream, physical, _, models = setup(
        monkeypatch,
        allowed_policies=[policy],
        strategy=policy,
        coverage="focused" if policy == "retrieve_read" else "exhaustive",
    )
    models["strong"].capabilities = {"text", "json_schema"}
    request = body(
        0,
        messages=[
            {
                "role": "user",
                "content": "Summarize deployment qualifications. <prism-source>Deployment requires approval.</prism-source>",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object", "required": ["answer"]},
            },
        },
    )
    async with upstream:
        response, trace = await send(app, request)
    assert response.status_code == 200, response.text
    assert trace["stage_models"]["synthesizer"] == "strong"
    if policy != "retrieve_read":
        assert trace["stage_models"]["worker"] == "mid"
    assert set(trace["coverage"]["required_partitions"]) == set(
        trace["coverage"]["validated_partitions"]
    )
    assert all("context_management" not in call for call in physical.calls)


@pytest.mark.asyncio
async def test_partition_cache_depends_on_worker_model(monkeypatch):
    app, upstream, _, _, models = setup(
        monkeypatch, allowed_policies=["evidence_map"], strategy="evidence_map"
    )
    # Leave room for the extraction contract while still forcing smaller chunks.
    models["cheap"].context_window = 4100
    models["cheap"].max_output_tokens = 512
    source = "Deployment requires approval.\n" * 160
    execution = app.state.engine.prepare(
        body(
            0,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize qualifications. <prism-source>"
                    + source
                    + "</prism-source>",
                }
            ],
        )
    )
    cached = {key[0]: parts for key, parts in execution["partition_cache"].items()}
    assert len(cached["cheap"]) > len(cached["strong"])
    for ref, partitions in cached.items():
        assert "".join(part.text for part in partitions) == source
        for partition in partitions:
            check_context(
                app.state.engine._worker_messages(execution["arena"], partition),
                {"temperature": 0, "response_format": {"type": "json_object"}},
                128,
                models[ref],
            )
    await upstream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_and_cancel_release_unstarted_stages(monkeypatch, cancel):
    app, upstream, _, _, _ = setup(
        monkeypatch,
        strategy="draft_review",
        limits={"deadline_seconds": 0.1 if not cancel else 5},
    )
    started = asyncio.Event()

    async def blocked(request):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(upstream, "_transport", httpx.MockTransport(blocked))
    engine = app.state.engine
    execution = engine.prepare(body(0))
    async with upstream:
        task = asyncio.create_task(engine.execute(execution))
        await asyncio.wait_for(started.wait(), 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(PrismError) as error:
                await task
            assert error.value.code == "deadline_exceeded"
    usage = execution["trace"]["execution_usage"]
    assert not usage["outstanding_nodes"]
    assert len(usage["calls"]) == 1 and usage["calls"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_openai_sdk_extra_body(monkeypatch):
    app, upstream, physical, _, _ = setup(monkeypatch, allowed_policies=["direct"])
    async with (
        upstream,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client,
    ):
        sdk = AsyncOpenAI(
            base_url="http://prism/v1",
            api_key="optimization-test",
            http_client=client,
            max_retries=0,
        )
        result = await sdk.chat.completions.create(
            model="prism",
            messages=body()["messages"],
            extra_body={"context_management": [{"prism_cost_priority": 0.8}]},
        )
    assert result.choices[0].message.content == "final answer"
    assert physical.calls[0]["model"] == "physical-cheap"


@pytest.mark.integration
def test_optimization_controls_over_real_http(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "optimization-test")
    upstream = FastAPI()
    calls = []

    @upstream.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        calls.append(payload)
        return completion("final answer")

    with serve(upstream) as backend_url:
        _, unused, _, config, models = setup(monkeypatch, allowed_policies=["direct"])
        asyncio.run(unused.aclose())
        for model in models.values():
            model.base_url = backend_url + "/v1"
        app = create_app(config, models=models, decision_agent=Agent("abstain"))
        with (
            serve(app) as prism_url,
            OpenAI(
                base_url=prism_url + "/v1", api_key="optimization-test", max_retries=0
            ) as sdk,
        ):
            for priority in (1, 0):
                result = sdk.chat.completions.create(
                    model="prism",
                    messages=body()["messages"],
                    extra_body={
                        "context_management": [{"prism_cost_priority": priority}]
                    },
                )
                assert result.choices[0].message.content == "final answer"
        assert [call["model"] for call in calls] == [
            "physical-cheap",
            "physical-strong",
        ]
        assert all("context_management" not in call for call in calls)


@pytest.mark.asyncio
async def test_metadata_error_is_clear_and_redacted(monkeypatch, tmp_path, capsys):
    _, upstream, _, config, models = setup(monkeypatch)
    models["cheap"].power_rating = None
    models["cheap"].api_key = "private-backend-secret"
    (tmp_path / "models.json").write_text(
        json.dumps(
            {"models": [model.model_dump(mode="json") for model in models.values()]}
        )
    )
    (tmp_path / "prism.json").write_text(config.model_dump_json())
    assert main(["validate", "--config", str(tmp_path / "prism.json")]) == 2
    output = capsys.readouterr().out
    assert "power_rating" in output and "prices" in output
    assert "private-backend-secret" not in output
    await upstream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["direct", "draft_review", "verified_map"])
async def test_exact_compiled_cost_ceiling_is_admitted(monkeypatch, policy):
    app, upstream, _, _, _ = setup(
        monkeypatch, allowed_policies=[policy], strategy=policy
    )
    request = body(0)
    if policy == "verified_map":
        request["messages"][0]["content"] = (
            "Summarize qualifications. <prism-source>Deployment requires approval.</prism-source>"
        )
    execution = app.state.engine.prepare(request)
    ceiling = execution["optimization_candidates"][policy]["cost_upper_estimate_usd"]
    request["context_management"].append({"prism_max_cost_usd": ceiling})
    async with upstream:
        response, trace = await send(app, request)
    assert response.status_code == 200, response.text
    assert trace["optimization"]["selected"]["cost_upper_estimate_usd"] <= ceiling


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", ["context", "output", "capability"])
async def test_strongest_infeasible_backend_is_excluded(monkeypatch, constraint):
    app, upstream, physical, _, models = setup(monkeypatch, allowed_policies=["direct"])
    request = body(0)
    if constraint == "context":
        models["strong"].context_window = 1024
        models["strong"].max_output_tokens = 512
        request["messages"][0]["content"] = "Explain this design. " * 100
    elif constraint == "output":
        models["strong"].max_output_tokens = 128
    else:
        models["strong"].capabilities.remove("stream")
        request["stream"] = True
    async with upstream:
        response, trace = await send(app, request)
    assert response.status_code == 200, response.text
    assert physical.calls[0]["model"] == "physical-mid"
    assert trace["optimization"]["feasible_assignments"] == 2


@pytest.mark.asyncio
async def test_fixed_policy_and_direct_requirement_cannot_be_overridden(monkeypatch):
    app, upstream, physical, _, _ = setup(monkeypatch, strategy="draft_review")
    async with upstream:
        excludes_fixed, _ = await send(
            app, body(context_management=[{"prism_allowed_policies": ["direct"]}])
        )
        needs_direct, _ = await send(app, body(seed=42))
        exact_count, _ = await send(
            app, body(messages=[{"role": "user", "content": "Count every record."}])
        )
    assert (
        excludes_fixed.status_code
        == needs_direct.status_code
        == exact_count.status_code
        == 400
    )
    assert not physical.calls


def test_example_configuration_validates():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/standalone/optimization/prism.json"
    )
    config, models = load_config(path)
    assert config.profiles["prism-optimized"].optimization.default_cost_priority == 0.5
    assert set(models) == {"small", "strong"}


@pytest.mark.asyncio
async def test_heterogeneous_output_caps_admitted_per_request(monkeypatch, tmp_path):
    old_app, upstream, physical, config, models = setup(
        monkeypatch, allowed_policies=["direct"]
    )
    models["cheap"].max_output_tokens = 128
    (tmp_path / "models.json").write_text(
        json.dumps(
            {"models": [model.model_dump(mode="json") for model in models.values()]}
        )
    )
    (tmp_path / "prism.json").write_text(config.model_dump_json())
    loaded, raw = load_config(tmp_path / "prism.json")
    app = create_app(
        loaded,
        models=raw,
        backend=old_app.state.engine.backend,
        decision_agent=Agent("abstain"),
    )
    async with upstream:
        large, large_trace = await send(app, body(1))
        small, small_trace = await send(app, body(1, max_completion_tokens=128))
    assert large.status_code == small.status_code == 200
    assert [call["model"] for call in physical.calls] == [
        "physical-mid",
        "physical-cheap",
    ]
    assert large_trace["optimization"]["feasible_assignments"] == 2
    assert small_trace["optimization"]["feasible_assignments"] == 3
    config.profiles["prism"].optimization = None
    (tmp_path / "prism.json").write_text(config.model_dump_json())
    with pytest.raises(ValueError, match="public output cap"):
        load_config(tmp_path / "prism.json")
