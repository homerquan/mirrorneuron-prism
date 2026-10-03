"""Regression coverage for live-model failures, recovery, and rolling evidence memory."""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from prism.backends import OpenAIBackend
from prism.config import Limits, RawModel, load_config
from prism.decision import LayaDecision
from prism.engine import ExecutionEngine
from prism.errors import PrismError
from prism.optimization import adequate_stage_power
from prism.runtime import Ledger

from .test_optimization import PhysicalBackend
from .test_policies import request_body, send, setup
from .test_proxy import completion


@pytest.mark.asyncio
async def test_aliases_share_physical_concurrency_and_operator_thinking_setting():
    model = RawModel(
        id="a",
        name="physical",
        base_url="http://fixture/v1",
        concurrency=1,
        enable_thinking=False,
    )
    alias = model.model_copy(
        update={"id": "b", "context_window": 8192, "concurrency": 2}
    )
    active = peak = 0
    seen = []

    async def handle(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append(json.loads(request.content))
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=completion("answer"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        backend = OpenAIBackend({"a": model, "b": alias}, client)
        ledger = Ledger(Limits())
        reservations = [
            await ledger.reserve(m.id, m, 1024, 128) for m in (model, alias)
        ]
        await asyncio.gather(
            *(
                backend.complete(
                    m, [{"role": "user", "content": "task"}], {}, 128, ledger, r
                )
                for m, r in zip((model, alias), reservations, strict=True)
            )
        )
        assert peak == 1
        assert all(
            body["chat_template_kwargs"] == {"enable_thinking": False} for body in seen
        )
        # Public reasoning instructions take precedence over the operator default.
        request = backend.request(model, [], {"reasoning_effort": "high"}, 128)
        assert "chat_template_kwargs" not in json.loads(request.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_upstream_context_error_is_classified_without_leaking_body(stream):
    model = RawModel(id="m", name="physical", base_url="http://fixture/v1")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                500,
                json={
                    "error": {
                        "message": "Context size has been exceeded. secret-provider-detail"
                    }
                },
            )
        )
    ) as client:
        backend = OpenAIBackend({"m": model}, client)
        ledger = Ledger(Limits())
        reservation = await ledger.reserve("node", model, 512, 128)
        with pytest.raises(PrismError, match="runtime context") as caught:
            if stream:
                async for _ in backend.stream(model, [], {}, 128, ledger, reservation):
                    pass
            else:
                await backend.complete(model, [], {}, 128, ledger, reservation)
        assert caught.value.code == "upstream_context_length_exceeded"
        assert ledger.usage[0]["upstream_status"] == 500
        assert "secret" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["evidence_map", "batched_map", "verified_map"])
async def test_failed_worker_recovers_once_without_losing_successful_partitions(
    monkeypatch, policy
):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, models = setup()
    profile = config.profiles[policy]
    profile.worker_fallback = "verify"
    models["worker"].power_rating = 3
    models["verify"].power_rating = models["synth"].power_rating = 9
    failed = False

    def handler(request):
        nonlocal failed
        body = json.loads(request.content)
        if body["model"] == "physical-worker" and not failed:
            failed = True
            physical.calls.append(body)
            return httpx.Response(
                200,
                json=completion(
                    '{"status":"incomplete","records":[],"needs":["missing definition"]}'
                ),
            )
        return physical.respond(request)

    # The fixture takes an HTTP request, like the real adapter.
    upstream._transport = httpx.MockTransport(handler)
    try:
        response, trace = await send(app, request_body(policy))
        assert response.status_code == 200, response.text
        assert trace["recovery"] and all(
            e["status"] == "complete" for e in trace["recovery"]
        )
        coverage = trace["coverage"]
        assert set(coverage["validated_partitions"]) == set(
            coverage["required_partitions"]
        )
        assert len(coverage["validated_partitions"]) == len(
            set(coverage["validated_partitions"])
        )
        assert all(
            c["status"] != "cancelled" for c in trace["execution_usage"]["calls"]
        )
        assert not trace["execution_usage"]["outstanding_nodes"]
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_recovery_budget_is_reserved_before_dispatch(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, _ = setup()
    config.profiles["evidence_map"].worker_fallback = "verify"
    config.profiles[
        "evidence_map"
    ].limits.max_calls = 4  # ordinary graph fits; recovery does not
    try:
        response, _ = await send(app, request_body("evidence_map"))
        assert response.status_code == 413
        assert physical.calls == []
    finally:
        await upstream.aclose()


def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/standalone/docker-spark/benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("reliability_runner", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_repair", [False, True])
async def test_verification_repair_is_fresh_bounded_and_rechecked(
    monkeypatch, reject_repair
):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, _ = setup()
    config.profiles["verified_map"].worker_fallback = "verify"
    config.profiles["verified_map"].limits.max_parallel = 1

    def handler(request):
        body = json.loads(request.content)
        if body.get("response_format"):
            packet = json.loads(body["messages"][-1]["content"])
            if "records" in packet and (
                reject_repair
                or not any(
                    "-repair-record-" in r["record_id"] for r in packet["records"]
                )
            ):
                physical.calls.append(body)
                return httpx.Response(
                    200,
                    json=completion(
                        json.dumps(
                            {
                                "supported": False,
                                "checked_record_ids": [
                                    r["record_id"] for r in packet["records"]
                                ],
                                "issues": ["unsupported interpretation"],
                            }
                        )
                    ),
                )
        return physical.respond(request)

    upstream._transport = httpx.MockTransport(handler)
    try:
        response, trace = await send(app, request_body("verified_map"))
        assert response.status_code == (502 if reject_repair else 200), response.text
        events = trace["recovery"]
        assert events and all(e["parent_node_id"].startswith("verify-") for e in events)
        assert all(
            e["replaced_evidence_sha256"] and e["replaced_record_count"] == 1
            for e in events
        )
        assert len(events) <= len(trace["coverage"]["required_partitions"])
        repair_checks = [
            json.loads(b["messages"][-1]["content"])
            for b in physical.calls
            if b.get("response_format")
            and "-repair-record-" in b["messages"][-1]["content"]
        ]
        assert repair_checks
        if reject_repair:
            assert response.json()["error"]["code"] == "unverified_evidence"
            assert not any(
                c["node_id"] == "synthesis" for c in trace["execution_usage"]["calls"]
            )
        else:
            final_packet = json.loads(physical.calls[-1]["messages"][-1]["content"])
            assert all(
                "-repair-record-" in r["record_id"]
                for r in final_packet["prism_evidence"]
            )
            assert all(e["status"] == "complete" for e in events)
            assert all(
                "source" in json.loads(b["messages"][-1]["content"])
                for b in physical.calls
                if "recovery_instruction" in b["messages"][-1]["content"]
            )
        assert not trace["execution_usage"]["outstanding_nodes"]
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_verification_repair_budget_admitted_before_dispatch(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, _ = setup()
    config.profiles["verified_map"].worker_fallback = "verify"
    config.profiles["verified_map"].limits.max_calls = 15
    try:
        response, _ = await send(app, request_body("verified_map"))
        assert response.status_code == 413
        assert not physical.calls
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [None, "json", "reference", "bytes", "length", "calls"]
)
async def test_overflow_compaction_is_bounded_validated_and_accounted(
    tmp_path, failure
):
    module = runner()
    path = module.configure(
        tmp_path,
        "http://local/v1",
        "http://remote/v1",
        "gemma",
        "nemotron",
        18080,
        1024,
    )
    config, models = load_config(path)
    profile = config.profiles["small-evidence_map-gemma"]
    if failure == "calls":
        profile.evidence_compaction.max_calls = 1
    physical = PhysicalBackend()
    compact_payloads = []

    def handler(request):
        body = json.loads(request.content)
        packet = json.loads(body["messages"][-1]["content"])
        if packet.get("prism_compaction"):
            compact_payloads.append(packet)
            if failure == "json":
                return httpx.Response(200, json=completion("not JSON"))
            memory = {
                "summary": "Deployment qualifications.",
                "unresolved": [],
                "record_ids": [],
            }
            if failure == "reference":
                memory["record_ids"] = ["invented"]
            if failure == "bytes":
                memory["summary"] = "x" * 2048
            return httpx.Response(
                200,
                json=completion(
                    json.dumps(memory),
                    finish="length" if failure == "length" else "stop",
                ),
            )
        return physical.respond(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        engine = ExecutionEngine(
            config, models, OpenAIBackend(models, client), LayaDecision(config.decision)
        )
        execution = engine.prepare(
            {
                **module.large_cases()[0]["request"],
                "model": "small-evidence_map-gemma",
                "max_completion_tokens": 1024,
            }
        )
        original = [
            engine._worker_messages(execution["arena"], p)[-1]["content"]
            for p in execution["policy_plans"]["evidence_map"]["partitions"]
        ]
        if failure:
            with pytest.raises(PrismError) as caught:
                await engine.execute(execution)
            assert caught.value.code in {
                "invalid_compaction",
                "incomplete_compaction",
                "resource_limit",
            }
            assert not any(
                c["node_id"] == "synthesis" for c in execution["ledger"].usage
            )
        else:
            await engine.execute(execution)
            assert len(compact_payloads) > 1
            assert compact_payloads[0]["memory"] is None
            assert all(p["memory"] is not None for p in compact_payloads[1:])
            assert execution["trace"]["compaction"]["lossy"] is True
            assert all(
                u["memory_bytes"] <= 1024
                for u in execution["trace"]["compaction"]["updates"]
            )
            assert any(
                c["node_id"].startswith("compact-") for c in execution["ledger"].usage
            )
        assert original == [
            engine._worker_messages(execution["arena"], p)[-1]["content"]
            for p in execution["policy_plans"]["evidence_map"]["partitions"]
        ]
        assert not execution["ledger"].snapshot()["outstanding_nodes"]


def test_harder_stages_cannot_select_weaker_or_equal_nonmaximal_models():
    models = {
        key: RawModel(
            id=key, name=key, base_url="http://fixture/v1", power_rating=rating
        )
        for key, rating in [("small", 3), ("mid", 6), ("strong", 9)]
    }
    assert not adequate_stage_power(
        {"worker": "small", "synthesizer": "small"}, models, models
    )
    assert not adequate_stage_power(
        {"worker": "mid", "verifier": "strong", "synthesizer": "mid"}, models, models
    )
    assert adequate_stage_power(
        {"worker": "small", "verifier": "mid", "synthesizer": "strong"}, models, models
    )
    assert adequate_stage_power(
        {"worker": "strong", "synthesizer": "strong"}, models, models
    )
    assert adequate_stage_power(
        {"worker": "small", "synthesizer": "small"}, models, ["small"]
    )


@pytest.mark.asyncio
async def test_empty_evidence_needs_no_model_verification(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, _, _ = setup()
    original = physical.respond

    def handler(request):
        body = json.loads(request.content)
        if body["model"] == "physical-worker":
            physical.calls.append(body)
            return httpx.Response(
                200, json=completion('{"status":"complete","records":[],"needs":[]}')
            )
        return original(request)

    upstream._transport = httpx.MockTransport(handler)
    try:
        response, trace = await send(app, request_body("verified_map"))
        assert response.status_code == 200
        assert all(
            method == "empty_record_set"
            for method in trace["verification"]["methods"].values()
        )
        assert not any(body["model"] == "physical-verify" for body in physical.calls)
        assert set(trace["verification"]["verified_partitions"]) == set(
            trace["coverage"]["required_partitions"]
        )
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_cancellation_during_recovery_stops_further_calls(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, _ = setup()
    profile = config.profiles["evidence_map"]
    profile.worker_fallback = "verify"
    profile.limits.max_parallel = 1
    entered = asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        physical.calls.append(body)
        if body["model"] == "physical-worker":
            return httpx.Response(
                200, json=completion('{"status":"incomplete","records":[],"needs":[]}')
            )
        entered.set()
        await asyncio.Event().wait()

    upstream._transport = httpx.MockTransport(handler)
    engine = app.state.engine
    execution = engine.prepare(request_body("evidence_map"))
    task = asyncio.create_task(engine.execute(execution))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(physical.calls) == 2
        assert execution["trace"]["recovery"][0]["status"] == "cancelled"
        assert not execution["ledger"].snapshot()["outstanding_nodes"]
    finally:
        task.cancel()
        await upstream.aclose()


@pytest.mark.asyncio
async def test_power_ordering_uses_feasible_stages_and_skips_ineligible_policies(
    monkeypatch,
):
    from .test_optimization import body
    from .test_optimization import setup as optimized_setup

    app, upstream, _, config, _ = optimized_setup(
        monkeypatch, allowed_policies=["direct", "evidence_map"]
    )
    config.profiles["prism"].optimization.enforce_stage_power_order = True
    source = "Deployment requires approval.\n" * 2400
    execution = app.state.engine.prepare(
        body(
            1,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize deployment qualifications. <prism-source>"
                    + source
                    + "</prism-source>",
                }
            ],
        )
    )
    assert "direct" not in execution["policy_plans"]
    for candidate in execution["optimization_candidates"].values():
        assert candidate["stage_models"]["synthesizer"] != "cheap"
    await upstream.aclose()
