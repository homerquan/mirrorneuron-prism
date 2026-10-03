"""Hierarchical state bounds, provenance, lazy lookup and scratch-file lifetime."""

import asyncio
import json

import httpx
import pytest

from prism.config import EvidenceReduction
from prism.contracts import byte_tokens, check_context
from prism.errors import PrismError
from prism.reduction import EvidenceStore, ReasoningState, bound_state, normalize_state

from .test_policies import send, setup
from .test_proxy import completion


def request(policy):
    source = "\n".join(
        f"Sensor {i:03d} measured stable voltage at station {i:03d}." for i in range(40)
    )
    source += "\nStandard deployments require manager approval."
    return {
        "model": policy,
        "messages": [
            {
                "role": "user",
                "content": "Who approves standard deployments? Return JSON standard_approver. <prism-source>"
                + source
                + "</prism-source>",
            }
        ],
        "max_completion_tokens": 256,
        "response_format": {"type": "json_object"},
    }


def configure(config, models, policy):
    profile = config.profiles[policy]
    profile.partition_bytes = 512
    profile.worker_output_tokens = 1024
    profile.limits.max_calls = 128
    profile.limits.max_output_tokens = 262144
    profile.limits.max_parallel = 2
    profile.evidence_reduction = EvidenceReduction(
        model="worker",
        state_max_tokens=512,
        output_tokens=1024,
        max_calls=16,
        evidence_max_tokens=1536,
        verification_max_extra_calls=48,
    )
    for model in models.values():
        model.context_window = 8192


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["evidence_map", "batched_map", "verified_map"])
async def test_tree_and_lazy_lookup_keep_all_packets_bounded(monkeypatch, policy):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, models = setup()
    configure(config, models, policy)
    scratch = []
    original_init = EvidenceStore.__init__

    def record_store(self, arena, records):
        original_init(self, arena, records)
        scratch.append(self.root)
        assert list(self.root.glob("*.md"))

    monkeypatch.setattr(EvidenceStore, "__init__", record_store)
    lookup_count = 0

    def handler(req):
        nonlocal lookup_count
        body = json.loads(req.content)
        physical.calls.append(body)
        packet = json.loads(body["messages"][-1]["content"])
        if "prism_reasoning_reduce" in packet:
            assert "quote" not in json.dumps(packet["prism_reasoning_reduce"])
            observations = [
                o for s in packet["prism_reasoning_reduce"] for o in s["facts"]
            ]
            observations.sort(key=lambda o: "manager" not in o["text"])
            state = ReasoningState(facts=observations[:2]).model_dump()
            state["evidence_refs"] = sorted(
                {r for o in state["facts"] for r in o["evidence_refs"]}
            )
            assert byte_tokens(state) <= packet["state_max_bytes"]
            result = state
        elif "prism_evidence_lookup" in packet:
            lookup_count += 1
            result = {
                "evidence_refs": [],
                "query": "manager approval" if lookup_count == 1 else "",
            }
        elif "prism_reasoning_state" in packet:
            assert any(
                "manager approval" in e["quote"] for e in packet["retrieved_evidence"]
            )
            result = {"standard_approver": "manager"}
        elif "records" in packet:
            assert len(packet["records"]) == 1
            result = {
                "supported": True,
                "checked_record_ids": [r["record_id"] for r in packet["records"]],
                "issues": [],
            }
        else:

            def artifact(part):
                return {
                    "status": "complete",
                    "needs": [],
                    "records": [
                        {"quote": line, "fact": line}
                        for line in part["source"].splitlines()
                        if line
                    ],
                }

            result = (
                {
                    "partitions": [
                        {"partition_id": p["partition_id"], **artifact(p)}
                        for p in packet["partitions"]
                    ]
                }
                if "partitions" in packet
                else artifact(packet)
            )
        return httpx.Response(200, json=completion(json.dumps(result)))

    upstream._transport = httpx.MockTransport(handler)
    try:
        response, trace = await send(app, request(policy))
        assert response.status_code == 200, response.text
        assert (
            json.loads(response.json()["choices"][0]["message"]["content"])[
                "standard_approver"
            ]
            == "manager"
        )
        reduction = trace["reduction"]
        assert reduction["level_count"] >= 2
        assert reduction["lookups"] and reduction["temporary_files_cleaned"]
        assert reduction["final_state_bytes"] <= reduction["state_max_bytes"] <= 512
        assert all(n["state_bytes"] <= 512 for n in reduction["levels"])
        assert all(not path.exists() for path in scratch)
        assert not trace["execution_usage"]["outstanding_nodes"]
        for body in physical.calls:
            model = next(m for m in models.values() if m.name == body["model"])
            parameters = {
                "response_format": body["response_format"],
                **(
                    {"temperature": body["temperature"]}
                    if "temperature" in body
                    else {}
                ),
            }
            bound = check_context(
                body["messages"], parameters, body["max_completion_tokens"], model
            )
            if "prism_reasoning_reduce" in body["messages"][-1]["content"]:
                assert bound <= profile_input(config, policy)
    finally:
        await upstream.aclose()


def profile_input(config, policy):
    return config.profiles[policy].evidence_reduction.max_input_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["malformed", "reference", "oversize", "truncated", "lookup", "budget", "cancel"],
)
async def test_invalid_state_lookup_or_budget_never_reaches_final(monkeypatch, failure):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")
    app, upstream, physical, config, models = setup()
    configure(config, models, "evidence_map")
    if failure == "budget":
        config.profiles["evidence_map"].limits.max_calls = 3
    scratch = []
    original_init = EvidenceStore.__init__

    def record_store(self, arena, records):
        original_init(self, arena, records)
        scratch.append(self.root)

    monkeypatch.setattr(EvidenceStore, "__init__", record_store)
    entered = asyncio.Event()

    async def handler(req):
        body = json.loads(req.content)
        packet = json.loads(body["messages"][-1]["content"])
        if "prism_reasoning_reduce" in packet:
            physical.calls.append(body)
            if failure == "cancel":
                entered.set()
                await asyncio.Event().wait()
            state = ReasoningState().model_dump()
            if failure == "reference":
                state["evidence_refs"] = ["unknown"]
            if failure == "oversize":
                state["open_questions"] = [{"text": "x" * 2000, "evidence_refs": []}]
            return httpx.Response(
                200,
                json=completion(
                    "bad json" if failure == "malformed" else json.dumps(state),
                    "length" if failure == "truncated" else "stop",
                ),
            )
        if "prism_evidence_lookup" in packet:
            physical.calls.append(body)
            return httpx.Response(
                200,
                json=completion('{"evidence_refs":["../../outside.md"],"query":""}'),
            )
        return physical.respond(req)

    upstream._transport = httpx.MockTransport(handler)
    engine = app.state.engine
    task = None
    try:
        if failure == "budget":
            with pytest.raises(PrismError, match="budget"):
                engine.prepare(request("evidence_map"))
            assert not physical.calls
            return
        execution = engine.prepare(request("evidence_map"))
        task = asyncio.create_task(engine.execute(execution))
        if failure == "cancel":
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(PrismError) as caught:
                await task
            assert caught.value.code in {
                "invalid_reasoning_state",
                "invalid_evidence_lookup",
            }
        assert scratch and all(not p.exists() for p in scratch)
        assert execution["trace"]["reduction"]["temporary_files_cleaned"]
        assert not any(c["node_id"] == "synthesis" for c in execution["ledger"].usage)
        assert not execution["ledger"].snapshot()["outstanding_nodes"]
    finally:
        if task:
            task.cancel()
        await upstream.aclose()


def test_state_dedup_preserves_provenance_and_rejects_ungrounded_facts():
    state = ReasoningState(
        facts=[
            {"text": "Manager approval", "evidence_refs": ["a"]},
            {"text": "manager  approval", "evidence_refs": ["b"]},
        ]
    ).model_dump()
    result = normalize_state(state, {"a", "b"})
    assert len(result["facts"]) == 1
    assert result["facts"][0]["evidence_refs"] == ["a", "b"]
    state["facts"][0]["evidence_refs"] = []
    with pytest.raises(PrismError, match="need evidence"):
        normalize_state(state, {"a", "b"})


def test_oversize_state_is_ranked_bounded_and_marks_omissions():
    facts = [
        {"text": f"Unrelated sensor {i} is stable.", "evidence_refs": [f"r{i}"]}
        for i in range(12)
    ]
    facts.append(
        {
            "text": "Standard deployments need manager approval.",
            "evidence_refs": ["important"],
        }
    )
    state = ReasoningState(facts=facts).model_dump()
    result, pruned = bound_state(
        state,
        {f"r{i}" for i in range(12)} | {"important"},
        512,
        "Who approves standard deployments?",
    )
    assert byte_tokens(result) <= 512 and pruned > 0
    assert "important" in result["evidence_refs"]
    assert result["open_questions"]


def test_evidence_store_rejects_paths_and_detects_changed_markdown(tmp_path):
    from prism.context import SourceArena

    arena = SourceArena(
        [{"role": "user", "content": "<prism-source>Fact.</prism-source>"}]
    )
    record = {"record_id": "r", "fact": "Fact.", "quote": "Fact.", "source_ref": {}}
    store = EvidenceStore(arena, [record])
    try:
        with pytest.raises(PrismError):
            store.read(str(tmp_path / "outside.md"))
        path = store.root / store.index["r"]["path"]
        path.write_text("changed")
        with pytest.raises(PrismError, match="changed"):
            store.read("r")
    finally:
        store.close()


def test_lazy_search_recovers_unmapped_utf8_facts_with_exact_excerpt_spans():
    from prism.context import SourceArena

    text = (
        "Telemetry stays stable.\n" * 100
        + "Standard deployments require José approval.\n"
        + "Telemetry stays stable.\n" * 100
    )
    arena = SourceArena(
        [{"role": "user", "content": "<prism-source>" + text + "</prism-source>"}]
    )
    store = EvidenceStore(arena, [])  # Mapper missed every relevant fact.
    try:
        excerpts = store.retrieve([], "standard_approval", 768)
        assert excerpts and "José approval" in excerpts[0]["quote"]
        assert byte_tokens(excerpts) <= 768
        ref = excerpts[0]["source_ref"]
        assert (
            arena.sources[ref["source_id"]]
            .data[ref["byte_start"] : ref["byte_end"]]
            .decode()
            == excerpts[0]["quote"]
        )
        assert excerpts[0]["excerpt_truncated"]
        assert excerpts[0]["evidence_pointer"].startswith("source-0000.md#L")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_reduction_helpers_respect_request_model_permissions(monkeypatch):
    from .test_optimization import body
    from .test_optimization import setup as optimized_setup

    app, upstream, physical, config, _ = optimized_setup(
        monkeypatch, allowed_policies=["evidence_map"]
    )
    profile = config.profiles["prism"]
    profile.evidence_reduction = EvidenceReduction(model="cheap", lookup_rounds=0)
    try:
        with pytest.raises(PrismError, match="excludes the configured reducer"):
            app.state.engine.prepare(
                body(
                    context_management=[
                        {"prism_cost_priority": 0.5, "prism_model_ids": ["strong"]}
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": "Summarize qualifications. <prism-source>Manager approval is required.</prism-source>",
                        }
                    ],
                )
            )
        assert not physical.calls
    finally:
        await upstream.aclose()
