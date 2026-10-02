"""Different execution graphs, constrained routing, coverage, and streaming."""

import json

import httpx
import pytest

from prism.api import create_app
from prism.backends import OpenAIBackend
from prism.config import DecisionConfig, PrismConfig, RawModel
from prism.contracts import PUBLIC_PARAMETERS, check_context
from prism.decision import LayaDecision
from prism.policies import POLICIES

from .test_proxy import completion

SOURCES = [
    "Standard deployments require manager approval.",
    "Emergency deployments require retrospective review within one business day.",
    "Rollback during an outage requires incident commander notification.",
]


class Agent:
    def __init__(self, choice="batched_map", probability=0.99, truncated=False):
        self.choice, self.probability, self.truncated = choice, probability, truncated
        self.calls = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions))
        return {
            "answers": {
                "strategy": {
                    "choice": self.choice,
                    "answer_confidence": self.probability,
                    "confidence": 1.0,
                }
            },
            "usage": {"truncated": self.truncated},
        }


class Backend:
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
                            "delta": {"content": "answer"},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 20},
                },
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n",
            )
        if body.get("response_format", {}).get("type") != "json_object":
            return httpx.Response(
                200,
                json=completion(
                    "manager approval; retrospective review; rollback notification"
                ),
            )
        packet = json.loads(body["messages"][-1]["content"])
        if "records" in packet:
            ids = [record["record_id"] for record in packet["records"]]
            verdict = {
                "supported": self.failure != "unsupported",
                "checked_record_ids": ids,
                "issues": [],
            }
            if self.failure == "missing_record":
                verdict["checked_record_ids"] = []
            if self.failure == "duplicate_record":
                verdict["checked_record_ids"] += ids
            if self.failure == "issues":
                verdict["issues"] = ["lost qualification"]
            if self.failure == "string_supported":
                verdict["supported"] = "true"
            return httpx.Response(
                200,
                json=completion(
                    json.dumps(verdict),
                    "length" if self.failure == "verify_truncated" else "stop",
                ),
            )

        def artifact(part):
            return {
                "status": "complete",
                "needs": [],
                "records": [{"quote": part["source"], "fact": part["source"]}],
            }

        if "partitions" in packet:
            parts = [
                {"partition_id": part["partition_id"], **artifact(part)}
                for part in packet["partitions"]
            ]
            if self.failure == "missing_partition":
                parts.pop()
            if self.failure == "duplicate_partition":
                parts.append(parts[0])
            if self.failure == "cross_partition_quote":
                parts[0]["records"][0]["quote"] = packet["partitions"][1]["source"]
            if self.failure == "needs":
                parts[0]["needs"] = ["missing definition"]
            result = {"partitions": parts}
        else:
            result = artifact(packet)
        return httpx.Response(200, json=completion(json.dumps(result)))


def setup(agent=None, failure=None, mode="route"):
    models = {
        identifier: RawModel(
            id=identifier, name="physical-" + identifier, base_url="http://fixture/v1"
        )
        for identifier in ("worker", "synth", "verify")
    }
    profiles = {
        "prism": {
            "direct": "worker",
            "worker": "worker",
            "synthesizer": "synth",
            "verifier": "verify",
        }
    }
    for name in POLICIES:
        profiles[name] = {**profiles["prism"], "strategy": name}
    profiles["retrieve_read"]["coverage"] = "focused"
    profiles["retrieve_read"]["retrieval_top_k"] = 1
    config = PrismConfig(profiles=profiles, decision=DecisionConfig(mode=mode))
    physical = Backend(failure)
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(physical.respond))
    app = create_app(
        config,
        models=models,
        backend=OpenAIBackend(models, upstream),
        decision_agent=agent,
    )
    return app, upstream, physical, config, models


def request_body(model="prism", query="Summarize deployment qualifications."):
    blocks = "".join(
        f'<prism-source id="source-{index}">{text}</prism-source>'
        for index, text in enumerate(SOURCES)
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": query
                + "\n"
                + blocks
                + "\nKeep the conditions attached to requirements.",
            }
        ],
        "max_completion_tokens": 512,
    }


async def send(app, body):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://prism",
        headers={"authorization": "Bearer policy-key"},
    ) as client:
        response = await client.post("/v1/chat/completions", json=body)
        trace = None
        if "x-request-id" in response.headers:
            trace = (
                await client.get("/v1/prism/traces/" + response.headers["x-request-id"])
            ).json()
        return response, trace


@pytest.fixture(autouse=True)
def credential(monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "policy-key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,calls",
    [
        ("direct", 1),
        ("evidence_map", 4),
        ("batched_map", 2),
        ("verified_map", 7),
        ("retrieve_read", 1),
    ],
)
async def test_policies_have_distinct_graphs_and_full_physical_accounting(
    policy, calls
):
    agent = Agent()
    app, upstream, backend, _, models = setup(agent)
    body = request_body(
        policy,
        "Which deployment requires retrospective review?"
        if policy == "retrieve_read"
        else "Summarize deployment qualifications.",
    )
    async with upstream:
        response, trace = await send(app, body)
    assert response.status_code == 200, response.text
    assert trace["strategy"] == policy and response.headers["x-prism-policy"] == policy
    assert len(backend.calls) == len(trace["execution_usage"]["calls"]) == calls
    assert not trace["execution_usage"]["outstanding_nodes"]
    assert not agent.calls  # No ranking overhead for a forced single choice.
    for call in backend.calls:
        model = next(model for model in models.values() if model.name == call["model"])
        check_context(
            call["messages"],
            {key: value for key, value in call.items() if key in PUBLIC_PARAMETERS},
            call["max_completion_tokens"],
            model,
        )
    if policy in {"evidence_map", "batched_map", "verified_map"}:
        assert (
            len(trace["coverage"]["required_partitions"])
            == len(trace["coverage"]["validated_partitions"])
            == 3
        )
        evidence = json.loads(backend.calls[-1]["messages"][-1]["content"])[
            "prism_evidence"
        ]
        assert [record["quote"] for record in evidence] == SOURCES
        if policy == "verified_map":
            assert len(trace["verification"]["verified_partitions"]) == 3
            assert (
                len([node for node in trace["plan"] if node["operator"] == "verify"])
                == 3
            )
    if policy == "retrieve_read":
        assert (
            trace["coverage"]["scope"]
            == response.headers["x-prism-coverage"]
            == "focused"
        )
        assert trace["retrieval"]["selected_partitions"] == ["partition-1"]
        assert len(trace["retrieval"]["unselected_partitions"]) == 2
        view = json.loads(backend.calls[0]["messages"][-1]["content"])[
            "prism_retrieved_sources"
        ]
        assert [part["text"] for part in view] == [SOURCES[1]]
    assert not any(text in json.dumps(trace) for text in SOURCES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["missing_partition", "duplicate_partition", "cross_partition_quote", "needs"],
)
async def test_batch_cannot_skip_sources_or_borrow_sibling_quotes(failure):
    app, upstream, backend, _, _ = setup(Agent(), failure)
    async with upstream:
        response, trace = await send(app, request_body("batched_map"))
    assert response.status_code == 502
    assert trace["stop_reason"] in {"invalid_evidence", "incomplete_evidence"}
    assert len(backend.calls) == 1 and backend.calls[0]["model"] == "physical-worker"
    assert not trace["execution_usage"]["outstanding_nodes"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "unsupported",
        "missing_record",
        "duplicate_record",
        "issues",
        "string_supported",
        "verify_truncated",
    ],
)
async def test_verification_failure_never_synthesizes_or_changes_evidence(failure):
    app, upstream, backend, _, _ = setup(Agent(), failure)
    async with upstream:
        response, trace = await send(app, request_body("verified_map"))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unverified_evidence"
    assert "physical-synth" not in [call["model"] for call in backend.calls]
    assert not trace["execution_usage"]["outstanding_nodes"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice,probability,truncated,selected",
    [
        ("batched_map", 0.99, False, "batched_map"),
        ("verified_map", 0.99, False, "verified_map"),
        ("evidence_map", 0.1, False, "direct"),
        ("evidence_map", True, False, "direct"),
        ("evidence_map", None, False, "direct"),
        ("evidence_map", float("nan"), False, "direct"),
        ("evidence_map", 0.99, True, "direct"),
        ("retrieve_read", 0.99, False, "direct"),
        ("abstain", 0.99, False, "direct"),
    ],
)
async def test_laya_routes_only_allowed_usable_proposals(
    choice, probability, truncated, selected
):
    agent = Agent(choice, probability, truncated)
    app, upstream, _, _, _ = setup(agent)
    async with upstream:
        response, trace = await send(app, request_body())
    assert response.status_code == 200, response.text
    assert trace["strategy"] == selected and len(agent.calls) == 1
    state, questions = agent.calls[0]
    assert set(questions["strategy"]["criteria"]) == {
        "direct",
        "evidence_map",
        "batched_map",
        "verified_map",
        "abstain",
    }
    assert "retrieve_read" not in state["eligible_policies"]
    assert not any(source in json.dumps(state) for source in SOURCES)
    assert (
        trace["decision"].get("quality_calibrated") is False
        or trace["decision"]["reason"] == "decision_unavailable"
    )


@pytest.mark.asyncio
async def test_budget_removes_expensive_choices_before_laya_and_before_dispatch():
    agent = Agent("batched_map")
    app, upstream, backend, config, _ = setup(agent)
    config.profiles["prism"].limits.max_calls = 2
    async with upstream:
        response, trace = await send(app, request_body())
    assert response.status_code == 200, response.text
    assert set(agent.calls[0][1]["strategy"]["criteria"]) == {
        "direct",
        "batched_map",
        "abstain",
    }
    assert len(backend.calls) == 2
    assert trace["ineligible_policies"]["verified_map"] in {
        "invalid_plan",
        "resource_limit",
    }
    agent = Agent()
    app, upstream, backend, config, _ = setup(agent)
    config.profiles["batched_map"].limits.max_calls = 1
    async with upstream:
        response, _ = await send(app, request_body("batched_map"))
    assert response.status_code in {400, 413} and not backend.calls and not agent.calls


@pytest.mark.asyncio
async def test_routing_before_stream_commit_and_controller_runs_once():
    for policy, mode in [("batched_map", "buffered"), ("direct", "forwarded")]:
        agent = Agent(policy)
        app, upstream, backend, _, _ = setup(agent)
        body = {
            **request_body(),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with upstream:
            response, trace = await send(app, body)
        assert response.status_code == 200, response.text
        assert response.headers["x-prism-stream-mode"] == mode
        assert response.headers["x-prism-policy"] == policy
        assert response.text.rstrip().endswith("data: [DONE]")
        assert len(agent.calls) == 1 and trace["strategy"] == policy
        assert all(call["stream"] == (policy == "direct") for call in backend.calls)


@pytest.mark.asyncio
async def test_shadow_and_direct_only_parameters_cannot_change_execution():
    agent = Agent("verified_map")
    app, upstream, backend, _, _ = setup(agent, mode="shadow")
    async with upstream:
        response, trace = await send(app, request_body())
    assert response.status_code == 200 and trace["strategy"] == "direct"
    assert trace["decision"]["reason"] == "shadow_mode" and len(backend.calls) == 1
    agent = Agent("verified_map")
    app, upstream, backend, _, models = setup(agent)
    body = {**request_body(), "seed": 42}
    models["worker"].capabilities.add("seed")
    async with upstream:
        response, trace = await send(app, body)
    assert response.status_code == 200 and not agent.calls
    assert (
        backend.calls[0]["messages"] == body["messages"]
        and backend.calls[0]["seed"] == 42
    )
    assert trace["eligible_policies"] == ["direct"]


@pytest.mark.asyncio
async def test_retrieval_has_explicit_scope_and_no_match_is_not_absence():
    agent = Agent("retrieve_read")
    app, upstream, backend, config, _ = setup(agent)
    config.profiles["prism"].coverage = "focused"
    config.profiles["prism"].retrieval_top_k = 1
    async with upstream:
        response, trace = await send(
            app, request_body(query="Which deployment requires retrospective review?")
        )
        missing, _ = await send(
            app, request_body("retrieve_read", "Locate the violet zeppelin.")
        )
    assert response.status_code == 200 and trace["strategy"] == "retrieve_read"
    assert response.headers["x-prism-coverage"] == "focused"
    assert "retrieve_read" in agent.calls[0][1]["strategy"]["criteria"]
    assert (
        missing.status_code == 400
        and missing.json()["error"]["code"] == "retrieval_no_match"
    )
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_direct_only_request_respects_operator_allowlist():
    agent = Agent()
    app, upstream, backend, config, models = setup(agent)
    models["worker"].capabilities.add("seed")
    config.profiles["prism"].allowed_policies = ["batched_map"]
    async with upstream:
        response, _ = await send(app, {**request_body(), "seed": 42})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"
    assert not agent.calls and not backend.calls


@pytest.mark.asyncio
async def test_batch_size_bound_produces_two_complete_groups():
    app, upstream, backend, config, _ = setup(Agent())
    config.profiles["batched_map"].batch_max_partitions = 2
    async with upstream:
        response, trace = await send(app, request_body("batched_map"))
    assert response.status_code == 200, response.text
    assert len(backend.calls) == 3
    assert len(trace["coverage"]["validated_partitions"]) == 3
    view = json.loads(backend.calls[-1]["messages"][-1]["content"])["prism_evidence"]
    assert [record["quote"] for record in view] == SOURCES


@pytest.mark.asyncio
async def test_required_checkpoint_failure_prevents_service_startup(monkeypatch):
    app, upstream, backend, config, _ = setup()
    assert config.decision.mode == "route"
    with pytest.raises(ValueError):
        DecisionConfig(mode="off")

    def unavailable(decision):
        raise ValueError("checkpoint unavailable")

    monkeypatch.setattr(LayaDecision, "prepare", unavailable)
    async with upstream:
        with pytest.raises(ValueError, match="checkpoint unavailable"):
            async with app.router.lifespan_context(app):
                pytest.fail("service must not become ready without its required model")
    assert not backend.calls
