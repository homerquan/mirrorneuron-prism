"""Repeatable runner fixtures, admission evidence, and owned-process cleanup."""

import argparse
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from prism.config import load_config
from prism.engine import ExecutionEngine


@pytest.fixture
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/standalone/docker-spark/benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("docker_spark_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_large_fixture_forces_automatic_lossless_partitioning(
    runner, tmp_path
):
    path = runner.configure(
        tmp_path,
        "http://local/v1",
        "http://remote/v1",
        "gemma",
        "nemotron",
        18080,
        1024,
    )
    config, models = load_config(path)
    engine = ExecutionEngine(config, models, None, None)
    cases = runner.large_cases()
    assert len(cases) == 3
    for case in cases:
        assert len(case["request"]["messages"][0]["content"].encode()) == 12786
        body = {
            **case["request"],
            "model": "reverse-auto-small",
            "max_completion_tokens": 1024,
        }
        execution = engine.prepare(body)
        assert "direct" not in execution["policy_plans"]
        assert (
            execution["trace"]["ineligible_policies"]["direct"]
            == "context_length_exceeded"
        )
        compiled = execution["policy_plans"]["batched_map"]
        assert len(compiled["partitions"]) >= 6
        assert len(compiled["jobs"]) > 1
        assert "".join(p.text for p in compiled["partitions"]) == "".join(
            execution["arena"].resolve(ref) for ref in execution["arena"].documents
        )
        assert compiled["plan"].nodes[-1].operator == "synthesize"


def test_inline_fixture_preserves_reference_rubrics(runner):
    original = [case.model_dump(mode="json") for case in runner.load_cases()]
    inline = runner.inline_cases()
    assert len(inline) == len(original) == 6
    for before, after in zip(original, inline, strict=True):
        assert before["checks"] == after["checks"]
        assert "<prism-source" not in after["request"]["messages"][0]["content"]
        assert (
            before["request"]["response_format"] == after["request"]["response_format"]
        )


@pytest.mark.asyncio
async def test_failure_cleans_up_owned_server_and_tunnel(runner, monkeypatch, tmp_path):
    processes = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.terminated = False
            processes.append(self)

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=0):
            return 0

    async def ready(url, process, timeout):
        return (
            {
                "data": [
                    {"id": "ai/nemotron-3.5-lightning:latest"},
                    {"id": "ai/gemma4:E2B"},
                ]
            }
            if url.endswith("/models")
            else {"status": "ok"}
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "wait_ready", ready)
    monkeypatch.setattr(runner, "runtime_configuration", lambda *args: [])

    async def probes(models, timeout):
        return []

    monkeypatch.setattr(runner, "probe_models", probes)
    monkeypatch.setattr(
        runner.httpx,
        "AsyncClient",
        lambda **kw: original_client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, json={"data": [{"id": "ai/gemma4:E2B"}]}
                )
            ),
            **kw,
        ),
    )

    async def failed_run(**kwargs):
        raise RuntimeError("injected benchmark failure")

    monkeypatch.setattr(runner, "run", failed_run)
    args = argparse.Namespace(
        out_dir=str(tmp_path / "new-run"),
        tunnel_port=12435,
        proxy_port=18080,
        remote_port=12434,
        spark="spark",
        startup_timeout=5,
        local_url="http://local/v1",
        local_model=None,
        remote_model=None,
        metadata=None,
        nemotron_worker_tokens=1024,
        suites="direct",
        repeats=1,
        warmup=0,
        output_tokens=8192,
        timeout=10,
        max_cases=None,
        review_max_cases=2,
    )
    with pytest.raises(RuntimeError, match="injected benchmark failure"):
        await runner.execute(args)
    assert len(processes) == 2
    assert all(process.terminated for process in processes)
    saved = json.loads((tmp_path / "new-run/prism.json").read_text())
    assert saved["server"]["api_key_env"] == "PRISM_BENCHMARK_KEY"
    assert "api_key" not in (tmp_path / "new-run/models.json").read_text()
    with pytest.raises(FileExistsError):
        await runner.execute(args)
    assert len(processes) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["gemma", "nemotron", "spark-gemma"])
@pytest.mark.parametrize(
    "policy", ["evidence_map", "batched_map", "verified_map", "retrieve_read"]
)
async def test_fixed_policy_matrix_executes_complete_graph(
    runner, tmp_path, family, policy
):
    from prism.backends import OpenAIBackend
    from prism.decision import LayaDecision

    from .test_optimization import PhysicalBackend

    path = runner.configure(
        tmp_path,
        "http://local/v1",
        "http://remote/v1",
        "gemma",
        "nemotron",
        18080,
        1024,
    )
    config, models = load_config(path)
    physical = PhysicalBackend()

    def respond(request):
        response = physical.respond(request)
        packet = json.loads(request.content)["messages"][-1]["content"]
        if policy == "verified_map" and "recovery_instruction" in packet:
            # The generic fixture copies a full source into each quote. A recovery
            # extractor returns a narrow original span, as real extraction should.
            data = response.json()
            artifact = json.loads(data["choices"][0]["message"]["content"])
            artifact["records"][0]["quote"] = json.loads(packet)["source"].splitlines()[
                0
            ]
            data["choices"][0]["message"]["content"] = json.dumps(artifact)
            return httpx.Response(200, json=data)
        return response

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    backend = OpenAIBackend(models, client=client)
    try:
        engine = ExecutionEngine(config, models, backend, LayaDecision(config.decision))
        case = (
            runner.focused_cases()
            if policy == "retrieve_read"
            else runner.large_cases()
        )[0]
        alias = f"small-{policy}-{family}"
        execution = engine.prepare(
            {**case["request"], "model": alias, "max_completion_tokens": 1024}
        )
        assert list(execution["policy_plans"]) == [policy]
        await engine.execute(execution)
        trace = execution["trace"]
        assert trace["strategy"] == policy
        assert trace["decision"]["disposition"] == "rules_only"
        assert trace["coverage"]["scope"] == (
            "focused" if policy == "retrieve_read" else "full"
        )
        assert set(trace["coverage"]["validated_partitions"]) == set(
            trace["coverage"]["required_partitions"]
        )
        calls = trace["execution_usage"]["calls"]
        assert len(calls) <= execution["policy_plans"][policy]["calls"]
        profile = config.profiles[alias]
        operators = {
            node["id"]: node["operator"]
            for node in [*trace["plan"], *trace.get("compaction_plan", [])]
        }
        assignments = {
            "extract": profile.worker,
            "verify": profile.verifier,
            "synthesize": profile.synthesizer,
            "compact": profile.evidence_compaction.model
            if profile.evidence_compaction
            else None,
        }
        recovery_nodes = {
            n["id"]: n["model_id"] for n in trace.get("recovery_plan", [])
        }
        assert all(
            call["model_id"]
            == (
                recovery_nodes[call["node_id"]]
                if call["node_id"] in recovery_nodes
                else assignments[operators[call["node_id"]]]
            )
            for call in calls
        )
        audit = runner.policy_audit(
            [{"trace": trace, "completed": True}], config.profiles[alias].model_dump()
        )
        assert (
            audit["selection_mismatches"] == audit["unexpected_completed_graphs"] == 0
        )
        assert audit["completed_graphs"] == audit["valid_plans"] == 1
        if policy == "verified_map":
            assert len(trace["verification"]["verified_partitions"]) >= 6
            assert sum(call["node_id"].startswith("verify-") for call in calls) == len(
                trace["verification"]["required_partitions"]
            )
        if policy == "retrieve_read":
            assert len(calls) == 1
            assert len(trace["coverage"]["required_partitions"]) <= 2
            assert len(trace["coverage"]["required_partitions"]) < 6
        else:
            partitions = execution["policy_plans"][policy]["partitions"]
            assert len(partitions) >= 6
            if policy == "batched_map":
                assert any(
                    len(job["partitions"]) > 1
                    for job in execution["policy_plans"][policy]["jobs"]
                )
    finally:
        await client.aclose()


def test_suite_groups_and_policy_audit_detect_wrong_or_missing_execution(runner):
    suites = runner.expand_suites("policies,large-evidence-gemma")
    assert len(suites) == 9
    assert "review" in suites
    assert all(
        any(policy in name for name in suites)
        for policy in ["evidence", "batched", "verified", "retrieve"]
    )
    assert len(runner.expand_suites("all")) == len(runner.SUITES)
    with pytest.raises(ValueError):
        runner.expand_suites("unknown")
    trace = {
        "strategy": "batched_map",
        "plan": [{"id": "synthesis", "operator": "synthesize"}],
        "execution_usage": {"calls": [{"node_id": "synthesis", "status": "complete"}]},
    }
    audit = runner.policy_audit(
        [{"trace": trace, "completed": True}, {"trace": None, "completed": False}],
        {"strategy": "verified_map"},
    )
    assert audit["selection_mismatches"] == 1
    assert audit["unexpected_completed_graphs"] == 1
    assert audit["completed_graphs"] == 0
    assert audit["requests_without_trace"] == 1
    trace = {
        "strategy": "verified_map",
        "plan": [
            {"id": node, "operator": operator}
            for node, operator in [
                ("extract", "extract"),
                ("verify-a", "verify"),
                ("verify-b", "verify"),
                ("final", "synthesize"),
            ]
        ],
        "execution_usage": {
            "calls": [
                {"node_id": node, "status": "complete"}
                for node in ["extract", "verify-a", "final"]
            ]
        },
    }
    audit = runner.policy_audit(
        [{"trace": trace, "completed": True}], {"strategy": "verified_map"}
    )
    assert audit["valid_plans"] == 1
    assert audit["completed_graphs"] == 0
    assert audit["unexpected_completed_graphs"] == 1
