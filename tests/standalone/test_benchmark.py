"""Reference scoring, full physical accounting, durable artifacts, and CLI runs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from prism.api import create_app
from prism.backends import OpenAIBackend
from prism.benchmark import Case, compare, grade, load_cases, run, sample, work_and_cost
from prism.cli import main
from prism.config import PrismConfig, RawModel
from prism.errors import PrismError

from .decision_stub import RulesAgent
from .test_http import serve


def reference_case():
    return {
        "id": "reference",
        "description": "Known numeric fact in two sources",
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": 'Return value as JSON. <prism-source id="a">The value is 7.</prism-source><prism-source id="b">The unit is seconds.</prism-source>',
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "value",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        "checks": [{"type": "json_equals", "path": ["value"], "value": 7}],
    }


def configuration(tmp_path, backend_url="http://fixture"):
    models = {
        identifier: RawModel(
            id=identifier,
            name="physical-" + identifier,
            base_url=backend_url + "/v1",
            api_key="backend-secret",
            input_cost_per_million=1,
            output_cost_per_million=2,
        )
        for identifier in ("worker", "synth")
    }
    config = PrismConfig(
        profiles={
            "prism-direct": {"direct": "worker", "strategy": "direct"},
            "prism-evidence": {
                "direct": "worker",
                "worker": "worker",
                "synthesizer": "synth",
                "strategy": "evidence_map",
            },
        }
    )
    (tmp_path / "models.json").write_text(
        json.dumps(
            {"models": [model.model_dump(mode="json") for model in models.values()]}
        )
    )
    config_path = tmp_path / "prism.json"
    config_path.write_text(config.model_dump_json())
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(reference_case()) + "\n")
    return config, models, config_path, cases_path


def respond(body):
    if body.get("response_format", {}).get("type") == "json_object":
        partition = json.loads(body["messages"][-1]["content"])
        content = {
            "status": "complete",
            "needs": [],
            "records": [{"quote": partition["source"], "fact": partition["source"]}],
        }
    else:
        # Deliberately different quality to establish an independently known delta.
        content = {"value": 7 if body["model"] == "physical-synth" else 6}
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_packaged_reference_suite_and_quality_edge_cases():
    cases = load_cases()
    assert len(cases) == 6
    assert {tag for case in cases for tag in case.tags} >= {
        "unicode",
        "units",
        "negation",
        "dates",
        "cross-source",
    }
    assert (
        grade(
            cases[0],
            json.dumps(
                {
                    "standard_approver": "manager",
                    "emergency_authority": "incident_commander",
                    "retrospective_business_days": 1,
                    "rollback_during_outage": True,
                }
            ),
            True,
        )["score"]
        == 1
    )
    case = Case.model_validate(reference_case())
    assert grade(case, '{"value":7}', True)["passed"]
    assert grade(case, '{"value":7}', False)["score"] == 0
    assert not grade(case, '{"value":6}', True)["passed"]
    assert not grade(case, "not JSON", True)["passed"]
    raw = reference_case()
    raw["checks"] = [{"type": "json_equals", "path": ["value"], "value": None}]
    assert not grade(Case.model_validate(raw), "not JSON", True)["passed"]
    raw["checks"][0]["value"] = True
    assert not grade(Case.model_validate(raw), '{"value":1}', True)["passed"]
    raw["checks"] = [
        {
            "type": "contains_any",
            "terms": ["Manager approval", "supervisor authorization"],
        },
        {"type": "not_contains", "terms": ["approval optional"]},
    ]
    assert grade(Case.model_validate(raw), "MANAGER\nAPPROVAL is required.", True)[
        "passed"
    ]
    assert (
        grade(Case.model_validate(raw), "Manager approval optional", True)["score"]
        == 0.5
    )


@pytest.mark.parametrize(
    "field,value",
    [("stream", True), ("model", "override"), ("max_completion_tokens", 99)],
)
def test_cases_cannot_change_comparison_controls(field, value):
    raw = reference_case()
    raw["request"][field] = value
    with pytest.raises(ValueError):
        Case.model_validate(raw)


@pytest.mark.asyncio
async def test_paired_artifacts_quality_cost_warmup_and_comparison(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("PRISM_API_KEY", "caller-secret")
    config, models, config_path, cases_path = configuration(tmp_path)

    def physical(request):
        assert request.headers["authorization"] == "Bearer backend-secret"
        return httpx.Response(200, json=respond(json.loads(request.content)))

    async with httpx.AsyncClient(transport=httpx.MockTransport(physical)) as upstream:
        app = create_app(config, models=models, backend=OpenAIBackend(models, upstream))
        result = await run(
            api_key="caller-secret",
            config_path=config_path,
            cases_path=cases_path,
            base_url="http://prism/v1",
            out_dir=tmp_path / "first",
            repeats=2,
            warmup=1,
            transport=httpx.ASGITransport(app=app),
        )
        second = await run(
            api_key="caller-secret",
            config_path=config_path,
            cases_path=cases_path,
            base_url="http://prism/v1",
            out_dir=tmp_path / "second",
            repeats=2,
            warmup=1,
            transport=httpx.ASGITransport(app=app),
        )
    folder = Path(result["run_dir"])
    assert {path.name for path in folder.iterdir()} == {
        "manifest.json",
        "cases.jsonl",
        "requests.jsonl",
        "warmup.jsonl",
        "summary.json",
        "report.md",
    }
    rows = [
        json.loads(line)
        for line in (folder / "requests.jsonl").read_text().splitlines()
    ]
    assert [row["route"] for row in rows] == [
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert len((folder / "warmup.jsonl").read_text().splitlines()) == 2
    summary = result["summary"]
    baseline, candidate = (
        summary["routes"][route] for route in ("baseline", "candidate")
    )
    assert baseline["requests"] == candidate["requests"] == 2
    assert baseline["quality"]["mean_score"] == 0
    assert candidate["quality"]["mean_score"] == 1
    assert candidate["work"]["backend_calls_known"] == 6
    assert candidate["work"]["reported_input_tokens"] == 60
    assert candidate["work"]["reported_output_tokens"] == 30
    assert candidate["cost"]["estimated_total_usd"] == pytest.approx(0.00012)
    assert baseline["cost"]["estimated_total_usd"] == pytest.approx(0.00004)
    assert (
        candidate["cost"]["upper_estimate_total_usd"]
        > candidate["cost"]["estimated_total_usd"]
    )
    assert summary["candidate_vs_baseline"]["quality_score_delta"] == 1
    assert summary["candidate_vs_baseline"]["mean_cost_ratio"] == pytest.approx(3)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["planned_pairs"] == 2
    for path in folder.iterdir():
        assert (
            "caller-secret" not in path.read_text()
            and "backend-secret" not in path.read_text()
        )
    comparison = compare([folder, second["run_dir"]], tmp_path / "comparison")
    assert comparison["comparable_workload"]
    assert (
        comparison["changes_vs_first"][0]["routes"]["candidate"]["quality_score_delta"]
        == 0
    )
    assert (tmp_path / "comparison" / "report.md").exists()
    monkeypatch.delenv("PRISM_API_KEY")
    assert main(["benchmark", "compare", str(folder), second["run_dir"]]) == 0
    assert json.loads(capsys.readouterr().out)["comparable_workload"]
    manifest["settings"]["output_tokens"] = 512
    (folder / "manifest.json").write_text(json.dumps(manifest))
    mismatch = compare([folder, second["run_dir"]])
    assert not mismatch["comparable_workload"] and mismatch["warnings"]
    assert mismatch["changes_vs_first"][0]["routes"]["candidate"] is None
    with pytest.raises(PrismError, match="already exists"):
        await run(api_key="key", out_dir=folder, cases_path=cases_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "length",
        "empty",
        "bad_json",
        "trace_failure",
        "invalid_trace",
        "transport",
        "upstream_error",
    ],
)
async def test_failed_or_untraced_requests_do_not_gain_quality_or_free_cost(mode):
    def physical(request):
        if request.method == "GET":
            if mode == "invalid_trace":
                return httpx.Response(
                    200, json={"request_id": "prism-test", "execution_usage": None}
                )
            return httpx.Response(503, text="unavailable")
        if mode == "transport":
            raise httpx.ReadTimeout("fixture timeout")
        if mode == "bad_json":
            return httpx.Response(
                200, text="invalid", headers={"x-request-id": "prism-test"}
            )
        if mode == "upstream_error":
            return httpx.Response(502, json={"error": {"code": "upstream_error"}})
        return httpx.Response(
            200,
            json={
                "model": "prism-direct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": " " if mode == "empty" else '{"value":7}',
                        },
                        "finish_reason": "length" if mode == "length" else "stop",
                    }
                ],
            },
            headers={"x-request-id": "prism-test"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(physical)) as client:
        record = await sample(
            client,
            Case.model_validate(reference_case()),
            alias="prism-direct",
            base_url="http://fixture/v1",
            output_tokens=2048,
            temperature=0,
            models={},
        )
    assert record["completed"] == (mode in {"trace_failure", "invalid_trace"})
    assert record["quality"]["score"] == (
        1 if mode in {"trace_failure", "invalid_trace"} else 0
    )
    assert record["work"]["estimated_cost_usd"] is None
    assert record["work"]["backend_calls"] is None
    assert record["latency_ms"] >= 0


def test_unknown_usage_missing_prices_and_explicit_zero_rates(tmp_path):
    _, models, _, _ = configuration(tmp_path)
    trace = {
        "execution_usage": {
            "calls": [
                {
                    "model_id": "worker",
                    "provider_usage": None,
                    "cost_upper_estimate_usd": 0.1,
                }
            ]
        }
    }
    result = work_and_cost(trace, models)
    assert result["estimated_cost_usd"] is None
    assert (
        result["cost_upper_estimate_usd"] == 0.1 and result["unknown_usage_calls"] == 1
    )
    trace["execution_usage"]["calls"][0]["provider_usage"] = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    models["worker"].input_cost_per_million = None
    assert work_and_cost(trace, models)["estimated_cost_usd"] is None
    models["worker"].input_cost_per_million = models[
        "worker"
    ].output_cost_per_million = 0
    assert work_and_cost(trace, models)["estimated_cost_usd"] == 0


@pytest.mark.asyncio
async def test_partial_results_survive_interruption(tmp_path):
    _, _, _, cases_path = configuration(tmp_path)

    def stop_after_sample(record):
        raise RuntimeError("simulated interruption")

    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": {"code": "fixture_failure"}})
    )
    folder = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="interruption"):
        await run(
            api_key="key",
            out_dir=folder,
            cases_path=cases_path,
            repeats=2,
            warmup=0,
            transport=transport,
            progress=stop_after_sample,
        )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["measured_requests"] == 1
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["routes"]["baseline"]["quality"]["mean_score"] == 0
    assert summary["routes"]["candidate"]["quality"]["mean_score"] is None
    assert len((folder / "requests.jsonl").read_text().splitlines()) == 1
    assert not compare([folder, folder])["comparable_workload"]


@pytest.mark.integration
def test_benchmark_command_over_http(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", "benchmark-socket-key")
    physical = FastAPI()

    @physical.post("/v1/chat/completions")
    async def chat(request: Request):
        assert request.headers["authorization"] == "Bearer backend-secret"
        return respond(await request.json())

    with serve(physical) as physical_url:
        config, models, config_path, cases_path = configuration(tmp_path, physical_url)
        with serve(
            create_app(config, models=models, decision_agent=RulesAgent())
        ) as url:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "prism",
                    "benchmark",
                    "run",
                    "--base-url",
                    url + "/v1",
                    "--config",
                    str(config_path),
                    "--cases",
                    str(cases_path),
                    "--repeats",
                    "1",
                    "--warmup",
                    "0",
                    "--out-dir",
                    str(tmp_path / "socket-run"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=os.environ.copy(),
            )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["summary"]["routes"]["candidate"]["quality"]["pass_rate"] == 1
    assert result["summary"]["routes"]["candidate"]["work"]["backend_calls_known"] == 3
    assert "measured reference" in completed.stderr
