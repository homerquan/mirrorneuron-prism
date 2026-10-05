import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from prism.evaluation import evaluate, read_jsonl, summarize


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", ["", "test-key"])
async def test_paired_evaluation_retains_http_protocol_and_transport_failures(
    api_key, monkeypatch
):
    calls = []
    client_type = httpx.AsyncClient

    def handler(request):
        calls.append(request)
        assert request.headers.get("authorization") == (
            f"Bearer {api_key}" if api_key else None
        )
        if request.method == "GET":
            return httpx.Response(200, json={"execution_usage": {"calls": []}})
        body = json.loads(request.content)
        if body["model"] == "baseline":
            return httpx.Response(
                200,
                headers={"x-request-id": "prism-test"},
                json={
                    "choices": [{"message": {"content": "Expected answer"}}],
                    "usage": {"total_tokens": 42},
                },
            )
        if body["messages"][0]["content"] == "http":
            return httpx.Response(429, json={"error": {"code": "upstream_rate_limit"}})
        if body["messages"][0]["content"] == "protocol":
            return httpx.Response(200, json={"choices": []})
        raise httpx.ConnectError("private transport detail", request=request)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    )
    rows = await evaluate(
        [
            {
                "id": "http",
                "messages": [{"role": "user", "content": "http"}],
                "expected_contains": "Expected",
            },
            {
                "id": "protocol",
                "messages": [{"role": "user", "content": "protocol"}],
                "expected_contains": "Expected",
            },
            {"id": "network", "messages": [{"role": "user", "content": "network"}]},
        ],
        base_url="http://prism/v1/",
        api_key=api_key,
        baseline="baseline",
        candidate="candidate",
    )
    summary = summarize(rows)
    assert summary["baseline"]["completed"] == 3
    assert summary["baseline"]["fixture_accuracy"] == 1
    assert summary["candidate"]["failures"] == 3
    assert summary["candidate"]["fixture_accuracy"] == 0
    assert rows[0]["routes"]["baseline"]["execution_usage"] == {"calls": []}
    assert rows[0]["routes"]["candidate"]["error_code"] == "upstream_rate_limit"
    assert (
        rows[1]["routes"]["candidate"]["error_code"] == "transport_or_protocol_failure"
    )
    assert rows[2]["routes"]["candidate"]["correct"] is None
    assert "private transport detail" not in json.dumps(rows)
    assert len(calls) == 9
    assert summarize([])["baseline"]["mean_latency_ms"] is None
    assert summarize([])["candidate"]["fixture_accuracy"] is None


def test_read_jsonl_skips_blank_lines(tmp_path):
    file = tmp_path / "cases.jsonl"
    file.write_text('\n{"id":1}\n\n{"id":2}\n')
    assert read_jsonl(file) == [{"id": 1}, {"id": 2}]


def test_pilot_savings_use_matching_known_pairs(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/standalone/openrouter_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("prism_pilot_test", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    records = []
    for index, case in enumerate(pilot.cases()):
        for route, prompt in [("baseline", 10000), ("candidate", 0)]:
            complete = not (index == 0 and route == "candidate")
            records.append(
                {
                    "case_id": case.id,
                    "repeat": 0,
                    "route": route,
                    "completed": complete,
                    "quality": {"passed": complete},
                    "latency_ms": 1000,
                    "trace": {
                        "execution_usage": {
                            "calls": [
                                {
                                    "model_id": "nemotron-super-reasoning",
                                    "provider_usage": {
                                        "prompt_tokens": prompt,
                                        "completion_tokens": 1000,
                                    },
                                }
                            ]
                        }
                    }
                    if complete
                    else None,
                }
            )
    pilot.report(records, tmp_path, 1)
    report = (tmp_path / "benchmark-results.md").read_text()
    assert "Matched completed pairs with known usage: 5/6" in report
    assert "| astra_standard | $0.75000 | $0.25000 | 66.7%" in report
    assert "Unknown-usage requests: 1" in report
    assert "never priced at zero" in (tmp_path / "marketing-narrative.md").read_text()
