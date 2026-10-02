"""Paired HTTP evaluation: keep failures and provider work in the denominator."""

import json
import time

import httpx


async def evaluate(cases, *, base_url, api_key, baseline, candidate, output_tokens=512):
    rows = []
    async with httpx.AsyncClient(
        timeout=300, trust_env=False, headers={"authorization": f"Bearer {api_key}"}
    ) as client:
        for index, case in enumerate(cases):
            pair = {"id": case.get("id", str(index)), "routes": {}}
            for label, alias in (("baseline", baseline), ("candidate", candidate)):
                started = time.monotonic()
                try:
                    response = await client.post(
                        base_url.rstrip("/") + "/chat/completions",
                        json={
                            "model": alias,
                            "messages": case["messages"],
                            "max_completion_tokens": output_tokens,
                        },
                    )
                    latency = (time.monotonic() - started) * 1000
                    data = response.json()
                    success = response.status_code == 200
                    text = (
                        data["choices"][0]["message"].get("content", "")
                        if success
                        else ""
                    )
                    expected = case.get("expected_contains")
                    trace_id = response.headers.get("x-request-id")
                    trace = None
                    if trace_id:
                        trace_response = await client.get(
                            base_url.rstrip("/") + "/prism/traces/" + trace_id
                        )
                        if trace_response.status_code == 200:
                            trace = trace_response.json()
                    pair["routes"][label] = {
                        "model": alias,
                        "status": response.status_code,
                        "latency_ms": latency,
                        "completed": success,
                        "correct": (expected in text if success else False)
                        if expected is not None
                        else None,
                        "logical_usage": data.get("usage"),
                        "execution_usage": trace.get("execution_usage")
                        if trace
                        else None,
                        "error_code": data.get("error", {}).get("code"),
                    }
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pair["routes"][label] = {
                        "model": alias,
                        "completed": False,
                        "correct": False
                        if case.get("expected_contains") is not None
                        else None,
                        "error_code": "transport_or_protocol_failure",
                        "latency_ms": (time.monotonic() - started) * 1000,
                    }
            rows.append(pair)
    return rows


def summarize(rows):
    result = {
        "requests": len(rows),
        "quality_metric": "expected_contains fixture check; not general answer correctness",
    }
    for label in ("baseline", "candidate"):
        routes = [row["routes"][label] for row in rows]
        judged = [r for r in routes if r.get("correct") is not None]
        result[label] = {
            "completed": sum(r["completed"] for r in routes),
            "failures": sum(not r["completed"] for r in routes),
            "fixture_accuracy": sum(r["correct"] for r in judged) / len(judged)
            if judged
            else None,
            "mean_latency_ms": sum(r["latency_ms"] for r in routes) / len(routes)
            if routes
            else None,
        }
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
