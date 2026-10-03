"""Measured stage timing and physical-token subtotals from authenticated traces."""

import math
from collections import Counter, defaultdict


def finite_number(value):
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stage_measurements(trace):
    if not isinstance(trace, dict):
        return {}
    usage = trace.get("execution_usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("calls"), list):
        return {}
    if any(not isinstance(call, dict) for call in usage["calls"]):
        return {}
    operators = {
        node["id"]: node["operator"]
        for node in [
            *(trace.get("plan") or []),
            *(trace.get("recovery_plan") or []),
            *(trace.get("compaction_plan") or []),
            *(trace.get("reduction_plan") or []),
        ]
        if isinstance(node, dict) and "id" in node and "operator" in node
    }
    stages = defaultdict(list)
    for call in usage["calls"]:
        node = call.get("node_id", "")
        stage = operators.get(node, "generate" if node == "direct" else "unknown")
        stages[stage].append(call)
    result = {}
    for stage, calls in stages.items():
        durations = [
            c["elapsed_ms"] for c in calls if finite_number(c.get("elapsed_ms"))
        ]
        intervals = [
            (c["started_ms"], c["finished_ms"])
            for c in calls
            if finite_number(c.get("started_ms"))
            and finite_number(c.get("finished_ms"))
            and c["finished_ms"] >= c["started_ms"]
        ]
        usages = [c.get("provider_usage") for c in calls]
        known = [
            u
            for u in usages
            if isinstance(u, dict)
            and all(
                type(u.get(k)) is int and u[k] >= 0
                for k in ("prompt_tokens", "completion_tokens")
            )
        ]
        result[stage] = {
            "calls": len(calls),
            "backend_completed_calls": sum(
                c.get("status") == "complete" for c in calls
            ),
            "invalid_artifact_calls": sum(
                c.get("artifact_status") == "invalid" for c in calls
            ),
            "models": dict(Counter(c.get("model_id", "unknown") for c in calls)),
            "timed_calls": len(durations),
            "call_elapsed_values_ms": durations,
            "call_elapsed_sum_ms": sum(durations) if durations else None,
            "call_elapsed_mean_ms": sum(durations) / len(durations)
            if durations
            else None,
            "phase_wall_ms": max(end for _, end in intervals)
            - min(start for start, _ in intervals)
            if len(intervals) == len(calls)
            else None,
            "reported_input_tokens": sum(u["prompt_tokens"] for u in known),
            "reported_output_tokens": sum(u["completion_tokens"] for u in known),
            "unknown_usage_calls": len(calls) - len(known),
        }
    decision = trace.get("decision") or {}
    elapsed = decision.get("elapsed_ms")
    result["decision"] = {
        "inference_attempts": int(finite_number(elapsed)),
        "elapsed_ms": elapsed if finite_number(elapsed) else None,
        "disposition": decision.get("disposition", "unavailable"),
        "proposal": decision.get("proposal"),
        "provider_usage": decision.get("usage"),
    }
    return result


def summarize_stages(records):
    grouped = defaultdict(list)
    for record in records:
        for stage, values in stage_measurements(record.get("trace")).items():
            grouped[stage].append(values)
    result = {}
    for stage, rows in grouped.items():
        if stage == "decision":
            times = [r["elapsed_ms"] for r in rows if r["elapsed_ms"] is not None]
            result[stage] = {
                "inference_attempts": sum(r["inference_attempts"] for r in rows),
                "elapsed_sum_ms": sum(times) if times else None,
                "elapsed_mean_ms": sum(times) / len(times) if times else None,
                "elapsed_p50_ms": percentile(times, 0.5),
                "elapsed_p95_ms": percentile(times, 0.95),
                "dispositions": dict(Counter(r["disposition"] for r in rows)),
                "reported_input_tokens": sum(
                    r["provider_usage"].get("input_tokens", 0)
                    for r in rows
                    if isinstance(r.get("provider_usage"), dict)
                    and type(r["provider_usage"].get("input_tokens")) is int
                ),
                "reported_output_tokens": sum(
                    r["provider_usage"].get("output_tokens", 0)
                    for r in rows
                    if isinstance(r.get("provider_usage"), dict)
                    and type(r["provider_usage"].get("output_tokens")) is int
                ),
                "unknown_usage_attempts": sum(
                    r["inference_attempts"]
                    and not (
                        isinstance(r.get("provider_usage"), dict)
                        and all(
                            type(r["provider_usage"].get(k)) is int
                            for k in ("input_tokens", "output_tokens")
                        )
                    )
                    for r in rows
                ),
                "token_usage": "Inspect per-request decision.provider_usage; decision tokens are separate from physical backend tokens.",
            }
            continue
        durations = [
            r["call_elapsed_sum_ms"]
            for r in rows
            if r["call_elapsed_sum_ms"] is not None
        ]
        wall = [r["phase_wall_ms"] for r in rows if r["phase_wall_ms"] is not None]
        timed = sum(r["timed_calls"] for r in rows)
        call_times = [value for row in rows for value in row["call_elapsed_values_ms"]]
        result[stage] = {
            "requests": len(rows),
            "calls": sum(r["calls"] for r in rows),
            "backend_completed_calls": sum(r["backend_completed_calls"] for r in rows),
            "invalid_artifact_calls": sum(r["invalid_artifact_calls"] for r in rows),
            "models": dict(sum((Counter(r["models"]) for r in rows), Counter())),
            "call_elapsed_sum_ms": sum(durations) if durations else None,
            "call_elapsed_mean_ms": sum(durations) / timed if timed else None,
            "call_elapsed_p50_ms": percentile(call_times, 0.5),
            "call_elapsed_p95_ms": percentile(call_times, 0.95),
            "phase_wall_mean_ms": sum(wall) / len(wall) if wall else None,
            "requests_with_wall_timing": len(wall),
            "reported_input_tokens": sum(r["reported_input_tokens"] for r in rows),
            "reported_output_tokens": sum(r["reported_output_tokens"] for r in rows),
            "unknown_usage_calls": sum(r["unknown_usage_calls"] for r in rows),
        }
    return result


def stages_table(summary):
    lines = [
        "| Route | Stage | Model assignments | Calls / inference attempts | Mean call ms | P50 call ms | P95 call ms | Mean phase wall ms | Input tokens | Output tokens | Unknown usage calls |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "unknown" if value is None else f"{value:.2f}"

    for route, data in summary["routes"].items():
        for stage, metrics in data.get("stages", {}).items():
            if stage == "decision":
                lines.append(
                    f"| {route} | decision | Laya CPU | {metrics['inference_attempts']} | {number(metrics['elapsed_mean_ms'])} | {number(metrics['elapsed_p50_ms'])} | {number(metrics['elapsed_p95_ms'])} | {number(metrics['elapsed_mean_ms'])} | {metrics['reported_input_tokens']} (Laya) | {metrics['reported_output_tokens']} (Laya) | {metrics['unknown_usage_attempts']} |"
                )
            else:
                models = (
                    ", ".join(
                        f"{model} ({count})"
                        for model, count in metrics["models"].items()
                    )
                    .replace("|", "\\|")
                    .replace("\n", " ")
                )
                lines.append(
                    f"| {route} | {stage} | {models} | {metrics['calls']} | {number(metrics['call_elapsed_mean_ms'])} | {number(metrics['call_elapsed_p50_ms'])} | {number(metrics['call_elapsed_p95_ms'])} | {number(metrics['phase_wall_mean_ms'])} | {metrics['reported_input_tokens']} | {metrics['reported_output_tokens']} | {metrics['unknown_usage_calls']} |"
                )
    return "\n".join(lines)
