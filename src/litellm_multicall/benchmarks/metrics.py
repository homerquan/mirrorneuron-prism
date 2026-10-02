"""Every planned unit stays in denominators; unknown resources stay unknown."""

import math
import statistics


def compute(outcomes, measurements, planned_count, labeled):
    completed = [r for r in outcomes if r["status"] == "completed"]
    accepted = [r for r in completed if r["disposition"] == "accept"]
    known = [r for r in outcomes if r.get("correct") is not None]
    errors = sum(not r["correct"] for r in accepted if r.get("correct") is not None)
    latencies = sorted(
        m["total_ms"]
        for m in measurements
        if m.get("role") == "measured" and "total_ms" in m
    )

    def total(name):
        values = [m.get(name) for m in measurements]
        return sum(values) if values and all(v is not None for v in values) else None

    return {
        "planned_decisions": planned_count,
        "recorded_decisions": len(outcomes),
        "completed_decisions": len(completed),
        "failed_decisions": planned_count - len(completed),
        "independent_groups": len({r["group_id"] for r in outcomes}),
        "accuracy": sum(bool(r.get("correct")) for r in outcomes) / planned_count
        if labeled and planned_count
        else None,
        "eligible_labeled_decisions": planned_count if labeled else 0,
        "known_labeled_decisions": len(known),
        "scored_coverage": sum(r.get("winner_option_id") is not None for r in completed)
        / planned_count
        if planned_count
        else None,
        "accepted_coverage": len(accepted) / planned_count if planned_count else None,
        "selective_risk": errors / len(accepted) if accepted and labeled else None,
        "latency_ms": {
            "median_per_batch": statistics.median(latencies) if latencies else None,
            "p95_per_batch": latencies[
                min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
            ]
            if latencies
            else None,
        },
        "controller_process_cpu_seconds": total("process_cpu_seconds"),
        "logical_input_tokens": total("logical_input_tokens"),
        "startup_process_cpu_seconds": total("startup_process_cpu_seconds"),
        "model_forward_passes": total("model_forward_passes"),
        "generated_output_tokens": total("generated_output_tokens"),
        "backend_duration_sum_ms": total("backend_total_ms"),
        "gpu_active_seconds": None,
        "energy_joules": None,
        "cost_usd": None,
        "usage_complete": all(m.get("usage_complete", True) for m in measurements),
        "limitations": [
            "Native decisions, not end-to-end task success",
            "Resource totals include explicit warmup; durations are sums, not wall time",
            "Unknown energy/GPU active time/cost remain null",
        ],
    }


def paired_quality(baseline, candidate, margin=0.02, minimum_groups=20):
    left = {(r["id"], r["trial"]): r for r in baseline}
    right = {(r["id"], r["trial"]): r for r in candidate}
    if left.keys() != right.keys() or any(
        r.get("correct") is None for r in list(left.values()) + list(right.values())
    ):
        return {
            "status": "invalid",
            "reason": "incomplete paired labeled outcomes",
            "lower_bound": None,
        }
    grouped = {}
    for key, a in left.items():
        b = right[key]
        if a["group_id"] != b["group_id"]:
            return {
                "status": "invalid",
                "reason": "source groups differ",
                "lower_bound": None,
            }
        grouped.setdefault(a["group_id"], []).append(
            int(b["correct"]) - int(a["correct"])
        )
    values = [statistics.mean(v) for v in grouped.values()]
    n = len(values)
    if not n:
        return {"status": "invalid", "reason": "empty denominator", "lower_bound": None}
    mean = statistics.mean(values)
    # Distribution-free independent-group bound for differences in [-1,1].
    lower = max(-1, mean - math.sqrt(2 * math.log(20) / n))
    return {
        "status": "established"
        if n >= minimum_groups and lower >= -margin
        else "not_established",
        "mean_group_quality_difference": mean,
        "lower_bound": lower,
        "independent_groups": n,
        "paired_trials": len(left),
        "minimum_independent_groups": minimum_groups,
        "margin": margin,
        "method": "paired source-group Hoeffding one-sided alpha=0.05 (conservative; no zero-width boundary intervals)",
    }
