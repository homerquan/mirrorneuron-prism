"""Stage measurements distinguish parallel work, elapsed time, and unknown usage."""

from prism.benchmark_stages import stage_measurements, summarize_stages


def call(node, start, end, usage=None):
    return {
        "node_id": node,
        "model_id": "fixture",
        "status": "complete",
        "started_ms": start,
        "finished_ms": end,
        "elapsed_ms": end - start,
        "provider_usage": usage,
    }


def test_parallel_stage_wall_time_and_decision_are_separate():
    trace = {
        "plan": [
            {"id": "a", "operator": "extract"},
            {"id": "b", "operator": "extract"},
            {"id": "final", "operator": "synthesize"},
        ],
        "decision": {
            "elapsed_ms": 5,
            "disposition": "accept",
            "usage": {"input_tokens": 50, "output_tokens": 0},
        },
        "execution_usage": {
            "calls": [
                call("a", 5, 25, {"prompt_tokens": 10, "completion_tokens": 4}),
                call("b", 10, 30, {"prompt_tokens": 12, "completion_tokens": 6}),
                call("final", 30, 40),
            ]
        },
    }
    stages = stage_measurements(trace)
    assert stages["extract"]["phase_wall_ms"] == 25
    assert stages["extract"]["call_elapsed_sum_ms"] == 40
    assert stages["extract"]["reported_input_tokens"] == 22
    assert stages["extract"]["reported_output_tokens"] == 10
    assert stages["synthesize"]["unknown_usage_calls"] == 1
    assert stages["decision"]["elapsed_ms"] == 5
    summary = summarize_stages([{"trace": trace}, {"trace": trace}])
    assert summary["extract"]["calls"] == 4
    assert summary["extract"]["phase_wall_mean_ms"] == 25
    assert summary["decision"]["inference_attempts"] == 2
    assert summary["decision"]["reported_input_tokens"] == 100
    assert summary["decision"]["reported_output_tokens"] == 0
    assert summary["decision"]["unknown_usage_attempts"] == 0
    assert summary["extract"]["call_elapsed_p95_ms"] == 20


def test_legacy_timing_and_skipped_decision_remain_unknown():
    trace = {
        "decision": {"disposition": "rules_only"},
        "execution_usage": {
            "calls": [{"node_id": "direct", "elapsed_ms": 12, "status": "complete"}]
        },
    }
    stages = stage_measurements(trace)
    assert stages["generate"]["phase_wall_ms"] is None
    assert stages["generate"]["call_elapsed_mean_ms"] == 12
    assert stages["decision"]["inference_attempts"] == 0
    assert stages["decision"]["elapsed_ms"] is None
    assert stage_measurements({"execution_usage": None}) == {}


async def test_ledger_records_relative_call_intervals():
    from prism.config import Limits, RawModel
    from prism.runtime import Ledger

    ledger = Ledger(Limits())
    model = RawModel(id="fixture", name="fixture", base_url="http://fixture/v1")
    reservation = await ledger.reserve("direct", model, 20, 10)
    await ledger.start(reservation)
    await ledger.finish(
        reservation, {"prompt_tokens": 12, "completion_tokens": 6}, elapsed_ms=1
    )
    recorded = ledger.snapshot()["calls"][0]
    assert recorded["started_ms"] >= 0
    assert recorded["finished_ms"] >= recorded["started_ms"]
    assert recorded["elapsed_ms"] == 1
