import json

import pytest

from litellm_multicall.benchmarks import datasets
from litellm_multicall.benchmarks.metrics import compute, paired_quality
from litellm_multicall.cli import main
from litellm_multicall.errors import PrismError
from litellm_multicall.storage import read_json


@pytest.fixture
def source(monkeypatch):
    original = datasets.subprocess.check_output
    rows = []
    for group in range(30):
        for variant in range(2):
            rows.append(
                {
                    "id": f"g{group}-v{variant}",
                    "group_id": f"g{group}",
                    "family": "test",
                    "split": "upstream",
                    "state": "Evidence is ambiguous.",
                    "question": "Is the claim supported?",
                    "options": [
                        {"id": "yes", "description": "Yes"},
                        {"id": "no", "description": "No"},
                    ],
                    "label": variant,
                    "provenance": {"rationale": "EVALUATOR_SECRET"},
                    "target_distribution": [1, 0],
                    "expected_answer": "EVALUATOR_SECRET",
                }
            )

    def show(argv, **kwargs):
        if argv[0] != "git":
            return original(argv, **kwargs)
        if argv[-1].endswith("authored144.jsonl") or argv[-1].endswith(
            "shape777.jsonl"
        ):
            return ("\n".join(json.dumps(r) for r in rows)).encode()
        if argv[-1] == "HEAD":
            return "a" * 40 if kwargs.get("text") else b"a" * 40
        return "MIT test notice" if kwargs.get("text") else b"MIT test notice"

    monkeypatch.setattr(datasets.subprocess, "check_output", show)
    return rows


def test_preparation_strips_gold_groups_and_tamper(tmp_path, source):
    manifest = datasets.prepare(tmp_path, "semif-authored", tmp_path, "a" * 40)
    assert datasets.verify(tmp_path, "semif-authored") == manifest
    base = datasets.paths(tmp_path, "semif-authored")
    content = (base / "inputs.jsonl").read_text()
    assert "EVALUATOR_SECRET" not in content
    assert "target_distribution" not in content and '"label"' not in content
    splits = {}
    for task in manifest["tasks"]:
        splits.setdefault(task["group_id"], set()).add(task["split"])
        assert task["upstream_split"] == "upstream"
    assert all(len(v) == 1 for v in splits.values())
    labels = [
        json.loads(llm_path)
        for llm_path in (base / "evaluator.jsonl").read_text().splitlines()
    ]
    assert labels[0]["gold_option_id"] == "yes" and labels[1]["gold_option_id"] == "no"
    with pytest.raises(PrismError, match="already prepared"):
        datasets.prepare(tmp_path, "semif-authored", tmp_path, "a" * 40)
    (base / "inputs.jsonl").write_text(content + "\n")
    with pytest.raises(PrismError, match="changed"):
        datasets.verify(tmp_path, "semif-authored")


def test_rules_experiment_end_to_end(project, source, capsys):
    datasets.prepare(project / ".prism", "semif-authored", project, "a" * 40)
    capsys.readouterr()
    plan_path = project / "plan.json"
    assert (
        main(
            [
                "benchmark",
                "plan",
                "semif-authored",
                "--treatment",
                "rules",
                "--out",
                str(plan_path),
                "--repetitions",
                "1",
                "--format",
                "json",
            ]
        )
        == 0
    )
    planned = json.loads(capsys.readouterr().out)["data"]["plan"]
    assert len(planned["tasks"]) == 4
    assert main(["benchmark", "run", "--plan", str(plan_path), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["state"] == "completed"
    assert data["metrics"]["accuracy"] == 0
    assert (
        data["metrics"]["accepted_coverage"] == 0
        and data["metrics"]["selective_risk"] is None
    )
    run_id = data["run_id"]
    path = project / ".prism/runs" / run_id
    for name in (
        "manifest.json",
        "events.jsonl",
        "decisions.jsonl",
        "outcomes.jsonl",
        "measurements.jsonl",
        "metrics.json",
        "metrics.csv",
        "report.md",
    ):
        assert (path / name).exists()
    assert "EVALUATOR_SECRET" not in "".join(
        p.read_text() for p in path.iterdir() if p.is_file()
    )
    assert (
        main(
            [
                "benchmark",
                "compare",
                run_id,
                run_id,
                "--require-parity",
                "--format",
                "json",
            ]
        )
        == 8
    )
    comparison = json.loads(capsys.readouterr().out)["data"]
    assert (
        comparison["quality"]["status"] == "not_established"
        and comparison["ECR_at_QP"] is None
    )
    (path / "outcomes.jsonl").write_text("{}\n")
    assert main(["benchmark", "report", run_id, "--format", "json"]) == 7


def test_plan_immutable(project, source, capsys):
    datasets.prepare(project / ".prism", "semif-authored", project, "a" * 40)
    plan_path = project / "plan.json"
    assert (
        main(
            [
                "benchmark",
                "plan",
                "semif-authored",
                "--treatment",
                "rules",
                "--out",
                str(plan_path),
                "--quiet",
            ]
        )
        == 0
    )
    plan = read_json(plan_path)
    plan["seed"] = 999
    plan_path.write_text(json.dumps(plan))
    assert main(["benchmark", "run", "--plan", str(plan_path), "--format", "json"]) == 7
    assert "plan hash mismatch" in capsys.readouterr().out


def test_metrics_unknown_and_failed_denominator():
    outcomes = [
        {
            "id": "1",
            "trial": 0,
            "group_id": "g",
            "status": "completed",
            "correct": True,
            "disposition": "accept",
            "winner_option_id": "yes",
        },
        {
            "id": "2",
            "trial": 0,
            "group_id": "g",
            "status": "failed",
            "correct": False,
            "disposition": "abstain",
        },
    ]
    result = compute(
        outcomes,
        [{"role": "measured", "total_ms": 2, "usage_complete": False}],
        3,
        True,
    )
    assert result["accuracy"] == 1 / 3 and result["failed_decisions"] == 2
    assert result["accepted_coverage"] == 1 / 3 and result["selective_risk"] == 0
    assert (
        result["controller_process_cpu_seconds"] is None
        and result["gpu_active_seconds"] is None
    )
    assert not result["usage_complete"]


@pytest.mark.parametrize("n", [1, 2, 10, 20])
def test_tiny_all_success_has_uncertainty(n):
    rows = [
        {"id": str(i), "trial": 0, "group_id": str(i), "correct": True}
        for i in range(n)
    ]
    result = paired_quality(rows, rows)
    assert result["status"] == "not_established" and result["lower_bound"] < 0


def test_repeated_trials_not_independent():
    rows = [
        {"id": "one", "trial": i, "group_id": "one", "correct": True}
        for i in range(100)
    ]
    assert paired_quality(rows, rows)["independent_groups"] == 1
    assert paired_quality(rows, [])["status"] == "invalid"
