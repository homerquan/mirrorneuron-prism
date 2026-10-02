from __future__ import annotations

import asyncio
import csv
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path

from ..classifier.artifacts import verify_controller
from ..classifier.executor import DecisionExecutor
from ..classifier.policy import DecisionPolicy
from ..classifier.types import DecisionRequest
from ..errors import PrismError
from ..meter import RequestMeter
from ..storage import (
    append_jsonl,
    atomic_write,
    digest,
    file_hash,
    locked,
    read_json,
    write_json,
)
from ..telemetry import log_event
from . import datasets
from .metrics import compute
from .reporting import markdown
from .types import Plan, RunManifest


def code_hash():
    package = Path(__file__).parents[1]
    return digest(
        {
            str(p.relative_to(package)): file_hash(p)
            for p in sorted(package.rglob("*.py"))
        }
    )


def create_plan(ctx):
    a = ctx.args
    manifest = datasets.verify(ctx.root, a.suite)
    profile = ctx.controller(a.controller).model_copy(deep=True)
    if a.treatment == "rules":
        profile.backend = "rules"
        profile.scoring_mode = "direct"
        artifact = None
    else:
        if profile.backend != "jev_cpu":
            raise PrismError("CPU treatment requires a jev_cpu controller")
        profile.scoring_mode = {
            "cpu_direct": "direct",
            "cpu_shared": "shared",
            "cpu_compact_generation": "compact_generation",
        }[a.treatment]
        artifact = verify_controller(ctx.artifact_lock, profile)
    seed = a.seed if a.seed is not None else ctx.project.benchmarks.seed
    repetitions = (
        a.repetitions
        if a.repetitions is not None
        else ctx.project.benchmarks.repetitions
    )
    split = "systems" if a.suite == "semif-shape" else a.split
    body = {
        "schema_version": 1,
        "suite": a.suite,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "dataset_revision": manifest["revision"],
        "selection": a.selection,
        "split": split,
        "tasks": datasets.select(manifest, split, a.selection, seed),
        "treatment": a.treatment,
        "controller": a.controller,
        "profile": profile.model_dump(mode="json"),
        "model_artifact_sha256": artifact["identity_sha256"] if artifact else None,
        "calibration_sha256": None,
        "calibration_path": None,
        "seed": seed,
        "repetitions": repetitions,
        "warmup_requests": 1,
        "concurrency": 1,
        "quality_margin": a.quality_margin,
        "statistical_method": "paired independent-source-group Hoeffding, one-sided alpha=0.05",
        "resource_metric": "controller_process_cpu_seconds",
        "split_method": "prism-group-hash-v1",
        "privacy_log_content": False,
        "package_code_sha256": code_hash(),
        "project_sha256": digest(ctx.project.model_dump(mode="json")),
    }
    if a.calibration:
        if a.treatment == "rules" or split == "all":
            raise PrismError(
                "calibrated gates require a CPU treatment and a named split"
            )
        calibration_path = ctx.explicit(a.calibration)
        calibration = read_json(calibration_path)
        from ..classifier.calibration import identity_for_profile, verify

        verify(calibration, identity_for_profile(profile, artifact))
        if calibration["dataset_provenance"] != manifest[
            "manifest_sha256"
        ] or calibration["split_hashes"].get(split) != manifest[
            "split_group_hashes"
        ].get(split):
            raise PrismError(
                "calibration dataset/split provenance mismatch", 7, "split_leakage"
            )
        body.update(
            calibration_sha256=calibration["artifact_sha256"],
            calibration_path=str(calibration_path),
        )
    plan = Plan.model_validate({**body, "plan_sha256": digest(body)})
    write_json(ctx.explicit(a.out), plan.model_dump(mode="json"))
    return {
        "plan": plan.model_dump(mode="json"),
        "semantic_decision_ceiling": len(plan.tasks) * plan.repetitions,
        "worker_call_ceiling": 0,
        "network_calls": 0,
        "diagnostic": plan.split == "all",
    }


def _environment(ctx):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ctx.invocation,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ctx.invocation,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ctx.invocation,
                stderr=subprocess.DEVNULL,
            )
        )
        import hashlib

        source = {
            "commit": commit,
            "dirty": dirty,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except subprocess.CalledProcessError:
        source = {"commit": None, "dirty": None, "tracked_diff_sha256": None}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependency_versions": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "source": source,
    }


def execute(ctx):
    raw_plan = read_json(ctx.explicit(ctx.args.plan))
    if raw_plan.get("plan_sha256") != digest(
        {k: v for k, v in raw_plan.items() if k != "plan_sha256"}
    ):
        raise PrismError(
            "plan hash mismatch; plans are immutable", 7, "identity_mismatch"
        )
    plan = Plan.model_validate(raw_plan)
    manifest = datasets.verify(ctx.root, plan.suite)
    if (
        manifest["manifest_sha256"] != plan.dataset_manifest_sha256
        or code_hash() != plan.package_code_sha256
        or digest(ctx.project.model_dump(mode="json")) != plan.project_sha256
    ):
        raise PrismError(
            "plan code/config/dataset identity changed; create a new plan",
            7,
            "identity_mismatch",
        )
    expected_tasks = datasets.select(manifest, plan.split, plan.selection, plan.seed)
    if [t.model_dump() for t in plan.tasks] != expected_tasks:
        raise PrismError(
            "plan selection/source groups differ from the pinned dataset",
            7,
            "identity_mismatch",
        )
    artifact = (
        verify_controller(ctx.artifact_lock, plan.profile)
        if plan.treatment != "rules"
        else None
    )
    if artifact and artifact["identity_sha256"] != plan.model_artifact_sha256:
        raise PrismError("plan model identity changed", 7, "identity_mismatch")
    calibration = (
        read_json(Path(plan.calibration_path)) if plan.calibration_path else None
    )
    if calibration and calibration["artifact_sha256"] != plan.calibration_sha256:
        raise PrismError("plan calibration changed", 7, "identity_mismatch")
    if calibration:
        from ..classifier.calibration import identity_for_profile, verify

        verify(calibration, identity_for_profile(plan.profile, artifact))
    base = datasets.paths(ctx.root, plan.suite)
    requests = {
        r["request_id"]: DecisionRequest.model_validate(r)
        for r in (
            json.loads(line)
            for line in (base / "inputs.jsonl").read_text().splitlines()
        )
    }
    # Evaluator data stays in the coordinator, never in the classifier process.
    gold = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in (base / "evaluator.jsonl").read_text().splitlines()
        )
    }
    tasks = {t.id: t for t in plan.tasks}
    if any(t.id not in requests for t in plan.tasks):
        raise PrismError("plan references absent dataset tasks", 7, "identity_mismatch")
    grouped = {}
    for task in plan.tasks:
        request = requests[task.id]
        grouped.setdefault((task.group_id, request.state_version), []).append(request)
    batches = []
    for group in grouped.values():
        for start in range(0, len(group), plan.profile.max_criteria_per_batch):
            chunk = group[start : start + plan.profile.max_criteria_per_batch]
            first = chunk[0]
            batches.append(
                DecisionRequest(
                    request_id=digest([r.request_id for r in chunk])[:24],
                    logical_request_id=first.logical_request_id,
                    state_version=first.state_version,
                    state=first.state,
                    criteria=[r.criteria[0] for r in chunk],
                )
            )
    run_id = uuid.uuid4().hex
    run = ctx.run_root / run_id
    run.mkdir(parents=True, exist_ok=False)
    environment = _environment(ctx)
    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "plan": plan.model_dump(mode="json"),
        "dataset_limitation": manifest["limitation"],
        "environment": environment,
        "hardware": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "gpu_active_time": "unavailable",
            "energy": "unavailable",
            "cpu_affinity": "unsupported",
        },
        "cache_policy": "no cross-request prefix cache",
        "warmup_policy": "one explicit first-batch warmup; counted separately",
        "failure_policy": "failed task-trial units retained; no automatic retries",
        "privacy": {"log_content": False},
    }
    run_manifest["manifest_sha256"] = digest(run_manifest)
    run_manifest = RunManifest.model_validate(run_manifest).model_dump(mode="json")
    write_json(run / "manifest.json", run_manifest)
    write_json(
        run / "effective_config.redacted.json",
        {
            "project_sha256": plan.project_sha256,
            "controller": plan.profile.model_dump(),
        },
    )
    write_json(run / "environment.json", environment)
    write_json(run / "hardware.json", run_manifest["hardware"])
    write_json(run / "task_selection.json", [t.model_dump() for t in plan.tasks])
    status = {
        "schema_version": 1,
        "state": "running",
        "run_id": run_id,
        "artifact_hashes": {},
    }
    write_json(run / "status.json", status)
    outcomes, decisions = [], []
    meter = RequestMeter(run_id)
    executor = DecisionExecutor(plan.profile, artifact)
    gate = DecisionPolicy(calibration)
    started = time.monotonic()
    events = []
    for name in (
        "events.jsonl",
        "decisions.jsonl",
        "predictions.jsonl",
        "outcomes.jsonl",
        "measurements.jsonl",
    ):
        atomic_write(run / name, "")

    def persist(name, value):
        append_jsonl(run / name, value)

    def event(kind, **data):
        events.append(
            log_event(
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "event_id": uuid.uuid4().hex,
                    "kind": kind,
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                    "parent_span_id": run_id,
                    "role": "controller",
                    "backend_name": plan.profile.backend,
                    "model": plan.profile.model_source
                    if plan.profile.backend != "rules"
                    else None,
                    "device": "cpu",
                    "provenance": "prism-native-runner-v1",
                    **data,
                }
            )
        )
        persist("events.jsonl", events[-1])

    async def run_all():
        try:
            event("run_started")
            try:
                warm = await executor.decide(batches[0], timeout=ctx.args.timeout)
                meter.record_controller(
                    warm.measurements, role="warmup", request_id=warm.request_id
                )
                persist("measurements.jsonl", meter.controller_batches[-1])
                event("warmup_completed", backend=warm.backend_metadata)
            except PrismError as exc:
                meter.record_controller(
                    {"usage_complete": False, "failure": exc.kind},
                    role="warmup",
                    request_id=batches[0].request_id,
                )
                persist("measurements.jsonl", meter.controller_batches[-1])
                event("warmup_failed", reason=exc.kind)
            for trial in range(plan.repetitions):
                for request in batches:
                    event(
                        "batch_started",
                        trial=trial,
                        logical_request_id=request.logical_request_id,
                        request_id=request.request_id,
                        input_sha256=digest(request.model_dump()),
                        criterion_ids=[c.id for c in request.criteria],
                        role="controller",
                    )
                    try:
                        batch = gate.apply(
                            await executor.decide(request, timeout=ctx.args.timeout),
                            task_families={
                                c.id: gold[c.id]["family"]
                                for c in request.criteria
                                if c.id in gold
                            },
                        )
                        meter.record_controller(
                            batch.measurements,
                            role="measured",
                            request_id=batch.request_id,
                        )
                        persist("measurements.jsonl", meter.controller_batches[-1])
                        for result in batch.results:
                            data = result.model_dump(mode="json")
                            decisions.append(
                                {
                                    "schema_version": 1,
                                    "run_id": run_id,
                                    "trial": trial,
                                    **data,
                                }
                            )
                            outcomes.append(
                                {
                                    "schema_version": 1,
                                    "run_id": run_id,
                                    "id": result.criterion_id,
                                    "trial": trial,
                                    "group_id": tasks[result.criterion_id].group_id,
                                    "status": "completed",
                                    "winner_option_id": result.winner_option_id,
                                    "disposition": result.disposition,
                                    "correct": result.winner_option_id
                                    == gold[result.criterion_id]["gold_option_id"]
                                    if result.criterion_id in gold
                                    else None,
                                }
                            )
                            persist("decisions.jsonl", decisions[-1])
                            persist("predictions.jsonl", decisions[-1])
                            persist("outcomes.jsonl", outcomes[-1])
                        event(
                            "batch_completed",
                            trial=trial,
                            request_id=request.request_id,
                        )
                    except PrismError as exc:
                        ctx.args.exit_code = exc.code
                        meter.record_controller(
                            {"usage_complete": False, "failure": exc.kind},
                            role="measured",
                            request_id=request.request_id,
                        )
                        persist("measurements.jsonl", meter.controller_batches[-1])
                        for c in request.criteria:
                            outcomes.append(
                                {
                                    "schema_version": 1,
                                    "run_id": run_id,
                                    "id": c.id,
                                    "trial": trial,
                                    "group_id": tasks[c.id].group_id,
                                    "status": "failed",
                                    "winner_option_id": None,
                                    "disposition": "abstain",
                                    "correct": False if c.id in gold else None,
                                    "failure": exc.kind,
                                }
                            )
                            persist("outcomes.jsonl", outcomes[-1])
                        event(
                            "batch_failed",
                            trial=trial,
                            request_id=request.request_id,
                            reason=exc.kind,
                        )
        finally:
            await executor.close()

    state = "completed"
    with locked(run / ".run.lock"):
        try:
            asyncio.run(run_all())
            if getattr(ctx.args, "exit_code", 0):
                state = "failed"
        except KeyboardInterrupt:
            state = "cancelled"
            ctx.args.exit_code = 130
        except Exception:
            state = "failed"
            raise
        finally:
            metrics = compute(
                outcomes,
                meter.controller_batches,
                len(plan.tasks) * plan.repetitions,
                bool(gold),
            )
            metrics["wall_time_seconds"] = time.monotonic() - started
            metrics["wasted_work"] = executor.wasted_work
            event("run_finished", state=state)
            for name, rows in (
                ("events.jsonl", events),
                ("decisions.jsonl", decisions),
                ("predictions.jsonl", decisions),
                ("outcomes.jsonl", outcomes),
                ("measurements.jsonl", meter.controller_batches),
            ):
                atomic_write(
                    run / name,
                    "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows),
                    force=True,
                )
            write_json(run / "metrics.json", metrics)
            csv_stream = io.StringIO()
            writer = csv.writer(csv_stream)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                if isinstance(v, (int, float)) or v is None:
                    writer.writerow([k, "" if v is None else v])
            atomic_write(run / "metrics.csv", csv_stream.getvalue())
            atomic_write(run / "report.md", markdown(run_manifest, metrics))
            status.update(
                state=state,
                artifact_hashes={
                    p.name: file_hash(p)
                    for p in run.iterdir()
                    if p.is_file() and p.name not in ("status.json", ".run.lock")
                },
            )
            write_json(run / "status.json", status, force=True)
    return {"run_id": run_id, "state": state, "directory": str(run), "metrics": metrics}
