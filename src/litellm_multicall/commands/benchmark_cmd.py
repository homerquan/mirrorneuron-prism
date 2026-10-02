from ..benchmarks import datasets
from ..benchmarks.metrics import paired_quality
from ..benchmarks.registry import SUITES
from ..benchmarks.reporting import load_run, markdown
from ..benchmarks.runner import create_plan, execute
from ..errors import PrismError
from ..storage import atomic_write


def list_suites(ctx):
    return {
        "suites": [{"name": name, **data} for name, data in SUITES.items()],
        "treatments": ["rules", "cpu_direct", "cpu_shared", "cpu_compact_generation"],
    }


def prepare(ctx):
    return datasets.prepare(
        ctx.root, ctx.args.suite, ctx.explicit(ctx.args.source), ctx.args.revision
    )


def plan(ctx):
    return create_plan(ctx)


def run(ctx):
    return execute(ctx)


def run_path(ctx, name):
    import re

    if not re.fullmatch(r"[0-9a-f]{32}", name):
        raise PrismError("run ID must be the identifier returned by benchmark run")
    return ctx.run_root / name


def report(ctx):
    manifest, metrics, _ = load_run(run_path(ctx, ctx.args.run_id))
    text = markdown(manifest, metrics)
    if ctx.args.output:
        atomic_write(ctx.explicit(ctx.args.output), text)
    return {"run_id": ctx.args.run_id, "markdown": text, "metrics": metrics}


def compare(ctx):
    a, am, ao = load_run(run_path(ctx, ctx.args.baseline))
    b, bm, bo = load_run(run_path(ctx, ctx.args.candidate))
    fields = (
        "suite",
        "dataset_manifest_sha256",
        "dataset_revision",
        "tasks",
        "seed",
        "repetitions",
        "split",
        "package_code_sha256",
        "statistical_method",
        "warmup_requests",
        "concurrency",
        "resource_metric",
    )
    differences = [field for field in fields if a["plan"][field] != b["plan"][field]]
    if (
        a["hardware"] != b["hardware"]
        or a["environment"]["dependency_versions"]
        != b["environment"]["dependency_versions"]
    ):
        differences.append("environment/hardware")
    # A rules arm is allowed; CPU arms must use the same immutable model and threads.
    if a["plan"]["treatment"] != "rules" and b["plan"]["treatment"] != "rules":
        for key in ("model_artifact_sha256",):
            if a["plan"][key] != b["plan"][key]:
                differences.append(key)
        for key in (
            "threads",
            "interop_threads",
            "max_input_tokens",
            "max_criteria_per_batch",
            "dtype",
        ):
            if a["plan"]["profile"][key] != b["plan"]["profile"][key]:
                differences.append("profile." + key)
    if differences and not ctx.args.exploratory:
        raise PrismError(
            "runs are not comparable: " + ", ".join(differences),
            7,
            "comparison_contract",
        )
    exploratory = (
        bool(differences)
        or ctx.args.quality_margin != a["plan"]["quality_margin"]
        or ctx.args.quality_margin != b["plan"]["quality_margin"]
    )
    quality = paired_quality(ao, bo, ctx.args.quality_margin)
    if exploratory:
        quality["status"] = "invalid"
        quality["reason"] = "exploratory comparison; no preregistered claim"
    resource = a["plan"]["resource_metric"]
    denominator, numerator = am.get(resource), bm.get(resource)
    reduction = (
        1 - numerator / denominator
        if denominator is not None and denominator > 0 and numerator is not None
        else None
    )
    if ctx.args.require_parity and quality["status"] != "established":
        ctx.args.exit_code = 8
    return {
        "baseline": ctx.args.baseline,
        "candidate": ctx.args.candidate,
        "quality": quality,
        "exploratory": exploratory,
        "differences": differences,
        "resource_metric": resource,
        "raw_resource_reduction": reduction,
        "ECR_at_QP": reduction if quality["status"] == "established" else None,
        "scope": "native decision quality; no whole-system or worker savings claim",
    }


def calibration_input(ctx):
    """Evaluator-side export; gold labels never go back to an inference worker."""
    from ..benchmarks.reporting import read_lines
    from ..classifier.calibration import identity
    from ..storage import write_json

    a, _, _ = load_run(run_path(ctx, ctx.args.calibration_run))
    b, _, _ = load_run(run_path(ctx, ctx.args.validation_run))
    if (
        a["plan"]["suite"] != "semif-authored"
        or b["plan"]["suite"] != "semif-authored"
        or a["plan"]["split"] != "calibration"
        or b["plan"]["split"] != "validation"
    ):
        raise PrismError(
            "calibration input requires distinct authored calibration/validation runs",
            7,
            "split_leakage",
        )
    for field in (
        "dataset_manifest_sha256",
        "model_artifact_sha256",
        "treatment",
        "package_code_sha256",
        "profile",
    ):
        if a["plan"][field] != b["plan"][field]:
            raise PrismError(
                "calibration runs have incompatible identities",
                7,
                "calibration_identity",
            )
    manifest = datasets.verify(ctx.root, "semif-authored")
    if manifest["manifest_sha256"] != a["plan"]["dataset_manifest_sha256"]:
        raise PrismError("calibration dataset changed", 7, "calibration_identity")
    base = datasets.paths(ctx.root, "semif-authored")
    gold = {r["id"]: r for r in read_lines(base / "evaluator.jsonl")}

    def predictions(run, run_id):
        records = []
        for row in read_lines(run_path(ctx, run_id) / "predictions.jsonl"):
            if row["trial"] != 0:
                continue
            if row["option_logits"] is None:
                raise PrismError("temperature fitting requires real model logits")
            records.append(
                {
                    "id": row["criterion_id"],
                    "group_id": gold[row["criterion_id"]]["group_id"],
                    "primitive": row["primitive"],
                    "family": gold[row["criterion_id"]]["family"],
                    "option_ids": row["option_ids"],
                    "option_logits": row["option_logits"],
                    "gold_option_id": gold[row["criterion_id"]]["gold_option_id"],
                }
            )
        backend = next(
            e["backend"]
            for e in read_lines(run_path(ctx, run_id) / "events.jsonl")
            if e["kind"] == "warmup_completed"
        )
        return records, identity(backend)

    calibration, ai = predictions(a, ctx.args.calibration_run)
    validation, bi = predictions(b, ctx.args.validation_run)
    if ai != bi:
        raise PrismError("scoring identities differ", 7, "calibration_identity")
    bundle = {
        "identity": ai,
        "dataset_provenance": manifest["manifest_sha256"],
        "calibration": calibration,
        "validation": validation,
        "test_groups": sorted(
            {t["group_id"] for t in manifest["tasks"] if t["split"] == "test"}
        ),
    }
    write_json(ctx.explicit(ctx.args.output), bundle)
    return {
        "output": str(ctx.explicit(ctx.args.output)),
        "calibration_predictions": len(calibration),
        "validation_predictions": len(validation),
        "test_predictions_included": False,
        "trial_policy": "first trial only; repeated deterministic trials are not extra calibration support",
    }
