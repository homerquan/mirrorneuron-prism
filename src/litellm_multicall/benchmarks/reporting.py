from ..storage import read_json


def markdown(manifest, metrics):
    plan = manifest["plan"]
    return (
        f"# Prism native decision experiment {manifest['run_id']}\n\n"
        f"Suite: {plan['suite']}; treatment: {plan['treatment']}; split: {plan['split']}.\n\n"
        f"Pinned dataset: `{plan['dataset_revision']}`. Plan: `{plan['plan_sha256']}`.\n\n"
        f"Planned decisions: {metrics['planned_decisions']}; completed: {metrics['completed_decisions']}; failed: {metrics['failed_decisions']}. "
        f"Independent source groups: {metrics['independent_groups']}.\n\n"
        f"Accuracy (all planned labeled decisions): {metrics['accuracy']}. "
        f"Accepted coverage: {metrics['accepted_coverage']}; selective risk: {metrics['selective_risk']}.\n\n"
        f"Median delivered batch latency: {metrics['latency_ms']['median_per_batch']} ms. "
        f"Controller process CPU seconds, including warmup: {metrics['controller_process_cpu_seconds']}. "
        f"Model forwards: {metrics['model_forward_passes']}; logical input tokens: {metrics['logical_input_tokens']}; "
        f"generated output tokens: {metrics['generated_output_tokens']}.\n\n"
        "Startup and warmup are recorded separately in measurements.jsonl. Shared mode has per-batch physical prefix/suffix/padding accounting. "
        "No worker inference was eliminated or measured by this standalone experiment. GPU active time, energy, and cost are unavailable.\n\n"
        f"Dataset limitation: {manifest['dataset_limitation']}.\n\n"
        "These are native decision results, not BFCL scores, agent task success, or evidence of whole-system compute savings. "
        "Raw option probabilities are conditional scores; no acceptance is claimed without a compatible calibration artifact. "
        "Private input content is not stored in run artifacts; exact reconstruction also requires the separately prepared dataset.\n"
    )


def read_lines(path):
    import json

    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_run(path):
    from ..errors import PrismError
    from ..storage import file_hash

    manifest = read_json(path / "manifest.json")
    from ..storage import digest

    if digest(
        {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    ) != manifest.get("manifest_sha256"):
        raise PrismError("run manifest identity changed", 7, "corrupt_run")
    status = read_json(path / "status.json")
    if status["state"] != "completed":
        raise PrismError(
            "run is not completed; comparison/report requires complete evidence",
            7,
            "incomplete_run",
        )
    for name, sha in status["artifact_hashes"].items():
        if file_hash(path / name) != sha:
            raise PrismError("run artifact changed after completion", 7, "corrupt_run")
    return (
        manifest,
        read_json(path / "metrics.json"),
        read_lines(path / "outcomes.jsonl"),
    )
