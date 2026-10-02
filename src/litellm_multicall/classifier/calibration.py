"""Temperature fitting and conservative grouped threshold validation."""

import math

from .._vendor.semif.core import softmax
from ..errors import PrismError
from ..storage import digest

IDENTITY_FIELDS = (
    "revision",
    "tokenizer_revision",
    "artifact_sha256",
    "dtype",
    "prompt_version",
    "scoring_implementation",
    "criterion_schema_version",
    "state_builder",
    "scoring_mode",
    "torch_version",
    "transformers_version",
)


def identity_for_profile(profile, artifact):
    """Expected runtime identity without loading or importing Torch."""
    from importlib.metadata import PackageNotFoundError, version

    from .._vendor.semif.direct import PROMPT_VERSION

    try:
        libraries = {
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
        }
    except PackageNotFoundError as exc:
        raise PrismError(
            "compatible calibration requires the optional CPU dependencies",
            4,
            "missing_dependency",
        ) from exc

    return {
        "revision": artifact["revision"],
        "tokenizer_revision": artifact["tokenizer_revision"],
        "artifact_sha256": artifact["identity_sha256"],
        "dtype": profile.dtype,
        "prompt_version": PROMPT_VERSION,
        "scoring_implementation": "semif-b49b5bf5776af78495fa4900996042726f8e7c10",
        "criterion_schema_version": 1,
        "state_builder": "native_decision_v1",
        "scoring_mode": profile.scoring_mode,
        **libraries,
    }


def identity(metadata):
    if any(k not in metadata for k in IDENTITY_FIELDS):
        raise PrismError(
            "calibration requires complete model/prompt/scoring identity",
            7,
            "calibration_identity",
        )
    return {k: metadata[k] for k in IDENTITY_FIELDS}


def _validate_records(records):
    if not records:
        raise PrismError("calibration and validation predictions must be nonempty")
    for r in records:
        if (
            not r.get("group_id")
            or not r.get("id")
            or not r.get("primitive")
            or not r.get("family")
        ):
            raise PrismError(
                "prediction records require ID, source group, family and primitive"
            )
        ids, logits = r["option_ids"], r["option_logits"]
        if (
            not 2 <= len(ids) <= 16
            or len(set(ids)) != len(ids)
            or len(logits) != len(ids)
            or r["gold_option_id"] not in ids
        ):
            raise PrismError("invalid labeled calibration prediction")
        softmax(logits)


def fit(bundle, risk_target=0.05, min_support=20):
    if not 0 < risk_target < 1 or min_support < 2:
        raise PrismError(
            "risk target must be between 0 and 1; minimum support must be at least 2"
        )
    metadata = identity(bundle["identity"])
    if not bundle.get("dataset_provenance"):
        raise PrismError("dataset provenance is required")
    calibration, validation = bundle["calibration"], bundle["validation"]
    _validate_records(calibration)
    _validate_records(validation)
    groups = [
        set(r["group_id"] for r in calibration),
        set(r["group_id"] for r in validation),
        set(bundle["test_groups"]),
    ]
    if not groups[2] or any(
        groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise PrismError(
            "calibration/validation/test source groups must be nonempty and disjoint",
            7,
            "split_leakage",
        )
    ids = [r["id"] for r in calibration + validation]
    if len(ids) != len(set(ids)):
        raise PrismError(
            "calibration/validation prediction IDs must be unique", 7, "split_leakage"
        )

    def loss(t):
        return sum(
            -math.log(
                max(
                    1e-300,
                    softmax([v / t for v in r["option_logits"]])[
                        r["option_ids"].index(r["gold_option_id"])
                    ],
                )
            )
            for r in calibration
        ) / len(calibration)

    temperatures = [math.exp(-3 + i * 6 / 240) for i in range(241)]
    temperature = min(temperatures, key=loss)
    scored = []
    for r in validation:
        probs = softmax([v / temperature for v in r["option_logits"]])
        winner = max(range(len(probs)), key=probs.__getitem__)
        scored.append(
            (max(probs), r["option_ids"][winner] != r["gold_option_id"], r["group_id"])
        )
    candidates = sorted({score for score, _, _ in scored})
    # Uniform bound across searched thresholds; related variants count as one group.
    radius_factor = math.log(max(1, len(candidates)) / 0.05)
    chosen = None
    diagnostics = []
    for threshold in candidates:
        accepted = [r for r in scored if r[0] >= threshold]
        group_errors = {}
        group_counts = {}
        for _, error, group in accepted:
            group_errors[group] = max(group_errors.get(group, 0), int(error))
            group_counts[group] = group_counts.get(group, 0) + 1
        n = len(group_errors)
        error_rate = sum(error for _, error, _ in accepted) / len(accepted)
        weights = {g: count / len(accepted) for g, count in group_counts.items()}
        upper = min(
            1,
            sum(weights[g] * group_errors[g] for g in group_errors)
            + math.sqrt(radius_factor * sum(w * w for w in weights.values()) / 2),
        )
        row = {
            "threshold": threshold,
            "accepted": len(accepted),
            "independent_groups": n,
            "coverage": len(accepted) / len(scored),
            "selective_error": error_rate,
            "risk_upper_bound": upper,
        }
        diagnostics.append(row)
        if chosen is None and n >= min_support and upper <= risk_target:
            chosen = row
    data = {
        "schema_version": 1,
        "method": "temperature-grid-v1",
        "identity": metadata,
        "temperature": temperature,
        "calibration_nll": loss(temperature),
        "risk_target": risk_target,
        "minimum_validation_groups": min_support,
        "threshold": chosen["threshold"] if chosen else None,
        "validation": chosen,
        "threshold_diagnostics": diagnostics,
        "uncertainty_method": "accepted-count-weighted group-max-error Hoeffding union bound over threshold search, alpha=0.05; empirical validation only",
        "split_hashes": {
            name: digest(sorted(g))
            for name, g in zip(("calibration", "validation", "test"), groups)
        },
        "dataset_provenance": bundle["dataset_provenance"],
        "source_prediction_sha256": digest(bundle),
        "supported_primitives": sorted(
            {r["primitive"] for r in calibration + validation}
        ),
        "supported_task_families": sorted(
            {r["family"] for r in calibration + validation}
        ),
        "option_counts": sorted(
            {len(r["option_ids"]) for r in calibration + validation}
        ),
        "test_performance": None,
        "status": "validated_gate"
        if chosen
        else "insufficient_selective_risk_evidence",
    }
    return {**data, "artifact_sha256": digest(data)}


def verify(calibration, metadata):
    data = {k: v for k, v in calibration.items() if k != "artifact_sha256"}
    if (
        calibration.get("artifact_sha256") != digest(data)
        or calibration.get("schema_version") != 1
    ):
        raise PrismError("calibration artifact corrupted", 7, "calibration_identity")
    if calibration["identity"] != identity(metadata):
        raise PrismError(
            "calibration is incompatible with this model/prompt/dtype/scoring mode",
            7,
            "calibration_identity",
        )
    if (
        calibration.get("method") != "temperature-grid-v1"
        or not math.isfinite(calibration.get("temperature", 0))
        or calibration["temperature"] <= 0
    ):
        raise PrismError(
            "unsupported/invalid calibration method or temperature",
            7,
            "calibration_identity",
        )
    threshold = calibration.get("threshold")
    validation = calibration.get("validation")
    if threshold is not None and (
        not validation
        or threshold != validation["threshold"]
        or not 0 <= threshold <= 1
        or validation["independent_groups"] < calibration["minimum_validation_groups"]
        or validation["risk_upper_bound"] > calibration["risk_target"]
    ):
        raise PrismError(
            "calibration threshold lacks its declared validation support",
            7,
            "calibration_identity",
        )
