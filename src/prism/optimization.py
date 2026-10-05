"""Request-local model assignment and deterministic cost/power optimization."""

from itertools import product

from .config import validate_optimization
from .contracts import context_controls
from .errors import PrismError
from .planning import compile_policy, stage_bound

ROLE_WEIGHTS = {
    "direct": {"direct": 1.0},
    "retrieve_read": {"synthesizer": 1.0},
    "evidence_map": {"worker": 0.5, "synthesizer": 0.5},
    "batched_map": {"worker": 0.5, "synthesizer": 0.5},
    "verified_map": {"worker": 0.25, "verifier": 0.25, "synthesizer": 0.5},
    "draft_review": {"worker": 0.25, "verifier": 0.25, "synthesizer": 0.5},
    "vision_synthesis": {"worker": 0.5, "synthesizer": 0.5},
    "text_synthesis": {"worker": 0.5, "synthesizer": 0.5},
}


def adequate_stage_power(assignments, models, available):
    """Review and source integration need more power than a narrow worker task.

    Equal ratings are allowed at the best available rating, so a pool with only
    one strength (or an all-strong assignment) remains feasible.
    """
    if "worker" not in assignments:
        return True
    worker = models[assignments["worker"]].power_rating
    maximum = max(models[ref].power_rating for ref in available)
    previous = worker
    for role in ("verifier", "synthesizer"):
        if role not in assignments:
            continue
        rating = models[assignments[role]].power_rating
        if rating < previous or rating == worker and worker < maximum:
            return False
        previous = rating
    return True


def effective_profile(profile, body, models):
    controls = context_controls(body.get("context_management", []))
    if profile.optimization is None:
        if controls:
            raise PrismError(
                "profile does not enable optimization",
                "unsupported_feature",
                param="context_management",
            )
        return profile, None
    validate_optimization(profile, models)
    effective = {
        "prism_cost_priority": profile.optimization.default_cost_priority,
        "prism_model_ids": list(profile.optimization.model_ids),
        "prism_allowed_policies": list(profile.allowed_policies),
        "prism_max_calls": profile.limits.max_calls,
        "prism_max_cost_usd": profile.limits.max_cost_usd,
        **controls,
    }
    for key, allowed in (
        ("prism_model_ids", profile.optimization.model_ids),
        ("prism_allowed_policies", profile.allowed_policies),
    ):
        if not set(effective[key]) <= set(allowed):
            raise PrismError(
                "request cannot expand profile model or policy permissions",
                param="context_management",
            )
    if (
        profile.strategy != "auto"
        and profile.strategy not in effective["prism_allowed_policies"]
    ):
        raise PrismError(
            "request excludes the profile's fixed strategy", param="context_management"
        )
    limits = profile.limits.model_dump()
    for key, field in (
        ("prism_max_calls", "max_calls"),
        ("prism_max_cost_usd", "max_cost_usd"),
    ):
        proposed = effective[key]
        if limits[field] is not None and proposed > limits[field]:
            raise PrismError(
                "request cannot relax profile resource limits",
                param="context_management",
            )
        limits[field] = proposed
    effective["prism_model_ids"] = sorted(effective["prism_model_ids"])
    return profile.model_copy(
        update={
            "limits": profile.limits.model_validate(limits),
            "allowed_policies": effective["prism_allowed_policies"],
        }
    ), effective


def ranking_key(candidate, priority, recommended=None):
    fit = float(candidate["policy"] == recommended)
    power = 0.9 * candidate["normalized_model_power"] + 0.1 * fit
    score = (1 - priority) * power - priority * candidate["normalized_cost"]
    return (
        -score,
        candidate["cost_upper_estimate_usd"],
        -candidate["model_power"],
        candidate["compiled"]["calls"],
        candidate["policy"],
        tuple(sorted(candidate["stage_models"].items())),
    )


def summary(candidate, models, priority, recommended=None):
    fit = float(candidate["policy"] == recommended)
    power = 0.9 * candidate["normalized_model_power"] + 0.1 * fit
    return {
        "policy": candidate["policy"],
        "stage_models": candidate["stage_models"],
        "power_ratings": {
            role: models[ref].power_rating
            for role, ref in candidate["stage_models"].items()
        },
        "calls": candidate["compiled"]["calls"],
        "coverage": "focused" if candidate["policy"] == "retrieve_read" else "full",
        "cost_upper_estimate_usd": candidate["cost_upper_estimate_usd"],
        "model_power": candidate["model_power"],
        "normalized_model_power": candidate["normalized_model_power"],
        "normalized_cost": candidate["normalized_cost"],
        "task_fit": fit,
        "power_score": power,
        "score": (1 - priority) * power - priority * candidate["normalized_cost"],
    }


def prepare_optimized_choices(engine, execution, direct_only):
    profile, controls = execution["profile"], execution["optimization_controls"]
    if direct_only and (
        "direct" not in profile.allowed_policies
        or profile.strategy not in {"auto", "direct"}
    ):
        raise PrismError(
            "request requires a permitted direct strategy", "unsupported_feature"
        )
    policies = (
        ["direct"]
        if direct_only
        else profile.allowed_policies
        if profile.strategy == "auto"
        else [profile.strategy]
    )
    # Shared request-local caches survive the temporary candidate executions.
    for name in ("stage_cache", "partition_cache", "job_cache", "retrieval_cache"):
        execution[name] = {}
    errors = {}

    stage_models = {}

    def candidates(enforce_order=True):
        for policy in policies:
            if (
                enforce_order
                and profile.optimization.enforce_stage_power_order
                and policy not in stage_models
            ):
                continue
            roles = ROLE_WEIGHTS[policy]
            for refs in product(controls["prism_model_ids"], repeat=len(roles)):
                if not execution["ledger"].remaining_seconds():
                    raise PrismError(
                        "request deadline exceeded during optimization",
                        "deadline_exceeded",
                        504,
                    )
                assignments = dict(zip(roles, refs, strict=True))
                if (
                    enforce_order
                    and profile.optimization.enforce_stage_power_order
                    and not adequate_stage_power(
                        assignments, engine.models, stage_models[policy]
                    )
                ):
                    errors[policy] = PrismError(
                        "stage assignments lack adequate relative model power",
                        "unsupported_feature",
                    )
                    continue
                candidate_profile = profile.model_copy(update=assignments)
                candidate_execution = {**execution, "profile": candidate_profile}
                try:
                    if policy == "direct":
                        candidate_execution["direct_input"] = stage_bound(
                            candidate_execution,
                            "direct",
                            engine.models[assignments["direct"]],
                            execution["body"]["messages"],
                            execution["parameters"],
                            execution["output"],
                        )
                    compiled = compile_policy(engine, candidate_execution, policy)
                except PrismError as error:
                    errors[policy] = error
                    continue
                power = sum(
                    engine.models[assignments[role]].power_rating * weight
                    for role, weight in roles.items()
                )
                yield {
                    "policy": policy,
                    "profile": candidate_profile,
                    "stage_models": assignments,
                    "compiled": compiled,
                    "direct_input": candidate_execution["direct_input"],
                    "cost_upper_estimate_usd": compiled["cost_upper_estimate_usd"],
                    "model_power": power,
                    "normalized_model_power": (power - 1) / 9,
                }

    # Power ordering uses models that can actually serve the harder stage under
    # this request's limits, rather than an unavailable high-rated pool member.
    if profile.optimization.enforce_stage_power_order:
        for candidate in candidates(enforce_order=False):
            ref = candidate["stage_models"].get(
                "synthesizer", candidate["stage_models"].get("direct")
            )
            stage_models.setdefault(candidate["policy"], set()).add(ref)
    # Two ranking passes avoid retaining every graph in a Cartesian space.
    minimum, maximum, count = float("inf"), 0.0, 0
    for candidate in candidates():
        cost = candidate["cost_upper_estimate_usd"]
        minimum, maximum, count = min(minimum, cost), max(maximum, cost), count + 1
    if not count:
        preferred = "evidence_map" if execution["arena"].explicit else "direct"
        raise errors.get(preferred, next(iter(errors.values())))
    priority = controls["prism_cost_priority"]
    best = {}
    for candidate in candidates():
        candidate["normalized_cost"] = (
            (candidate["cost_upper_estimate_usd"] - minimum) / (maximum - minimum)
            if maximum > minimum
            else 0.0
        )
        policy = candidate["policy"]
        if policy not in best or ranking_key(candidate, priority) < ranking_key(
            best[policy], priority
        ):
            best[policy] = candidate
    selected = min(
        best.values(), key=lambda candidate: ranking_key(candidate, priority)
    )
    execution.update(
        strategy=selected["policy"],
        policy_selected=False,
        optimization_candidates=best,
        policy_plans={
            policy: candidate["compiled"] for policy, candidate in best.items()
        },
    )
    execution["trace"].update(
        strategy=selected["policy"],
        rules_strategy=selected["policy"],
        eligible_policies=list(best),
        ineligible_policies={
            policy: error.code for policy, error in errors.items() if policy not in best
        },
        policy_call_bounds={
            policy: candidate["compiled"]["calls"] for policy, candidate in best.items()
        },
        optimization={
            "effective_controls": controls,
            "feasible_assignments": count,
            "cost_range_usd": [minimum, maximum],
            "quality_calibrated": False,
            "enforce_stage_power_order": profile.optimization.enforce_stage_power_order,
            "candidates": [
                summary(candidate, engine.models, priority)
                for candidate in best.values()
            ],
        },
    )


def select_optimized_candidate(engine, execution, decision):
    candidates = execution["optimization_candidates"]
    priority = execution["optimization_controls"]["prism_cost_priority"]
    recommended = (
        decision.get("proposal") if decision.get("disposition") == "accept" else None
    )
    selected = min(
        candidates.values(),
        key=lambda candidate: ranking_key(candidate, priority, recommended),
    )
    execution.update(
        profile=selected["profile"],
        direct_input=selected["direct_input"],
        strategy=selected["policy"],
    )
    execution["trace"]["stage_models"] = selected["stage_models"]
    execution["trace"]["optimization"].update(
        candidates=[
            summary(candidate, engine.models, priority, recommended)
            for candidate in candidates.values()
        ],
        selected=summary(selected, engine.models, priority, recommended),
        laya_recommendation=decision.get("proposal"),
        selection_reason="cost_power_ranking_with_task_fit"
        if recommended in candidates
        else "cost_power_ranking",
    )
