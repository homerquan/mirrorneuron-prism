"""Process-lifetime token spend and explicitly estimated direct-route savings."""

import time
from threading import RLock

from .contracts import check_context, has_images
from .errors import PrismError


def usage_cost(model, usage):
    if (
        model.input_cost_per_million is None
        or model.output_cost_per_million is None
        or not isinstance(usage, dict)
        or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        )
    ):
        return None
    return (
        usage["prompt_tokens"] * model.input_cost_per_million
        + usage["completion_tokens"] * model.output_cost_per_million
    ) / 1_000_000


def estimate_prompt_tokens(body):
    """Shared text tokenizer estimate; never download images or provider tokenizers."""
    if has_images(body["messages"]):
        return None
    try:
        import litellm

        return litellm.token_counter(
            model="gpt-4o",
            messages=body["messages"],
            tools=body.get("tools"),
        )
    except Exception:
        # Missing tokenizer data is unknown, never a byte-based savings claim.
        return None


class CostTracker:
    def __init__(self, models, publish=None):
        self.models = models
        self._lock = RLock()
        self.publish = publish
        self.started_at = int(time.time())
        self.calls = 0
        self.reported_cost = 0.0
        self.reported_input_tokens = 0
        self.reported_output_tokens = 0
        self.unpriced_calls = 0
        self.unreported_usage_calls = 0
        self.unknown_usage_upper_estimate = 0.0
        self.requests = 0
        self.compared_requests = 0
        self.compared_cost = 0.0
        self.baseline_cost = 0.0
        self.last_request = None
        self.model_usage = {
            ref: {
                "model_id": ref,
                "physical_calls": 0,
                "reported_input_tokens": 0,
                "reported_output_tokens": 0,
                "known_cost_usd": 0.0,
                "unpriced_calls": 0,
                "unreported_usage_calls": 0,
                "input_cost_per_million": model.input_cost_per_million,
                "output_cost_per_million": model.output_cost_per_million,
                "cost_rates_are_hypothetical": model.cost_rates_are_hypothetical,
            }
            for ref, model in sorted(models.items())
        }

    def record_call(self, model, usage, reservation):
        with self._lock:
            self._record_call(model, usage, reservation)

    def _record_call(self, model, usage, reservation):
        if not reservation.started:
            return
        self.calls += 1
        row = self.model_usage[model.id]
        row["physical_calls"] += 1
        priced = (
            model.input_cost_per_million is not None
            and model.output_cost_per_million is not None
        )
        reported = usage_cost(model, usage)
        if not priced:
            self.unpriced_calls += 1
            row["unpriced_calls"] += 1
        valid_usage = isinstance(usage, dict) and not any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        if valid_usage:
            self.reported_input_tokens += usage["prompt_tokens"]
            self.reported_output_tokens += usage["completion_tokens"]
            row["reported_input_tokens"] += usage["prompt_tokens"]
            row["reported_output_tokens"] += usage["completion_tokens"]
        else:
            self.unreported_usage_calls += 1
            row["unreported_usage_calls"] += 1
            self.unknown_usage_upper_estimate += reservation.cost_usd or 0
        if reported is not None:
            self.reported_cost += reported
            row["known_cost_usd"] += reported

    def record_execution(self, execution):
        with self._lock:
            self._record_execution(execution)
        self.emit()

    def _record_execution(self, execution):
        self.requests += 1
        self.last_request = {
            "profile": execution["trace"].get("model"),
            "strategy": execution["strategy"],
            "stop_reason": execution["trace"]["stop_reason"],
            "elapsed_ms": execution["trace"].get("elapsed_ms"),
        }
        calls = execution["ledger"].usage
        costs = [
            usage_cost(self.models[c["model_id"]], c["provider_usage"]) for c in calls
        ]
        successful = (
            execution["trace"]["stop_reason"] == "complete"
            and execution["trace"].get("delivery_status", "complete") == "complete"
        )
        if successful and calls and all(cost is not None for cost in costs):
            profile = execution["profile"]
            baseline = self.models[
                profile.cost_baseline_model or profile.synthesizer or profile.direct
            ]
            final = next(
                (c for c in reversed(calls) if c["node_id"] in {"direct", "synthesis"}),
                None,
            )
            baseline_cost = None
            if (
                execution["strategy"] == "direct"
                and len(calls) == 1
                and calls[0]["model_id"] == baseline.id
            ):
                # Same measured call: do not invent savings from token estimates.
                baseline_cost = costs[0]
            elif (
                final
                and final["provider_usage"]
                and baseline.input_cost_per_million is not None
                and baseline.output_cost_per_million is not None
            ):
                try:
                    check_context(
                        execution["body"]["messages"],
                        execution["parameters"],
                        execution["output"],
                        baseline,
                    )
                except PrismError:
                    pass  # An impossible direct call is not a comparable baseline.
                else:
                    tokens = estimate_prompt_tokens(execution["body"])
                    if tokens is not None:
                        baseline_cost = usage_cost(
                            baseline,
                            {
                                "prompt_tokens": tokens,
                                "completion_tokens": final["provider_usage"][
                                    "completion_tokens"
                                ],
                            },
                        )
            if baseline_cost is not None:
                self.compared_requests += 1
                self.compared_cost += sum(costs)
                self.baseline_cost += baseline_cost

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def _snapshot(self):
        savings = (
            self.baseline_cost - self.compared_cost if self.compared_requests else None
        )
        return {
            "object": "costs",
            "started_at": self.started_at,
            "scope": "since_process_start",
            "pricing_mode": "hypothetical"
            if any(m.cost_rates_are_hypothetical for m in self.models.values())
            else "configured",
            "models": [
                {
                    **row,
                    "total_cost_usd": row["known_cost_usd"]
                    if not row["unpriced_calls"] and not row["unreported_usage_calls"]
                    else None,
                }
                for row in self.model_usage.values()
            ],
            "last_request": dict(self.last_request) if self.last_request else None,
            "physical_calls": self.calls,
            "requests": self.requests,
            "reported_input_tokens": self.reported_input_tokens,
            "reported_output_tokens": self.reported_output_tokens,
            "total_cost_usd": self.reported_cost
            if not self.unpriced_calls and not self.unreported_usage_calls
            else None,
            "known_cost_usd": self.reported_cost,
            "unpriced_calls": self.unpriced_calls,
            "unreported_usage_calls": self.unreported_usage_calls,
            "unknown_usage_upper_estimate_usd": self.unknown_usage_upper_estimate,
            "compared_requests": self.compared_requests,
            "excluded_requests": self.requests - self.compared_requests,
            "compared_cost_usd": self.compared_cost,
            "estimated_baseline_cost_usd": self.baseline_cost
            if self.compared_requests
            else None,
            "estimated_saved_usd": savings,
            "estimated_saved_percent": 100 * savings / self.baseline_cost
            if self.baseline_cost and savings is not None
            else None,
            "baseline_method": "same-call reported usage for identical direct routes; o200k text estimate plus final reported output for other routes",
        }

    def emit(self):
        if self.publish:
            self.publish(self.snapshot())
