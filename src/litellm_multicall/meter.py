from dataclasses import dataclass, field


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


@dataclass
class RequestMeter:
    logical_request_id: str
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    controller_batches: list[dict] = field(default_factory=list)

    def record_controller(self, measurements: dict, *, role: str, request_id: str):
        """One physical batch record; shared prefix work is never counted per criterion."""
        self.controller_batches.append(
            {**measurements, "role": role, "request_id": request_id}
        )

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        u = self.model_usage.setdefault(model, ModelUsage())
        u.calls += 1
        u.prompt_tokens += prompt_tokens
        u.completion_tokens += completion_tokens

    def aggregate(self):
        total_prompt = sum(u.prompt_tokens for u in self.model_usage.values())
        total_completion = sum(u.completion_tokens for u in self.model_usage.values())
        return {
            "model_usage": {k: vars(v) for k, v in self.model_usage.items()},
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "controller_batches": self.controller_batches,
        }


def estimate_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    # No pinned pricing source exists yet. Unknown cost is never zero.
    return None


def get_price_estimate(meter: RequestMeter) -> dict[str, float | None]:
    estimates = {}
    for model, usage in meter.model_usage.items():
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        estimates[model] = round(cost, 6) if cost is not None else None
    return estimates
