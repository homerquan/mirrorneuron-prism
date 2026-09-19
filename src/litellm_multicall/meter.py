from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

@dataclass
class RequestMeter:
    """Offline per-logical-request usage ledger (Step 01 baseline).

    Aggregates known token counts per physical model group. It creates no
    charges and owns no billing: leaf-call accounting through LiteLLM lands
    in the governed profile (compat/accounting.py) in a later step.
    """
    logical_request_id: str
    model_usage: Dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        u = self.model_usage.setdefault(model, ModelUsage())
        u.calls += 1
        u.prompt_tokens += prompt_tokens
        u.completion_tokens += completion_tokens

    def aggregate(self):
        total_prompt = sum(u.prompt_tokens for u in self.model_usage.values())
        total_completion = sum(u.completion_tokens for u in self.model_usage.values())
        return {
            "model_usage": {k: vars(v) for k,v in self.model_usage.items()},
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Return a cost estimate in USD, or None when pricing is unknown.

    Step 01 baseline: LiteLLM price-table lookup is NOT wired yet, so pricing
    is always unknown. Returning None (not 0.0) is deliberate: a silent zero
    would misreport consumed compute as free and could bypass budget checks.
    """
    return None

def get_price_estimate(meter: RequestMeter) -> Dict[str, Optional[float]]:
    """Per-model cost estimates; values are None while pricing is unwired."""
    return {
        model: estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        for model, usage in meter.model_usage.items()
    }
