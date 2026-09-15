from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

@dataclass
class RequestMeter:
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

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Placeholder: real implementation would query litellm pricing table
    # Local models are free for demo purposes
    return 0.0

def get_price_estimate(meter: RequestMeter) -> Dict[str, float]:
    estimates = {}
    for model, usage in meter.model_usage.items():
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        estimates[model] = round(cost, 6)
    return estimates
