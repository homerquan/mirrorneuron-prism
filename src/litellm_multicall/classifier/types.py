from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..config import Name, StrictModel

Primitive = Literal["route", "relevance", "sufficient", "retry", "escalate", "stop"]


class Option(StrictModel):
    id: Name
    description: Name

    @field_validator("id", "description")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Criterion(StrictModel):
    id: Name
    primitive: Primitive
    question: Name
    options: list[Option] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def unique(self):
        if not self.id.strip() or not self.question.strip():
            raise ValueError("criterion ID/question must not be blank")
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("option IDs must be unique")
        return self


class DecisionRequest(StrictModel):
    schema_version: Literal[1] = 1
    request_id: Name
    logical_request_id: Name | None = None
    state_version: Name
    state: Any
    criteria: list[Criterion] = Field(min_length=1, max_length=64)

    @field_validator("state")
    @classmethod
    def finite_state(cls, value):
        if not isinstance(value, (str, dict, list)) or not value:
            raise ValueError("state must be a nonempty string, object or array")

        def check(item):
            if isinstance(item, dict):
                if any(not isinstance(k, str) for k in item):
                    raise ValueError("JSON object keys must be strings")
                for v in item.values():
                    check(v)
            elif isinstance(item, list):
                for v in item:
                    check(v)
            elif item is not None and not isinstance(item, (str, int, float, bool)):
                raise ValueError("state must be JSON-compatible")

        check(value)
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def unique(self):
        if not self.request_id.strip() or not self.state_version.strip():
            raise ValueError("request ID/state version must not be blank")
        if len({c.id for c in self.criteria}) != len(self.criteria):
            raise ValueError("criterion IDs must be unique")
        return self

    def rows(self):
        # This allowlist is the entire inference boundary: no evaluator metadata.
        return [
            {
                "id": c.id,
                "state": self.state,
                "question": c.question,
                "options": [o.model_dump() for o in c.options],
            }
            for c in self.criteria
        ]


class DecisionResult(StrictModel):
    schema_version: Literal[1] = 1
    request_id: Name
    logical_request_id: Name | None = None
    state_version: Name
    criterion_id: Name
    primitive: Primitive
    backend: str
    winner_option_id: str | None = None
    accepted_option_id: str | None = None
    option_ids: list[str]
    option_logits: list[float] | None = None
    option_probabilities: list[float] | None = None
    probability_status: str = "uncalibrated_conditional_option_scores"
    confidence: float | None = None
    disposition: Literal["scored", "accept", "abstain", "reject"] = "scored"
    reason: str = "standalone_scoring"
    scoring_mode: str = "direct"
    input_tokens: int | None = None
    generated_output_tokens: int = 0
    model_forward_passes: int | None = 1
    timing_ms: dict[str, float] = Field(default_factory=dict)
    prompt_sha256: str | None = None
    max_score: float | None = None
    score_margin: float | None = None
    entropy: float | None = None

    @model_validator(mode="after")
    def scores(self):
        for values in (self.option_logits, self.option_probabilities):
            if values is not None and (
                len(values) != len(self.option_ids)
                or any(not math.isfinite(v) for v in values)
            ):
                raise ValueError("invalid score count or non-finite score")
        if self.option_probabilities is not None:
            if (
                any(p < 0 or p > 1 for p in self.option_probabilities)
                or abs(sum(self.option_probabilities) - 1) > 1e-6
            ):
                raise ValueError("option probabilities must form a distribution")
        if (
            self.winner_option_id is not None
            and self.winner_option_id not in self.option_ids
        ):
            raise ValueError("winner must reference a declared option")
        if self.accepted_option_id is not None and (
            self.disposition != "accept"
            or self.accepted_option_id != self.winner_option_id
        ):
            raise ValueError("accepted option must match an accepted winner")
        if any(v < 0 or not math.isfinite(v) for v in self.timing_ms.values()):
            raise ValueError("invalid timing")
        return self


class DecisionBatch(StrictModel):
    schema_version: Literal[1] = 1
    request_id: str
    results: list[DecisionResult]
    measurements: dict[str, Any] = Field(default_factory=dict)
    backend_metadata: dict[str, Any] = Field(default_factory=dict)


def result_from_score(request, criterion, raw, mode):
    ids = [o.id for o in criterion.options]
    if raw["option_ids"] != ids:
        raise ValueError("backend changed option ordering")
    probs = raw["probabilities"]
    winner = ids[max(range(len(probs)), key=probs.__getitem__)]
    ordered = sorted(probs, reverse=True)
    forward = raw.get("forward_seconds")
    total = raw.get("total_seconds")
    timing = (
        {}
        if total is None
        else {
            "forward": (forward or 0) * 1000,
            "total": total * 1000,
            "encode_and_readout": max(0, total - (forward or 0)) * 1000,
        }
    )
    return DecisionResult(
        request_id=request.request_id,
        logical_request_id=request.logical_request_id,
        state_version=request.state_version,
        criterion_id=criterion.id,
        primitive=criterion.primitive,
        backend="jev_cpu",
        winner_option_id=winner,
        option_ids=ids,
        option_logits=raw["option_logits"],
        option_probabilities=probs,
        scoring_mode=mode,
        input_tokens=raw["input_tokens"],
        model_forward_passes=None if mode == "shared" else 1,
        generated_output_tokens=1 if mode == "compact_generation" else 0,
        timing_ms=timing,
        prompt_sha256=raw["prompt_sha256"],
        max_score=ordered[0],
        score_margin=ordered[0] - ordered[1],
        entropy=-sum(p * math.log(p) for p in probs if p > 0),
    )
