"""Strict operator-owned JSON configuration. Physical endpoints never come from callers."""

import json
import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import OptimizationConfigurationError, PrismError
from .policies import PolicyName


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RawModel(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_url: str | None = None
    # Explicit SDK transport; OpenAI-compatible URLs default to openai.
    provider: str | None = None
    api_version: str | None = None
    parameters: dict = Field(default_factory=dict, repr=False)
    provider_options: dict = Field(default_factory=dict, repr=False)
    timeout_seconds: float = Field(default=120, gt=0, le=86400)
    rate_limit_rpm: int | None = Field(default=None, ge=1)
    image_token_reserve: int = Field(default=4096, ge=1)
    api_key: str | None = Field(default=None, repr=False)
    api_key_env: str | None = None
    context_window: int = Field(default=32768, ge=512)
    max_output_tokens: int = Field(default=4096, ge=1)
    safety_margin: int = Field(default=256, ge=0)
    concurrency: int = Field(default=2, ge=1, le=128)
    enable_thinking: bool | None = Field(default=None, strict=True)
    # Admission uses an explicit UTF-8 byte bound, including framing overhead.
    # Operators must validate this bound against their backend tokenizer/template.
    tokens_per_byte_bound: float = Field(default=1.0, ge=1.0)
    chat_overhead_tokens: int = Field(default=64, ge=16)
    capabilities: set[str] = Field(
        default_factory=lambda: {
            "text",
            "stream",
            "tools",
            "json_object",
            "json_schema",
            "image",
            "reasoning_effort",
        }
    )
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    cost_rates_are_hypothetical: bool = Field(default=False, strict=True)
    power_rating: int | None = Field(default=None, ge=1, le=10, strict=True)

    @field_validator("input_cost_per_million", "output_cost_per_million", mode="before")
    @classmethod
    def price_per_million(cls, value):
        if isinstance(value, str):
            match = re.fullmatch(
                r"\s*\$?([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:1?m|million)\s*", value, re.I
            )
            if not match:
                raise ValueError('price must be USD per million tokens, e.g. "$5/m"')
            return float(match[1])
        if isinstance(value, bool):
            raise ValueError("price must be a number or a per-million price string")
        return value

    @model_validator(mode="after")
    def endpoint(self):
        url = urlsplit(self.base_url or "")
        if self.base_url is not None and (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials/query"
            )
        if not self.base_url and not self.provider and "/" not in self.name:
            raise ValueError("native models require a LiteLLM provider or model prefix")
        # Defaults may tune inference, but cannot change identity, credentials,
        # accounting, retry behavior, or execute SDK callbacks/mock functions.
        from .registry import validate_parameters

        validate_parameters(self.parameters, self.provider_options)
        for key in ("max_tokens", "max_completion_tokens"):
            if self.parameters.get(key, 0) > self.max_output_tokens:
                raise ValueError("model output default exceeds backend output cap")
        if self.api_key and self.api_key_env:
            raise ValueError("use api_key or api_key_env, not both")
        if self.safety_margin + self.max_output_tokens >= self.context_window:
            raise ValueError("output reserve and margin must fit context_window")
        return self

    def credential(self):
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if not value:
                raise PrismError(
                    "backend credential environment variable is unset",
                    "configuration_error",
                    503,
                )
            return value
        return self.api_key

    def call_parameters(self, parameters, output):
        defaults = {k: v for k, v in self.parameters.items() if k != "stream"}
        spelling = next(
            (k for k in ("max_completion_tokens", "max_tokens") if k in parameters),
            next(
                (k for k in ("max_completion_tokens", "max_tokens") if k in defaults),
                "max_completion_tokens",
            ),
        )
        defaults.pop("max_tokens", None)
        defaults.pop("max_completion_tokens", None)
        result = {**defaults, **parameters, spelling: output}
        if self.enable_thinking is not None and "reasoning_effort" not in result:
            result["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return result


class Limits(StrictModel):
    max_calls: int = Field(default=64, ge=1, le=4096)
    max_input_tokens: int = Field(default=2_000_000, ge=1)
    max_output_tokens: int = Field(default=65536, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    deadline_seconds: float = Field(default=120, gt=0, le=86400)
    max_partitions: int = Field(default=60, ge=1, le=4096)
    max_parallel: int = Field(default=2, ge=1, le=128)


class OptimizationConfig(StrictModel):
    model_ids: list[str] = Field(min_length=1)
    default_cost_priority: float = Field(default=0.5, ge=0, le=1, strict=True)
    enforce_stage_power_order: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def unique_models(self):
        if any(not ref for ref in self.model_ids) or len(set(self.model_ids)) != len(
            self.model_ids
        ):
            raise ValueError("optimization model_ids must be nonempty and unique")
        return self


class EvidenceCompaction(StrictModel):
    model: str
    output_tokens: int = Field(default=512, ge=128, strict=True)
    memory_max_bytes: int = Field(default=1024, ge=128, le=16384, strict=True)
    max_calls: int = Field(default=16, ge=1, le=256, strict=True)


class EvidenceReduction(StrictModel):
    model: str
    max_input_tokens: int = Field(default=8000, ge=1024, strict=True)
    output_tokens: int = Field(default=1024, ge=128, strict=True)
    state_max_tokens: int = Field(default=1024, ge=256, le=4096, strict=True)
    fanout: int = Field(default=8, ge=2, le=32, strict=True)
    max_calls: int = Field(default=16, ge=1, le=256, strict=True)
    lookup_rounds: int = Field(default=2, ge=0, le=8, strict=True)
    evidence_max_tokens: int = Field(default=1024, ge=256, strict=True)
    verification_max_extra_calls: int = Field(default=32, ge=0, le=256, strict=True)


class Profile(StrictModel):
    direct: str
    worker: str | None = None
    worker_fallback: str | None = None
    evidence_compaction: EvidenceCompaction | None = None
    evidence_reduction: EvidenceReduction | None = None
    synthesizer: str | None = None
    structured_output_model: str | None = None
    verifier: str | None = None
    cost_baseline_model: str | None = None
    strategy: Literal[
        "auto",
        "direct",
        "evidence_map",
        "batched_map",
        "verified_map",
        "retrieve_read",
        "draft_review",
        "vision_synthesis",
        "text_synthesis",
    ] = "auto"
    allowed_policies: list[PolicyName] = Field(
        default_factory=lambda: [
            "direct",
            "evidence_map",
            "batched_map",
            "verified_map",
            "retrieve_read",
        ],
        min_length=1,
    )
    coverage: Literal["exhaustive", "focused"] = "exhaustive"
    batch_max_partitions: int = Field(default=4, ge=1, le=32)
    retrieval_top_k: int = Field(default=4, ge=1, le=128)
    public_max_output_tokens: int = Field(default=2048, ge=1)
    worker_output_tokens: int = Field(default=1024, ge=1)
    partition_bytes: int = Field(default=6000, ge=128)
    intermediate_max_bytes: int = Field(default=4096, ge=1, le=65536)
    optimization: OptimizationConfig | None = None
    limits: Limits = Field(default_factory=Limits)


class DecisionConfig(StrictModel):
    mode: Literal["route", "shadow"] = "route"
    model: str = "convaiinnovations/laya-typed-decisions"
    revision: str | None = None
    expected_sha256: dict[str, str] | None = None
    max_len: int = Field(default=1024, ge=128, le=8192)
    max_state_bytes: int = Field(default=1200, ge=128, le=8192)
    min_option_confidence: float = Field(default=0.7, ge=0, le=1)


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key_env: str = "PRISM_API_KEY"
    max_payload_bytes: int = Field(default=16_000_000, ge=1024)
    max_virtual_input_bytes: int = Field(default=12_000_000, ge=1024)
    max_concurrent_requests: int = Field(default=16, ge=1, le=1024)
    trace_capacity: int = Field(default=256, ge=0, le=10000)
    trace_ttl_seconds: float = Field(default=3600, gt=0)
    public_url: str | None = None


class CapacityConfig(StrictModel):
    ttl_seconds: float = Field(default=60, ge=0, le=86400)
    probe_timeout_seconds: float = Field(default=30, gt=0, le=600)
    output_tokens: int = Field(default=512, ge=64, le=4096)
    max_parallel: int = Field(default=2, ge=1, le=16)


class PrismConfig(StrictModel):
    schema_version: Literal[1] = 1
    models_file: str = "models.json"
    server: ServerConfig = Field(default_factory=ServerConfig)
    profiles: dict[str, Profile] = Field(default_factory=dict)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)


class ModelFile(StrictModel):
    models: list[RawModel]


def validate_optimization(profile, models):
    if profile.optimization is None:
        return
    for ref in profile.optimization.model_ids:
        model = models.get(ref)
        if model is None:
            raise OptimizationConfigurationError(
                "optimization references an unknown raw model"
            )
        if (
            model.power_rating is None
            or model.input_cost_per_million is None
            or model.output_cost_per_million is None
        ):
            raise OptimizationConfigurationError(
                "optimization requires power_rating and both model prices"
            )


def load_config(path):
    path = Path(path).resolve()
    config = PrismConfig.model_validate(json.loads(path.read_text()))
    from .registry import read_models

    raw = [
        RawModel.model_validate(item)
        for item in read_models(path.parent / config.models_file)
    ]
    models = {model.id: model for model in raw}
    if not models or len(models) != len(raw):
        raise ValueError("raw model IDs must be nonempty and unique")
    return validate_config(config, models, expose_raw_aliases=True)


def validate_config(config, models, *, expose_raw_aliases=False):
    """Apply the same admission checks to standalone profiles and legacy configs."""
    raw = list(models.values())
    # Profiles only define orchestration. Every registry combination is callable
    # without repeating any physical model metadata in prism.json.
    for model in raw if expose_raw_aliases else []:
        config.profiles.setdefault(
            model.id,
            Profile(
                direct=model.id,
                strategy="direct",
                allowed_policies=["direct"],
                public_max_output_tokens=min(
                    model.max_output_tokens,
                    model.parameters.get(
                        "max_completion_tokens",
                        model.parameters.get("max_tokens", 2048),
                    ),
                ),
                worker_output_tokens=min(1024, model.max_output_tokens),
            ),
        )
    own_urls = [f"http://{config.server.host}:{config.server.port}"]
    if config.server.public_url:
        own_urls.append(config.server.public_url)
    loopback = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}
    for model in raw:
        if not model.base_url:
            continue
        target = urlsplit(model.base_url)
        target_port = target.port or (443 if target.scheme == "https" else 80)
        for own in own_urls:
            url = urlsplit(own)
            port = url.port or (443 if url.scheme == "https" else 80)
            same_host = target.hostname == url.hostname or (
                target.hostname in loopback and url.hostname in loopback
            )
            if same_host and port == target_port:
                raise ValueError("self-referential Prism backend endpoint")
    for alias, profile in config.profiles.items():
        validate_optimization(profile, models)
        if not alias or any(
            ref and ref not in models
            for ref in (
                profile.direct,
                profile.worker,
                profile.worker_fallback,
                profile.synthesizer,
                profile.structured_output_model,
                profile.verifier,
                profile.cost_baseline_model,
                profile.evidence_compaction.model
                if profile.evidence_compaction
                else None,
                profile.evidence_reduction.model
                if profile.evidence_reduction
                else None,
            )
        ):
            raise ValueError("profile references an unknown raw model")
        # Optimized assignments are admitted per request/output size. Unused
        # legacy stage references must not constrain heterogeneous model pools.
        if profile.optimization is None:
            for ref in {
                profile.direct,
                profile.synthesizer or profile.direct,
                profile.structured_output_model or profile.direct,
            }:
                if profile.public_max_output_tokens > models[ref].max_output_tokens:
                    raise ValueError("public output cap exceeds backend output cap")
            worker = models[profile.worker or profile.direct]
            if profile.worker_output_tokens > worker.max_output_tokens:
                raise ValueError("worker output cap exceeds backend output cap")
            if profile.worker_fallback:
                fallback = models[profile.worker_fallback]
                if profile.worker_output_tokens > fallback.max_output_tokens:
                    raise ValueError("worker output cap exceeds fallback output cap")
                if (
                    worker.power_rating is not None
                    and fallback.power_rating is not None
                    and fallback.power_rating < worker.power_rating
                ):
                    raise ValueError("worker fallback must be at least as powerful")
            verifier = models[profile.verifier or profile.worker or profile.direct]
            if {"verified_map", "draft_review"} & set(
                profile.allowed_policies
            ) and profile.worker_output_tokens > verifier.max_output_tokens:
                raise ValueError("verification output cap exceeds backend output cap")
        if profile.strategy == "retrieve_read" and profile.coverage != "focused":
            raise ValueError("retrieve_read requires explicitly focused coverage")
        if len(set(profile.allowed_policies)) != len(profile.allowed_policies):
            raise ValueError("allowed_policies must be unique")
        if (
            profile.strategy != "auto"
            and profile.strategy not in profile.allowed_policies
        ):
            raise ValueError("forced strategy must be present in allowed_policies")
    return config, models
