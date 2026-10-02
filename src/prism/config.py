"""Strict operator-owned JSON configuration. Physical endpoints never come from callers."""

import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import PrismError
from .policies import PolicyName


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RawModel(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_url: str
    api_key: str | None = Field(default=None, repr=False)
    api_key_env: str | None = None
    context_window: int = Field(default=32768, ge=512)
    max_output_tokens: int = Field(default=4096, ge=1)
    safety_margin: int = Field(default=256, ge=0)
    concurrency: int = Field(default=2, ge=1, le=128)
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
        }
    )
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def endpoint(self):
        url = urlsplit(self.base_url)
        if (
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


class Limits(StrictModel):
    max_calls: int = Field(default=64, ge=1, le=4096)
    max_input_tokens: int = Field(default=2_000_000, ge=1)
    max_output_tokens: int = Field(default=65536, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    deadline_seconds: float = Field(default=120, gt=0, le=86400)
    max_partitions: int = Field(default=60, ge=1, le=4096)
    max_parallel: int = Field(default=2, ge=1, le=128)


class Profile(StrictModel):
    direct: str
    worker: str | None = None
    synthesizer: str | None = None
    verifier: str | None = None
    strategy: Literal[
        "auto", "direct", "evidence_map", "batched_map", "verified_map", "retrieve_read"
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
    worker_output_tokens: int = Field(default=1024, ge=128)
    partition_bytes: int = Field(default=6000, ge=128)
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


class PrismConfig(StrictModel):
    schema_version: Literal[1] = 1
    models_file: str = "models.json"
    server: ServerConfig = Field(default_factory=ServerConfig)
    profiles: dict[str, Profile]
    decision: DecisionConfig = Field(default_factory=DecisionConfig)


class ModelFile(StrictModel):
    models: list[RawModel]


def load_config(path):
    path = Path(path).resolve()
    config = PrismConfig.model_validate(json.loads(path.read_text()))
    data = json.loads((path.parent / config.models_file).read_text())
    # Existing {models: [{name, base_url, api_key}]} files continue to work.
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        for model in data["models"]:
            if isinstance(model, dict) and "id" not in model and "name" in model:
                model["id"] = model["name"]
    raw = ModelFile.model_validate(data).models
    models = {model.id: model for model in raw}
    if not models or len(models) != len(raw):
        raise ValueError("raw model IDs must be nonempty and unique")
    if not config.profiles:
        raise ValueError("at least one virtual model profile is required")
    own_urls = [f"http://{config.server.host}:{config.server.port}"]
    if config.server.public_url:
        own_urls.append(config.server.public_url)
    loopback = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}
    for model in raw:
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
        if not alias or any(
            ref and ref not in models
            for ref in (
                profile.direct,
                profile.worker,
                profile.synthesizer,
                profile.verifier,
            )
        ):
            raise ValueError("profile references an unknown raw model")
        for ref in {profile.direct, profile.synthesizer or profile.direct}:
            if profile.public_max_output_tokens > models[ref].max_output_tokens:
                raise ValueError("public output cap exceeds backend output cap")
        worker = models[profile.worker or profile.direct]
        if profile.worker_output_tokens > worker.max_output_tokens:
            raise ValueError("worker output cap exceeds backend output cap")
        verifier = models[profile.verifier or profile.worker or profile.direct]
        if (
            "verified_map" in profile.allowed_policies
            and profile.worker_output_tokens > verifier.max_output_tokens
        ):
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
