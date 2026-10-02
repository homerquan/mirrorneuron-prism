"""Offline, strict schema-v1 policy validation."""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Name = Annotated[str, Field(min_length=1)]
Positive = Annotated[int, Field(gt=0, strict=True)]


class Runtime(StrictModel):
    backend: Literal["proxy_router"] = "proxy_router"
    integration_profile: Literal["trusted_local", "authenticated_proxy"] = (
        "trusted_local"
    )
    max_inflight_children_per_worker: Positive = 4
    max_inflight_children_per_request: Positive = 2
    child_cache: Literal["disabled"] = "disabled"
    outer_cache: Literal["disabled"] = "disabled"
    log_content: bool = False


class Generation(StrictModel):
    model: Name
    max_completion_tokens: Positive = 1500
    temperature: float = Field(default=0, ge=0, le=2)


class CandidateConfig(Generation):
    samples: Positive = 1
    prompt_variants: list[Name] = Field(default_factory=list)


class Selection(Generation):
    type: Literal["llm_judge"] = "llm_judge"
    allow_abstain: bool = True


class Budgets(StrictModel):
    max_model_calls: Positive
    max_backend_attempts: Positive | None = None
    max_total_tokens: Positive | None = None
    deadline_ms: Positive | None = None
    child_timeout_ms: Positive | None = None

    @model_validator(mode="after")
    def attempts(self):
        if (
            self.max_backend_attempts is not None
            and self.max_backend_attempts < self.max_model_calls
        ):
            raise ValueError("max_backend_attempts must cover max_model_calls")
        return self


class Adaptive(StrictModel):
    initial_samples: Positive = 1
    expansion_batches: list[Positive] = Field(default_factory=list)
    early_stop: Literal["semantic_verifier_pass"] = "semantic_verifier_pass"


class Decompose(StrictModel):
    planner_model: Name
    executor_model: Name
    min_subjobs: Positive
    max_subjobs: Positive
    decomposition_prompt: Name


class Execution(StrictModel):
    concurrency: Positive = 1
    max_retries: int = Field(default=0, ge=0)


class Combine(StrictModel):
    model: Name
    prompt: Name


class APISettings(StrictModel):
    unsupported: Literal["reject", "passthrough"] = "reject"
    passthrough_model: Name | None = None


class Policy(StrictModel):
    strategy: Literal["best_of_n", "adaptive", "decompose_map_reduce"]
    allowed_model_groups: list[Name] = Field(min_length=1)
    privacy: Literal["local_only", "allow_cloud"] = "local_only"
    candidates: list[CandidateConfig] = Field(default_factory=list)
    selection: Selection | None = None
    adaptive: Adaptive | None = None
    decompose: Decompose | None = None
    execution: Execution | None = None
    combine: Combine | None = None
    fallback: Generation | None = None
    budgets: Budgets
    on_unresolved: Literal["error", "fallback"] = "error"
    api: APISettings = Field(default_factory=APISettings)

    @model_validator(mode="after")
    def references(self):
        if len(set(self.allowed_model_groups)) != len(self.allowed_model_groups):
            raise ValueError("allowed_model_groups must be unique")
        if any(
            not g.strip() or g.startswith("multicall/")
            for g in self.allowed_model_groups
        ):
            raise ValueError("allowed groups must be nonempty physical model names")
        if self.fallback and self.on_unresolved != "fallback":
            raise ValueError("fallback requires on_unresolved: fallback")
        if self.api.unsupported == "passthrough" and not self.api.passthrough_model:
            raise ValueError("passthrough requires a model")
        refs = [c.model for c in self.candidates]
        refs += [s.model for s in (self.selection, self.fallback, self.combine) if s]
        if self.api.passthrough_model:
            refs.append(self.api.passthrough_model)
        if self.decompose:
            refs += [self.decompose.planner_model, self.decompose.executor_model]
        if set(refs) - set(self.allowed_model_groups):
            raise ValueError("every model reference must be in allowed_model_groups")
        if self.on_unresolved == "fallback" and not self.fallback:
            raise ValueError("fallback policy requires a declared fallback")
        if self.strategy in ("best_of_n", "adaptive"):
            if not self.candidates or not self.selection:
                raise ValueError("candidate strategy requires candidates and selection")
            if self.decompose or self.combine or self.execution:
                raise ValueError("decomposition settings require decompose_map_reduce")
            calls = sum(c.samples for c in self.candidates) + 1 + bool(self.fallback)
            if self.strategy == "adaptive":
                if not self.adaptive:
                    raise ValueError("adaptive strategy requires adaptive settings")
                if self.adaptive.initial_samples + sum(
                    self.adaptive.expansion_batches
                ) != sum(c.samples for c in self.candidates):
                    raise ValueError(
                        "adaptive sample schedule must equal candidate sample count"
                    )
            elif self.adaptive:
                raise ValueError("adaptive settings require adaptive strategy")
        else:
            if not self.decompose or not self.combine or not self.execution:
                raise ValueError(
                    "decomposition requires decompose, execution and combine"
                )
            if self.candidates or self.selection or self.adaptive:
                raise ValueError(
                    "candidate settings are incompatible with decomposition"
                )
            if self.decompose.min_subjobs > self.decompose.max_subjobs:
                raise ValueError("min_subjobs exceeds max_subjobs")
            calls = 2 + self.decompose.max_subjobs + bool(self.fallback)
        if self.budgets.max_model_calls < calls:
            raise ValueError("max_model_calls does not cover the declared plan")
        return self


class PrismConfig(StrictModel):
    schema_version: StrictInt
    runtime: Runtime = Field(default_factory=Runtime)
    policies: dict[Name, Policy] = Field(min_length=1)
    prompt_variants: dict[Name, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def version_one(cls, value):
        if value != 1:
            raise ValueError("schema_version must be 1")
        return value

    @field_validator("policies")
    @classmethod
    def policy_ids(cls, value):
        if not value or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in value):
            raise ValueError("invalid policy id")
        return value

    @model_validator(mode="after")
    def prompts(self):
        for policy in self.policies.values():
            refs = [p for c in policy.candidates for p in c.prompt_variants]
            if policy.decompose:
                refs.append(policy.decompose.decomposition_prompt)
            if policy.combine:
                refs.append(policy.combine.prompt)
            if set(refs) - set(self.prompt_variants):
                raise ValueError("unknown prompt variant")
        return self


def load_yaml(path: Path | str) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    return data


def load_policy_config(path: Path | str) -> PrismConfig:
    return PrismConfig.model_validate(load_yaml(path))


def local_endpoint(value: str) -> bool:
    if value.startswith("os.environ/"):
        value = os.environ.get(value.split("/", 1)[1], "")
    host = urlsplit(value).hostname
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host or "")
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def _validate_references(policy: dict, litellm: dict) -> None:
    entries = litellm.get("model_list")
    if not isinstance(entries, list) or not entries:
        raise ValueError("LiteLLM model_list must be a nonempty list")
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not entry.get("model_name")
            or not isinstance(entry.get("litellm_params"), dict)
        ):
            raise ValueError("invalid LiteLLM model entry")
        params = entry["litellm_params"]
        model = params.get("model", "")
        if not isinstance(model, str) or not model:
            raise ValueError("model entry requires a model string")
        groups.setdefault(entry["model_name"], []).append(params)
        if (
            model.startswith("multicall/")
            and model.split("/", 1)[1] not in policy["policies"]
        ):
            raise ValueError(f"logical alias references unknown policy: {model}")
    for p in policy["policies"].values():
        for name in p["allowed_model_groups"]:
            if name not in groups:
                raise ValueError(f"unknown physical model group: {name}")
            for params in groups[name]:
                if params["model"].startswith("multicall/"):
                    raise ValueError("recursive multicall model reference")
                if p["privacy"] == "local_only" and not (
                    params["model"].startswith(("ollama/", "ollama_chat/"))
                    or local_endpoint(params.get("api_base", ""))
                ):
                    raise ValueError(
                        f"local_only requires an explicitly local deployment: {name}"
                    )


# Public configuration API retained from remote main.
ConfigError = ValueError
MULTICALL_PREFIX = "multicall/"
RuntimeConfig = Runtime
CandidateSpec = CandidateConfig
SelectionSpec = Selection
BudgetSpec = Budgets
AdaptiveOptions = Adaptive
FallbackSpec = Generation
ApiSpec = APISettings
PolicySpec = Policy
load_litellm_config = load_yaml


def extract_litellm_model_groups(litellm_cfg: dict[str, Any]) -> set[str]:
    """Return the set of ``model_name`` values from a LiteLLM config mapping."""
    groups: set[str] = set()
    model_list = litellm_cfg.get("model_list", [])
    if not isinstance(model_list, list):
        raise ConfigError("litellm config 'model_list' must be a list")
    for entry in model_list:
        if isinstance(entry, dict) and isinstance(entry.get("model_name"), str):
            groups.add(entry["model_name"])
    return groups


def extract_multicall_policy_refs(
    litellm_cfg: dict[str, Any],
) -> dict[str, list[str]]:
    """Map ``multicall/<policy>`` targets to the public model names using them."""
    refs: dict[str, list[str]] = {}
    for entry in litellm_cfg.get("model_list", []) or []:
        if not isinstance(entry, dict):
            continue
        public = entry.get("model_name")
        params = entry.get("litellm_params", {})
        target = params.get("model") if isinstance(params, dict) else None
        if isinstance(public, str) and isinstance(target, str):
            if target == MULTICALL_PREFIX.rstrip("/"):
                continue
            if target.startswith(MULTICALL_PREFIX):
                policy_id = target[len(MULTICALL_PREFIX) :]
                refs.setdefault(policy_id, []).append(public)
    return refs


def validate_references(policy: PrismConfig | dict, litellm: dict) -> list[str]:
    if isinstance(policy, PrismConfig):
        policy = policy.model_dump(mode="json")
    try:
        _validate_references(policy, litellm)
    except ValueError as exc:
        return [str(exc)]
    return []


def validate_files(policy_path: Path | str, litellm_path: Path | str) -> list[str]:
    return validate_references(
        load_policy_config(policy_path), load_litellm_config(litellm_path)
    )


def prism_config_json_schema() -> dict:
    return PrismConfig.model_json_schema()
