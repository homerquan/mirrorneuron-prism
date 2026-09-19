"""Strict typed configuration for MirrorNeuron Prism (Step 02).

Policy/config errors must be detected before model execution. All
operator-facing models use ``extra="forbid"`` so unknown keys are rejected
instead of silently ignored. YAML/Pydantic errors are never swallowed:
loaders raise and the CLI converts them to a non-zero exit.

No model calls are made during validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

__all__ = [
    "AdaptiveOptions",
    "ApiSpec",
    "BudgetSpec",
    "CandidateSpec",
    "ConfigError",
    "FallbackSpec",
    "PolicySpec",
    "PrismConfig",
    "RuntimeConfig",
    "SelectionSpec",
    "extract_litellm_model_groups",
    "extract_multicall_policy_refs",
    "load_litellm_config",
    "load_policy_config",
    "load_yaml",
    "prism_config_json_schema",
    "validate_files",
    "validate_references",
]

POLICY_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

MULTICALL_PREFIX = "multicall/"


class ConfigError(ValueError):
    """Raised for malformed config files (empty, wrong shape, unreadable)."""


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(_Forbid):
    backend: Literal["proxy_router"] = "proxy_router"
    integration_profile: Literal["trusted_local"] = "trusted_local"
    max_inflight_children_per_worker: int = Field(default=4, ge=1)
    max_inflight_children_per_request: int = Field(default=2, ge=1)
    child_cache: Literal["disabled"] = "disabled"
    outer_cache: Literal["disabled"] = "disabled"
    log_content: bool = False


class CandidateSpec(_Forbid):
    model: str = Field(min_length=1)
    samples: int = Field(default=4, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_completion_tokens: int = Field(ge=1)
    prompt_variants: list[str] = Field(default_factory=list)


class SelectionSpec(_Forbid):
    type: Literal["llm_judge"] = "llm_judge"
    model: str = Field(min_length=1)
    max_completion_tokens: int = Field(default=160, ge=1)
    temperature: float = Field(default=0, ge=0, le=2)
    allow_abstain: bool = True


class BudgetSpec(_Forbid):
    max_model_calls: int = Field(ge=1)
    max_backend_attempts: Optional[int] = Field(default=None, ge=1)
    max_total_tokens: Optional[int] = Field(default=None, ge=1)
    deadline_ms: Optional[int] = Field(default=None, ge=1)
    child_timeout_ms: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_attempts_cover_calls(self) -> "BudgetSpec":
        if (
            self.max_backend_attempts is not None
            and self.max_backend_attempts < self.max_model_calls
        ):
            raise ValueError(
                "budgets.max_backend_attempts must be >= budgets.max_model_calls "
                f"({self.max_backend_attempts} < {self.max_model_calls})"
            )
        return self


class AdaptiveOptions(_Forbid):
    initial_samples: int = Field(default=1, ge=1)
    expansion_batches: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    early_stop: Literal["semantic_verifier_pass"] = "semantic_verifier_pass"


class FallbackSpec(_Forbid):
    model: str = Field(min_length=1)
    max_completion_tokens: int = Field(ge=1)


class ApiSpec(_Forbid):
    unsupported: Literal["reject", "passthrough"] = "reject"
    passthrough_model: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_passthrough(self) -> "ApiSpec":
        if self.unsupported == "passthrough" and not self.passthrough_model:
            raise ValueError(
                "api.passthrough_model is required when api.unsupported is 'passthrough'"
            )
        return self


class PolicySpec(_Forbid):
    strategy: Literal["best_of_n", "adaptive"]
    allowed_model_groups: list[str] = Field(min_length=1)
    privacy: Literal["local_only"] = "local_only"
    candidates: list[CandidateSpec] = Field(min_length=1)
    selection: SelectionSpec
    budgets: BudgetSpec
    adaptive: Optional[AdaptiveOptions] = None
    fallback: Optional[FallbackSpec] = None
    on_unresolved: Literal["error", "fallback"] = "error"
    api: ApiSpec = Field(default_factory=ApiSpec)

    @field_validator("allowed_model_groups")
    @classmethod
    def _groups_wellformed(cls, v: list[str]) -> list[str]:
        for g in v:
            if not g or not g.strip():
                raise ValueError(
                    "allowed_model_groups entries must be non-empty strings"
                )
            if g.startswith(MULTICALL_PREFIX):
                raise ValueError(
                    f"allowed model group {g!r} must not route to "
                    f"{MULTICALL_PREFIX}* (recursion guard)"
                )
        return v

    @model_validator(mode="after")
    def _check_coherence(self) -> "PolicySpec":
        allowed = set(self.allowed_model_groups)

        for i, c in enumerate(self.candidates):
            if c.model.startswith(MULTICALL_PREFIX):
                raise ValueError(
                    f"candidates[{i}].model {c.model!r} must not route to "
                    f"{MULTICALL_PREFIX}* (recursion guard)"
                )
            if c.model not in allowed:
                raise ValueError(
                    f"candidates[{i}].model {c.model!r} "
                    "is not in allowed_model_groups"
                )

        if self.selection.model.startswith(MULTICALL_PREFIX):
            raise ValueError(
                f"selection.model {self.selection.model!r} must not route to "
                f"{MULTICALL_PREFIX}* (recursion guard)"
            )
        if self.selection.model not in allowed:
            raise ValueError(
                f"selection.model {self.selection.model!r} "
                "is not in allowed_model_groups"
            )

        if self.strategy == "adaptive" and self.adaptive is None:
            raise ValueError(
                "adaptive block is required when strategy is 'adaptive'"
            )
        if self.strategy == "best_of_n" and self.adaptive is not None:
            raise ValueError(
                "adaptive block is only allowed when strategy is 'adaptive'"
            )

        if self.on_unresolved == "fallback" and self.fallback is None:
            raise ValueError(
                "fallback block is required when on_unresolved is 'fallback'"
            )
        if self.fallback is not None and self.on_unresolved != "fallback":
            raise ValueError("fallback block requires on_unresolved: 'fallback'")
        if self.fallback is not None:
            if self.fallback.model.startswith(MULTICALL_PREFIX):
                raise ValueError(
                    f"fallback.model {self.fallback.model!r} must not route to "
                    f"{MULTICALL_PREFIX}* (recursion guard)"
                )
            if self.fallback.model not in allowed:
                raise ValueError(
                    f"fallback.model {self.fallback.model!r} "
                    "is not in allowed_model_groups"
                )

        if self.api.unsupported == "passthrough" and self.api.passthrough_model:
            pm = self.api.passthrough_model
            if pm.startswith(MULTICALL_PREFIX):
                raise ValueError(
                    f"api.passthrough_model {pm!r} must not route to "
                    f"{MULTICALL_PREFIX}* (recursion guard)"
                )
            if pm not in allowed:
                raise ValueError(
                    f"api.passthrough_model {pm!r} is not in allowed_model_groups"
                )

        if self.strategy == "adaptive" and self.adaptive is not None:
            planned = self.adaptive.initial_samples + sum(
                self.adaptive.expansion_batches
            )
            plan_desc = (
                f"adaptive initial ({self.adaptive.initial_samples}) + "
                f"batches ({sum(self.adaptive.expansion_batches)})"
            )
        else:
            planned = sum(c.samples for c in self.candidates)
            plan_desc = f"candidate samples ({planned})"
        required = planned + 1  # selection step
        if self.fallback is not None:
            required += 1
        if self.budgets.max_model_calls < required:
            raise ValueError(
                f"budgets.max_model_calls={self.budgets.max_model_calls} is "
                f"insufficient: need >= {required} "
                f"({plan_desc} + 1 selection"
                f"{' + 1 fallback' if self.fallback is not None else ''})"
            )
        return self


class PrismConfig(_Forbid):
    """Root of multicall.yaml. ``schema_version`` is strictly integer 1."""

    schema_version: StrictInt
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    policies: dict[str, PolicySpec] = Field(min_length=1)
    prompt_variants: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _version_must_be_one(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"schema_version must be 1, got {v!r}")
        return v

    @field_validator("policies")
    @classmethod
    def _policy_ids_valid(cls, v: dict[str, PolicySpec]) -> dict[str, PolicySpec]:
        for key in v:
            if not key or not POLICY_ID_RE.match(key):
                raise ValueError(
                    f"invalid policy id {key!r}: use [A-Za-z0-9_-]+"
                )
        return v

    @model_validator(mode="after")
    def _check_prompt_variants(self) -> "PrismConfig":
        for pid, policy in self.policies.items():
            for i, cand in enumerate(policy.candidates):
                for name in cand.prompt_variants:
                    if name not in self.prompt_variants:
                        raise ValueError(
                            f"policies[{pid!r}].candidates[{i}]."
                            f"prompt_variants[{name!r}] is not defined "
                            "in top-level prompt_variants"
                        )
        return self


def _read_yaml_mapping(path: Path | str, *, what: str) -> dict[str, Any]:
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if raw is None:
        raise ConfigError(f"{what} is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{what} must be a YAML mapping at top level, "
            f"got {type(raw).__name__}: {path}"
        )
    return raw


def load_policy_config(path: Path | str) -> PrismConfig:
    """Load and strictly validate a multicall.yaml file.

    Raises :class:`ConfigError`, :class:`yaml.YAMLError`, or
    :class:`pydantic.ValidationError`. Never returns ``{}``.
    """
    raw = _read_yaml_mapping(path, what="policy config")
    return PrismConfig.model_validate(raw)


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping without swallowing parse errors."""
    return _read_yaml_mapping(path, what="YAML config")


def load_litellm_config(path: Path | str) -> dict[str, Any]:
    """Load a LiteLLM config file as a raw mapping (no network calls)."""
    return _read_yaml_mapping(path, what="litellm config")


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
                policy_id = target[len(MULTICALL_PREFIX):]
                refs.setdefault(policy_id, []).append(public)
    return refs


def validate_references(
    prism: PrismConfig, litellm_cfg: dict[str, Any]
) -> list[str]:
    """Cross-check a validated policy config against a LiteLLM config mapping.

    No network calls are made. Returns a list of human-readable errors
    (empty when everything resolves).
    """
    errors: list[str] = []
    try:
        groups = extract_litellm_model_groups(litellm_cfg)
    except ConfigError as exc:
        return [str(exc)]
    if not groups:
        return ["litellm config defines no model_list entries with model_name"]

    refs = extract_multicall_policy_refs(litellm_cfg)
    for policy_id, public_models in refs.items():
        if policy_id not in prism.policies:
            errors.append(
                f"litellm model(s) {public_models} route to "
                f"multicall/{policy_id} which is not defined in policies"
            )

    for pid, policy in prism.policies.items():
        for group in policy.allowed_model_groups:
            if group not in groups:
                errors.append(
                    f"policies[{pid!r}].allowed_model_groups[{group!r}] "
                    "is not defined in litellm model_list"
                )
        for i, cand in enumerate(policy.candidates):
            if cand.model not in groups:
                errors.append(
                    f"policies[{pid!r}].candidates[{i}].model[{cand.model!r}] "
                    "is not defined in litellm model_list"
                )
        if policy.selection.model not in groups:
            errors.append(
                f"policies[{pid!r}].selection.model[{policy.selection.model!r}] "
                "is not defined in litellm model_list"
            )
        if policy.fallback is not None and policy.fallback.model not in groups:
            errors.append(
                f"policies[{pid!r}].fallback.model[{policy.fallback.model!r}] "
                "is not defined in litellm model_list"
            )
        if (
            policy.api.unsupported == "passthrough"
            and policy.api.passthrough_model
            and policy.api.passthrough_model not in groups
        ):
            errors.append(
                f"policies[{pid!r}].api.passthrough_model"
                f"[{policy.api.passthrough_model!r}] "
                "is not defined in litellm model_list"
            )
    return errors


def validate_files(policy_path: Path | str, litellm_path: Path | str) -> list[str]:
    """Validate both config files. Returns errors; raises on parse failures.

    Raises :class:`ConfigError`, :class:`yaml.YAMLError`, or
    :class:`pydantic.ValidationError` for malformed files instead of
    returning an empty dict. Reference mismatches are returned as strings.
    """
    prism = load_policy_config(policy_path)
    litellm_cfg = load_litellm_config(litellm_path)
    return validate_references(prism, litellm_cfg)


def prism_config_json_schema() -> dict[str, Any]:
    """Return the JSON schema generated from the strict Pydantic model."""
    return PrismConfig.model_json_schema()
