"""Legacy-behavior regression lock for Step 01 (adaptive_v2 baseline).

These tests prove the CURRENT Best-of-N / legacy Adaptive / ordinary-alias
behavior is unchanged BEFORE adaptive_v2 is added:

- Legacy strategy configs load unchanged (exact budgets, no silent migration).
- Ordinary non-Prism LiteLLM aliases never route through MulticallProvider.
- No new calls/symbols appear for existing policies (adaptive_v2 is opt-in
  and does not exist yet).
- The tested LiteLLM version stays inside the declared dependency range.

Deterministic: file fixtures and package metadata only. No live proxy,
no GPU, no paid APIs, no network calls.
"""

from importlib.metadata import version as dist_version
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from litellm_multicall import provider as provider_mod
from litellm_multicall.budgets import BudgetTracker
from litellm_multicall.config import (
    PrismConfig,
    extract_multicall_policy_refs,
    load_litellm_config,
    load_policy_config,
    validate_references,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _best_of_n_dict():
    return {
        "schema_version": 1,
        "runtime": {"backend": "proxy_router", "integration_profile": "trusted_local"},
        "policies": {
            "default": {
                "strategy": "best_of_n",
                "allowed_model_groups": ["local-small"],
                "candidates": [
                    {
                        "model": "local-small",
                        "samples": 4,
                        "temperature": 0.7,
                        "max_completion_tokens": 1500,
                    }
                ],
                "selection": {
                    "type": "llm_judge",
                    "model": "local-small",
                    "max_completion_tokens": 160,
                    "temperature": 0,
                },
                "budgets": {
                    "max_model_calls": 5,
                    "max_total_tokens": 16000,
                    "deadline_ms": 120000,
                },
            }
        },
    }


def _adaptive_dict():
    return {
        "schema_version": 1,
        "runtime": {"backend": "proxy_router", "integration_profile": "trusted_local"},
        "policies": {
            "default": {
                "strategy": "adaptive",
                "allowed_model_groups": ["local-small", "local-strong"],
                "candidates": [
                    {
                        "model": "local-small",
                        "samples": 4,
                        "temperature": 0.7,
                        "max_completion_tokens": 1500,
                    }
                ],
                "adaptive": {"initial_samples": 1, "expansion_batches": [3]},
                "selection": {
                    "type": "llm_judge",
                    "model": "local-small",
                    "max_completion_tokens": 160,
                    "temperature": 0,
                },
                "fallback": {
                    "model": "local-strong",
                    "max_completion_tokens": 1500,
                },
                "on_unresolved": "fallback",
                # 1 initial + 3 expansion + 1 judge + 1 fallback.
                "budgets": {"max_model_calls": 6},
            }
        },
    }


def _litellm_dict_with_ordinary_and_prism():
    return {
        "model_list": [
            {
                "model_name": "local-small",
                "litellm_params": {"model": "openai/local-small-model"},
            },
            {
                "model_name": "plain-direct",
                "litellm_params": {"model": "openai/plain-model"},
            },
            {
                "model_name": "smart-local",
                "litellm_params": {"model": "multicall/default"},
            },
        ]
    }


# --- legacy strategy configs load unchanged -------------------------------


def test_legacy_best_of_n_config_loads_unchanged():
    prism = PrismConfig.model_validate(_best_of_n_dict())
    policy = prism.policies["default"]
    assert policy.strategy == "best_of_n"
    assert policy.adaptive is None
    assert policy.fallback is None
    assert policy.on_unresolved == "error"
    assert sum(c.samples for c in policy.candidates) == 4
    # 4 candidates + 1 judge, exactly; no room for extra calls.
    assert policy.budgets.max_model_calls == 5


def test_legacy_adaptive_config_loads_unchanged():
    prism = PrismConfig.model_validate(_adaptive_dict())
    policy = prism.policies["default"]
    assert policy.strategy == "adaptive"
    assert policy.adaptive is not None
    assert policy.adaptive.initial_samples == 1
    assert policy.adaptive.expansion_batches == [3]
    assert policy.adaptive.early_stop == "semantic_verifier_pass"
    # 1 initial + 3 expansion + 1 judge + 1 fallback, exactly.
    assert policy.budgets.max_model_calls == 6


def test_legacy_config_reload_is_stable_no_silent_migration(tmp_path):
    """Loading a legacy config twice yields identical policies: no upgrade."""
    path = tmp_path / "multicall.yaml"
    path.write_text(yaml.safe_dump(_best_of_n_dict()))
    first = load_policy_config(path).model_dump(mode="python")
    second = load_policy_config(path).model_dump(mode="python")
    assert first == second
    assert second["policies"]["default"]["strategy"] == "best_of_n"
    assert "adaptive_v2" not in yaml.safe_dump(second)


def test_example_pair_still_validates_cleanly():
    prism = load_policy_config(REPO_ROOT / "examples" / "multicall.yaml")
    litellm_cfg = load_litellm_config(REPO_ROOT / "examples" / "litellm.yaml")
    assert validate_references(prism, litellm_cfg) == []
    assert set(prism.policies) == {"default"}
    assert prism.policies["default"].strategy == "best_of_n"


# --- ordinary non-Prism aliases remain unchanged ---------------------------


def test_ordinary_aliases_untouched():
    """Ordinary aliases keep their backend; only multicall/* refs resolve."""
    litellm_cfg = _litellm_dict_with_ordinary_and_prism()
    refs = extract_multicall_policy_refs(litellm_cfg)
    assert refs == {"default": ["smart-local"]}
    ordinary = {
        e["model_name"]: e["litellm_params"]["model"]
        for e in litellm_cfg["model_list"]
        if e["model_name"] in ("local-small", "plain-direct")
    }
    assert ordinary == {
        "local-small": "openai/local-small-model",
        "plain-direct": "openai/plain-model",
    }
    assert not any(v.startswith("multicall/") for v in ordinary.values())


def test_ordinary_only_config_has_no_multicall_refs():
    litellm_cfg = {
        "model_list": [
            {
                "model_name": "plain-direct",
                "litellm_params": {"model": "openai/plain-model"},
            }
        ]
    }
    assert extract_multicall_policy_refs(litellm_cfg) == {}


def test_provider_only_claims_multicall_prefix():
    assert provider_mod.MulticallProvider.model_name == "multicall"
    assert isinstance(
        provider_mod.multicall_provider, provider_mod.MulticallProvider
    )


# --- no new calls for existing policies (adaptive_v2 is opt-in) ------------


@pytest.mark.parametrize("strategy", ["adaptive_v2", "adaptive-v2", "ADAPTIVE_V2"])
def test_adaptive_v2_strategy_rejected(strategy):
    data = _best_of_n_dict()
    data["policies"]["default"]["strategy"] = strategy
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_existing_policies_gain_no_new_calls():
    """Exact legacy call ledgers: 4+1 best_of_n, 1+3+1+1 adaptive."""
    best = PrismConfig.model_validate(_best_of_n_dict()).policies["default"]
    planned_best = sum(c.samples for c in best.candidates)
    assert best.budgets.max_model_calls == planned_best + 1  # +1 judge

    adap = PrismConfig.model_validate(_adaptive_dict()).policies["default"]
    assert adap.adaptive is not None
    planned_adap = adap.adaptive.initial_samples + sum(
        adap.adaptive.expansion_batches
    )
    assert planned_adap == 4
    assert adap.budgets.max_model_calls == planned_adap + 1 + 1  # judge + fallback

    # BudgetTracker enforces the exact ceiling: no hidden extra reservation.
    bt = BudgetTracker(max_model_calls=best.budgets.max_model_calls)
    for _ in range(planned_best + 1):
        bt.reserve()
    with pytest.raises(RuntimeError):
        bt.reserve()


def test_no_adaptive_v2_symbols_in_package():
    """Step 01 must not implement adaptive_v2 behavior yet."""
    pkg = REPO_ROOT / "src" / "litellm_multicall"
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in sorted(pkg.rglob("*.py"))
        if "adaptive_v2" in p.read_text() or "adaptive-v2" in p.read_text()
    ]
    assert offenders == []


# --- tested LiteLLM version is explicit ------------------------------------


def test_litellm_version_within_declared_range():
    declared = (REPO_ROOT / "pyproject.toml").read_text()
    assert "litellm>=1,<2" in declared
    installed = dist_version("litellm")
    assert installed.split(".")[0] == "1", f"unexpected litellm {installed}"
