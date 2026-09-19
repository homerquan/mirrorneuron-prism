"""Strict typed-configuration tests for Step 02.

Deterministic: fake file fixtures only, no live LiteLLM proxy, no GPU,
no network calls.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from litellm_multicall.config import (
    PrismConfig,
    extract_litellm_model_groups,
    extract_multicall_policy_refs,
    load_litellm_config,
    load_policy_config,
    prism_config_json_schema,
    validate_files,
    validate_references,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT / "src" / "litellm_multicall" / "resources" / "policy.schema.json"
)


def _valid_policy_dict(**overrides):
    base = {
        "schema_version": 1,
        "runtime": {
            "backend": "proxy_router",
            "integration_profile": "trusted_local",
        },
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
    base.update(overrides)
    return base


def _valid_litellm_dict():
    return {
        "model_list": [
            {
                "model_name": "local-small",
                "litellm_params": {"model": "openai/local-small-model"},
            },
            {
                "model_name": "smart-local",
                "litellm_params": {"model": "multicall/default"},
            },
        ]
    }


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


# --- valid configs -------------------------------------------------------


def test_valid_best_of_n_config_loads():
    prism = PrismConfig.model_validate(_valid_policy_dict())
    assert prism.schema_version == 1
    assert prism.policies["default"].strategy == "best_of_n"
    assert prism.policies["default"].budgets.max_model_calls == 5


def test_valid_adaptive_config_loads():
    data = _valid_policy_dict()
    policy = data["policies"]["default"]
    policy["strategy"] = "adaptive"
    policy["allowed_model_groups"] = ["local-small", "local-strong"]
    policy["adaptive"] = {"initial_samples": 1, "expansion_batches": [3]}
    policy["fallback"] = {"model": "local-strong", "max_completion_tokens": 1500}
    policy["on_unresolved"] = "fallback"
    policy["budgets"] = {"max_model_calls": 6}
    prism = PrismConfig.model_validate(data)
    assert prism.policies["default"].adaptive is not None
    assert prism.policies["default"].fallback is not None


def test_valid_config_with_prompt_variants_loads():
    data = _valid_policy_dict()
    data["prompt_variants"] = {
        "direct": "Produce a complete answer.",
        "independent_check": "Check assumptions, then answer.",
    }
    data["policies"]["default"]["candidates"][0]["prompt_variants"] = [
        "direct",
        "independent_check",
    ]
    prism = PrismConfig.model_validate(data)
    assert set(prism.prompt_variants) == {"direct", "independent_check"}


def test_load_policy_config_from_file(tmp_path):
    path = _write(tmp_path, "multicall.yaml", _valid_policy_dict())
    prism = load_policy_config(path)
    assert isinstance(prism, PrismConfig)
    assert "default" in prism.policies


def test_example_multicall_yaml_loads():
    path = REPO_ROOT / "examples" / "multicall.yaml"
    prism = load_policy_config(path)
    assert "default" in prism.policies


# --- schema_version ------------------------------------------------------


@pytest.mark.parametrize("bad_version", ["1", "1.0", 2, 0, 1.5, True])
def test_bad_schema_version_rejected(bad_version):
    data = _valid_policy_dict(schema_version=bad_version)
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_missing_schema_version_rejected():
    data = _valid_policy_dict()
    del data["schema_version"]
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


# --- unknown keys / legacy shape -----------------------------------------


def test_unknown_top_level_key_rejected():
    data = _valid_policy_dict()
    data["bogus_key"] = 1
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_unknown_runtime_key_rejected():
    data = _valid_policy_dict()
    data["runtime"]["bogus"] = True
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_unknown_candidate_key_rejected():
    data = _valid_policy_dict()
    data["policies"]["default"]["candidates"][0]["bogus"] = 1
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_legacy_singular_policy_key_rejected(tmp_path):
    # `mn_prism init` must generate `policies`, never `policy`.
    raw = {"schema_version": 1, "policy": {}}
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError):
        load_policy_config(path)


def test_empty_policies_rejected():
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(_valid_policy_dict(policies={}))


# --- malformed files never become {} -------------------------------------


def test_malformed_yaml_raises_not_empty_dict(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("policies: [unclosed\n  bad: : :\n")
    with pytest.raises(Exception):
        load_policy_config(path)


def test_empty_file_raises_not_empty_dict(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(Exception):
        load_policy_config(path)


def test_non_mapping_file_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(Exception):
        load_policy_config(path)


# --- policy coherence ----------------------------------------------------


def test_candidate_model_outside_allowed_groups_rejected():
    data = _valid_policy_dict()
    data["policies"]["default"]["candidates"][0]["model"] = "elsewhere"
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_selection_model_outside_allowed_groups_rejected():
    data = _valid_policy_dict()
    data["policies"]["default"]["selection"]["model"] = "elsewhere"
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_insufficient_budget_rejected():
    # 4 candidate samples + 1 judge call needs >= 5.
    data = _valid_policy_dict()
    data["policies"]["default"]["budgets"] = {"max_model_calls": 4}
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_backend_attempts_below_model_calls_rejected():
    data = _valid_policy_dict()
    data["policies"]["default"]["budgets"] = {
        "max_model_calls": 5,
        "max_backend_attempts": 4,
    }
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_adaptive_requires_adaptive_block():
    data = _valid_policy_dict()
    data["policies"]["default"]["strategy"] = "adaptive"
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_best_of_n_rejects_stray_adaptive_block():
    data = _valid_policy_dict()
    data["policies"]["default"]["adaptive"] = {
        "initial_samples": 1,
        "expansion_batches": [3],
    }
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_fallback_requires_on_unresolved_fallback():
    data = _valid_policy_dict()
    policy = data["policies"]["default"]
    policy["fallback"] = {"model": "local-small", "max_completion_tokens": 100}
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_on_unresolved_fallback_requires_fallback_block():
    data = _valid_policy_dict()
    data["policies"]["default"]["on_unresolved"] = "fallback"
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_multicall_recursion_rejected():
    data = _valid_policy_dict()
    policy = data["policies"]["default"]
    policy["allowed_model_groups"] = ["multicall/default"]
    policy["candidates"][0]["model"] = "multicall/default"
    policy["selection"]["model"] = "multicall/default"
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_undefined_prompt_variant_rejected():
    data = _valid_policy_dict()
    data["policies"]["default"]["candidates"][0]["prompt_variants"] = ["nope"]
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


def test_unsupported_strategy_rejected():
    data = _valid_policy_dict()
    data["policies"]["default"]["strategy"] = "decompose_map_reduce"
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(data)


# --- LiteLLM cross-references (no network) -------------------------------


def test_references_resolve_happy_path():
    prism = PrismConfig.model_validate(_valid_policy_dict())
    assert validate_references(prism, _valid_litellm_dict()) == []


def test_missing_model_group_reported():
    prism = PrismConfig.model_validate(_valid_policy_dict())
    litellm_cfg = {"model_list": []}
    errors = validate_references(prism, litellm_cfg)
    assert errors


def test_unknown_candidate_group_reported():
    prism = PrismConfig.model_validate(_valid_policy_dict())
    litellm_cfg = {
        "model_list": [
            {
                "model_name": "smart-local",
                "litellm_params": {"model": "multicall/default"},
            }
        ]
    }
    errors = validate_references(prism, litellm_cfg)
    assert any("local-small" in e for e in errors)


def test_multicall_ref_to_unknown_policy_reported():
    prism = PrismConfig.model_validate(_valid_policy_dict())
    litellm_cfg = {
        "model_list": [
            {
                "model_name": "smart-local",
                "litellm_params": {"model": "multicall/ghost"},
            }
        ]
    }
    errors = validate_references(prism, litellm_cfg)
    assert any("ghost" in e for e in errors)


def test_extract_helpers():
    groups = extract_litellm_model_groups(_valid_litellm_dict())
    assert groups == {"local-small", "smart-local"}
    refs = extract_multicall_policy_refs(_valid_litellm_dict())
    assert refs == {"default": ["smart-local"]}


def test_validate_files_happy_path(tmp_path):
    policy_path = _write(tmp_path, "multicall.yaml", _valid_policy_dict())
    litellm_path = _write(tmp_path, "litellm.yaml", _valid_litellm_dict())
    assert validate_files(policy_path, litellm_path) == []


def test_validate_files_reports_missing_groups(tmp_path):
    policy_path = _write(tmp_path, "multicall.yaml", _valid_policy_dict())
    litellm_path = _write(
        tmp_path, "litellm.yaml", {"model_list": [{"model_name": "other"}]}
    )
    assert validate_files(policy_path, litellm_path)


# --- JSON schema stays in sync -------------------------------------------


def test_policy_schema_json_in_sync_with_model():
    assert SCHEMA_PATH.exists(), "resources/policy.schema.json must exist"
    on_disk = json.loads(SCHEMA_PATH.read_text())
    generated = prism_config_json_schema()
    for key in ("title", "$defs", "properties", "required"):
        assert key in on_disk, f"schema missing {key}"
    # Compare modulo the informational envelope keys we add at write time.
    stripped = {
        k: v for k, v in on_disk.items() if k not in ("$schema", "title")
    }
    expected = {
        k: v for k, v in generated.items() if k not in ("$schema", "title")
    }
    assert stripped == expected


def test_schema_rejects_empty_object():
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({}, schema)


# --- CLI -----------------------------------------------------------------


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "litellm_multicall.cli", *argv],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_validate_accepts_valid_pair(tmp_path):
    policy_path = _write(tmp_path, "multicall.yaml", _valid_policy_dict())
    litellm_path = _write(tmp_path, "litellm.yaml", _valid_litellm_dict())
    proc = _run_cli(
        "validate",
        "--policy-config",
        str(policy_path),
        "--litellm-config",
        str(litellm_path),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Validate OK" in proc.stdout


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update({"schema_version": "1.0"}),
        lambda d: d.update({"policy": {}, "policies": {}}),
        lambda d: d.update({"unexpected": True}),
        lambda d: d["policies"]["default"].update({"strategy": "magic"}),
        lambda d: d["policies"]["default"].update({"allowed_model_groups": []}),
    ],
)
def test_cli_validate_rejects_malformed_policy(tmp_path, mutate):
    data = _valid_policy_dict()
    mutate(data)
    policy_path = _write(tmp_path, "multicall.yaml", data)
    litellm_path = _write(tmp_path, "litellm.yaml", _valid_litellm_dict())
    proc = _run_cli(
        "validate",
        "--policy-config",
        str(policy_path),
        "--litellm-config",
        str(litellm_path),
    )
    assert proc.returncode != 0


def test_cli_validate_rejects_missing_model_groups(tmp_path):
    policy_path = _write(tmp_path, "multicall.yaml", _valid_policy_dict())
    litellm_path = _write(
        tmp_path, "litellm.yaml", {"model_list": [{"model_name": "other"}]}
    )
    proc = _run_cli(
        "validate",
        "--policy-config",
        str(policy_path),
        "--litellm-config",
        str(litellm_path),
    )
    assert proc.returncode != 0


def test_cli_validate_rejects_invalid_schema_version(tmp_path):
    data = _valid_policy_dict(schema_version=2)
    policy_path = _write(tmp_path, "multicall.yaml", data)
    litellm_path = _write(tmp_path, "litellm.yaml", _valid_litellm_dict())
    proc = _run_cli(
        "validate",
        "--policy-config",
        str(policy_path),
        "--litellm-config",
        str(litellm_path),
    )
    assert proc.returncode != 0


def test_cli_init_generates_loadable_and_validatable_pair(tmp_path):
    out_dir = tmp_path / "gen"
    proc = _run_cli("init", "--out-dir", str(out_dir))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    policy_path = out_dir / "multicall.yaml"
    litellm_path = out_dir / "litellm.yaml"
    assert policy_path.exists() and litellm_path.exists()

    raw = yaml.safe_load(policy_path.read_text())
    # Generated YAML must match the loader: `policies`, int version.
    assert "policies" in raw and "policy" not in raw
    assert raw["schema_version"] == 1 and isinstance(raw["schema_version"], int)

    prism = load_policy_config(policy_path)
    litellm_cfg = load_litellm_config(litellm_path)
    assert validate_references(prism, litellm_cfg) == []

    proc = _run_cli(
        "validate",
        "--policy-config",
        str(policy_path),
        "--litellm-config",
        str(litellm_path),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
