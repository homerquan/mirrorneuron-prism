import pytest

from litellm_multicall.classifier.artifacts import register, verify_controller
from litellm_multicall.classifier.calibration import IDENTITY_FIELDS, fit, verify
from litellm_multicall.errors import PrismError
from litellm_multicall.project_config import Controller


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    path.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(b"safe checkpoint test fixture")
    return path


def test_preparation_immutable_and_tamper(tmp_path, snapshot):
    profile = Controller()
    lock = tmp_path / "artifacts.json"
    with pytest.raises(PrismError) as error:
        verify_controller(lock, profile)
    assert error.value.code == 3
    desc = register(lock, profile, snapshot, "a" * 40, "model-license")
    assert verify_controller(lock, profile) == desc
    (snapshot / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(PrismError, match="changed"):
        verify_controller(lock, profile)
    with pytest.raises(PrismError, match="immutable"):
        register(lock, profile, snapshot, "b" * 40, "model-license")


@pytest.mark.parametrize("revision", [None, "main", "v1", "abc"])
def test_mutable_revisions_rejected(tmp_path, snapshot, revision):
    with pytest.raises(PrismError) as error:
        register(tmp_path / "lock.json", Controller(), snapshot, revision, "license")
    assert error.value.code == 7


def test_incomplete_snapshot(tmp_path, snapshot):
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(PrismError, match="incomplete"):
        register(tmp_path / "lock.json", Controller(), snapshot, "a" * 40, "license")


def bundle(n=30):
    metadata = {k: "test" for k in IDENTITY_FIELDS}

    def records(prefix):
        return [
            {
                "id": f"{prefix}-{i}",
                "group_id": f"{prefix}-{i}",
                "primitive": "sufficient",
                "family": "fixture",
                "option_ids": ["yes", "no"],
                "option_logits": [2, 0],
                "gold_option_id": "yes",
            }
            for i in range(n)
        ]

    return {
        "identity": metadata,
        "dataset_provenance": "fixture-hash",
        "calibration": records("cal"),
        "validation": records("val"),
        "test_groups": ["test"],
    }


def test_calibration_small_sample_never_certifies():
    data = bundle(2)
    artifact = fit(data, risk_target=0.05, min_support=2)
    assert artifact["threshold"] is None
    assert artifact["test_performance"] is None
    assert artifact["status"] == "insufficient_selective_risk_evidence"


def test_calibration_gate_identity_and_no_leakage():
    data = bundle()
    artifact = fit(data, risk_target=0.5, min_support=20)
    assert artifact["threshold"] is not None
    verify(artifact, data["identity"])
    for field in IDENTITY_FIELDS:
        changed = {**data["identity"], field: "changed"}
        with pytest.raises(PrismError, match="incompatible"):
            verify(artifact, changed)
    data["test_groups"] = ["cal-0"]
    with pytest.raises(PrismError, match="disjoint"):
        fit(data)


def test_calibration_corruption():
    artifact = fit(bundle())
    artifact["temperature"] = 0.001
    with pytest.raises(PrismError, match="corrupted"):
        verify(artifact, bundle()["identity"])
