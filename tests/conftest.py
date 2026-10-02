import pytest


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch):
    monkeypatch.delenv("PRISM_CONFIG", raising=False)
    monkeypatch.delenv("PRISM_WORKDIR", raising=False)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture
def project(tmp_path, monkeypatch):
    from litellm_multicall.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["init", "--template", "cpu-bench", "--quiet"]) == 0
    return tmp_path


@pytest.fixture
def decision_request():
    from litellm_multicall.classifier.types import DecisionRequest

    return DecisionRequest.model_validate(
        {
            "request_id": "r1",
            "state_version": "s1",
            "state": {
                "operation": "read_repository_file",
                "result": "temporary timeout",
                "attempts": 1,
            },
            "criteria": [
                {
                    "id": "c1",
                    "primitive": "retry",
                    "question": "Next action?",
                    "options": [
                        {"id": "retry_read", "description": "Retry"},
                        {"id": "abstain", "description": "Abstain"},
                    ],
                }
            ],
        }
    )
