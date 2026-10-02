import signal
from pathlib import Path

import pytest

from litellm_multicall.commands.proxy_cmd import build_config, launch
from litellm_multicall.errors import PrismError


def test_complete_config_preserved():
    llm = {
        "model_list": [
            {"model_name": "worker", "litellm_params": {"model": "openai/worker"}}
        ],
        "general_settings": {"master_key": "secret"},
        "litellm_settings": {
            "callbacks": ["existing"],
            "custom_provider_map": [{"provider": "existing"}],
        },
    }
    result = build_config(llm)
    assert result["model_list"] == llm["model_list"]
    assert result["general_settings"] == llm["general_settings"]
    assert result["litellm_settings"]["callbacks"][0] == "existing"
    assert result["litellm_settings"]["custom_provider_map"][0] == {
        "provider": "existing"
    }
    assert len(llm["litellm_settings"]["callbacks"]) == 1


@pytest.mark.parametrize("exit_status,expected", [(17, 17), (-signal.SIGTERM, 143)])
def test_child_status_env_private_temp_cleanup(
    monkeypatch, tmp_path, exit_status, expected
):
    captured = {}

    class Process:
        def __init__(self, argv, **kwargs):
            captured.update(argv=argv, **kwargs)
            config_path = Path(argv[2])
            assert config_path.stat().st_mode & 0o777 == 0o600
            assert kwargs["env"]["PRISM_POLICY_CONFIG"] == str(tmp_path / "policy.yaml")
            assert "LITELLM_MASTER_KEY" not in kwargs["env"]

        def poll(self):
            return exit_status

        def wait(self, timeout=None):
            return exit_status

    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(
        "litellm_multicall.commands.proxy_cmd.subprocess.Popen", Process
    )
    assert (
        launch({"model_list": []}, tmp_path / "policy.yaml", None, "127.0.0.1", 4000)
        == expected
    )
    assert not Path(captured["argv"][2]).exists()


def test_nonloopback_auth():
    with pytest.raises(PrismError):
        launch(
            {"general_settings": {"master_key": "sk-test"}},
            Path("policy"),
            None,
            "0.0.0.0",
            4000,
        )
