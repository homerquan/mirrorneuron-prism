import json
import os
import subprocess
import sys

import pytest

from litellm_multicall.cli import main
from litellm_multicall.commands.context import Context
from litellm_multicall.commands.parser import build_parser


def output(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_init_valid_and_create_only(tmp_path, capsys):
    args = [
        "init",
        "--template",
        "cpu-bench",
        "--out-dir",
        str(tmp_path),
        "--format",
        "json",
    ]
    assert main(args) == 0
    assert output(capsys)["data"]["downloads"] == 0
    original = (tmp_path / "multicall.yaml").read_bytes()
    assert main(args) == 2
    assert output(capsys)["status"] == "error"
    assert (tmp_path / "multicall.yaml").read_bytes() == original
    assert main(args + ["--force"]) == 0


@pytest.mark.parametrize("placement", ["before", "middle", "after"])
def test_global_options_at_every_depth(project, capsys, placement):
    flags = ["--config", str(project / "prism.yaml"), "--format", "json", "--offline"]
    args = (
        flags + ["config", "validate"]
        if placement == "before"
        else ["config"] + flags + ["validate"]
        if placement == "middle"
        else ["config", "validate"] + flags
    )
    assert main(args) == 0
    assert output(capsys)["data"]["valid"]


def test_relative_path_precedence(project, monkeypatch, tmp_path):
    elsewhere = project / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("PRISM_CONFIG", str(project / "prism.yaml"))
    monkeypatch.setenv("PRISM_WORKDIR", "runtime")
    ctx = Context.resolve(
        build_parser().parse_args(["--workdir", "explicit", "config", "show"])
    )
    assert ctx.workdir == elsewhere / "explicit"
    assert ctx.configured("multicall.yaml") == project / "multicall.yaml"
    assert ctx.explicit("multicall.yaml") == elsewhere / "multicall.yaml"
    assert os.getcwd() == str(elsewhere)


def test_doctor_truthful_unavailable(project, capsys):
    assert main(["doctor", "--format", "json"]) == 0
    result = output(capsys)["data"]
    assert not result["ready"]
    assert not next(c for c in result["checks"] if c["name"] == "inference_runtime")[
        "ready"
    ]
    assert not next(c for c in result["checks"] if c["name"] == "quick.artifacts")[
        "ready"
    ]


def test_redaction(project, capsys):
    import yaml

    path = project / "litellm.yaml"
    config = yaml.safe_load(path.read_text())
    config["model_list"][0]["litellm_params"]["api_key"] = "super-secret-key"
    config["model_list"][0]["litellm_params"]["api_base"] = (
        "http://bob:secret@127.0.0.1:8000/v1?token=secret"
    )
    path.write_text(yaml.safe_dump(config))
    assert main(["config", "show", "--format", "json"]) == 0
    text = capsys.readouterr().out
    assert (
        "super-secret-key" not in text
        and "bob:secret" not in text
        and "token=secret" not in text
    )
    json.loads(text)


def test_help_listing_lightweight():
    code = "import sys; old=list(sys.path); from litellm_multicall.cli import main; assert main(['benchmark','list','--format','json'])==0; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'litellm' not in sys.modules; assert old==sys.path"
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["data"]["suites"]


def test_invalid_args_and_schema_machine_errors(project, capsys):
    assert main(["--format", "json", "nonsense"]) == 2
    assert output(capsys)["error"]["exit_code"] == 2
    (project / "prism.yaml").write_text(
        "schema_version: 2\nsecret: should-not-appear\n"
    )
    assert main(["config", "validate", "--format", "json"]) == 2
    text = capsys.readouterr().out
    assert "should-not-appear" not in text
    assert json.loads(text)["status"] == "error"


def test_explicit_unsupported_adapter(tmp_path, capsys):
    assert (
        main(
            [
                "benchmark",
                "prepare",
                "bfcl",
                "--source",
                str(tmp_path),
                "--revision",
                "a" * 40,
                "--format",
                "json",
            ]
        )
        == 3
    )
    assert output(capsys)["error"]["kind"] == "unsupported_capability"


def test_module_and_root_launchers():
    from pathlib import Path

    root = Path(__file__).parents[2]
    for args in (
        ["-m", "litellm_multicall", "--version"],
        [str(root / "mn_prism.py"), "--version"],
    ):
        completed = subprocess.run(
            [sys.executable, *args], capture_output=True, text=True
        )
        assert completed.returncode == 0
        assert completed.stdout.startswith("mn_prism ")


def test_no_dummy_proxy_or_nonloopback_demo(project, capsys):
    assert main(["proxy", "--format", "json"]) == 3
    assert output(capsys)["error"]["kind"] == "unsupported_capability"
    assert (
        main(
            ["serve", "--file", "unused.json", "--host", "0.0.0.0", "--format", "json"]
        )
        == 2
    )


def test_rules_cli_and_jsonl_rows(project, capsys):
    path = project / "examples/decision.json"
    assert (
        main(
            [
                "classify",
                "run",
                "--controller",
                "rules",
                "--input",
                str(path),
                "--format",
                "json",
            ]
        )
        == 0
    )
    data = output(capsys)["data"]
    assert data["results"][0]["winner_option_id"] == "retry_read"
    assert data["results"][0]["accepted_option_id"] is None
    batch = project / "batch.jsonl"
    batch.write_text(
        path.read_text().replace("\n", "")
        + "\n{broken\n"
        + path.read_text().replace("\n", "")
        + "\n"
    )
    assert (
        main(
            [
                "classify",
                "batch",
                "--controller",
                "rules",
                "--input",
                str(batch),
                "--format",
                "jsonl",
                "--continue-on-error",
            ]
        )
        == 2
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 4
    assert records[1]["data"]["request_id"] == "invalid-row-2"
    assert records[0]["data"]["request_id"] == "decision-001"
    assert records[-1]["data"]["model_loads"] == 1
    assert records[-1]["data"]["completed"] == 2 and records[-1]["data"]["errors"] == 1
