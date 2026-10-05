import io
import json
from importlib.resources import files
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from prism.cli import main, parser
from prism.cli_ui import Output
from prism.config import load_config

from .test_openrouter_combinations import setup
from .test_proxy import completion


@pytest.mark.parametrize("prefix", [[], ["--json"], ["--output", "json"]])
def test_json_flags_and_redirected_output(prefix, tmp_path, capsys):
    assert (
        main([*prefix, "init", "--preset", "providers", "--out-dir", str(tmp_path)])
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert all(
        Path(p).parent.name in {"models", "profiles"} for p in created["created"]
    )
    assert len(created["next_steps"]) == 4
    assert (
        main(
            [
                "profiles",
                "--profile",
                str(tmp_path / "profiles/nano-openai.json"),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["object"] == "profiles"
    assert any(p["id"] == "nano-openai" for p in result["data"])


def test_model_inventory_never_prints_keys(tmp_path, monkeypatch, capsys):
    main(["init", "--preset", "providers", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    monkeypatch.setenv("OPENAI_API_KEY", "private-secret")
    assert (
        main(["models", "--profile", str(tmp_path / "profiles/prism-openai.json")]) == 0
    )
    raw = capsys.readouterr().out
    assert "private-secret" not in raw
    data = json.loads(raw)["data"]
    assert (
        next(m for m in data if m["id"] == "openai-strong")["credential_ready"] is True
    )


def test_help_and_no_args_are_useful(capsys):
    assert main([]) == 0
    assert "Quick start" in capsys.readouterr().out
    help_text = parser().format_help()
    assert "profiles" in help_text and "--json" in help_text
    assert parser().parse_args(["--json", "validate"]).output == "json"
    assert parser().parse_args(["validate", "--json"]).output == "json"
    assert parser().parse_args(["benchmark", "run", "--no-auth"]).no_auth


@pytest.mark.parametrize("terminal", [False, True])
def test_human_tables_and_literal_markup(terminal):
    text = io.StringIO()
    console = Console(file=text, width=120, force_terminal=terminal, color_system=None)
    output = Output("table", console=console, errors=console)
    output.emit(
        {
            "object": "models",
            "data": [
                {
                    "id": "[bold]literal",
                    "name": "physical",
                    "context_window": 32768,
                    "max_output_tokens": 4096,
                    "capabilities": ["text"],
                    "credential_ready": False,
                    "credential_env": "TEST_KEY",
                }
            ],
        }
    )
    assert "[bold]literal" in text.getvalue()
    assert "TEST_KEY" in text.getvalue()
    output.emit({"error": {"code": "invalid_request", "message": "Fix the config"}})
    assert "Prism error" in text.getvalue()
    output.emit(
        {
            "object": "capacity",
            "data": [
                {
                    "model": "physical",
                    "capabilities": {
                        "image": {
                            "status": "unavailable",
                            "passed": 0,
                            "total": 1,
                            "elapsed_ms": 1,
                            "error_code": "upstream_rate_limit",
                        }
                    },
                }
            ],
        }
    )
    assert "unavailable" in text.getvalue()


def test_vendor_samples_and_packaged_preset_match():
    import litellm

    root = Path(__file__).resolve().parents[2]
    directory = root / "examples/standalone/providers"
    for path in directory.glob("prism*.json"):
        config, models = load_config(path)
        assert config.profiles
        for m in models.values():
            _, provider, _, _ = litellm.get_llm_provider(
                m.name, custom_llm_provider=m.provider
            )
            assert provider in {"openai", "anthropic", "gemini", "openrouter"}
            assert m.api_key is None and m.api_key_env
    packaged = files("prism").joinpath("resources", "providers")
    for name in ("prism.json", "models.json"):
        assert json.loads(packaged.joinpath(name).read_text()) == json.loads(
            (directory / name).read_text()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["json_object", "json_schema"])
async def test_json_uses_configured_structured_output_model(mode):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json=completion('{"ok":true}'))

    app, upstream = setup(
        handler,
        direct="vision",
        strategy="direct",
        allowed_policies=["direct"],
        structured_output_model="text",
    )
    fmt = {"type": mode}
    if mode == "json_schema":
        fmt["json_schema"] = {
            "name": "ok",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
        }
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        assert (
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mix",
                    "messages": [{"role": "user", "content": "Say OK."}],
                },
            )
        ).status_code == 200
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mix",
                "messages": [{"role": "user", "content": "Return JSON."}],
                "response_format": fmt,
            },
        )
        assert response.status_code == 200, response.text
    assert [c["model"] for c in calls] == ["physical-vision", "physical-text"]
    assert calls[-1]["response_format"]["type"] == mode


@pytest.mark.asyncio
async def test_text_synthesis_nano_never_receives_json_mode():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["model"] == "physical-vision":
            assert "response_format" not in body
            return httpx.Response(
                200, json=completion("Refund period: 14 days; receipt needed.")
            )
        assert "background detail" not in json.dumps(body["messages"])
        return httpx.Response(200, json=completion('{"refund_days":14}'))

    app, upstream = setup(
        handler,
        strategy="text_synthesis",
        allowed_policies=["direct", "text_synthesis"],
    )
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mix",
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return refund_days as JSON. <prism-source id="p">Refunds within 14 days. Unrelated background detail.</prism-source>',
                    }
                ],
                "response_format": {"type": "json_object"},
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-prism-policy"] == "text_synthesis"
        assert response.headers["x-prism-coverage"] == "text_observations"
    assert len(calls) == 2


@pytest.mark.parametrize("width", [60, 80, 120])
def test_readable_profile_tables_status_and_readiness(width):
    stream = io.StringIO()
    console = Console(file=stream, width=width, force_terminal=True, color_system=None)
    output = Output("auto", console=console, errors=console)
    with output.status("Checking"):
        output.emit(
            {
                "object": "profiles",
                "data": [
                    {
                        "id": "prism-balanced",
                        "strategy": "direct",
                        "direct": "nano",
                        "worker": None,
                        "synthesizer": None,
                        "structured_output_model": "super",
                    }
                ],
            }
        )
    output.emit({"created": ["prism.json"], "next_steps": ["prism serve"]})
    output.emit({"valid": True, "profiles": ["prism-balanced"]})
    output.emit(
        {
            "authentication_ready": False,
            "ready": False,
            "models": [
                {"id": "nano", "probed": False},
                {
                    "id": "super",
                    "probed": True,
                    "ready": False,
                    "error_code": "upstream_rate_limit",
                },
                {"id": "local", "probed": True, "ready": True},
            ],
        }
    )
    output.emit({"policies": [{"id": "direct", "description": "One call"}]})
    output.emit({"requests": 1})
    output.serving("127.0.0.1", 8080, 1, True)
    rendered = stream.getvalue()
    assert "prism-balanced" in rendered
    assert "JSON: super" in rendered
    assert "Set the configured Prism API key" in rendered
    assert "disabled (--no-auth)" in rendered
    assert "upstream_rate_limit" in rendered


@pytest.mark.parametrize(
    "fail,no_auth,probed,expected",
    [(False, True, True, 0), (True, True, True, 3), (False, False, False, 3)],
)
def test_doctor_readiness_and_failure_exit_codes(
    fail, no_auth, probed, expected, tmp_path, monkeypatch, capsys
):
    from prism.errors import PrismError

    instances = []

    class Backend:
        def __init__(self, models):
            self.closed = False
            instances.append(self)

        async def complete(self, *args):
            if fail:
                raise PrismError("private upstream detail", "upstream_rate_limit", 429)
            return completion()

        async def close(self):
            self.closed = True

    monkeypatch.setattr("prism.backends.LiteLLMBackend", Backend)
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    main(["init", "--preset", "local", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    command = [
        "doctor",
        "--profile",
        str(tmp_path / "profiles/prism-local-direct.json"),
    ]
    if no_auth:
        command.append("--no-auth")
    if probed:
        command.append("--probe-backends")
    assert main(command) == expected
    raw = capsys.readouterr().out
    assert "private upstream detail" not in raw
    assert json.loads(raw)["authentication_mode"] == (
        "disabled" if no_auth else "bearer"
    )
    assert instances[0].closed


def test_trace_auth_modes_missing_and_unavailable(monkeypatch, capsys):
    requests = []

    def get(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(
            200 if len(requests) == 1 else 404, json={"request_id": "prism-id"}
        )

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    assert main(["trace", "show", "prism-id"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]
    assert not requests
    assert main(["trace", "show", "prism-id", "--no-auth"]) == 0
    assert requests[0][1]["headers"] == {}
    assert json.loads(capsys.readouterr().out)["request_id"] == "prism-id"
    monkeypatch.setenv("PRISM_API_KEY", "test-secret")
    assert main(["trace", "show", "prism-id"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "trace_not_found"
    assert requests[-1][1]["headers"] == {"authorization": "Bearer test-secret"}


def test_capacity_cli_selects_physical_and_profile_evaluators(
    tmp_path, capsys, monkeypatch
):
    from prism.capacity import CapacityEvaluator, ProfileCapacityEvaluator

    selected = []

    async def query(self, aliases, refresh):
        selected.append((type(self).__name__, aliases, refresh))
        return [{"model": aliases[0], "capabilities": {}}]

    monkeypatch.setattr(CapacityEvaluator, "query", query)
    monkeypatch.setattr(ProfileCapacityEvaluator, "query", query)
    main(["init", "--preset", "openrouter", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    assert (
        main(
            [
                "capacity",
                "--profile",
                str(tmp_path / "profiles/prism-vision-llm.json"),
                "--model",
                "nemotron-ultra",
                "--model",
                "prism-balanced",
            ]
        )
        == 0
    )
    assert selected == [
        ("CapacityEvaluator", ["nemotron-ultra"], True),
        ("ProfileCapacityEvaluator", ["prism-balanced"], True),
    ]
    assert len(json.loads(capsys.readouterr().out)["data"]) == 2


def test_eval_cli_runs_compares_and_preserves_output(tmp_path, capsys, monkeypatch):
    called = []

    async def evaluate(cases, **kwargs):
        called.append(kwargs)
        return [
            {
                "id": "case",
                "routes": {
                    label: {
                        "completed": label == "baseline",
                        "correct": label == "baseline",
                        "latency_ms": 5,
                    }
                    for label in ("baseline", "candidate")
                },
            }
        ]

    monkeypatch.setattr("prism.evaluation.evaluate", evaluate)
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"messages":[]}\n')
    destination = tmp_path / "result.jsonl"
    command = ["eval", "run", "--cases", str(cases), "--out", str(destination)]
    assert main(command) == 2
    capsys.readouterr()
    assert main([*command, "--no-auth"]) == 0
    assert called[0]["api_key"] == ""
    assert json.loads(capsys.readouterr().out)["candidate"]["failures"] == 1
    original = destination.read_bytes()
    assert main([*command, "--no-auth"]) == 2
    capsys.readouterr()
    assert destination.read_bytes() == original
    assert main(["eval", "compare", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["baseline"]["fixture_accuracy"] == 1


def test_benchmark_cli_progress_and_auth(tmp_path, capsys, monkeypatch):
    called = []

    async def run(**kwargs):
        called.append(kwargs)
        kwargs["progress"](
            {
                "warmup": False,
                "case_id": "case",
                "route": "candidate",
                "latency_ms": 10,
                "quality": {"score": 1},
                "error_code": None,
            }
        )
        return {"saved": str(tmp_path)}

    monkeypatch.setattr("prism.benchmark.run", run)
    monkeypatch.setattr(
        "prism.benchmark.compare", lambda paths, destination: {"compared": len(paths)}
    )
    monkeypatch.delenv("PRISM_API_KEY", raising=False)
    assert main(["benchmark", "run"]) == 2
    capsys.readouterr()
    assert main(["benchmark", "run", "--no-auth", "--out-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["saved"] == str(tmp_path)
    assert "measured case candidate" in captured.err
    assert called[0]["api_key"] == ""
    assert main(["benchmark", "compare", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["compared"] == 1


def test_config_errors_are_sanitized(tmp_path, capsys):
    missing = tmp_path / "private-key-do-not-print.json"
    assert main(["validate", "--config", str(missing)]) == 2
    raw = capsys.readouterr().out
    assert "private-key-do-not-print" not in raw
    assert json.loads(raw)["error"]["code"] == "configuration_or_dependency_error"


def test_trace_network_failure_is_actionable_and_sanitized(monkeypatch, capsys):
    def get(*args, **kwargs):
        raise httpx.ConnectError("private network detail")

    monkeypatch.setattr(httpx, "get", get)
    assert main(["trace", "show", "prism-test", "--no-auth"]) == 2
    raw = capsys.readouterr().out
    assert "private network detail" not in raw
    assert json.loads(raw)["error"]["code"] == "transport_error"
    assert "--base-url" in raw
