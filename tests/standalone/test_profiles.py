import json
import runpy
from importlib.resources import files
from pathlib import Path

import pytest

from prism.cli import main, parser
from prism.errors import PrismError
from prism.profiles import initialize, load_profile, profile_inventory


@pytest.mark.parametrize(
    "preset", ["local", "openrouter", "openai", "claude", "gemini", "providers"]
)
def test_installed_presets_load_and_match_repository(preset, tmp_path):
    paths, default = initialize(tmp_path, preset)
    assert paths and not (tmp_path / "prism.json").exists()
    config, selected = load_profile(tmp_path / "profiles" / f"{default}.json")
    assert list(config.profiles) == [default]
    assert selected
    root = Path(__file__).resolve().parents[2]
    for path in paths:
        relative = path.relative_to(tmp_path)
        assert json.loads(path.read_text()) == json.loads((root / relative).read_text())
    for profile in (tmp_path / "profiles").glob("*.json"):
        config, models = load_profile(profile)
        assert len(config.profiles) == 1
        assert all(m.api_key is None for m in models.values())


def test_selected_profile_ignores_unrelated_credentials_and_invalid_models(
    tmp_path, monkeypatch, capsys
):
    initialize(tmp_path, "providers")
    (tmp_path / "models/claude-small.json").write_text("invalid unrelated JSON")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["validate", "--profile", "prism-openai", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["profiles"] == ["prism-openai"]
    assert set(result["raw_models"]) == {"openai-small", "openai-strong"}
    assert len(profile_inventory()) > 10


def test_profile_paths_resolve_models_relative_to_file(tmp_path, monkeypatch):
    initialize(tmp_path, "openai")
    monkeypatch.chdir(tmp_path.parent)
    _, models = load_profile(tmp_path / "profiles/prism-openai-reviewed.json")
    assert set(models) == {"openai-small", "openai-strong"}


@pytest.mark.parametrize(
    "change",
    ["missing", "mismatched", "inline-key", "unsafe-reference", "unknown-reference"],
)
def test_invalid_model_references_and_inline_secrets_are_rejected(
    change, tmp_path, capsys
):
    initialize(tmp_path, "openai")
    path = tmp_path / "models/openai-small.json"
    if change == "missing":
        path.unlink()
    elif change == "unsafe-reference":
        profile = tmp_path / "profiles/prism-openai-direct.json"
        value = json.loads(profile.read_text())
        value["profile"]["direct"] = "../private-secret"
        profile.write_text(json.dumps(value))
    elif change == "unknown-reference":
        profile = tmp_path / "profiles/prism-openai-direct.json"
        value = json.loads(profile.read_text())
        value["profile"]["structured_output_model"] = "missing-final"
        profile.write_text(json.dumps(value))
    else:
        value = json.loads(path.read_text())
        if change == "mismatched":
            value["id"] = "different-id"
        else:
            value.pop("api_key_env")
            value["api_key"] = "private-secret"
        path.write_text(json.dumps(value))
    assert (
        main(
            [
                "validate",
                "--profile",
                str(tmp_path / "profiles/prism-openai-direct.json"),
                "--json",
            ]
        )
        == 2
    )
    assert "private-secret" not in capsys.readouterr().out


def test_init_never_partially_overwrites_existing_files(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    existing = directory / "openai-strong.json"
    existing.write_text("existing")
    with pytest.raises(PrismError, match="never overwrites"):
        initialize(tmp_path, "openai")
    assert existing.read_text() == "existing"
    assert not (tmp_path / "profiles").exists()


def test_start_needs_only_selected_profile_keys_and_show_cost(
    tmp_path, monkeypatch, capsys
):
    import uvicorn

    initialize(tmp_path, "providers")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRISM_API_KEY", "client-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    command = ["start", "--profile", "prism-openai-direct", "--show-cost", "--json"]
    assert main(command) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().out
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-secret")
    called = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: called.append(app))
    assert main(command) == 0
    assert list(called[0].state.engine.config.profiles) == ["prism-openai-direct"]
    assert set(called[0].state.engine.models) == {"openai-small"}
    captured = capsys.readouterr()
    assert "since_process_start" in captured.err
    assert "upstream-secret" not in captured.out + captured.err
    assert (
        parser()
        .parse_args(["serve", "--profile", "prism-openai", "--show-cost"])
        .show_cost
    )
    with pytest.raises(SystemExit):
        parser().parse_args(["start"])


def test_all_packaged_profiles_have_pinned_decisions_and_real_transports():
    catalog = files("prism").joinpath("resources", "catalog")
    for path in catalog.joinpath("profiles").iterdir():
        if path.name.endswith(".json"):
            config, models = load_profile(str(path))
            assert config.decision.revision
            assert all(
                m.provider in {"openai", "anthropic", "gemini", "openrouter"}
                for m in models.values()
            )


def test_local_prism_and_openrouter_cost_demo_use_real_connections():
    from prism.config import RawModel

    root = Path(__file__).resolve().parents[2]
    local, models = load_profile(root / "profiles/prism.json")
    assert set(models) == {"local", "local-strong"}
    assert all(
        m.input_cost_per_million == m.output_cost_per_million == 0
        for m in models.values()
    )
    assert all(not m.cost_rates_are_hypothetical for m in models.values())
    demo, priced = load_profile(root / "profiles/prism-mock-cost.json")
    assert demo.profiles["prism-mock-cost"].strategy == "text_synthesis"
    assert all(m.cost_rates_are_hypothetical for m in priced.values())
    for ref, test_id in [
        ("nemotron-super-reasoning", "nemotron-super-mock-cost"),
        ("nemotron-ultra", "nemotron-ultra-mock-cost"),
    ]:
        model = RawModel.model_validate_json(
            (root / "models" / f"{ref}.json").read_text()
        )
        test_model = priced[test_id]
        assert (test_model.name, test_model.base_url, test_model.provider) == (
            model.name,
            model.base_url,
            model.provider,
        )
        assert test_model.input_cost_per_million > 0
        assert test_model.output_cost_per_million > 0
        assert test_model.api_key_env == "OPENROUTER_API_KEY"
        assert model.input_cost_per_million == model.output_cost_per_million == 0
    assert local.profiles["prism"].worker == "local"


@pytest.mark.asyncio
async def test_cost_demo_request_fits_both_models_and_compares_all_stage_usage(
    monkeypatch,
):
    import httpx

    from prism.api import create_app
    from prism.backends import LiteLLMBackend

    from .test_proxy import completion

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-upstream-key")
    root = Path(__file__).resolve().parents[2]
    body = runpy.run_path(str(root / "examples/standalone/cost_demo.py"))[
        "demo_request"
    ]()
    config, models = load_profile(root / "profiles/prism-mock-cost.json")
    prompts = []

    def handler(request):
        prompts.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=completion(
                "two reviewers; support in 30 minutes; logs retained 14 days"
            ),
        )

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        config, models=models, backend=LiteLLMBackend(models, upstream), no_auth=True
    )
    # Planning must fit both OpenRouter stages and the original direct baseline.
    planned = app.state.engine.prepare(body)
    assert planned["direct_input"] is not None
    assert planned["trace"]["eligible_policies"] == ["text_synthesis"]
    async with (
        upstream,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client,
    ):
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        stats = (await client.get("/v1/prism/costs")).json()
    assert len(prompts) == stats["physical_calls"] == 2
    assert "Team 01" in json.dumps(prompts[0]["messages"])
    assert "Team 01" not in json.dumps(prompts[1]["messages"])
    assert stats["pricing_mode"] == "hypothetical"
    assert stats["compared_requests"] == 1
    assert stats["total_cost_usd"] == pytest.approx(
        sum(m["total_cost_usd"] for m in stats["models"])
    )
    assert stats["estimated_saved_usd"] == pytest.approx(
        stats["estimated_baseline_cost_usd"] - stats["total_cost_usd"]
    )


def test_comparison_only_model_does_not_require_unused_provider_key(
    tmp_path, monkeypatch, capsys
):
    import uvicorn

    initialize(tmp_path, "providers")
    path = tmp_path / "profiles/prism-openai-direct.json"
    value = json.loads(path.read_text())
    value["profile"]["cost_baseline_model"] = "claude-small"
    path.write_text(json.dumps(value))
    monkeypatch.setenv("PRISM_API_KEY", "client-key")
    monkeypatch.setenv("OPENAI_API_KEY", "upstream-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    assert main(["start", "--profile", str(path)]) == 0
    capsys.readouterr()
