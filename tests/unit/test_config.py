import pytest
from pydantic import ValidationError

from litellm_multicall.commands.init_cmd import example_configs
from litellm_multicall.config import PrismConfig, validate_references
from litellm_multicall.project_config import Controller


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(schema_version=2),
        lambda c: c.update(unknown=True),
        lambda c: c["policies"]["default"].update(unknown=True),
        lambda c: c["policies"]["default"]["selection"].update(type="cpu_then_llm"),
        lambda c: c["policies"]["default"]["budgets"].update(max_model_calls=1),
        lambda c: c["policies"]["default"]["budgets"].update(deadline_ms=-1),
        lambda c: c["policies"]["default"]["candidates"][0].update(model="cloud"),
        lambda c: c["policies"]["default"].update(on_unresolved="fallback"),
        lambda c: c["policies"]["default"]["candidates"][0].update(
            prompt_variants=["missing"]
        ),
    ],
)
def test_strict_v1(change):
    c, _, _ = example_configs()
    change(c)
    with pytest.raises(ValidationError):
        PrismConfig.model_validate(c)


def test_v1_roundtrip_no_controller():
    c, llm, _ = example_configs()
    assert PrismConfig.model_validate(c).model_dump(mode="json", exclude_none=True) == c
    assert "controller" not in str(c)
    validate_references(c, llm)


@pytest.mark.parametrize(
    "model,url",
    [
        ("multicall/default", "http://localhost"),
        ("openai/cloud", "https://api.example.com"),
        ("openai/worker", "http://127.0.0.1"),
    ],
)
def test_physical_model_locality_and_recursion(model, url):
    c, llm, _ = example_configs()
    llm["model_list"][0]["litellm_params"].update(model=model, api_base=url)
    if model.startswith("multicall") or "example" in url:
        with pytest.raises(ValueError):
            validate_references(c, llm)
    else:
        validate_references(c, llm)


@pytest.mark.parametrize(
    "kw",
    [
        {"device": "mps"},
        {"dtype": "bfloat16"},
        {"threads": 0},
        {"max_inflight_batches": 2},
        {"arbitrary": "module"},
    ],
)
def test_controller_capabilities_strict(kw):
    with pytest.raises(ValidationError):
        Controller(**kw)
