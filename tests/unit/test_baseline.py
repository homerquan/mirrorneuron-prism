"""Baseline portability tests for Step 01.

These tests must run on any checkout without live model endpoints,
machine-specific paths, or network access.
"""

import subprocess
import sys

import pytest


def test_import_package_without_sys_path_mutation():
    import litellm_multicall

    assert litellm_multicall is not None
    # The package must be importable via the installed distribution;
    # this test file itself must not rely on machine-specific paths.
    machine_prefix = "/".join(["", "Users", "homer", "Sandbox"])
    assert not any("Sandbox" in entry for entry in sys.path)
    assert not any(machine_prefix in entry for entry in sys.path)


def test_import_package_has_no_side_effects():
    # Importing the top-level package in a fresh interpreter must succeed
    # quickly without starting servers or requiring network/credentials.
    proc = subprocess.run(
        [sys.executable, "-c", "import litellm_multicall"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_import_does_not_mutate_sys_path():
    before = list(sys.path)
    import litellm_multicall.cli  # noqa: F401

    assert list(sys.path) == before


def test_core_modules_importable():
    import litellm_multicall.backend  # noqa: F401
    import litellm_multicall.budgets  # noqa: F401
    import litellm_multicall.compat.accounting  # noqa: F401
    import litellm_multicall.compat.errors  # noqa: F401
    import litellm_multicall.compat.proxy_runtime  # noqa: F401
    import litellm_multicall.compat.request_context  # noqa: F401
    import litellm_multicall.compat.versions  # noqa: F401
    import litellm_multicall.config  # noqa: F401
    import litellm_multicall.engine  # noqa: F401
    import litellm_multicall.hooks  # noqa: F401
    import litellm_multicall.meter  # noqa: F401
    import litellm_multicall.models_loader  # noqa: F401
    import litellm_multicall.provider  # noqa: F401
    import litellm_multicall.selection  # noqa: F401
    import litellm_multicall.streaming  # noqa: F401
    import litellm_multicall.telemetry  # noqa: F401
    import litellm_multicall.types  # noqa: F401
    import litellm_multicall.validation  # noqa: F401


def test_provider_matches_installed_custom_llm_contract():
    import inspect

    from litellm.llms.custom_llm import CustomLLM

    from litellm_multicall import provider as provider_mod

    # get_instance_fn("litellm_multicall.provider.multicall_provider") returns
    # this object and custom_chat_llm_router accesses .acompletion on it, so it
    # must be a bound instance, not the class.
    assert isinstance(provider_mod.multicall_provider, CustomLLM)
    assert isinstance(provider_mod.multicall_provider, provider_mod.MulticallProvider)
    # The installed CustomLLM dispatches to acompletion/astreaming/completion/
    # streaming; a misspelled `acomplete` would never be called.
    assert not hasattr(provider_mod.MulticallProvider, "acomplete")
    for name in ("acompletion", "astreaming", "completion", "streaming"):
        assert callable(getattr(provider_mod.multicall_provider, name, None)), name
    # Base __init__ takes no arguments; the subclass must not forward **kwargs.
    assert tuple(inspect.signature(provider_mod.MulticallProvider.__init__).parameters) == ("self",)


def test_provider_calls_fail_closed():
    import asyncio

    from litellm_multicall.provider import multicall_provider

    with pytest.raises(NotImplementedError):
        asyncio.run(multicall_provider.acompletion(model="m", messages=[]))
    with pytest.raises(NotImplementedError):
        asyncio.run(multicall_provider.astreaming(model="m", messages=[]))


def test_no_machine_specific_paths_in_package_and_unit_tests():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    # Build needles without writing them literally so this test file itself
    # does not match the scan.
    needles = [
        "/".join(["", "Users", "homer"]),
        "Gomoku" + "Bench",
        "Sandbox" + "/mirrorneuron",
    ]
    offenders = []
    for base in (repo_root / "src", repo_root / "tests" / "unit"):
        for path in sorted(base.rglob("*.py")):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text()
            if any(n in text for n in needles):
                offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []


def test_budget_tracker_reserves_and_exhausts():
    from litellm_multicall.budgets import BudgetTracker

    bt = BudgetTracker(max_model_calls=2)
    bt.reserve()
    bt.reserve()
    with pytest.raises(RuntimeError):
        bt.reserve()
