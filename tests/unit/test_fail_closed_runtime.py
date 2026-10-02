import pytest


@pytest.mark.asyncio
async def test_provider_consumes_explicit_policy_and_fails_closed(project, monkeypatch):
    from litellm_multicall.provider import load_runtime_config, multicall_provider

    monkeypatch.setenv("PRISM_POLICY_CONFIG", str(project / "multicall.yaml"))
    assert load_runtime_config()["schema_version"] == 1
    with pytest.raises(NotImplementedError):
        await multicall_provider.acompletion(
            "default", [{"role": "user", "content": "test"}]
        )
    with pytest.raises(NotImplementedError):
        await multicall_provider.astreaming("default", [])
    monkeypatch.delenv("PRISM_POLICY_CONFIG")
    monkeypatch.delenv("MULTICALL_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="explicit policy"):
        load_runtime_config()


@pytest.mark.asyncio
async def test_empty_engine_and_first_candidate_no_longer_success():
    from litellm_multicall.engine import ExecutionEngine
    from litellm_multicall.selection import select_best

    with pytest.raises(NotImplementedError):
        await ExecutionEngine(None).run(None, None)
    with pytest.raises(NotImplementedError):
        select_best(["incorrect", "correct"])


def test_existing_router_injection():
    from litellm_multicall.compat.proxy_runtime import (
        inject_router,
        resolve_proxy_router,
    )

    router = object()
    try:
        inject_router(router)
        assert resolve_proxy_router() is router
    finally:
        inject_router(None)
