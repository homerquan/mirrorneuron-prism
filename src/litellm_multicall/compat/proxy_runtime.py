"""Isolated proxy-router access (Step 01 baseline).

This module is the ONLY place allowed to resolve LiteLLM proxy runtime
internals (SPEC section 3.3). Rules:

- Never import-and-capture ``llm_router`` at module import time.
- Prefer explicit router injection (tests, embedded use).
- Otherwise lazily resolve the running worker's initialized router from
  ``litellm.proxy.proxy_server`` and revalidate on every call.
- Never construct a fresh ``Router`` or call raw provider APIs here.
- Fail closed with :class:`IntegrationUnavailable` when no router exists.
"""

from __future__ import annotations

from typing import Any, Optional

from litellm_multicall.compat.errors import IntegrationUnavailable

_injected_router: Optional[Any] = None


def inject_router(router: Any) -> None:
    """Provide an explicit router (tests / optional embedded use)."""
    global _injected_router
    _injected_router = router


def clear_router() -> None:
    """Remove any explicitly injected router (test cleanup)."""
    global _injected_router
    _injected_router = None


def resolve_proxy_router(router_override: Optional[Any] = None) -> Any:
    """Return the initialized LiteLLM proxy router for child calls.

    Resolution order: explicit ``router_override`` argument, then the
    module-level injected router, then a lazy lookup of the running
    worker's ``llm_router``. Raises :class:`IntegrationUnavailable`
    instead of building a fallback router.
    """
    if router_override is not None:
        return router_override
    if _injected_router is not None:
        return _injected_router
    try:
        from litellm.proxy import proxy_server
    except Exception as exc:
        raise IntegrationUnavailable(
            "LiteLLM Proxy runtime is not importable in this process"
        ) from exc
    router = getattr(proxy_server, "llm_router", None)
    if router is None:
        raise IntegrationUnavailable(
            "LiteLLM Proxy router is not initialized"
        )
    return router
