"""The only module allowed to resolve LiteLLM proxy Router internals."""

_injected = None


def inject_router(router):
    global _injected
    _injected = router


def resolve_proxy_router():
    if _injected is not None:
        return _injected
    from litellm.proxy import proxy_server

    router = getattr(proxy_server, "llm_router", None)
    if router is None:
        raise RuntimeError("LiteLLM proxy Router is not initialized")
    return router
