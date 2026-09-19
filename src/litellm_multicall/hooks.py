"""Request hook instance (Step 01 baseline).

``multicall_hooks`` is a real LiteLLM ``CustomLogger`` subclass so the proxy
can load it from ``litellm_settings.callbacks`` without crashing. It performs
no inference planning and runs no second copy of the execution engine: hook
methods are pass-through observers only. Authenticated request context
propagation and accounting hooks land here in later steps.
"""

from __future__ import annotations

from litellm.integrations.custom_logger import CustomLogger


class MulticallHooks(CustomLogger):
    """Observer-only proxy hook; never executes the inference plan."""

    pass


multicall_hooks = MulticallHooks()
