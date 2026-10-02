"""Fail-closed integration errors (Step 01 baseline).

All LiteLLM-version-sensitive error mapping lives here per SPEC section 16.
Use :class:`IntegrationUnavailable` when the proxy router, request context,
or version support required for a virtual-model call is missing. Fail closed:
never silently downgrade to an untested path.
"""

from __future__ import annotations


class IntegrationUnavailable(RuntimeError):
    """Raised when required LiteLLM proxy runtime support is unavailable."""
