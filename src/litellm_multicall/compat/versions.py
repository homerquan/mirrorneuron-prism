"""Version-sensitive LiteLLM compatibility facts (Step 01 baseline).

``TESTED_LITELLM_VERSION`` is the exact LiteLLM release the current checkout
was verified against. ``DECLARED_RANGE`` mirrors the dependency constraint in
``pyproject.toml``. Bumping the tested version requires re-running the
provider-contract and real-proxy checks and updating the implementation map.
"""

from __future__ import annotations

TESTED_LITELLM_VERSION = "1.101.0"
DECLARED_RANGE = "litellm>=1,<2"
