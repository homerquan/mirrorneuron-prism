"""Package-local trace events (Step 01 baseline).

:func:`log_event` is a metadata-only no-op sink: it records nothing but the
caller-supplied metadata dict and never logs prompts, candidate contents,
credentials, or judge explanations. Real LiteLLM callback integration lands
in a later step.
"""


def log_event(event: dict):
    # Metadata-only no-op sink (see module docstring).
    pass
