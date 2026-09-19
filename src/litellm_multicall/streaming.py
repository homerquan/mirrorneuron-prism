"""Buffered-final streaming conversion (Step 01 baseline: NOT implemented).

Fail-closed stub: :func:`convert_to_chunks` raises
:class:`NotImplementedError` instead of returning a silent empty chunk list,
which would look like a successful but content-free stream.
"""


def convert_to_chunks(response):
    raise NotImplementedError(
        "convert_to_chunks is not implemented yet (Step 01 baseline)"
    )
