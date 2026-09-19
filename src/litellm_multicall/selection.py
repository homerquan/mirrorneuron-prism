"""Deterministic selection helpers (Step 01 baseline: STUB).

.. warning::
    :func:`select_best` currently returns the first candidate as a
    structural placeholder. First-surviving is NOT best, NOT correct, and
    NOT judge preference. Real judge/consensus selection with ID validation
    lands in a later step; nothing in the production path calls this yet
    (the provider fails closed).
"""


def select_best(candidates):
    if not candidates:
        return None
    # STUB: first candidate only; see module docstring.
    return candidates[0]
