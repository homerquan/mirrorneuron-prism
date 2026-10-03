"""Overlapping passage windows and deterministic lexical relevance."""

import re


def terms(text):
    return set(re.findall(r"[\w-]+", text.casefold().replace("_", " ")))


def passage_spans(text, size=384, overlap=96):
    """Character offsets; source bytes are derived from the unchanged original."""
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        yield start, end
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def span_key(excerpt):
    ref = excerpt["source_ref"]
    return (ref["source_id"], ref["source_sha256"], ref["byte_start"], ref["byte_end"])


def overlaps(left, right):
    a, b = left["source_ref"], right["source_ref"]
    return (
        a["source_id"] == b["source_id"]
        and a["source_sha256"] == b["source_sha256"]
        and max(a["byte_start"], b["byte_start"]) < min(a["byte_end"], b["byte_end"])
    )
