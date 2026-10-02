"""Versioned telemetry. No implicit stdout or private-content logging."""

import uuid


def log_event(event: dict, *, path=None):
    from .commands.output import redact
    from .storage import append_jsonl

    if any(key in event for key in ("messages", "prompt", "response", "gold", "rationale")) or (
        "state" in event and event["state"] not in ("planned", "running", "completed", "failed", "cancelled", "interrupted")
    ):
        raise ValueError(
            "content telemetry requires a separate explicitly enabled sink"
        )
    record = redact({"schema_version": 1, "event_id": uuid.uuid4().hex, **event})
    if path is not None:
        append_jsonl(path, record)
    return record
