import json
import re
from urllib.parse import urlsplit, urlunsplit


def redact(value, key=""):
    if any(
        word in key.lower()
        for word in (
            "api_key",
            "secret",
            "password",
            "authorization",
            "credential",
            "access_token",
            "master_key",
        )
    ):
        return "[redacted]"
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        if value.startswith("os.environ/"):
            return "[environment reference]"
        if "://" in value:
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                return urlunsplit(
                    (parsed.scheme, parsed.netloc.split("@")[-1], parsed.path, "", "")
                )
        return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", value)
    return value


def envelope(command, data=None, error=None, warnings=None):
    return {
        "schema_version": 1,
        "command": command,
        "status": "error" if error else "ok",
        "data": data,
        "warnings": warnings or [],
        "error": error,
    }


def render(args, record):
    if getattr(args, "format", "human") in ("json", "jsonl"):
        print(json.dumps(record, ensure_ascii=False, allow_nan=False))
    elif record["error"]:
        import sys

        print(record["error"]["message"], file=sys.stderr)
    elif not getattr(args, "quiet", False):
        print(json.dumps(record["data"], ensure_ascii=False, indent=2, allow_nan=False))
