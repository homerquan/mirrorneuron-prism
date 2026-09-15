from __future__ import annotations
from pathlib import Path

try:
    import yaml
    _yaml = yaml
except Exception:
    _yaml = None

def _read_yaml(path: Path | str) -> dict:
    txt = Path(path).read_text()
    if _yaml:
        return _yaml.safe_load(txt) or {}
    return {}

def load_policy_config(path: Path | str) -> dict:
    data = _read_yaml(path) or {}
    # Minimal validation
    return {
        "schema_version": data.get("schema_version", 1),
        "runtime": data.get("runtime", {}),
        "policies": data.get("policies", {}),
    }

def load_yaml(path: Path | str) -> dict:
    return _read_yaml(path) or {}

