import pytest

from litellm_multicall.errors import PrismError
from litellm_multicall.storage import append_jsonl, atomic_stream, atomic_write, locked


def test_create_only_private_atomic_rollback(tmp_path):
    path = tmp_path / "artifact.json"
    atomic_write(path, "one")
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(PrismError):
        atomic_write(path, "two")
    assert path.read_text() == "one"
    with pytest.raises(RuntimeError):
        with atomic_stream(tmp_path / "unfinished") as output:
            output.write("partial")
            raise RuntimeError("cancel")
    assert not (tmp_path / "unfinished").exists()
    assert not list(tmp_path.glob(".prism-*"))


def test_per_artifact_lock(tmp_path):
    with locked(tmp_path / ".run.lock"):
        with pytest.raises(PrismError, match="locked"):
            with locked(tmp_path / ".run.lock"):
                pass
    with locked(tmp_path / ".run.lock"):
        pass


def test_append_complete_records(tmp_path):
    import json

    path = tmp_path / "events.jsonl"
    append_jsonl(path, {"id": "a"})
    append_jsonl(path, {"id": "b"})
    assert [json.loads(row)["id"] for row in path.read_text().splitlines()] == [
        "a",
        "b",
    ]
