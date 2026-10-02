"""Private atomic artifacts and canonical identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .errors import PrismError


def canonical(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_write(path: Path, text: str, *, force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".prism-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise PrismError(f"refusing to overwrite {path}; use --force") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value, *, force: bool = False) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        force=force,
    )


def read_json(path: Path):
    def reject(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def append_jsonl(path: Path, value):
    data = (canonical(value) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_stream(path: Path):
    """Bounded output spool; create-only commit after the stream finishes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".prism-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
        except FileExistsError as exc:
            raise PrismError("refusing to overwrite output") from exc
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def locked(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PrismError(
                "artifact is locked by another process", 5, "locked"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
