"""Explicit preparation; runtime never discovers/downloads artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import PrismError
from ..storage import digest, file_hash, locked, read_json, write_json


def pinned(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise PrismError(
            "an immutable 40-character commit revision is required",
            7,
            "mutable_revision",
        )


def required_files(snapshot):
    names = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    if (snapshot / "model.safetensors.index.json").exists():
        index = read_json(snapshot / "model.safetensors.index.json")
        names.add("model.safetensors.index.json")
        names.update(index["weight_map"].values())
    else:
        names.add("model.safetensors")
    for name in names:
        if Path(name).name != name or not (snapshot / name).is_file():
            raise PrismError(
                f"incomplete or unsafe model snapshot: {name}", 7, "corrupt_artifact"
            )
    return names


def describe_snapshot(
    snapshot: Path, source: str, revision: str, license_reference: str
):
    pinned(revision)
    snapshot = snapshot.resolve()
    required_files(snapshot)
    files = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and (
            path.suffix in (".json", ".safetensors", ".txt", ".model", ".jinja")
            or path.name.startswith(("LICENSE", "README"))
        ):
            files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    identity = {
        "source": source,
        "revision": revision,
        "tokenizer_revision": revision,
        "files": files,
        "license_reference": license_reference,
        "complete": True,
    }
    return {
        **identity,
        "snapshot": str(snapshot),
        "identity_sha256": digest(identity),
        "weights_bytes": sum(
            v["bytes"] for k, v in files.items() if k.endswith(".safetensors")
        ),
    }


def register(lock_path, profile, snapshot, revision, license_reference):
    descriptor = describe_snapshot(
        snapshot, profile.model_source, revision, license_reference
    )
    with locked(lock_path.with_suffix(".lock")):
        lock = (
            read_json(lock_path)
            if lock_path.exists()
            else {"schema_version": 1, "artifacts": {}}
        )
        existing = lock["artifacts"].get(profile.artifact_ref)
        if existing and existing != descriptor:
            raise PrismError(
                "artifact reference is immutable; use a new artifact_ref for changed files/revision",
                7,
                "identity_mismatch",
            )
        lock["artifacts"][profile.artifact_ref] = descriptor
        write_json(lock_path, lock, force=lock_path.exists())
    return descriptor


def verify_controller(lock_path, profile):
    if not lock_path.exists():
        raise PrismError(
            "model not prepared: use models fetch explicitly", 3, "unprepared_model"
        )
    lock = read_json(lock_path)
    if lock.get("schema_version") != 1:
        raise PrismError("unsupported artifact lock version", 7, "corrupt_artifact")
    descriptor = lock.get("artifacts", {}).get(profile.artifact_ref)
    if not descriptor:
        raise PrismError(
            "controller artifact_ref is not prepared", 3, "unprepared_model"
        )
    if descriptor.get("source") != profile.model_source:
        raise PrismError("controller source differs from lock", 7, "identity_mismatch")
    actual = describe_snapshot(
        Path(descriptor["snapshot"]),
        descriptor["source"],
        descriptor["revision"],
        descriptor["license_reference"],
    )
    if actual != descriptor:
        raise PrismError(
            "model/tokenizer snapshot changed after preparation", 7, "corrupt_artifact"
        )
    return descriptor
