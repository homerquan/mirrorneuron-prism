"""Pinned fixtures with separate scorer allowlists and evaluator-held labels."""

import json
import random
import subprocess

from ..classifier.artifacts import pinned
from ..classifier.types import DecisionRequest
from ..errors import PrismError
from ..storage import atomic_write, digest, file_hash, read_json, write_json
from .registry import supported

UPSTREAM = "https://github.com/leesk212/JEV-CPU"


def paths(root, suite):
    supported(suite)
    return root / "datasets" / suite


def prepare(root, suite, source, revision):
    info = supported(suite)
    pinned(revision)

    def source_bytes(name):
        try:
            return subprocess.check_output(
                ["git", "-C", str(source), "show", f"{revision}:{name}"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise PrismError(
                "pinned dataset source unavailable in local checkout",
                5,
                "dataset_source",
            ) from exc

    raw = source_bytes("benchmarks/data/" + info["file"])
    license_text = source_bytes("LICENSE").decode()
    notices = source_bytes("THIRD_PARTY.md").decode()
    import hashlib

    rows = [json.loads(line) for line in raw.decode().splitlines()]
    seen = set()
    inputs, gold, tasks = [], [], []
    groups = sorted(
        {r["group_id"] for r in rows}, key=lambda g: digest(["prism-group-hash-v1", g])
    )
    split_groups = {
        "calibration": groups[: len(groups) // 3],
        "validation": groups[len(groups) // 3 : 2 * len(groups) // 3],
        "test": groups[2 * len(groups) // 3 :],
    }
    mapping = {g: split for split, gs in split_groups.items() for g in gs}
    for row in rows:
        if row["id"] in seen:
            raise PrismError("duplicate fixture IDs", 7, "corrupt_dataset")
        seen.add(row["id"])
        # Build using ONLY these fields, even if source adds adversarial metadata.
        criterion = {
            "id": row["id"],
            "primitive": "sufficient",
            "question": row["question"],
            "options": [
                {"id": o["id"], "description": o["description"]} for o in row["options"]
            ],
        }
        request = DecisionRequest.model_validate(
            {
                "request_id": row["id"],
                "logical_request_id": row["id"],
                "state_version": digest(row["state"]),
                "state": row["state"],
                "criteria": [criterion],
            }
        )
        inputs.append(request.model_dump(mode="json"))
        split = "systems" if suite == "semif-shape" else mapping[row["group_id"]]
        tasks.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "split": split,
                "upstream_split": row.get("split"),
                "family": row["family"],
            }
        )
        if suite == "semif-authored":
            label = row.get("label")
            if type(label) is not int or not 0 <= label < len(row["options"]):
                raise PrismError("invalid fixture label", 7, "corrupt_dataset")
            gold.append(
                {
                    "id": row["id"],
                    "gold_option_id": row["options"][label]["id"],
                    "group_id": row["group_id"],
                    "family": row["family"],
                }
            )
    base = paths(root, suite)
    if base.exists():
        raise PrismError(
            "dataset already prepared; use a new workdir rather than overwriting pinned evidence",
            7,
            "immutable_dataset",
        )
    base.mkdir(parents=True)
    try:
        atomic_write(
            base / "inputs.jsonl",
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in inputs),
        )
        atomic_write(
            base / "evaluator.jsonl", "".join(json.dumps(r) + "\n" for r in gold)
        )
        atomic_write(base / "LICENSE", license_text)
        atomic_write(base / "THIRD_PARTY.md", notices)
        manifest = {
            "schema_version": 1,
            "suite": suite,
            "repository": UPSTREAM,
            "revision": revision,
            "source_file": "benchmarks/data/" + info["file"],
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "input_sha256": file_hash(base / "inputs.jsonl"),
            "evaluator_sha256": file_hash(base / "evaluator.jsonl"),
            "transformation": "prism-native-allowlist-v1",
            "tasks": tasks,
            "source_group_count": len(groups),
            "split_method": "prism-group-hash-v1 (derived; upstream metadata retained)",
            "split_group_hashes": {
                k: digest(sorted(v)) for k, v in split_groups.items()
            },
            "license": "MIT project code/fixtures; upstream third-party notices retained",
            "limitation": info["limitation"],
        }
        manifest["manifest_sha256"] = digest(manifest)
        write_json(base / "manifest.json", manifest)
    except Exception:
        import shutil

        shutil.rmtree(base)
        raise
    return manifest


def verify(root, suite):
    base = paths(root, suite)
    if not (base / "manifest.json").exists():
        raise PrismError(
            "dataset is not prepared; use benchmark prepare --source --revision",
            3,
            "unprepared_dataset",
        )
    manifest = read_json(base / "manifest.json")
    if (
        digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        != manifest["manifest_sha256"]
    ):
        raise PrismError("dataset manifest hash mismatch", 7, "corrupt_dataset")
    for name, field in (
        ("inputs.jsonl", "input_sha256"),
        ("evaluator.jsonl", "evaluator_sha256"),
    ):
        if file_hash(base / name) != manifest[field]:
            raise PrismError("prepared dataset changed", 7, "corrupt_dataset")
    return manifest


def select(manifest, split, selection, seed):
    tasks = [t for t in manifest["tasks"] if split == "all" or t["split"] == split]
    if not tasks:
        raise PrismError("selected dataset split has no tasks")
    if selection == "smoke":
        # Select complete source groups; never split related rows across partitions.
        groups = sorted({t["group_id"] for t in tasks})
        random.Random(seed).shuffle(groups)
        selected = set(groups[:2])
        tasks = [t for t in tasks if t["group_id"] in selected]
    return [{k: t[k] for k in ("id", "group_id", "split")} for t in tasks]
