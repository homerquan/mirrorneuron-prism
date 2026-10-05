"""Standalone profiles select reusable, operator-owned model definitions."""

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import Field

from .config import (
    CapacityConfig,
    DecisionConfig,
    PrismConfig,
    Profile,
    RawModel,
    ServerConfig,
    StrictModel,
    validate_config,
)
from .errors import PrismError


class ProfileFile(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    models_dir: str = "../models"
    profile: Profile
    server: ServerConfig = Field(default_factory=ServerConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)


def model_references(profile, *, include_baseline=True):
    refs = {
        profile.direct,
        profile.worker,
        profile.worker_fallback,
        profile.synthesizer,
        profile.structured_output_model,
        profile.verifier,
    }
    if include_baseline:
        refs.add(profile.cost_baseline_model)
    for helper in (profile.evidence_compaction, profile.evidence_reduction):
        if helper:
            refs.add(helper.model)
    if profile.optimization:
        refs.update(profile.optimization.model_ids)
    return sorted(ref for ref in refs if ref is not None)


def resolve_profile(selection):
    path = Path(selection)
    if path.suffix != ".json" and len(path.parts) == 1:
        path = Path("profiles") / f"{selection}.json"
    if not path.is_file():
        raise PrismError(
            "profile not found; run prism profiles or pass a profiles/NAME.json path",
            "configuration_error",
        )
    return path.resolve()


def read_profile(path):
    return ProfileFile.model_validate(json.loads(Path(path).read_text()))


def load_profile(selection):
    path = resolve_profile(selection)
    definition = read_profile(path)
    directory = path.parent / definition.models_dir
    models = {}
    for ref in model_references(definition.profile):
        # Model IDs are filenames, independent of the provider's physical name.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ref):
            raise ValueError("standalone model IDs must be safe filenames")
        model_path = directory / f"{ref}.json"
        if not model_path.is_file():
            raise PrismError(
                f"missing model definition: models/{ref}.json",
                "configuration_error",
            )
        model = RawModel.model_validate(json.loads(model_path.read_text()))
        if model.id != ref:
            raise ValueError("model ID must match its definition filename")
        if model.api_key is not None:
            raise ValueError("standalone models must use api_key_env for credentials")
        models[ref] = model
    config = PrismConfig(
        models_file=str(directory.resolve()),
        profiles={definition.id: definition.profile},
        server=definition.server,
        decision=definition.decision,
        capacity=definition.capacity,
    )
    return validate_config(config, models)


def profile_inventory(directory=Path("profiles")):
    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise PrismError(
            "no profiles found; run prism init first", "configuration_error"
        )
    result = []
    ids = set()
    for path in paths:
        definition = read_profile(path)
        if definition.id in ids:
            raise ValueError("profile IDs must be unique")
        ids.add(definition.id)
        result.append(
            {"id": definition.id, "file": str(path), **definition.profile.model_dump()}
        )
    return result


def initialize(directory, preset):
    """Copy only a preset's referenced models; check every destination first."""
    resources = files("prism").joinpath("resources", "catalog")
    presets = json.loads(resources.joinpath("presets.json").read_text())
    selection = presets[preset]
    contents = {}
    refs = set()
    for name in selection["profiles"]:
        relative = Path("profiles") / f"{name}.json"
        content = resources.joinpath(*relative.parts).read_text()
        definition = ProfileFile.model_validate(json.loads(content))
        refs.update(model_references(definition.profile))
        contents[relative] = content
    for ref in sorted(refs):
        relative = Path("models") / f"{ref}.json"
        contents[relative] = resources.joinpath(*relative.parts).read_text()
    paths = [Path(directory) / relative for relative in contents]
    if any(path.exists() for path in paths):
        raise PrismError("configuration already exists; init never overwrites files")
    for relative, content in contents.items():
        path = Path(directory) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return paths, selection["default"]
