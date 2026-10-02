from typing import Literal

from pydantic import Field, model_validator

from ..config import StrictModel
from ..project_config import Controller
from ..storage import digest


class Task(StrictModel):
    id: str
    group_id: str
    split: str


class Plan(StrictModel):
    schema_version: Literal[1] = 1
    suite: Literal["semif-authored", "semif-shape"]
    dataset_manifest_sha256: str
    dataset_revision: str
    selection: Literal["smoke", "full"]
    split: Literal["calibration", "validation", "test", "all", "systems"]
    split_method: str = "prism-group-hash-v1"
    tasks: list[Task] = Field(min_length=1)
    treatment: Literal["rules", "cpu_direct", "cpu_shared", "cpu_compact_generation"]
    controller: str
    profile: Controller
    model_artifact_sha256: str | None = None
    calibration_sha256: str | None = None
    calibration_path: str | None = None
    seed: int = 42
    repetitions: int = Field(ge=1, le=100)
    warmup_requests: Literal[1] = 1
    concurrency: Literal[1] = 1
    quality_margin: float = Field(default=0.02, gt=0, lt=1)
    statistical_method: str = (
        "paired independent-source-group Hoeffding, one-sided alpha=0.05"
    )
    resource_metric: str = "controller_process_cpu_seconds"
    privacy_log_content: Literal[False] = False
    package_code_sha256: str
    project_sha256: str
    plan_sha256: str

    @model_validator(mode="after")
    def integrity(self):
        body = self.model_dump(mode="json", exclude={"plan_sha256"})
        if digest(body) != self.plan_sha256:
            raise ValueError("plan hash mismatch; plans are immutable")
        if len({t.id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task IDs")
        if self.treatment == "rules" and self.profile.backend != "rules":
            raise ValueError("rules treatment requires rules backend")
        if self.treatment != "rules" and (
            self.profile.backend != "jev_cpu" or self.model_artifact_sha256 is None
        ):
            raise ValueError("CPU treatments require a pinned JEV artifact")
        return self


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    plan: Plan
    dataset_limitation: str
    environment: dict
    hardware: dict
    cache_policy: str
    warmup_policy: str
    failure_policy: str
    privacy: dict
    manifest_sha256: str

    @model_validator(mode="after")
    def integrity(self):
        if (
            digest(self.model_dump(mode="json", exclude={"manifest_sha256"}))
            != self.manifest_sha256
        ):
            raise ValueError("run manifest hash mismatch")
        return self
