"""Project settings separate from LiteLLM's generative model registry."""

from typing import Literal

from pydantic import Field

from .config import Name, Positive, StrictModel


class Files(StrictModel):
    policy_config: str = "multicall.yaml"
    litellm_config: str = "litellm.yaml"
    artifact_lock: str = ".prism/artifacts.lock.json"


class Privacy(StrictModel):
    local_only: Literal[True] = True
    log_content: bool = False


class Controller(StrictModel):
    backend: Literal["jev_cpu", "rules"] = "jev_cpu"
    model_source: Name = "Qwen/Qwen3-0.6B"
    artifact_ref: Name = "quick"
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float32"] = "float32"
    scoring_mode: Literal["direct", "shared", "compact_generation"] = "direct"
    max_input_tokens: Positive = 1024
    max_request_bytes: Positive = 262144
    max_criterion_bytes: Positive = 32768
    max_criteria_per_batch: Positive = 8
    max_padded_suffix_tokens: Positive = 8192
    process_mode: Literal["spawned"] = "spawned"
    threads: int = Field(default=2, ge=1, le=64)
    interop_threads: int = Field(default=1, ge=1, le=16)
    max_inflight_batches: Literal[1] = 1
    queue_capacity: int = Field(default=8, ge=0, le=1024)
    inference_timeout_ms: Positive = 10000
    startup_timeout_ms: Positive = 120000
    max_restarts: int = Field(default=1, ge=0, le=3)
    on_unavailable: Literal["abstain", "error"] = "abstain"


class Measurements(StrictModel):
    sample_interval_ms: Positive = 250
    log_content: bool = False
    collect_gpu: Literal["auto", "disabled"] = "auto"
    collect_energy: Literal["auto", "disabled"] = "auto"


class Benchmarks(StrictModel):
    seed: int = 42
    repetitions: int = Field(default=3, ge=1, le=100)
    concurrency: Literal[1] = 1
    run_root: str = ".prism/runs"


class ProjectConfig(StrictModel):
    kind: Literal["prism_project"] = "prism_project"
    schema_version: Literal[1] = 1
    workdir: str = "."
    files: Files = Field(default_factory=Files)
    privacy: Privacy = Field(default_factory=Privacy)
    controllers: dict[Name, Controller] = Field(default_factory=dict)
    measurements: Measurements = Field(default_factory=Measurements)
    benchmarks: Benchmarks = Field(default_factory=Benchmarks)
