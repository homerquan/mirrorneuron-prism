from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import load_policy_config, load_yaml, validate_references
from ..errors import PrismError
from ..project_config import ProjectConfig


@dataclass
class Context:
    args: object
    invocation: Path
    project_path: Path | None
    project: ProjectConfig
    workdir: Path
    sources: dict

    @classmethod
    def resolve(cls, args):
        invocation = Path.cwd()
        selected = getattr(args, "config", None) or os.environ.get("PRISM_CONFIG")
        if not selected and (invocation / "prism.yaml").exists():
            selected = "prism.yaml"
        path = (invocation / selected).resolve() if selected else None
        project = (
            ProjectConfig.model_validate(load_yaml(path)) if path else ProjectConfig()
        )
        base = path.parent if path else invocation
        flag = getattr(args, "workdir", None)
        env = os.environ.get("PRISM_WORKDIR")
        workdir = (
            (invocation / (flag or env)).resolve()
            if flag or env
            else (base / project.workdir).resolve()
        )
        sources = {
            "project": "flag"
            if getattr(args, "config", None)
            else "environment"
            if os.environ.get("PRISM_CONFIG")
            else "cwd"
            if path
            else "default",
            "workdir": "flag"
            if flag
            else "environment"
            if env
            else "project"
            if path
            else "default",
        }
        return cls(args, invocation, path, project, workdir, sources)

    def explicit(self, value: str) -> Path:
        return (self.invocation / value).resolve()

    def configured(self, value: str) -> Path:
        return (
            (self.project_path.parent if self.project_path else self.invocation) / value
        ).resolve()

    @property
    def root(self) -> Path:
        return self.workdir / ".prism"

    @property
    def run_root(self) -> Path:
        return (self.workdir / self.project.benchmarks.run_root).resolve()

    @property
    def artifact_lock(self) -> Path:
        return self.configured(self.project.files.artifact_lock)

    def configs(self):
        p = getattr(self.args, "policy_config", None)
        llm_path = getattr(self.args, "litellm_config", None)
        policy_path = (
            self.explicit(p) if p else self.configured(self.project.files.policy_config)
        )
        llm_path = (
            self.explicit(llm_path)
            if llm_path
            else self.configured(self.project.files.litellm_config)
        )
        policy = load_policy_config(policy_path)
        llm = load_yaml(llm_path)
        validate_references(policy, llm)
        return policy, llm, policy_path, llm_path

    def controller(self, name: str):
        if name not in self.project.controllers:
            raise PrismError(f"unknown controller: {name}")
        return self.project.controllers[name]
