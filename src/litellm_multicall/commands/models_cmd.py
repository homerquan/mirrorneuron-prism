import asyncio

from ..classifier.artifacts import pinned, register, verify_controller
from ..classifier.executor import DecisionExecutor
from ..errors import PrismError


def list_models(ctx):
    from .output import redact

    _, llm, _, _ = ctx.configs()
    return {
        "resources": redact(
            [
                {
                    "name": m["model_name"],
                    "kind": "generative_alias",
                    "parameters": m["litellm_params"],
                }
                for m in llm["model_list"]
            ]
            + [
                {"name": name, "kind": "controller", "profile": p.model_dump()}
                for name, p in ctx.project.controllers.items()
            ]
        )
    }


def inspect(ctx):
    if ctx.args.name not in ctx.project.controllers:
        from .output import redact

        _, llm, _, _ = ctx.configs()
        entries = [m for m in llm["model_list"] if m["model_name"] == ctx.args.name]
        if not entries:
            raise PrismError("unknown generative alias/controller")
        return redact(
            {"name": ctx.args.name, "kind": "generative_alias", "deployments": entries}
        )
    profile = ctx.controller(ctx.args.name)
    return {
        "name": ctx.args.name,
        "kind": "controller",
        "profile": profile.model_dump(),
        "runtime_requires_prepared_artifact": profile.backend != "rules",
    }


def fetch(ctx):
    profile = ctx.controller(ctx.args.name)
    if profile.backend == "rules":
        return {"backend": "rules", "preparation_required": False}
    revision = ctx.args.revision
    if ctx.args.snapshot:
        if ctx.args.resolve_revision:
            raise PrismError(
                "local snapshot registration requires an explicit immutable --revision"
            )
        snapshot = ctx.explicit(ctx.args.snapshot)
        pinned(revision)
    else:
        if ctx.args.offline:
            raise PrismError("--offline forbids remote artifact preparation")
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise PrismError(
                "models fetch requires the classifier-jevcpu extra",
                4,
                "missing_dependency",
            ) from exc
        if ctx.args.resolve_revision:
            revision = (
                HfApi()
                .model_info(profile.model_source, revision=revision or "main")
                .sha
            )
        pinned(revision)
        snapshot = ctx.root / "models" / revision
        snapshot_download(
            profile.model_source,
            revision=revision,
            local_dir=snapshot,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
                "*.jinja",
                "LICENSE*",
                "README*",
            ],
        )
    descriptor = register(
        ctx.artifact_lock,
        profile,
        snapshot,
        revision,
        ctx.args.license_reference
        or f"https://huggingface.co/{profile.model_source}/blob/{revision}/README.md",
    )
    return {
        "artifact_ref": profile.artifact_ref,
        "artifact": descriptor,
        "license_review": "operator must review model terms at the recorded reference",
    }


def verify(ctx):
    profile = ctx.controller(ctx.args.name)
    artifact = (
        verify_controller(ctx.artifact_lock, profile)
        if profile.backend != "rules"
        else None
    )
    data = {
        "verified": True,
        "artifact": artifact,
        "device": "cpu",
        "load_tested": False,
    }
    if ctx.args.load:

        async def load():
            executor = DecisionExecutor(profile, artifact)
            try:
                await executor._start(
                    __import__("time").monotonic()
                    + (ctx.args.timeout or profile.startup_timeout_ms / 1000)
                )
                return executor.metadata
            finally:
                await executor.close()

        data.update(load_tested=True, runtime_metadata=asyncio.run(load()))
    return data
