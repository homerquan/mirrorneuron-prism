from ..config import PrismConfig
from ..project_config import ProjectConfig
from ..storage import write_json
from .output import redact


def validate(ctx):
    policy, llm, p, llm_path = ctx.configs()
    return {
        "valid": True,
        "policy_schema_version": policy["schema_version"],
        "policies": list(policy["policies"]),
        "policy_config": str(p),
        "litellm_config": str(llm_path),
        "controllers": list(ctx.project.controllers),
        "runtime_available": False,
    }


def show(ctx):
    policy, llm, p, llm_path = ctx.configs()
    return redact(
        {
            "project": ctx.project.model_dump(mode="json"),
            "policy": policy,
            "litellm": llm,
            "paths": {
                "project": str(ctx.project_path) if ctx.project_path else None,
                "policy": str(p),
                "litellm": str(llm_path),
                "workdir": str(ctx.workdir),
            },
            "sources": {
                **ctx.sources,
                "policy": "flag" if ctx.args.policy_config else "project/default",
                "litellm": "flag" if ctx.args.litellm_config else "project/default",
            },
        }
    )


def schema(ctx):
    from ..benchmarks.types import RunManifest
    from ..classifier.types import DecisionRequest, DecisionResult

    model = {
        "policy": PrismConfig,
        "project": ProjectConfig,
        "decision": DecisionRequest,
        "result": DecisionResult,
        "run": RunManifest,
    }[ctx.args.kind]
    data = model.model_json_schema()
    if ctx.args.output:
        write_json(ctx.explicit(ctx.args.output), data)
    return {"kind": ctx.args.kind, "schema_version": 1, "schema": data}
