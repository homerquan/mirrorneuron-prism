import yaml

from ..config import PrismConfig, validate_references
from ..errors import PrismError
from ..project_config import Controller, ProjectConfig
from ..storage import atomic_write


def example_configs(cpu=False):
    policy = PrismConfig.model_validate(
        {
            "schema_version": 1,
            "policies": {
                "default": {
                    "strategy": "best_of_n",
                    "allowed_model_groups": ["local-small"],
                    "candidates": [{"model": "local-small", "samples": 2}],
                    "selection": {"type": "llm_judge", "model": "local-small"},
                    "budgets": {
                        "max_model_calls": 3,
                        "max_total_tokens": 16000,
                        "deadline_ms": 120000,
                    },
                }
            },
        }
    ).model_dump(mode="json", exclude_none=True)
    llm = {
        "model_list": [
            {
                "model_name": "local-small",
                "litellm_params": {
                    "model": "openai/local-small",
                    "api_base": "http://127.0.0.1:8000/v1",
                    "api_key": "os.environ/LOCAL_LLM_API_KEY",
                },
            },
            {
                "model_name": "smart-local",
                "litellm_params": {"model": "multicall/default", "num_retries": 0},
            },
        ],
        "litellm_settings": {
            "custom_provider_map": [
                {
                    "provider": "multicall",
                    "custom_handler": "litellm_multicall.provider.multicall_provider",
                }
            ],
            "callbacks": ["litellm_multicall.hooks.multicall_hooks"],
        },
    }
    validate_references(policy, llm)
    project = ProjectConfig(
        controllers={"quick": Controller(), "rules": Controller(backend="rules")}
        if cpu
        else {}
    )
    return policy, llm, project.model_dump(mode="json")


def run(ctx):
    import json

    from ..classifier.types import DecisionRequest

    out = ctx.explicit(ctx.args.out_dir)
    policy, llm, project = example_configs(ctx.args.template == "cpu-bench")
    files = {
        "multicall.yaml": yaml.safe_dump(policy, sort_keys=False),
        "litellm.yaml": yaml.safe_dump(llm, sort_keys=False),
        "prism.yaml": yaml.safe_dump(project, sort_keys=False),
    }
    if ctx.args.template == "cpu-bench":
        request = DecisionRequest.model_validate(
            {
                "schema_version": 1,
                "request_id": "decision-001",
                "state_version": "v1",
                "state": {
                    "operation": "read_repository_file",
                    "result": "temporary timeout",
                    "attempts": 1,
                },
                "criteria": [
                    {
                        "id": "next_action",
                        "primitive": "retry",
                        "question": "Which next action is supported by the result?",
                        "options": [
                            {
                                "id": "retry_read",
                                "description": "Retry the read-only operation within its budget.",
                            },
                            {"id": "inspect", "description": "Inspect the failure."},
                            {"id": "abstain", "description": "Insufficient evidence."},
                        ],
                    }
                ],
            }
        )
        files["examples/decision.json"] = (
            json.dumps(request.model_dump(mode="json"), indent=2) + "\n"
        )
        files["experiments/native.yaml"] = yaml.safe_dump(
            {
                "suite": "semif-authored",
                "split": "validation",
                "treatments": ["rules", "cpu_direct", "cpu_compact_generation"],
                "seed": 42,
            }
        )
    if not ctx.args.force and any((out / name).exists() for name in files):
        raise PrismError(
            "destination contains generated files; use --force to overwrite"
        )
    created = []
    previous = (
        {name: (out / name).read_text() for name in files if (out / name).exists()}
        if ctx.args.force
        else {}
    )
    try:
        for name, content in files.items():
            target = out / name
            existed = target.exists()
            atomic_write(target, content, force=ctx.args.force)
            if not existed:
                created.append(target)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        for name, content in previous.items():
            atomic_write(out / name, content, force=True)
        raise
    return {
        "created": [str(out / name) for name in files],
        "inference_runtime": "planned",
        "downloads": 0,
    }
