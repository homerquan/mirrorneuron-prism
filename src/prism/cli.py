"""Small standalone CLI. Importing help never imports Torch or LiteLLM."""

import argparse
import asyncio
import json
import os
import sys
from importlib.resources import files
from pathlib import Path

from . import __version__
from .config import load_config
from .errors import PrismError


def parser():
    root = argparse.ArgumentParser(
        prog="prism", description="Standalone adaptive LLM proxy"
    )
    root.add_argument("--version", action="version", version=f"prism {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create JSON config and raw model registry")
    init.add_argument("--out-dir", type=Path, default=Path("."))
    for name in ("serve", "validate", "doctor"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--config", default="prism.json")
        if name == "serve":
            cmd.add_argument("--host")
            cmd.add_argument("--port", type=int)
        if name == "doctor":
            cmd.add_argument("--probe-backends", action="store_true")
    trace = commands.add_parser("trace")
    trace_commands = trace.add_subparsers(dest="trace_command", required=True)
    show = trace_commands.add_parser("show")
    show.add_argument("request_id")
    show.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    show.add_argument("--api-key-env", default="PRISM_API_KEY")
    evaluation = commands.add_parser("eval")
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    run = eval_commands.add_parser(
        "run", help="paired cases against two server aliases"
    )
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--baseline", default="prism-direct")
    run.add_argument("--candidate", default="prism")
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--api-key-env", default="PRISM_API_KEY")
    run.add_argument("--output-tokens", type=int, default=512)
    run.add_argument("--out", type=Path, required=True)
    compare = eval_commands.add_parser(
        "compare", help="summarize a paired evaluation JSONL"
    )
    compare.add_argument("run", type=Path)
    return root


def emit(value):
    print(json.dumps(value, indent=2))


async def doctor(config, models, probe):
    import importlib.metadata

    import httpx

    result = {
        "valid": True,
        "authentication_ready": bool(os.environ.get(config.server.api_key_env)),
        "decision_mode": config.decision.mode,
        "models": [],
    }
    try:
        result["laya_version"] = importlib.metadata.version("laya")
    except importlib.metadata.PackageNotFoundError:
        result["laya_version"] = None
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        for model in models.values():
            item = {"id": model.id, "probed": probe}
            if probe:
                try:
                    key = model.credential()
                    response = await client.get(
                        model.base_url.rstrip("/") + "/models",
                        headers={"authorization": f"Bearer {key}"} if key else {},
                    )
                    item["ready"] = response.status_code == 200
                    item["status"] = response.status_code
                except (httpx.HTTPError, PrismError):
                    item["ready"] = False
            result["models"].append(item)
    result["ready"] = (
        result["authentication_ready"]
        and (config.decision.mode == "off" or result["laya_version"] is not None)
        and (all(m.get("ready", False) for m in result["models"]) if probe else False)
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            paths = [args.out_dir / name for name in ("prism.json", "models.json")]
            if any(path.exists() for path in paths):
                raise PrismError(
                    "configuration already exists; init never overwrites files"
                )
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for path in paths:
                path.write_text(
                    files("prism").joinpath("resources", path.name).read_text()
                )
            emit({"created": [str(path.resolve()) for path in paths]})
            return 0
        if args.command in {"serve", "validate", "doctor"}:
            config, models = load_config(args.config)
            if args.command == "validate":
                emit(
                    {
                        "valid": True,
                        "profiles": list(config.profiles),
                        "raw_models": list(models),
                    }
                )
                return 0
            if args.command == "doctor":
                result = asyncio.run(doctor(config, models, args.probe_backends))
                emit(result)
                return 0 if result["ready"] else 3
            if not os.environ.get(config.server.api_key_env):
                raise PrismError(
                    "set the configured Prism API key environment variable before serving"
                )
            # Revalidate endpoint recursion after CLI binding overrides.
            if args.host or args.port:
                raise_if_recursive_override(config, models, args.host, args.port)
            import uvicorn

            from .api import create_app

            uvicorn.run(
                create_app(config, models=models),
                host=config.server.host,
                port=config.server.port,
            )
            return 0
        if args.command == "trace":
            import httpx

            key = os.environ.get(args.api_key_env)
            if not key:
                raise PrismError("Prism API key environment variable is unset")
            response = httpx.get(
                args.base_url.rstrip("/") + "/prism/traces/" + args.request_id,
                headers={"authorization": f"Bearer {key}"},
                timeout=10,
                trust_env=False,
            )
            if response.status_code != 200:
                raise PrismError(
                    "trace unavailable", "trace_not_found", response.status_code
                )
            emit(response.json())
            return 0
        from .evaluation import evaluate, read_jsonl, summarize

        if args.eval_command == "compare":
            emit(summarize(read_jsonl(args.run)))
            return 0
        key = os.environ.get(args.api_key_env)
        if not key:
            raise PrismError("Prism API key environment variable is unset")
        if args.out.exists():
            raise PrismError("evaluation output already exists")
        rows = asyncio.run(
            evaluate(
                read_jsonl(args.cases),
                base_url=args.base_url,
                api_key=key,
                baseline=args.baseline,
                candidate=args.candidate,
                output_tokens=args.output_tokens,
            )
        )
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
        emit(summarize(rows))
        return 0
    except PrismError as exc:
        emit(exc.body())
        return 2
    except (ValueError, OSError, ImportError):
        # Pydantic and filesystem messages can include credentials or input values.
        emit(
            {
                "error": {
                    "code": "configuration_or_dependency_error",
                    "message": "invalid/missing configuration or dependency; check the JSON config and installation extras",
                }
            }
        )
        return 2


def raise_if_recursive_override(config, models, host, port):
    from urllib.parse import urlsplit

    if host:
        config.server.host = host
    if port:
        if not 1 <= port <= 65535:
            raise PrismError("port must be between 1 and 65535")
        config.server.port = port
    local = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}
    for model in models.values():
        url = urlsplit(model.base_url)
        if (
            url.hostname == config.server.host
            or url.hostname in local
            and config.server.host in local
        ) and (
            url.port or (443 if url.scheme == "https" else 80)
        ) == config.server.port:
            raise PrismError("self-referential backend after server binding override")


if __name__ == "__main__":
    sys.exit(main())
