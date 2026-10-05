"""Small standalone CLI. Importing help never imports ML runtimes."""

import argparse
import asyncio
import json
import os
import shlex
import sys
from pathlib import Path

from rich_argparse import RichHelpFormatter

from . import __version__
from .cli_ui import Output
from .config import load_config
from .errors import OptimizationConfigurationError, PrismError
from .profiles import initialize, load_profile, model_references, profile_inventory


class PrismParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def parser():
    root = PrismParser(
        prog="prism",
        description="Prism · one API, bounded model workflows",
        epilog="Quick start: prism init --preset openrouter → prism start --profile prism-balanced\nInspect: prism profiles · prism models · prism capacity --profile NAME\nUse --json for scripts; NO_COLOR=1 disables terminal color.",
    )
    root.add_argument("--version", action="version", version=f"prism {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("policies", help="list executable routing policies")
    init = commands.add_parser(
        "init", help="create reusable models/ and profiles/ JSON files"
    )
    init.add_argument("--out-dir", type=Path, default=Path("."))
    init.add_argument(
        "--preset",
        choices=("local", "openrouter", "openai", "claude", "gemini", "providers"),
        default="openrouter",
        help="choose a provider; providers includes native and mixed workflows",
    )
    descriptions = {
        "start": "start one profile from profiles/NAME.json or an explicit path",
        "serve": "start the OpenAI-compatible server",
        "validate": "check model and profile configuration without inference",
        "doctor": "check credentials and optionally probe backends",
        "capacity": "measure JSON, vision, and reasoning behavior",
        "models": "inspect configured physical models and credentials",
        "profiles": "inspect virtual aliases and stage assignments",
    }
    for name, description in descriptions.items():
        cmd = commands.add_parser(name, help=description)
        selector = cmd.add_mutually_exclusive_group(required=name == "start")
        selector.add_argument("--profile", help="profile name or JSON path")
        if name != "start":
            selector.add_argument("--config", help="legacy combined prism.json config")
        if name in {"start", "serve"}:
            cmd.add_argument(
                "--show-cost",
                action="store_true",
                help="show a live full-screen cost/savings dashboard (JSON reports when redirected)",
            )
            cmd.add_argument("--host")
            cmd.add_argument("--port", type=int)
            cmd.add_argument(
                "--no-auth",
                action="store_true",
                help="explicitly disable Prism client authentication (upstream keys still apply)",
            )
        if name == "doctor":
            cmd.add_argument("--probe-backends", action="store_true")
            cmd.add_argument(
                "--no-auth",
                action="store_true",
                help="check readiness for an explicitly unauthenticated deployment",
            )
        if name == "capacity":
            cmd.add_argument(
                "--model",
                action="append",
                help="physical model or profile alias; repeat to select several",
            )
    trace = commands.add_parser("trace", help="inspect a request's execution trace")
    trace_commands = trace.add_subparsers(dest="trace_command", required=True)
    show = trace_commands.add_parser(
        "show", help="fetch metadata from a running server"
    )
    show.add_argument("request_id")
    show.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    show.add_argument("--api-key-env", default="PRISM_API_KEY")
    show.add_argument("--no-auth", action="store_true")
    evaluation = commands.add_parser(
        "eval", help="run and summarize your task fixtures"
    )
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)
    run = eval_commands.add_parser(
        "run", help="paired cases against two server aliases"
    )
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--baseline", default="prism-direct")
    run.add_argument("--candidate", default="prism")
    run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    run.add_argument("--api-key-env", default="PRISM_API_KEY")
    run.add_argument("--no-auth", action="store_true")
    run.add_argument("--output-tokens", type=int, default=512)
    run.add_argument("--out", type=Path, required=True)
    compare = eval_commands.add_parser(
        "compare", help="summarize a paired evaluation JSONL"
    )
    compare.add_argument("run", type=Path)
    benchmark = commands.add_parser(
        "benchmark", help="speed, reference quality, and cost benchmarks"
    )
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command", required=True
    )
    bench_run = benchmark_commands.add_parser(
        "run", help="save a paired benchmark in a new run folder"
    )
    bench_run.add_argument(
        "--cases",
        type=Path,
        help="JSONL cases; defaults to the packaged reference suite",
    )
    bench_run.add_argument(
        "--config",
        type=Path,
        help="JSON config for a redacted settings/pricing snapshot",
    )
    bench_run.add_argument("--baseline", default="prism-direct")
    bench_run.add_argument("--candidate", default="prism-evidence")
    bench_run.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    bench_run.add_argument("--api-key-env", default="PRISM_API_KEY")
    bench_run.add_argument("--no-auth", action="store_true")
    bench_run.add_argument("--repeats", type=int, default=3)
    bench_run.add_argument("--warmup", type=int, default=1)
    bench_run.add_argument("--output-tokens", type=int, default=2048)
    bench_run.add_argument("--temperature", type=float, default=0.0)
    bench_run.add_argument("--timeout", type=float, default=180.0)
    bench_run.add_argument("--max-cases", type=int)
    bench_run.add_argument(
        "--out-dir",
        type=Path,
        help="new run folder; defaults to benchmark-results/<run-id>",
    )
    bench_compare = benchmark_commands.add_parser(
        "compare", help="compare saved run folders without calling models"
    )
    bench_compare.add_argument("runs", type=Path, nargs="+")
    bench_compare.add_argument(
        "--out-dir", type=Path, help="save comparison JSON and Markdown in a new folder"
    )

    def output_flags(cmd, top=False):
        cmd.add_argument(
            "--output",
            choices=("auto", "json", "table"),
            default="auto" if top else argparse.SUPPRESS,
            help="terminal tables or script-friendly JSON (default: auto)",
        )
        cmd.add_argument(
            "--json",
            dest="output",
            action="store_const",
            const="json",
            default=argparse.SUPPRESS,
            help="emit plain JSON, including when attached to a terminal",
        )
        for action in cmd._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    output_flags(child)

    output_flags(root, True)
    return root


async def doctor(config, models, probe, no_auth=False):
    import importlib.metadata

    from .backends import LiteLLMBackend
    from .config import Limits
    from .contracts import check_context
    from .runtime import Ledger

    result = {
        "valid": True,
        "authentication_ready": no_auth
        or bool(os.environ.get(config.server.api_key_env)),
        "authentication_mode": "disabled" if no_auth else "bearer",
        "decision_mode": config.decision.mode,
        "models": [],
    }
    try:
        result["laya_version"] = importlib.metadata.version("laya")
    except importlib.metadata.PackageNotFoundError:
        result["laya_version"] = None
    transport = LiteLLMBackend(models)
    try:
        for model in models.values():
            item = {"id": model.id, "probed": probe}
            if probe:
                try:
                    messages = [{"role": "user", "content": "Reply OK."}]
                    output = min(64, model.max_output_tokens)
                    tokens = check_context(messages, {}, output, model)
                    ledger = Ledger(Limits(max_calls=1, deadline_seconds=10))
                    reservation = await ledger.reserve("doctor", model, tokens, output)
                    await transport.complete(
                        model, messages, {}, output, ledger, reservation
                    )
                    item["ready"] = True
                except PrismError as exc:
                    item.update(ready=False, error_code=exc.code)
            result["models"].append(item)
    finally:
        await transport.close()
    result["ready"] = (
        result["authentication_ready"]
        and result["laya_version"] is not None
        and (all(m.get("ready", False) for m in result["models"]) if probe else False)
    )
    return result


def main(argv=None):
    root = parser()
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        root.print_help()
        return 0
    args = root.parse_args(arguments)
    output = Output(args.output)
    emit = output.emit
    try:
        if args.command == "policies":
            from .policies import POLICIES

            emit(
                {
                    "policies": [
                        {
                            "id": name,
                            "description": description,
                            "coverage": {
                                "retrieve_read": "focused",
                                "vision_synthesis": "visual_observations",
                                "text_synthesis": "text_observations",
                            }.get(name, "full"),
                        }
                        for name, description in POLICIES.items()
                    ],
                    "decision_mode": "route",
                    "laya_required": True,
                }
            )
            return 0
        if args.command == "init":
            paths, default_profile = initialize(args.out_dir, args.preset)
            profile_path = shlex.quote(
                str((args.out_dir / "profiles" / f"{default_profile}.json").resolve())
            )
            config, models = load_profile(
                args.out_dir / "profiles" / f"{default_profile}.json"
            )
            credentials = sorted(
                {m.api_key_env for m in models.values() if m.api_key_env}
            )
            emit(
                {
                    "created": [str(path.resolve()) for path in paths],
                    "next_steps": [
                        *[f"export {env}='your-provider-key'" for env in credentials],
                        f"export {config.server.api_key_env}='your-prism-client-secret'",
                        f"prism validate --profile {profile_path}",
                        f"prism start --profile {profile_path}",
                    ],
                }
            )
            return 0
        if args.command in {
            "start",
            "serve",
            "validate",
            "doctor",
            "capacity",
            "models",
            "profiles",
        }:
            if args.profile:
                config, models = load_profile(args.profile)
            elif args.config:
                config, models = load_config(args.config)
            elif args.command in {"profiles", "models"}:
                inventory = profile_inventory()
                if args.command == "profiles":
                    emit({"object": "profiles", "data": inventory})
                    return 0
                models = {}
                for item in inventory:
                    _, selected = load_profile(item["file"])
                    for ref, model in selected.items():
                        if ref in models and models[ref] != model:
                            raise ValueError(
                                "conflicting model definitions across profiles"
                            )
                        models[ref] = model
            elif Path("prism.json").is_file():
                config, models = load_config("prism.json")
            else:
                raise PrismError(
                    "select a profile with --profile NAME; run prism profiles to list choices"
                )
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
                with output.status("Checking backend readiness…"):
                    result = asyncio.run(
                        doctor(config, models, args.probe_backends, args.no_auth)
                    )
                emit(result)
                return 0 if result["ready"] else 3
            if args.command == "models":
                emit(
                    {
                        "object": "models",
                        "data": [
                            {
                                "id": m.id,
                                "name": m.name,
                                "provider": m.provider,
                                "base_url": m.base_url,
                                "api_version": m.api_version,
                                "timeout_seconds": m.timeout_seconds,
                                "rate_limit_rpm": m.rate_limit_rpm,
                                "context_window": m.context_window,
                                "max_output_tokens": m.max_output_tokens,
                                "capabilities": sorted(m.capabilities),
                                "credential_env": m.api_key_env,
                                "credential_ready": bool(os.environ.get(m.api_key_env))
                                if m.api_key_env
                                else True,
                                "input_cost_per_million": m.input_cost_per_million,
                                "output_cost_per_million": m.output_cost_per_million,
                                "cost_rates_are_hypothetical": m.cost_rates_are_hypothetical,
                            }
                            for m in models.values()
                        ],
                    }
                )
                return 0
            if args.command == "profiles":
                emit(
                    {
                        "object": "profiles",
                        "data": [
                            {
                                "id": alias,
                                **profile.model_dump(
                                    include={
                                        "strategy",
                                        "direct",
                                        "worker",
                                        "synthesizer",
                                        "verifier",
                                        "structured_output_model",
                                    }
                                ),
                            }
                            for alias, profile in config.profiles.items()
                        ],
                    }
                )
                return 0
            if args.command == "capacity":
                from .backends import LiteLLMBackend
                from .capacity import CapacityEvaluator, ProfileCapacityEvaluator
                from .decision import LayaDecision
                from .engine import ExecutionEngine

                async def capacity_run():
                    transport = LiteLLMBackend(models)
                    try:
                        physical = CapacityEvaluator(models, transport, config.capacity)
                        engine = ExecutionEngine(
                            config, models, transport, LayaDecision(config.decision)
                        )
                        profiles = ProfileCapacityEvaluator(engine, config.capacity)
                        results = []
                        for alias in args.model or (
                            list(config.profiles) if args.profile else list(models)
                        ):
                            evaluator = physical if alias in models else profiles
                            results.extend(await evaluator.query([alias], True))
                        return results
                    finally:
                        await transport.close()

                with output.status("Running live capability challenges…"):
                    results = asyncio.run(capacity_run())
                emit(
                    {"object": "capacity", "source": "live_evaluation", "data": results}
                )
                return 0
            if not args.no_auth and not os.environ.get(config.server.api_key_env):
                raise PrismError(
                    f"set {config.server.api_key_env} before starting Prism"
                )
            active_ids = {
                ref
                for profile in config.profiles.values()
                for ref in model_references(profile, include_baseline=False)
            }
            missing = sorted(
                {
                    m.api_key_env
                    for ref, m in models.items()
                    if ref in active_ids
                    if m.api_key_env and not os.environ.get(m.api_key_env)
                }
            )
            if missing:
                raise PrismError(
                    "set upstream credential environment variables: "
                    + ", ".join(missing)
                )
            # Revalidate endpoint recursion after CLI binding overrides.
            if args.host or args.port:
                raise_if_recursive_override(config, models, args.host, args.port)
            import uvicorn

            from .api import create_app

            app = create_app(
                config,
                models=models,
                no_auth=args.no_auth,
                cost_reporter=output.costs if args.show_cost else None,
            )
            with output.cost_display(
                args.show_cost,
                app.state.costs.snapshot,
                host=config.server.host,
                port=config.server.port,
                profiles=config.profiles,
                no_auth=args.no_auth,
            ) as fullscreen:
                uvicorn.run(
                    app,
                    host=config.server.host,
                    port=config.server.port,
                    **({"log_config": None, "log_level": "info"} if fullscreen else {}),
                )
            return 0
        if args.command == "trace":
            import httpx

            key = os.environ.get(args.api_key_env)
            if not args.no_auth and not key:
                raise PrismError("Prism API key environment variable is unset")
            try:
                response = httpx.get(
                    args.base_url.rstrip("/") + "/prism/traces/" + args.request_id,
                    headers={} if args.no_auth else {"authorization": f"Bearer {key}"},
                    timeout=10,
                    trust_env=False,
                )
            except httpx.HTTPError as exc:
                raise PrismError(
                    "unable to reach Prism; check --base-url and server readiness",
                    "transport_error",
                    503,
                ) from exc
            if response.status_code != 200:
                raise PrismError(
                    "trace unavailable", "trace_not_found", response.status_code
                )
            emit(response.json())
            return 0
        if args.command == "benchmark":
            from .benchmark import compare, run

            if args.benchmark_command == "compare":
                emit(compare(args.runs, args.out_dir))
                return 0
            key = os.environ.get(args.api_key_env)
            if not args.no_auth and not key:
                raise PrismError("Prism API key environment variable is unset")

            def progress(record):
                phase = "warmup" if record["warmup"] else "measured"
                print(
                    f"{phase} {record['case_id']} {record['route']}: "
                    f"{record['latency_ms']:.0f} ms, quality={record['quality']['score']:.2f}, "
                    f"{record['error_code'] or 'complete'}",
                    file=sys.stderr,
                    flush=True,
                )

            emit(
                asyncio.run(
                    run(
                        api_key=key or "",
                        baseline=args.baseline,
                        candidate=args.candidate,
                        base_url=args.base_url,
                        cases_path=args.cases,
                        config_path=args.config,
                        out_dir=args.out_dir,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        output_tokens=args.output_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        max_cases=args.max_cases,
                        progress=progress,
                    )
                )
            )
            return 0
        from .evaluation import evaluate, read_jsonl, summarize

        if args.eval_command == "compare":
            emit(summarize(read_jsonl(args.run)))
            return 0
        key = os.environ.get(args.api_key_env)
        if not args.no_auth and not key:
            raise PrismError("Prism API key environment variable is unset")
        if args.out.exists():
            raise PrismError("evaluation output already exists")
        rows = asyncio.run(
            evaluate(
                read_jsonl(args.cases),
                base_url=args.base_url,
                api_key=key or "",
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
    except OptimizationConfigurationError as exc:
        emit({"error": {"code": "configuration_error", "message": str(exc)}})
        return 2
    except (ValueError, OSError, ImportError):
        # Pydantic and filesystem messages can include credentials or input values.
        emit(
            {
                "error": {
                    "code": "configuration_or_dependency_error",
                    "message": "invalid/missing configuration or dependency; check the JSON config and installed dependencies",
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
        if not model.base_url:
            continue
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
