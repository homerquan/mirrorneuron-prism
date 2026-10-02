import argparse

from ..errors import PrismError


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise PrismError(message)


def common(parser):
    for name in ("config", "workdir"):
        parser.add_argument(f"--{name}", default=argparse.SUPPRESS)
    parser.add_argument(
        "--format", choices=("human", "json", "jsonl"), default=argparse.SUPPRESS
    )
    for name in ("quiet", "verbose", "offline"):
        parser.add_argument(f"--{name}", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=argparse.SUPPRESS)


def paths(parser):
    parser.add_argument("--policy-config")
    parser.add_argument("--litellm-config")


def command(sub, name, handler=None, command_id=None):
    parser = sub.add_parser(name)
    common(parser)
    if handler:
        parser.set_defaults(handler=handler, command_id=command_id or name)
    return parser


def build_parser():
    parser = Parser(prog="mn_prism")
    common(parser)
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    init = command(sub, "init", "init_cmd:run")
    init.add_argument("--out-dir", default=".")
    init.add_argument("--force", action="store_true")
    init.add_argument("--template", choices=("default", "cpu-bench"), default="default")
    config = command(sub, "config").add_subparsers(required=True)
    for name in ("validate", "show"):
        paths(command(config, name, f"config_cmd:{name}", f"config.{name}"))
    schema = command(config, "schema", "config_cmd:schema", "config.schema")
    schema.add_argument(
        "kind", choices=("project", "policy", "decision", "result", "run")
    )
    schema.add_argument("--output")
    paths(command(sub, "validate", "config_cmd:validate", "config.validate"))
    paths(command(sub, "doctor", "doctor_cmd:run"))
    proxy = command(sub, "proxy", "proxy_cmd:run")
    paths(proxy)
    proxy.add_argument("--model-name")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=4000)
    serve = command(sub, "serve", "serve_cmd:run")
    serve.add_argument("--file", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=4001)
    models = command(sub, "models").add_subparsers(required=True)
    command(models, "list", "models_cmd:list_models", "models.list")
    inspect = command(models, "inspect", "models_cmd:inspect", "models.inspect")
    inspect.add_argument("name")
    verify = command(models, "verify", "models_cmd:verify", "models.verify")
    verify.add_argument("name")
    verify.add_argument("--load", action="store_true")
    fetch = command(models, "fetch", "models_cmd:fetch", "models.fetch")
    fetch.add_argument("name")
    fetch.add_argument(
        "--snapshot", help="explicitly register a complete existing local snapshot"
    )
    fetch.add_argument("--revision")
    fetch.add_argument("--resolve-revision", action="store_true")
    fetch.add_argument("--license-reference")
    classify = command(sub, "classify").add_subparsers(required=True)
    for name in ("run", "batch"):
        c = command(classify, name, f"classify_cmd:{name}", f"classify.{name}")
        c.add_argument("--controller", default="quick")
        c.add_argument("--input", required=True)
        c.add_argument("--output")
        c.add_argument("--calibration")
        c.add_argument("--continue-on-error", action="store_true")
    calibrate = command(
        classify, "calibrate", "classify_cmd:calibrate", "classify.calibrate"
    )
    calibrate.add_argument(
        "--input",
        required=True,
        help="calibration prediction bundle with distinct grouped splits",
    )
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--risk-target", type=float, default=0.05)
    calibrate.add_argument("--min-support", type=int, default=20)
    benchmark = command(sub, "benchmark").add_subparsers(required=True)
    command(benchmark, "list", "benchmark_cmd:list_suites", "benchmark.list")
    prepare = command(
        benchmark, "prepare", "benchmark_cmd:prepare", "benchmark.prepare"
    )
    prepare.add_argument("suite")
    prepare.add_argument(
        "--source",
        required=True,
        help="local JEV-CPU checkout; no implicit dataset downloads",
    )
    prepare.add_argument("--revision", required=True)
    plan = command(benchmark, "plan", "benchmark_cmd:plan", "benchmark.plan")
    plan.add_argument("suite")
    plan.add_argument(
        "--treatment",
        choices=("rules", "cpu_direct", "cpu_shared", "cpu_compact_generation"),
        default="cpu_direct",
    )
    plan.add_argument("--controller", default="quick")
    plan.add_argument("--selection", choices=("smoke", "full"), default="smoke")
    plan.add_argument(
        "--split",
        choices=("calibration", "validation", "test", "all", "systems"),
        default="test",
    )
    plan.add_argument("--repetitions", type=int)
    plan.add_argument("--seed", type=int)
    plan.add_argument("--out", required=True)
    plan.add_argument("--calibration")
    plan.add_argument("--quality-margin", type=float, default=0.02)
    run = command(benchmark, "run", "benchmark_cmd:run", "benchmark.run")
    run.add_argument("--plan", required=True)
    report = command(benchmark, "report", "benchmark_cmd:report", "benchmark.report")
    report.add_argument("run_id")
    report.add_argument("--output")
    compare = command(
        benchmark, "compare", "benchmark_cmd:compare", "benchmark.compare"
    )
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--quality-margin", type=float, default=0.02)
    compare.add_argument("--exploratory", action="store_true")
    compare.add_argument("--require-parity", action="store_true")
    bundle = command(
        benchmark,
        "calibration-input",
        "benchmark_cmd:calibration_input",
        "benchmark.calibration-input",
    )
    bundle.add_argument("calibration_run")
    bundle.add_argument("validation_run")
    bundle.add_argument("--output", required=True)
    runs = command(sub, "runs").add_subparsers(required=True)
    command(runs, "list", "runs_cmd:list_runs", "runs.list")
    for name in ("show", "events", "export"):
        r = command(runs, name, f"runs_cmd:{name}", f"runs.{name}")
        r.add_argument("run_id")
        if name == "export":
            r.add_argument("--output", required=True)
    return parser
