"""Repeatable paired benchmarks with inspectable, versioned run artifacts."""

import hashlib
import json
import math
import platform
import re
import time
import uuid
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator

from . import __version__
from .backends import validate_completion
from .config import StrictModel, load_config
from .contracts import parse_json, validate_request
from .errors import PrismError

SCHEMA_VERSION = 1
QUALITY_REVISION = "reference-checks-v1"


class Check(StrictModel):
    type: Literal["json_equals", "contains_any", "not_contains"]
    path: list[str | int] = Field(default_factory=list)
    value: Any = None
    terms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_check(self):
        if self.type == "json_equals":
            if self.terms or "value" not in self.model_fields_set:
                raise ValueError("JSON checks use path/value")
        elif not self.terms or any(not term.strip() for term in self.terms):
            raise ValueError("text checks need nonempty terms")
        elif self.path or self.value is not None:
            raise ValueError("text checks use terms")
        return self


class Case(StrictModel):
    id: str = Field(min_length=1)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    request: dict
    checks: list[Check] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_request(self):
        # These fields are controlled identically for both routes by the runner.
        if set(self.request) & {
            "model",
            "stream",
            "stream_options",
            "max_tokens",
            "max_completion_tokens",
        }:
            raise ValueError(
                "case must not override route, streaming, or output budget"
            )
        validate_request({"model": "benchmark-validation", **self.request})
        return self


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def load_cases(path=None, max_cases=None):
    text = (
        Path(path).read_text()
        if path is not None
        else files("prism").joinpath("resources", "benchmark-cases.jsonl").read_text()
    )
    cases = [
        Case.model_validate(parse_json(line))
        for line in text.splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("benchmark needs nonempty cases with unique IDs")
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    return cases


def grade(case, content, completed):
    normalized = re.sub(r"\s+", " ", content.casefold())
    try:
        document = parse_json(content)
        parsed = True
    except ValueError:
        document = None
        parsed = False
    checks = []
    for check in case.checks:
        if check.type == "json_equals":
            observed = document
            try:
                if not parsed:
                    raise ValueError("invalid JSON answer")
                for part in check.path:
                    if isinstance(part, int) and (
                        not isinstance(observed, list) or part < 0
                    ):
                        raise KeyError(part)
                    observed = observed[part]
                # Distinguish boolean true from numeric 1 in reference checks.
                passed = json_text(observed) == json_text(check.value)
            except (KeyError, IndexError, TypeError, ValueError):
                passed = False
        else:
            matches = [
                re.sub(r"\s+", " ", term.casefold()) in normalized
                for term in check.terms
            ]
            passed = any(matches) if check.type == "contains_any" else not any(matches)
        checks.append({**check.model_dump(), "passed": passed})
    score = sum(check["passed"] for check in checks) / len(checks) if completed else 0.0
    return {"score": score, "passed": completed and score == 1, "checks": checks}


def work_and_cost(trace, models):
    execution = trace.get("execution_usage") if isinstance(trace, dict) else None
    calls = execution.get("calls") if isinstance(execution, dict) else None
    result = {
        "backend_calls": None,
        "reported_input_tokens": 0,
        "reported_output_tokens": 0,
        "unknown_usage_calls": None,
        "estimated_cost_usd": None,
        "known_cost_subtotal_usd": 0.0,
        "cost_upper_estimate_usd": None,
        "cost_basis": "operator prices × provider-reported tokens; excludes local hardware costs",
    }
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        return result
    result["backend_calls"] = len(calls)
    result["unknown_usage_calls"] = 0
    known = True
    upper_known = True
    upper = 0.0
    for call in calls:
        usage = call.get("provider_usage")
        reported = isinstance(usage, dict) and all(
            type(usage.get(key)) is int and usage[key] >= 0
            for key in ("prompt_tokens", "completion_tokens")
        )
        if reported:
            result["reported_input_tokens"] += usage["prompt_tokens"]
            result["reported_output_tokens"] += usage["completion_tokens"]
        else:
            result["unknown_usage_calls"] += 1
        model = models.get(call.get("model_id"))
        priced = (
            model is not None
            and model.input_cost_per_million is not None
            and model.output_cost_per_million is not None
        )
        if reported and priced:
            result["known_cost_subtotal_usd"] += (
                usage["prompt_tokens"] * model.input_cost_per_million
                + usage["completion_tokens"] * model.output_cost_per_million
            ) / 1_000_000
        else:
            known = False
        bound = call.get("cost_upper_estimate_usd")
        if type(bound) in {int, float} and math.isfinite(bound) and bound >= 0:
            upper += bound
        else:
            upper_known = False
    if known:
        result["estimated_cost_usd"] = result["known_cost_subtotal_usd"]
    if upper_known:
        result["cost_upper_estimate_usd"] = upper
    return result


async def sample(client, case, *, alias, base_url, output_tokens, temperature, models):
    body = {
        "temperature": temperature,
        **case.request,
        "model": alias,
        "max_completion_tokens": output_tokens,
    }
    record = {
        "case_id": case.id,
        "model": alias,
        "completed": False,
        "http_success": False,
        "status": None,
        "error_code": None,
        "finish_reason": None,
        "response": None,
        "trace": None,
        "trace_status": "unavailable",
        "logical_usage": None,
    }
    started = time.perf_counter()
    try:
        response = await client.post(base_url + "/chat/completions", json=body)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        record["status"] = response.status_code
        record["http_success"] = response.status_code == 200
        data = parse_json(response.content)
        if not isinstance(data, dict):
            raise ValueError("response must be an object")
        record["response"] = data
        record["logical_usage"] = data.get("usage")
        if response.status_code == 200:
            validate_completion(data)
            if data.get("model") != alias:
                raise ValueError("response alias mismatch")
            record["finish_reason"] = data["choices"][0]["finish_reason"]
            record["completed"] = record["finish_reason"] in {"stop", "tool_calls"}
            if (
                record["finish_reason"] == "stop"
                and not (data["choices"][0]["message"].get("content") or "").strip()
            ):
                record["completed"] = False
                record["error_code"] = "empty_answer"
            if not record["completed"]:
                record["error_code"] = record["error_code"] or (
                    "truncated"
                    if record["finish_reason"] == "length"
                    else "content_filter"
                )
        else:
            error = data.get("error")
            record["error_code"] = (
                error.get("code", "http_error")
                if isinstance(error, dict)
                else "http_error"
            )
    except (httpx.HTTPError, ValueError, TypeError, KeyError, PrismError):
        record.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        record["error_code"] = "transport_or_protocol_failure"
    # Trace retrieval is outside request latency, and cannot turn a good answer
    # into a transport failure. Missing traces make work/cost unknown.
    if record["status"] is not None and response.headers.get("x-request-id"):
        request_id = response.headers["x-request-id"]
        try:
            trace_response = await client.get(
                base_url + "/prism/traces/" + request_id, timeout=10
            )
            trace = parse_json(trace_response.content)
            if (
                trace_response.status_code == 200
                and isinstance(trace, dict)
                and trace.get("request_id") == request_id
            ):
                record["trace"] = trace
                execution = trace.get("execution_usage")
                calls = execution.get("calls") if isinstance(execution, dict) else None
                record["trace_status"] = (
                    "available"
                    if isinstance(calls, list)
                    and all(isinstance(call, dict) for call in calls)
                    else "invalid"
                )
        except (httpx.HTTPError, ValueError):
            pass
    content = ""
    if record["completed"]:
        content = record["response"]["choices"][0]["message"].get("content") or ""
    record["quality"] = grade(case, content, record["completed"])
    record["work"] = work_and_cost(record["trace"], models)
    return record


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(records):
    result = {
        "schema_version": SCHEMA_VERSION,
        "quality_revision": QUALITY_REVISION,
        "quality_metric": "equal-weight reference checks per case; failures/truncation score zero",
        "routes": {},
    }
    for label in ("baseline", "candidate"):
        route = [record for record in records if record["route"] == label]
        latencies = [record["latency_ms"] for record in route]
        completed = [record for record in route if record["completed"]]
        passed = sum(record["quality"]["passed"] for record in route)
        known_costs = [
            record["work"]["estimated_cost_usd"]
            for record in route
            if record["work"]["estimated_cost_usd"] is not None
        ]
        known_upper = [
            record["work"]["cost_upper_estimate_usd"]
            for record in route
            if record["work"]["cost_upper_estimate_usd"] is not None
        ]
        total = sum(known_costs) if len(known_costs) == len(route) and route else None
        result["routes"][label] = {
            "model": route[0]["model"] if route else None,
            "requests": len(route),
            "completed": len(completed),
            "failures": len(route) - len(completed),
            "truncated": sum(record["finish_reason"] == "length" for record in route),
            "quality": {
                "mean_score": mean([record["quality"]["score"] for record in route]),
                "passed": passed,
                "pass_rate": passed / len(route) if route else None,
            },
            "speed": {
                "mean_latency_ms": mean(latencies),
                "p50_latency_ms": percentile(latencies, 0.5),
                "p95_latency_ms": percentile(latencies, 0.95),
                "successful_mean_latency_ms": mean(
                    [record["latency_ms"] for record in completed]
                ),
                "sequential_requests_per_second": len(route) * 1000 / sum(latencies)
                if sum(latencies)
                else None,
            },
            "work": {
                "backend_calls_known": sum(
                    record["work"]["backend_calls"] or 0 for record in route
                ),
                "requests_without_trace": sum(
                    record["trace_status"] != "available" for record in route
                ),
                "reported_input_tokens": sum(
                    record["work"]["reported_input_tokens"] for record in route
                ),
                "reported_output_tokens": sum(
                    record["work"]["reported_output_tokens"] for record in route
                ),
                "unknown_usage_calls": sum(
                    record["work"]["unknown_usage_calls"] or 0 for record in route
                ),
            },
            "cost": {
                "estimated_total_usd": total,
                "known_subtotal_usd": sum(known_costs),
                "requests_with_unknown_cost": len(route) - len(known_costs),
                "mean_request_usd": total / len(route) if total is not None else None,
                "per_completed_request_usd": total / len(completed)
                if total is not None and completed
                else None,
                "per_passing_request_usd": total / passed
                if total is not None and passed
                else None,
                "upper_estimate_total_usd": sum(known_upper)
                if len(known_upper) == len(route) and route
                else None,
            },
        }
    baseline, candidate = (
        result["routes"][label] for label in ("baseline", "candidate")
    )
    a, b = (
        baseline["speed"]["successful_mean_latency_ms"],
        candidate["speed"]["successful_mean_latency_ms"],
    )
    qa, qb = baseline["quality"]["mean_score"], candidate["quality"]["mean_score"]
    ca, cb = baseline["cost"]["mean_request_usd"], candidate["cost"]["mean_request_usd"]
    result["candidate_vs_baseline"] = {
        "successful_latency_speedup": a / b if a is not None and b else None,
        "quality_score_delta": qb - qa if qa is not None and qb is not None else None,
        "mean_cost_ratio": cb / ca if cb is not None and ca else None,
    }
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def create_folder(folder):
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PrismError(
            "benchmark output folder already exists", "benchmark_output_exists"
        ) from exc


def report_table(summary):
    lines = [
        "| Route | Model | Complete | Quality score | Pass rate | Mean ms | P95 ms | Estimated USD | Backend calls* |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value, digits=2):
        return "unknown" if value is None else f"{value:.{digits}f}"

    for label, route in summary["routes"].items():
        model = (route["model"] or "unknown").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {label} | {model} | {route['completed']}/{route['requests']} | "
            f"{number(route['quality']['mean_score'], 3)} | {number(route['quality']['pass_rate'], 3)} | "
            f"{number(route['speed']['mean_latency_ms'])} | {number(route['speed']['p95_latency_ms'])} | "
            f"{number(route['cost']['estimated_total_usd'], 6)} | {route['work']['backend_calls_known']} |"
        )
    return "\n".join(lines)


def write_report(folder, manifest, summary):
    text = (
        f"# Prism benchmark {manifest['run_id']}\n\nStatus: {manifest['status']}. "
        f"Measured requests exclude {manifest['settings']['warmup']} warmup pair(s).\n\n"
        + report_table(summary)
        + "\n\nQuality is a deterministic reference rubric, not a general semantic judge. "
        "Failures and truncated answers score zero. Latency includes HTTP generation time; "
        "trace retrieval is excluded. Throughput is sequential, not a concurrent load test.\n\n"
        "Estimated cost uses configured per-model prices and reported physical tokens across "
        "all calls, including failed work. Unknown prices/usage remain unknown. Local hardware "
        "and electricity are not measured. Reservation upper estimates are separate from costs. "
        "Warmup responses, work, and costs are retained separately in warmup.jsonl.\n\n"
        "*Backend calls and physical tokens are known subtotals; see requests_without_trace and "
        "unknown_usage_calls in summary.json. Inspect requests.jsonl for outputs, errors, checks, and traces.\n"
    )
    (folder / "report.md").write_text(text)


async def run(
    *,
    api_key,
    baseline="prism-direct",
    candidate="prism-evidence",
    base_url="http://127.0.0.1:8080/v1",
    cases_path=None,
    config_path=None,
    out_dir=None,
    repeats=3,
    warmup=1,
    output_tokens=2048,
    temperature=0.0,
    timeout=180.0,
    max_cases=None,
    progress=None,
    transport=None,
):
    if (
        repeats < 1
        or warmup < 0
        or output_tokens < 1
        or not 0 <= temperature <= 2
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("invalid benchmark settings")
    url = urlsplit(base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("base URL must not contain credentials, query, or fragment")
    cases = load_cases(cases_path, max_cases)
    config, models = load_config(config_path) if config_path else (None, {})
    snapshot = (
        {
            "schema_version": config.schema_version,
            "server": config.server.model_dump(
                mode="json", exclude={"api_key_env", "public_url"}
            ),
            "profiles": {
                key: profile.model_dump(mode="json")
                for key, profile in config.profiles.items()
            },
            "decision": config.decision.model_dump(mode="json"),
            "models": {
                key: model.model_dump(mode="json", exclude={"api_key", "api_key_env"})
                for key, model in models.items()
            },
        }
        if config
        else None
    )
    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    )
    folder = Path(out_dir) if out_dir else Path("benchmark-results") / run_id
    create_folder(folder)
    serialized_cases = [case.model_dump(mode="json") for case in cases]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "quality_revision": QUALITY_REVISION,
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "prism_version": __version__,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cases_sha256": digest(serialized_cases),
        "case_count": len(cases),
        "config_sha256": digest(snapshot) if snapshot else None,
        "config": snapshot,
        "settings": {
            "baseline": baseline,
            "candidate": candidate,
            "base_url": base_url.rstrip("/"),
            "repeats": repeats,
            "warmup": warmup,
            "output_tokens": output_tokens,
            "temperature": temperature,
            "timeout_seconds": timeout,
            "execution_order": "alternate route order per pair; sequential HTTP requests",
        },
        "planned_pairs": len(cases) * repeats,
    }
    write_json(folder / "manifest.json", manifest)
    (folder / "cases.jsonl").write_text(
        "".join(json_text(case) + "\n" for case in serialized_cases)
    )
    records = []
    warmups = []
    status = "interrupted"
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
            headers={"authorization": f"Bearer {api_key}"},
            transport=transport,
        ) as client:
            with (
                (folder / "requests.jsonl").open("w") as output,
                (folder / "warmup.jsonl").open("w") as warming,
            ):
                schedule = [
                    (True, index, cases[index % len(cases)]) for index in range(warmup)
                ] + [
                    (False, repeat, case) for repeat in range(repeats) for case in cases
                ]
                for pair_index, (is_warmup, repeat, case) in enumerate(schedule):
                    routes = [("baseline", baseline), ("candidate", candidate)]
                    if pair_index % 2:
                        routes.reverse()
                    for label, alias in routes:
                        record = await sample(
                            client,
                            case,
                            alias=alias,
                            base_url=base_url.rstrip("/"),
                            output_tokens=output_tokens,
                            temperature=temperature,
                            models=models,
                        )
                        record.update(
                            route=label,
                            repeat=repeat,
                            warmup=is_warmup,
                            pair_index=pair_index,
                        )
                        stream = warming if is_warmup else output
                        stream.write(json_text(record) + "\n")
                        stream.flush()
                        (warmups if is_warmup else records).append(record)
                        if progress:
                            progress(record)
        status = "complete"
    finally:
        summary = summarize(records)
        summary["warmup"] = summarize(warmups)
        summary["run_id"] = run_id
        summary["expected_requests_per_route"] = len(cases) * repeats
        manifest.update(
            status=status,
            finished_at=datetime.now(UTC).isoformat(),
            measured_requests=len(records),
        )
        write_json(folder / "summary.json", summary)
        write_json(folder / "manifest.json", manifest)
        write_report(folder, manifest, summary)
    return {"run_dir": str(folder.resolve()), "summary": summary}


def compare(folders, out_dir=None):
    if len(folders) < 2:
        raise ValueError("comparison needs at least two run folders")
    runs = []
    for folder in folders:
        folder = Path(folder)
        manifest = parse_json((folder / "manifest.json").read_text())
        summary = parse_json((folder / "summary.json").read_text())
        if not isinstance(manifest, dict) or not isinstance(summary, dict):
            raise ValueError("invalid benchmark artifacts")
        if (
            manifest["schema_version"] != SCHEMA_VERSION
            or summary["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("unsupported benchmark artifact schema")
        runs.append(
            {"run_dir": str(folder.resolve()), "manifest": manifest, "summary": summary}
        )
    first = runs[0]["manifest"]
    warnings = []
    comparable = True
    for run_info in runs:
        manifest = run_info["manifest"]
        if manifest["status"] != "complete":
            warnings.append(f"{manifest['run_id']}: incomplete run")
            comparable = False
        if (
            manifest["cases_sha256"] != first["cases_sha256"]
            or manifest["quality_revision"] != first["quality_revision"]
        ):
            warnings.append(f"{manifest['run_id']}: different cases or quality rubric")
            comparable = False
        for setting in ("output_tokens", "temperature", "repeats", "warmup"):
            if manifest["settings"][setting] != first["settings"][setting]:
                warnings.append(f"{manifest['run_id']}: different {setting}")
                if setting in {"output_tokens", "temperature"}:
                    comparable = False
    changes = []
    for run_info in runs[1:]:
        deltas = {}
        for label in ("baseline", "candidate"):
            old = runs[0]["summary"]["routes"][label]
            new = run_info["summary"]["routes"][label]
            old_speed = old["speed"]["successful_mean_latency_ms"]
            new_speed = new["speed"]["successful_mean_latency_ms"]
            old_quality = old["quality"]["mean_score"]
            new_quality = new["quality"]["mean_score"]
            old_cost = old["cost"]["mean_request_usd"]
            new_cost = new["cost"]["mean_request_usd"]
            deltas[label] = (
                {
                    "successful_latency_speedup": old_speed / new_speed
                    if old_speed is not None and new_speed
                    else None,
                    "quality_score_delta": new_quality - old_quality
                    if old_quality is not None and new_quality is not None
                    else None,
                    "mean_cost_delta_usd": new_cost - old_cost
                    if old_cost is not None and new_cost is not None
                    else None,
                }
                if comparable
                else None
            )
        changes.append({"run_id": run_info["manifest"]["run_id"], "routes": deltas})
    result = {
        "schema_version": SCHEMA_VERSION,
        "comparable_workload": comparable,
        "warnings": warnings,
        "runs": runs,
        "changes_vs_first": changes,
    }
    if out_dir is not None:
        folder = Path(out_dir)
        create_folder(folder)
        write_json(folder / "comparison.json", result)
        text = (
            "# Prism benchmark comparison\n\nComparable workload: "
            + str(comparable)
            + ".\n\n"
        )
        text += (
            "\n".join("- " + warning for warning in warnings) + "\n\n"
            if warnings
            else ""
        )
        for run_info in runs:
            text += (
                "## "
                + run_info["manifest"]["run_id"]
                + "\n\n"
                + report_table(run_info["summary"])
                + "\n\n"
            )
        (folder / "report.md").write_text(text)
        result["comparison_dir"] = str(folder.resolve())
    return result
