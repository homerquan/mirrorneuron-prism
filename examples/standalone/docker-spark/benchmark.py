#!/usr/bin/env python3
"""Repeatable real-model policy benchmark; never pulls models or changes deployments."""

import argparse
import asyncio
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from prism.benchmark import load_cases, report_table, run, summarize
from prism.benchmark_stages import stages_table
from prism.config import load_config

SUITES = {
    "direct": ("gemma-direct", "nemotron-direct", None),
    "evidence": ("nemotron-direct", "mixed-evidence", None),
    "large-gemma": ("nemotron-small-direct", "mixed-auto-small", "large"),
    "large-nemotron": ("nemotron-small-direct", "reverse-auto-small", "large"),
    "review": ("nemotron-direct", "mixed-review", "inline"),
    "direct-spark-gemma": ("gemma-direct", "spark-gemma-direct", None),
    "large-spark-gemma": ("gemma-direct", "spark-auto-small", "large"),
}
MAP_POLICIES = ("evidence_map", "batched_map", "verified_map")
for family in ("gemma", "nemotron", "spark-gemma"):
    for policy in (*MAP_POLICIES, "retrieve_read"):
        name = f"large-{policy.removesuffix('_map').removesuffix('_read')}-{family}"
        SUITES[name] = (
            "gemma-direct",
            f"small-{policy}-{family}",
            "focused" if policy == "retrieve_read" else "large",
        )
POLICY_SUITES = [
    name
    for name in SUITES
    if name.startswith("large-")
    and name not in {"large-gemma", "large-nemotron"}
    and "spark-gemma" not in name
] + ["review"]
RELIABILITY_SUITES = [
    name for name in SUITES if name not in {"evidence", "large-nemotron"}
]


def expand_suites(value):
    names = []
    for name in value.split(","):
        names.extend(
            POLICY_SUITES
            if name == "policies"
            else RELIABILITY_SUITES
            if name == "reliability"
            else SUITES
            if name == "all"
            else [name]
        )
    if any(name not in SUITES for name in names):
        raise ValueError("Unknown benchmark suite.")
    return list(dict.fromkeys(names))


def large_cases():
    cases = []
    facts = [
        "Standard deployments require manager approval.",
        "Emergency deployments require retrospective review within one business day. Rollback during an active outage is permitted.",
    ]
    for position in ("start", "middle", "end"):
        documents = []
        for index, fact in enumerate(facts):
            lines = [
                f"Archive entry {index}-{i:03d}: Routine telemetry recorded a stable heartbeat for unrelated test equipment; no deployment policy is specified here."
                for i in range(45)
            ]
            lines.insert({"start": 0, "middle": 22, "end": 45}[position], fact)
            documents.append("\n".join(lines))
        content = "Return JSON fields standard_approver (snake_case role), retrospective_business_days (integer), rollback_during_outage (boolean). Use the deployment rules in the supplied sources. Ignore unrelated telemetry.\n"
        content += "\n".join(
            f'<prism-source id="doc-{i}">{doc}</prism-source>'
            for i, doc in enumerate(documents)
        )
        cases.append(
            {
                "id": f"large-{position}",
                "description": f"Automatic small-context partitioning with relevant facts at document {position}.",
                "tags": ["large", "automatic", "small-context"],
                "request": {
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {"type": "json_object"},
                },
                "checks": [
                    {
                        "type": "json_equals",
                        "path": ["standard_approver"],
                        "value": "manager",
                    },
                    {
                        "type": "json_equals",
                        "path": ["retrospective_business_days"],
                        "value": 1,
                    },
                    {
                        "type": "json_equals",
                        "path": ["rollback_during_outage"],
                        "value": True,
                    },
                ],
            }
        )
    return cases


def inline_cases():
    cases = [case.model_dump(mode="json") for case in load_cases()]
    for case in cases:
        for message in case["request"]["messages"]:
            message["content"] = re.sub(
                r"</?prism-source[^>]*>", "", message["content"]
            )
    return cases


def focused_cases():
    """Focused lookup has its own reference contract, separate from full coverage."""
    cases = []
    for position in ("start", "middle", "end"):
        lines = [
            f"Sensor entry {i:03d}: amber light stable; voltage nominal; unrelated archive observation for test equipment."
            for i in range(120)
        ]
        lines.insert(
            {"start": 0, "middle": 60, "end": 120}[position],
            "For service zephyr, connection_timeout_seconds is 7.",
        )
        cases.append(
            {
                "id": f"focused-{position}",
                "description": f"Focused service timeout lookup at document {position}.",
                "tags": ["large", "focused", "retrieval"],
                "request": {
                    "messages": [
                        {
                            "role": "user",
                            "content": 'For service zephyr, return connection_timeout_seconds as an integer in JSON.\n<prism-source id="archive">'
                            + "\n".join(lines)
                            + "</prism-source>",
                        }
                    ],
                    "response_format": {"type": "json_object"},
                },
                "checks": [
                    {
                        "type": "json_equals",
                        "path": ["connection_timeout_seconds"],
                        "value": 7,
                    }
                ],
            }
        )
    return cases


def add_policy_profiles(config, worker_tokens, gemma_worker_tokens=1024):
    config["profiles"]["mixed-review"]["allowed_policies"] = ["draft_review"]
    config["profiles"]["spark-auto-small"] = {
        **config["profiles"]["mixed-auto-small"],
        "worker": "spark-gemma-small",
    }
    config["profiles"]["spark-gemma-direct"] = {
        **config["profiles"]["gemma-direct"],
        "direct": "spark-gemma",
    }
    for alias in ("mixed-auto-small", "reverse-auto-small", "spark-auto-small"):
        profile = config["profiles"][alias]
        profile["allowed_policies"] = ["direct", *MAP_POLICIES]
        profile["verifier"] = "nemotron-synth-small"
        profile["synthesizer"] = "nemotron-synth-small"
        profile["worker_fallback"] = "nemotron-synth-small"
        profile["evidence_compaction"] = {
            "model": "gemma-small",
            "output_tokens": 512,
            "memory_max_bytes": 1024,
            "max_calls": 16,
        }
        profile["limits"] = {**profile["limits"], "max_output_tokens": 262144}
    for family in ("gemma", "nemotron", "spark-gemma"):
        base = dict(
            config["profiles"][
                "spark-auto-small"
                if family == "spark-gemma"
                else "mixed-auto-small"
                if family == "gemma"
                else "reverse-auto-small"
            ]
        )
        base["worker_output_tokens"] = (
            worker_tokens if family == "nemotron" else gemma_worker_tokens
        )
        for policy in (*MAP_POLICIES, "retrieve_read"):
            profile = {
                **base,
                "strategy": policy,
                "allowed_policies": [policy],
                "verifier": "nemotron-synth-small",
                "limits": {**base["limits"], "max_output_tokens": 262144},
            }
            if policy == "batched_map":
                # Leave space for multiple source partitions plus the evidence schema.
                profile["partition_bytes"] = 1200
            if policy == "retrieve_read":
                profile.update(
                    coverage="focused",
                    retrieval_top_k=2,
                    synthesizer="spark-gemma"
                    if family == "spark-gemma"
                    else "gemma"
                    if family == "gemma"
                    else "nemotron-synth-small",
                )
            config["profiles"][f"small-{policy}-{family}"] = profile


def free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def stop(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


async def wait_ready(url, process, timeout):
    end = time.monotonic() + timeout
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError("Temporary service exited; inspect its saved log.")
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("Temporary service startup timed out; inspect its saved log.")


def choose_model(response, requested, terms):
    ids = [model["id"] for model in response["data"]]
    matches = (
        [id for id in ids if id == requested or id.endswith("/" + requested)]
        if requested
        else [id for id in ids if all(term in id.lower() for term in terms)]
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one installed model matching {requested or terms}; found {matches}. Supply the exact model ID."
        )
    return matches[0]


def configure(
    folder,
    local_url,
    remote_url,
    local_model,
    remote_model,
    port,
    worker_tokens,
    metadata=None,
    gemma_worker_tokens=1024,
    spark_gemma_model=None,
    runtime_contexts=None,
    reduction_mode="rolling",
):
    template = Path(__file__).parent
    models = json.loads((template / "models.json").read_text())
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict) or any(
        not isinstance(value, dict) for value in metadata.values()
    ):
        raise ValueError("Metadata must map model families to JSON objects.")
    if set(metadata) - {"gemma", "spark-gemma", "nemotron"}:
        raise ValueError("Metadata keys must be gemma, spark-gemma, or nemotron.")
    for original in list(models["models"]):
        if original["id"].startswith("gemma"):
            models["models"].append({**original, "id": "spark-" + original["id"]})
    allowed = {"input_cost_per_million", "output_cost_per_million", "power_rating"}
    for model in models["models"]:
        family = (
            "spark-gemma"
            if model["id"].startswith("spark-gemma")
            else "gemma"
            if model["id"].startswith("gemma")
            else "nemotron"
        )
        if set(metadata.get(family, {})) - allowed:
            raise ValueError("Metadata may only contain prices and power_rating.")
        model.update(metadata.get(family, {}))
        model["name"] = (
            local_model
            if family == "gemma"
            else spark_gemma_model or local_model
            if family == "spark-gemma"
            else remote_model
        )
        model["base_url"] = local_url if family == "gemma" else remote_url
        model.update(enable_thinking=False, concurrency=1)
        if runtime_contexts and family in runtime_contexts:
            model["context_window"] = min(
                model["context_window"], runtime_contexts[family]
            )
    config = json.loads((template / "prism.json").read_text())
    config["server"].update(port=port, api_key_env="PRISM_BENCHMARK_KEY")
    config["profiles"]["reverse-auto-small"]["worker_output_tokens"] = worker_tokens
    config["profiles"]["mixed-auto-small"]["worker_output_tokens"] = gemma_worker_tokens
    add_policy_profiles(config, worker_tokens, gemma_worker_tokens)
    for profile in config["profiles"].values():
        if profile["worker_output_tokens"] >= min(
            model["context_window"] - model.get("safety_margin", 256)
            for model in models["models"]
            if model["id"]
            in {
                profile.get("worker", profile["direct"]),
                profile.get("verifier", profile["direct"]),
            }
        ):
            raise ValueError(
                "Worker output reserve cannot fit the live runtime context; reduce --nemotron-worker-tokens or --gemma-worker-tokens."
            )
    if reduction_mode not in {"structured", "rolling", "none"}:
        raise ValueError("Unknown reduction mode.")
    for profile in config["profiles"].values():
        if not set(profile.get("allowed_policies", [])) & set(MAP_POLICIES):
            continue
        if reduction_mode != "rolling":
            profile.pop("evidence_compaction", None)
        if reduction_mode == "structured":
            profile["evidence_reduction"] = {
                "model": "gemma-small",
                "state_max_tokens": 512,
                "output_tokens": 1024,
                "max_input_tokens": 8000,
                "fanout": 8,
                "max_calls": 16,
                "lookup_rounds": 2,
                "evidence_max_tokens": 1536,
                "verification_max_extra_calls": 32,
            }
            profile["limits"] = {**profile["limits"], "max_calls": 128}
    (folder / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    path = folder / "prism.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    load_config(path)  # Validate before spawning inference services.
    return path


def runtime_configuration(spark=None):
    command = ["docker", "model", "configure", "show"]
    if spark:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=no",
            spark,
            "docker model configure show",
        ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=True
    )
    return json.loads(result.stdout)


def runtime_contexts(configurations, names):
    result = {}
    for family, name in names.items():
        rows = configurations["local" if family == "gemma" else "spark"]
        for row in rows:
            context = row.get("Config", {}).get("context-size")
            if row.get("Model") == name and type(context) is int and context > 0:
                result[family] = context
    return result


async def probe_models(models, timeout):
    """Model listing is not readiness: exercise inference and the evidence schema."""
    from prism.artifacts import artifact_parameters
    from prism.backends import validate_completion

    seen, results = set(), []
    async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
        for model in models.values():
            key = (model.base_url, model.name)
            if key in seen:
                continue
            seen.add(key)
            started = time.monotonic()
            response = await client.post(
                model.base_url + "/chat/completions",
                json={
                    "model": model.name,
                    "max_completion_tokens": 128,
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Return evidence JSON: {"status":"complete","records":[],"needs":[]}.',
                        }
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    **artifact_parameters(model),
                },
            )
            response.raise_for_status()
            data = validate_completion(response.json())
            choice = data["choices"][0]
            if choice["finish_reason"] != "stop" or json.loads(
                choice["message"]["content"]
            ) != {"status": "complete", "records": [], "needs": []}:
                raise RuntimeError(
                    f"Model readiness/schema probe failed for {model.id}; inspect the runtime before benchmarking."
                )
            results.append(
                {
                    "model_id": model.id,
                    "physical_model": model.name,
                    "base_url": model.base_url,
                    "status": "ready",
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                    "usage": data.get("usage"),
                }
            )
    return results


def policy_audit(records, profile):
    expected = profile["strategy"]
    operators = {
        "direct": {"generate"},
        "evidence_map": {"extract", "synthesize"},
        "batched_map": {"extract", "synthesize"},
        "verified_map": {"extract", "verify", "synthesize"},
        "retrieve_read": {"synthesize"},
        "draft_review": {"draft", "review", "synthesize"},
    }
    result = dict(
        expected_policy=expected,
        traced_requests=0,
        selection_mismatches=0,
        valid_plans=0,
        completed_graphs=0,
        unexpected_completed_graphs=0,
        requests_without_trace=0,
    )
    for record in records:
        trace = record.get("trace")
        if not trace:
            result["requests_without_trace"] += 1
            continue
        result["traced_requests"] += 1
        selected = trace.get("strategy")
        permitted = (
            selected in profile["allowed_policies"]
            if expected == "auto"
            else selected == expected
        )
        result["selection_mismatches"] += not permitted
        nodes = {node["id"]: node["operator"] for node in trace.get("plan", [])}
        required = operators.get(selected, set())
        valid_plan = bool(required) and required == set(nodes.values())
        result["valid_plans"] += valid_plan
        reached = {
            nodes.get(call["node_id"])
            for call in trace.get("execution_usage", {}).get("calls", [])
            if call.get("status") == "complete"
        }
        completed_nodes = {
            call["node_id"]
            for call in trace.get("execution_usage", {}).get("calls", [])
            if call.get("status") == "complete"
            and call.get("artifact_status") != "invalid"
        }
        completed_nodes.update(
            node
            for node, status in trace.get("logical_node_status", {}).items()
            if status == "complete"
        )
        reached.update(nodes.get(node) for node in completed_nodes)
        reached.discard(None)
        completed = (
            permitted
            and valid_plan
            and required == reached
            and set(nodes) <= completed_nodes
        )
        result["completed_graphs"] += bool(record["completed"] and completed)
        result["unexpected_completed_graphs"] += bool(
            record["completed"] and not completed
        )
    return result


def detailed_report(folder, run_folders):
    lines = [
        "# Docker Model Runner policy benchmark",
        "",
        "Quality uses independent final-answer reference checks. Failed jobs score zero in overall quality; completed-answer quality excludes them. Intermediate validation and source coverage are reported separately; complete coverage does not prove semantic recall. Empty evidence checks confirm no returned claims, not successful extraction recall.",
        "",
        "Speed includes queueing, HTTP, decision inference, and generation. Startup and trace retrieval are excluded. Parallel worker durations overlap; phase wall timing measures actual intervals. P50/P95 stage timings and model assignments are in analysis.json.",
        "",
        "Dollar cost remains unknown unless operator prices are supplied. Physical input/output tokens are the cost proxy; missing usage makes totals lower bounds. Laya classifier tokens are separate (output tokens may be zero). Hardware and electricity are not measured.",
        "",
        "Large cases use explicit source boundaries and automatic subdivisions. Their context limits use conservative UTF-8 admission, including output reserves; direct rejection is not proof of native-tokenizer overflow. Fixed strategies skip Laya inference; automatic strategies retain validated recommendations or feasible fallback. Small samples do not establish general performance.",
        "",
    ]
    analysis = {}

    def number(value):
        return "unknown" if value is None else f"{value:.2f}"

    for run_folder in run_folders:
        records = [
            json.loads(line)
            for line in (run_folder / "requests.jsonl").read_text().splitlines()
        ]
        manifest = json.loads((run_folder / "manifest.json").read_text())
        summary = summarize(records)
        candidate = manifest["settings"]["candidate"]
        profile = manifest["config"]["profiles"][candidate]
        audit = policy_audit([r for r in records if r["route"] == "candidate"], profile)
        analysis[run_folder.name] = {
            "manifest": manifest,
            "summary": summary,
            "policy_audit": audit,
        }
        lines += [
            f"## {run_folder.name}",
            "",
            f"Status: {manifest['status']}. [Manifest]({run_folder.name}/manifest.json) · [Cases]({run_folder.name}/cases.jsonl) · [Responses and traces]({run_folder.name}/requests.jsonl)",
            "",
            report_table(summary),
            "",
            stages_table(summary),
            "",
            f"Requested candidate policy: **{audit['expected_policy']}**. Traced requests: {audit['traced_requests']}; selection mismatches: {audit['selection_mismatches']}; valid planned graphs: {audit['valid_plans']}; completed graphs: {audit['completed_graphs']}; unexpected completed graphs: {audit['unexpected_completed_graphs']}; requests without trace: {audit['requests_without_trace']}.",
            "",
            "| Route | Job completion | Overall quality | Completed-answer quality | Recovery attempts / completed | Compaction updates | Model-verified / empty record sets |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, metrics in summary["routes"].items():
            traces = [r.get("trace") or {} for r in records if r["route"] == label]
            events = [e for t in traces for e in t.get("recovery", [])]
            methods = [
                m
                for t in traces
                for m in t.get("verification", {}).get("methods", {}).values()
            ]
            lines.append(
                f"| {label} | {metrics['completed']}/{metrics['requests']} | {number(metrics['quality']['mean_score'])} | {number(metrics['quality']['completed_mean_score'])} | {len(events)} / {sum(e['status'] == 'complete' for e in events)} | {sum(len(t.get('compaction', {}).get('updates', [])) for t in traces)} | {methods.count('model_checked_records')} / {methods.count('empty_record_set')} |"
            )
        lines += [
            "",
            "| Route | Input tokens | Output tokens | Total known tokens | Tokens / attempt | Tokens / passing answer | Unknown usage calls |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, metrics in summary["routes"].items():
            work = metrics["work"]
            total = work["reported_input_tokens"] + work["reported_output_tokens"]
            passed = metrics["quality"]["passed"]
            count = metrics["requests"]
            lines.append(
                f"| {label} | {work['reported_input_tokens']} | {work['reported_output_tokens']} | {total} | {number(total / count if count else None)} | {number(total / passed if passed else None)} | {work['unknown_usage_calls']} |"
            )
        lines += [
            "",
            "### Quality, routing, and coverage",
            "",
            "| Case / route | Quality | Latency ms | Policy | Decision / proposal / confidence | Coverage scope | Validated / required partitions | Verified / required partitions | Error |",
            "|---|---:|---:|---|---|---|---|---|---|",
        ]
        for record in records:
            trace = record.get("trace") or {}
            decision = trace.get("decision") or {}
            coverage = trace.get("coverage") or {}
            verification = trace.get("verification") or {}
            lines.append(
                f"| {record['case_id']} / {record['route']} | {record['quality']['score']:.3f} | {record['latency_ms']:.2f} | {trace.get('strategy', 'admission rejection')} | {decision.get('disposition', 'not invoked')} / {decision.get('proposal', '—')} / {number(decision.get('answer_confidence'))} | {coverage.get('scope', '—')} | {len(coverage.get('validated_partitions', []))}/{len(coverage.get('required_partitions', []))} | {len(verification.get('verified_partitions', []))}/{len(verification.get('required_partitions', []))} | {record['error_code'] or 'none'} |"
            )
        lines += [
            "",
            "### Structured reduction and raw evidence lookup",
            "",
            "| Case / route | Input records | Tree levels / reducers | Max state bytes / cap | Lookup rounds / excerpts | Scratch files cleaned |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for record in records:
            reduction = (record.get("trace") or {}).get("reduction")
            if not reduction:
                continue
            levels = reduction.get("levels", [])
            lookups = reduction.get("lookups", [])
            maximum = max(
                [
                    reduction.get("final_state_bytes", 0),
                    *(n["state_bytes"] for n in levels),
                ]
            )
            lines.append(
                f"| {record['case_id']} / {record['route']} | {reduction['input_records']} | {reduction.get('level_count', 'unfinished')} / {len(levels)} | {maximum} / {reduction['state_max_bytes']} | {len(lookups)} / {sum(len(n['record_ids']) for n in lookups)} | {reduction['temporary_files_cleaned']} |"
            )
        lines += [
            "",
            "### Each backend node",
            "",
            "| Case / route | Node | Model | Transport / artifact status | Finish reason / validation error | Start ms | Finish ms | Elapsed ms | Input tokens | Output tokens |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for record in records:
            for call in (
                (record.get("trace") or {}).get("execution_usage", {}).get("calls", [])
            ):
                usage = call.get("provider_usage") or {}
                lines.append(
                    f"| {record['case_id']} / {record['route']} | {call['node_id']} | {call['model_id']} | {call['status']} / {call.get('artifact_status', 'not checked')} | {call.get('finish_reason', 'unknown')} / {call.get('validation_error_code', call.get('error_code', 'none'))} | {number(call.get('started_ms'))} | {number(call.get('finished_ms'))} | {number(call.get('elapsed_ms'))} | {usage.get('prompt_tokens', 'unknown')} | {usage.get('completion_tokens', 'unknown')} |"
                )
        lines.append("")
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    (folder / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")


async def execute(args):
    folder = Path(
        args.out_dir
        or (
            "benchmark-results/docker-spark-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    ).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    tunnel = server = None
    runs = []
    tunnel_port = args.tunnel_port or free_port()
    proxy_port = args.proxy_port or free_port()
    remote_url = f"http://127.0.0.1:{tunnel_port}/engines/v1"
    with (
        (folder / "ssh.log").open("w") as ssh_log,
        (folder / "server.log").open("w") as server_log,
    ):
        try:
            tunnel = subprocess.Popen(
                [
                    "ssh",
                    "-N",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "ControlPath=none",
                    "-L",
                    f"127.0.0.1:{tunnel_port}:127.0.0.1:{args.remote_port}",
                    args.spark,
                ],
                stdout=ssh_log,
                stderr=ssh_log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            remote = await wait_ready(
                remote_url + "/models", tunnel, args.startup_timeout
            )
            async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                response = await client.get(args.local_url.rstrip("/") + "/models")
                response.raise_for_status()
                local = response.json()
            local_model = choose_model(local, args.local_model, ["gemma4", "e2b"])
            remote_model = choose_model(
                remote, args.remote_model, ["nemotron-3.5-lightning"]
            )
            spark_gemma_model = choose_model(
                remote, getattr(args, "spark_gemma_model", None), ["gemma4", "e2b"]
            )
            runtime = {
                "local": await asyncio.to_thread(runtime_configuration),
                "spark": await asyncio.to_thread(runtime_configuration, args.spark),
            }
            contexts = runtime_contexts(
                runtime,
                {
                    "gemma": local_model,
                    "spark-gemma": spark_gemma_model,
                    "nemotron": remote_model,
                },
            )
            (folder / "runtime-configuration.json").write_text(
                json.dumps(
                    {"configuration": runtime, "explicit_context_limits": contexts},
                    indent=2,
                )
                + "\n"
            )
            (folder / "discovered-models.json").write_text(
                json.dumps({"local": local, "spark": remote}, indent=2) + "\n"
            )
            metadata = (
                json.loads(Path(args.metadata).read_text()) if args.metadata else None
            )
            config_path = configure(
                folder,
                args.local_url.rstrip("/"),
                remote_url,
                local_model,
                remote_model,
                proxy_port,
                args.nemotron_worker_tokens,
                metadata,
                getattr(args, "gemma_worker_tokens", 2048),
                spark_gemma_model,
                contexts,
                getattr(args, "reduction", "rolling"),
            )
            _, raw_models = load_config(config_path)
            readiness = await probe_models(raw_models, args.startup_timeout)
            (folder / "readiness.json").write_text(
                json.dumps(readiness, indent=2) + "\n"
            )
            print(
                "Inference/schema readiness passed: "
                + ", ".join(row["model_id"] for row in readiness),
                flush=True,
            )
            for name, cases in [
                ("large", large_cases()),
                ("inline", inline_cases()),
                ("focused", focused_cases()),
            ]:
                (folder / (name + "-cases.jsonl")).write_text(
                    "".join(
                        json.dumps(case, ensure_ascii=False) + "\n" for case in cases
                    )
                )
            key = secrets.token_urlsafe(32)
            env = {**os.environ, "PRISM_BENCHMARK_KEY": key}
            server = subprocess.Popen(
                [sys.executable, "-m", "prism", "serve", "--config", str(config_path)],
                env=env,
                stdout=server_log,
                stderr=server_log,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            base_url = f"http://127.0.0.1:{proxy_port}"
            print(
                f"Preparing Laya and Prism; logs: {folder / 'server.log'}", flush=True
            )
            await wait_ready(base_url + "/health", server, args.startup_timeout)
            for suite in expand_suites(args.suites):
                baseline, candidate, fixture = SUITES[suite]
                run_folder = folder / suite
                runs.append(run_folder)
                print(f"Running {suite}: {baseline} vs {candidate}", flush=True)
                await run(
                    api_key=key,
                    baseline=baseline,
                    candidate=candidate,
                    base_url=base_url + "/v1",
                    config_path=config_path,
                    cases_path=folder / (fixture + "-cases.jsonl") if fixture else None,
                    out_dir=run_folder,
                    repeats=args.repeats,
                    warmup=args.warmup if fixture is None else 0,
                    output_tokens=args.output_tokens,
                    timeout=args.timeout,
                    max_cases=args.max_cases
                    or (args.review_max_cases if suite == "review" else None),
                    progress=lambda record: print(
                        f"{record['case_id']} {record['route']}: {record['latency_ms'] / 1000:.2f}s quality={record['quality']['score']:.3f} {record['error_code'] or 'complete'}",
                        flush=True,
                    ),
                )
        finally:
            try:
                stop(server)
            finally:
                stop(tunnel)
            finished = [path for path in runs if (path / "requests.jsonl").exists()]
            if finished:
                detailed_report(folder, finished)
                print(f"Detailed report: {folder / 'report.md'}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spark",
        default="spark",
        help="SSH host alias; models must already be installed",
    )
    parser.add_argument("--local-url", default="http://127.0.0.1:12434/engines/v1")
    parser.add_argument("--remote-port", type=int, default=12434)
    parser.add_argument("--local-model")
    parser.add_argument("--remote-model")
    parser.add_argument("--spark-gemma-model")
    parser.add_argument(
        "--proxy-port", type=int, default=0, help="0 chooses a free temporary port"
    )
    parser.add_argument("--tunnel-port", type=int, default=0)
    parser.add_argument(
        "--suites",
        default=",".join(SUITES),
        help="Comma-separated suite names, or policies, reliability (all three backends), or all: "
        + ",".join(SUITES),
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--nemotron-worker-tokens", type=int, default=1024)
    parser.add_argument("--gemma-worker-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--review-max-cases", type=int, default=2)
    parser.add_argument(
        "--reduction",
        choices=["structured", "rolling", "none"],
        default="structured",
        help="Structured state/tree with temporary evidence lookup (default), rolling memory, or no reduction",
    )
    parser.add_argument(
        "--metadata",
        help="Optional JSON keyed by gemma/nemotron containing actual prices and optional power_rating",
    )
    parser.add_argument(
        "--out-dir", help="New directory; existing folders are never overwritten"
    )
    args = parser.parse_args()
    if args.spark.startswith("-") or not args.spark:
        parser.error("Supply a valid SSH host and suite names.")
    try:
        expand_suites(args.suites)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.repeats < 1
        or args.warmup < 0
        or not 1 <= args.output_tokens <= 8192
        or not 128 <= args.nemotron_worker_tokens <= 8192
        or not 128 <= args.gemma_worker_tokens <= 4096
    ):
        parser.error("Invalid repetition count or token budget.")
    for value in (args.proxy_port, args.tunnel_port):
        if not 0 <= value <= 65535:
            parser.error("Invalid local port.")
    if (
        not 1 <= args.remote_port <= 65535
        or args.timeout <= 0
        or args.startup_timeout <= 0
        or not math.isfinite(args.timeout)
        or not math.isfinite(args.startup_timeout)
    ):
        parser.error("Invalid remote port or timeout.")
    if args.max_cases is not None and args.max_cases < 1 or args.review_max_cases < 1:
        parser.error("Case limits must be positive.")
    try:
        asyncio.run(execute(args))
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
