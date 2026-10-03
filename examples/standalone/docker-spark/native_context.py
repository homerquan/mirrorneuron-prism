#!/usr/bin/env python3
"""Native-sized, seeded qualification of Prism on one fixed physical model.

Runs the real execution engine and backend HTTP adapter in process. Fixed policies
skip learned decisions. No deployment changes, weight pulls, or hidden truncation.
"""

import argparse
import asyncio
import hashlib
import json
import math
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from prism.backends import OpenAIBackend, provider_error, validate_completion
from prism.benchmark import Case, grade, work_and_cost
from prism.config import EvidenceReduction, Limits, PrismConfig, Profile, RawModel
from prism.decision import LayaDecision
from prism.engine import ExecutionEngine
from prism.long_context_cases import FAMILIES, build_case, evidence_recall
from prism.native_tokens import NativeCounter, size_case


def make_config(model, deadline, lookup_rounds):
    limits = Limits(
        max_calls=512,
        max_partitions=512,
        max_input_tokens=4_000_000,
        max_output_tokens=524288,
        deadline_seconds=deadline,
        max_parallel=1,
    )
    reduction = EvidenceReduction(
        model=model.id,
        state_max_tokens=512,
        output_tokens=1024,
        max_calls=64,
        lookup_rounds=lookup_rounds,
        evidence_max_tokens=1536,
    )
    profiles = {}
    for policy in ("direct", "evidence_map", "batched_map"):
        profiles[policy] = Profile(
            direct=model.id,
            worker=model.id,
            worker_fallback=model.id,
            synthesizer=model.id,
            strategy=policy,
            allowed_policies=[policy],
            partition_bytes=4200,
            worker_output_tokens=1024,
            limits=limits.model_copy(deep=True),
            evidence_reduction=reduction.model_copy(deep=True)
            if policy != "direct"
            else None,
        )
    return PrismConfig(profiles=profiles)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["route"], row["family"], row["sizing"]["target_prompt_tokens"])
        groups.setdefault(key, []).append(row)
    result = []
    for (route, family, target), items in groups.items():
        passing = sum(r["quality"]["passed"] for r in items)
        trials = len(items)
        latencies = sorted(r["latency_ms"] for r in items)
        known = sum(
            r["work"]["reported_input_tokens"] + r["work"]["reported_output_tokens"]
            for r in items
        )
        recalls = [
            r["retrieved_evidence_recall"]
            for r in items
            if r["retrieved_evidence_recall"] is not None
        ]
        final_recalls = [
            r["final_evidence_recall"]
            for r in items
            if r["final_evidence_recall"] is not None
        ]
        mapped_recalls = [
            r["mapped_evidence_recall"]
            for r in items
            if r.get("mapped_evidence_recall") is not None
        ]
        result.append(
            {
                "route": route,
                "family": family,
                "target_tokens": target,
                "requests": trials,
                "distinct_cases": len({r["case_id"] for r in items}),
                "completed": sum(r["completed"] for r in items),
                "passed": passing,
                "accuracy": passing / trials,
                "mean_quality": sum(r["quality"]["score"] for r in items) / trials,
                "p50_ms": latencies[(trials - 1) // 2],
                "p95_ms": latencies[math.ceil(trials * 0.95) - 1],
                "retrieved_evidence_recall": sum(recalls) / len(recalls)
                if recalls
                else None,
                "final_evidence_recall": sum(final_recalls) / len(final_recalls)
                if final_recalls
                else None,
                "mapped_evidence_recall": sum(mapped_recalls) / len(mapped_recalls)
                if mapped_recalls
                else None,
                "requests_with_reported_stage_fit": sum(
                    r["reported_stage_prompts_fit"] and not r["unknown_stage_usage"]
                    for r in items
                ),
                "known_tokens_per_passing_answer": known / passing if passing else None,
                "unknown_usage_calls": sum(
                    r["work"].get("unknown_usage_calls") or 0 for r in items
                ),
            }
        )
    return result


def save_report(folder, rows, manifest):
    summary = summarize(rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Native context qualification",
        "",
        f"Status: {manifest['status']}. Runtime context: {manifest['context_window']} native tokens. Same physical model for every stage and route.",
        "",
        "Input lengths use the installed native chat template/tokenizer, calibrated against provider prompt usage. Prism admission remains conservative UTF-8 accounting. Direct admission rejection is a negative control, never a successful generation baseline. Native direct controls bypass Prism admission to distinguish actual backend rejection from byte bounds.",
        "",
        "Latency includes backend queueing and inference, but excludes fixture sizing, tokenizer calibration, and native direct controls. Evidence recall checks complete gold byte spans in mapped quotes, lookup excerpts, and final state references/excerpts separately; it does not establish semantic entailment. Failed jobs score zero. Physical tokens are work proxies; missing usage remains unknown. Repeated attempts share a case and do not count as independent examples.",
        "",
        "| Route | Family | Target native tokens | Distinct cases | Complete | Fully correct | Mean quality | P50 ms | P95 ms | Mapped / retrieved / final span recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary:
        recall = item["retrieved_evidence_recall"]
        mapped, final = item["mapped_evidence_recall"], item["final_evidence_recall"]
        lines.append(
            f"| {item['route']} | {item['family']} | {item['target_tokens']} | {item['distinct_cases']} | {item['completed']}/{item['requests']} | {item['passed']}/{item['requests']} | {item['mean_quality']:.3f} | {item['p50_ms']:.0f} | {item['p95_ms']:.0f} | {mapped if mapped is not None else 'unknown'} / {recall if recall is not None else 'unknown'} / {final if final is not None else 'unknown'} |"
        )
    lines += [
        "",
        "Artifacts: [manifest](manifest.json), [cases](cases.jsonl), [gold spans and token sizes](sizing.jsonl), [responses and traces](requests.jsonl), [native direct controls](native-direct.jsonl), [summary](summary.json).",
        "",
        "This run establishes only the measured task families, lengths, and samples. Dependency repair, semantic final-answer verification, and exhaustive aggregation are not implemented by this change.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


async def native_direct(client, model, case, measured, output):
    response = await client.post(
        model.base_url + "/chat/completions",
        json={
            "model": model.name,
            **case["request"],
            "max_completion_tokens": output,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    result = {
        "case_id": case["id"],
        "native_prompt_tokens": measured,
        "status": response.status_code,
    }
    if response.status_code >= 400:
        result["error_code"] = provider_error(
            response.status_code, response.content[:8192]
        ).code
        result["completed"] = False
    else:
        data = validate_completion(response.json())
        usage = data.get("usage", {})
        result.update(
            completed=data["choices"][0]["finish_reason"] == "stop",
            provider_usage=usage,
            response=data,
        )
        result["prompt_count_matches"] = usage.get("prompt_tokens") == measured
        result["possible_truncation"] = (
            type(usage.get("prompt_tokens")) is int
            and usage["prompt_tokens"] < measured
        )
    if "checks" in case:
        content = (
            result.get("response", {})
            .get("choices", [{"message": {}}])[0]["message"]
            .get("content", "")
        )
        result["quality"] = grade(
            Case.model_validate(case),
            content,
            result["completed"] and not result.get("possible_truncation", False),
        )
    return result


async def execute(args):
    folder = Path(
        args.out_dir
        or "benchmark-results/native-context-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    model = RawModel(
        id="native",
        name=args.model,
        base_url=args.base_url.rstrip("/"),
        context_window=args.context_window,
        max_output_tokens=2048,
        concurrency=1,
        enable_thinking=False,
    )
    counter = NativeCounter(
        base_url=args.tokenizer_url,
        command=shlex.split(args.tokenizer_command) if args.tokenizer_command else None,
    )
    config = make_config(model, args.timeout, args.lookup_rounds)
    manifest = {
        "status": "preparing",
        "started_at": datetime.now(UTC).isoformat(),
        "context_window": model.context_window,
        "physical_model": model.name,
        "base_url": model.base_url,
        "config": config.model_dump(mode="json"),
        "model": model.model_dump(mode="json", exclude={"api_key"}),
        "native_counter": "llama.cpp apply-template + tokenize, add_special/parse_special",
        "tokenizer_transport": "argv" if args.tokenizer_command else "http",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("src/prism")
            .glob("*.py")
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {
            "sizes": args.sizes,
            "families": args.families,
            "seeds": args.seeds,
            "positions": args.positions,
            "repeats": args.repeats,
            "policies": args.policies,
            "output_tokens": args.output_tokens,
            "native_direct": args.native_direct,
        },
    }
    rows = []
    try:
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
            # Two lengths exercise the same JSON mode and disabled-thinking template.
            calibration = []
            for content in [
                "Return JSON ok=true.",
                "Return JSON ok=true. " + "José measured stable voltage. " * 80,
            ]:
                case = {
                    "id": "calibration",
                    "request": {
                        "messages": [{"role": "user", "content": content}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0,
                    },
                }
                count = await counter.count(case["request"]["messages"])
                control = await native_direct(client, model, case, count, 64)
                calibration.append(control)
                if control.get("provider_usage", {}).get("prompt_tokens") != count:
                    raise RuntimeError(
                        "Native tokenizer/template does not match provider prompt usage; qualification stopped"
                    )
            manifest["calibration"] = calibration
            properties = await counter.properties()
            runtime_context = properties.get("default_generation_settings", {}).get(
                "n_ctx"
            )
            if (
                type(runtime_context) is not int
                or runtime_context != model.context_window
            ):
                raise RuntimeError(
                    "Declared runtime context does not match native tokenizer server properties; qualification stopped"
                )
            manifest["native_runtime"] = {
                "n_ctx": runtime_context,
                "total_slots": properties.get("total_slots"),
                "model_alias_sha256": hashlib.sha256(
                    str(properties.get("model_alias", "")).encode()
                ).hexdigest(),
            }
            engine = ExecutionEngine(
                config,
                {model.id: model},
                OpenAIBackend({model.id: model}, client),
                LayaDecision(config.decision),
            )
            manifest["status"] = "running"
            with (
                (folder / "cases.jsonl").open("w") as cases_file,
                (folder / "sizing.jsonl").open("w") as sizes_file,
                (folder / "requests.jsonl").open("w") as requests_file,
                (folder / "native-direct.jsonl").open("w") as controls_file,
            ):
                for target in args.sizes:
                    for family in args.families:
                        for seed in args.seeds:
                            for position in args.positions:

                                def build(count):
                                    return build_case(family, seed, position, count)[0]

                                case, sizing = await size_case(
                                    build, target, counter.count
                                )
                                case["id"] += f"-n{target}"
                                gold = build_case(
                                    family, seed, position, sizing["filler_records"]
                                )[1]
                                sizing.update(
                                    case_id=case["id"],
                                    gold_spans=gold,
                                    prompt_exceeds_runtime=sizing[
                                        "native_prompt_tokens"
                                    ]
                                    > model.context_window,
                                )
                                cases_file.write(
                                    json.dumps(case, ensure_ascii=False) + "\n"
                                )
                                cases_file.flush()
                                sizes_file.write(json.dumps(sizing) + "\n")
                                sizes_file.flush()
                                if args.native_direct:
                                    control = await native_direct(
                                        client,
                                        model,
                                        case,
                                        sizing["native_prompt_tokens"],
                                        args.output_tokens,
                                    )
                                    controls_file.write(json.dumps(control) + "\n")
                                    controls_file.flush()
                                for repetition in range(args.repeats):
                                    routes = (
                                        args.policies
                                        if repetition % 2 == 0
                                        else list(reversed(args.policies))
                                    )
                                    for policy in routes:
                                        started = time.monotonic()
                                        execution, response, error = None, None, None
                                        try:
                                            execution = engine.prepare(
                                                {
                                                    "model": policy,
                                                    **case["request"],
                                                    "max_completion_tokens": args.output_tokens,
                                                }
                                            )
                                            response = await engine.execute(execution)
                                        except Exception as exc:
                                            error = getattr(
                                                exc, "code", type(exc).__name__
                                            )
                                        elapsed = (time.monotonic() - started) * 1000
                                        trace = (
                                            execution["trace"] if execution else None
                                        )
                                        completed = (
                                            response is not None
                                            and response["choices"][0]["finish_reason"]
                                            == "stop"
                                        )
                                        content = (
                                            response["choices"][0]["message"].get(
                                                "content", ""
                                            )
                                            if response
                                            else ""
                                        )
                                        if response is not None and not completed:
                                            error = "incomplete_answer"
                                        reduction = (trace or {}).get("reduction", {})
                                        spans = [
                                            s
                                            for lookup in reduction.get("lookups", [])
                                            for s in lookup.get("source_spans", [])
                                        ]
                                        spans += reduction.get(
                                            "initial_lookup", {}
                                        ).get("source_spans", [])
                                        work = work_and_cost(trace, {model.id: model})
                                        calls = (
                                            (trace or {})
                                            .get("execution_usage", {})
                                            .get("calls", [])
                                        )
                                        fits = [
                                            c.get("provider_usage", {}).get(
                                                "prompt_tokens"
                                            )
                                            + c["reserved_output_tokens"]
                                            <= model.context_window
                                            for c in calls
                                            if c.get("provider_usage")
                                        ]
                                        row = {
                                            "case_id": case["id"],
                                            "route": policy,
                                            "family": family,
                                            "seed": seed,
                                            "position": position,
                                            "repetition": repetition,
                                            "sizing": sizing,
                                            "completed": completed,
                                            "error_code": error,
                                            "response": response,
                                            "trace": trace,
                                            "latency_ms": elapsed,
                                            "quality": grade(
                                                Case.model_validate(case),
                                                content,
                                                completed,
                                            ),
                                            "work": work,
                                            "retrieved_evidence_recall": evidence_recall(
                                                gold, spans
                                            )
                                            if trace
                                            else None,
                                            "final_evidence_recall": evidence_recall(
                                                gold,
                                                reduction.get(
                                                    "final_evidence_spans", []
                                                )
                                                + reduction.get(
                                                    "state_evidence_spans", []
                                                ),
                                            )
                                            if trace
                                            else None,
                                            "reported_stage_prompts_fit": bool(fits)
                                            and all(fits),
                                            "unknown_stage_usage": len(calls)
                                            - len(fits),
                                            "mapped_evidence_recall": evidence_recall(
                                                gold,
                                                reduction.get(
                                                    "mapped_evidence_spans", []
                                                ),
                                            )
                                            if trace
                                            else None,
                                        }
                                        rows.append(row)
                                        requests_file.write(
                                            json.dumps(row, ensure_ascii=False) + "\n"
                                        )
                                        requests_file.flush()
                                        print(
                                            f"{case['id']} {policy}: {elapsed / 1000:.2f}s score={row['quality']['score']:.3f} {error or 'complete'}",
                                            flush=True,
                                        )
            manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["measured_requests"] = len(rows)
        cases_path = folder / "cases.jsonl"
        if cases_path.exists():
            manifest["cases_sha256"] = hashlib.sha256(
                cases_path.read_bytes()
            ).hexdigest()
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save_report(folder, rows, manifest)
        print(f"Native qualification report: {folder / 'report.md'}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--context-window",
        type=int,
        required=True,
        help="Actual configured native runtime slot, not advertised model maximum",
    )
    token = parser.add_mutually_exclusive_group(required=True)
    token.add_argument("--tokenizer-url", help="Native llama.cpp base URL")
    token.add_argument(
        "--tokenizer-command",
        help="Curl-compatible argv prefix, e.g. ssh host docker exec -i runner curl --unix-socket /app/inference-runner-0.sock",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[4096, 16384, 32768, 65536]
    )
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=["start", "middle", "end"],
        default=["start", "middle", "end"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["direct", "evidence_map", "batched_map"],
        default=["direct", "evidence_map"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--lookup-rounds", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--native-direct",
        action="store_true",
        help="Also exercise unchanged backend HTTP requests, with explicit truncation/count checks",
    )
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    if (
        args.repeats < 1
        or any(size < 512 or size > 262144 for size in args.sizes)
        or not 1 <= args.lookup_rounds <= 8
        or not 1 <= args.output_tokens <= 2048
        or not math.isfinite(args.timeout)
        or not 0 < args.timeout <= 86400
    ):
        parser.error("Invalid sizes, repetitions, deadline, or token budgets")
    try:
        asyncio.run(execute(args))
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
        print(f"Native qualification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
