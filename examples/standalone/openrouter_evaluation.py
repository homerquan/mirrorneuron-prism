"""Small, reproducible live evaluation; all inference uses explicit :free models.

Run a local server first, then:
python examples/standalone/openrouter_evaluation.py --repeats 2 --no-auth
Costs are counterfactual estimates applied AFTER inference, never provider prices.
"""

import argparse
import ast
import asyncio
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from prism.benchmark import Case, grade, json_text, sample
from prism.config import load_config

ROOT = Path(__file__).resolve().parents[2]
PRICES = {
    "astra_standard": {
        "input": 10.0,
        "output": 50.0,
        "source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
    },
    "claude_opus_5_5": {
        "input": 4.0,
        "output": 20.0,
        "source": "https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5",
    },
}


def source(facts, long=True):
    # Synthetic long business context, deliberately labeled as such in the report.
    background = (
        "\n".join(
            f"Archive entry {i:03d}: the facilities team reviewed desk allocation for branch {i % 19}; this historical note concerns office furniture and has no effect on the current task."
            for i in range(180)
        )
        if long
        else ""
    )
    return f'<prism-source id="brief">{background[: len(background) // 2]}\n{facts}\n{background[len(background) // 2 :]}</prism-source>'


def cases():
    def case(identifier, category, prompt, facts, checks, long=True):
        return Case.model_validate(
            {
                "id": identifier,
                "description": category,
                "tags": [category, "long-context" if long else "short-context"],
                "request": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return only the requested valid JSON object, without markdown. Follow the user's output constraints.",
                        },
                        {
                            "role": "user",
                            "content": prompt + "\n" + source(facts, long),
                        },
                    ],
                    "response_format": {"type": "json_object"},
                },
                "checks": checks,
            }
        )

    def contains(*terms):
        return {"type": "contains_any", "terms": list(terms)}

    def equals(path, value):
        return {"type": "json_equals", "path": path, "value": value}

    return [
        case(
            "coding-stock-fix",
            "coding",
            'Return {"code":"Python source"} containing only a function reconcile(start, deltas). Use no imports. Fix the behavior according to the specification; no explanation.',
            "Current code: def reconcile(start, deltas): return start + sum(deltas). Specification: start is nonnegative. Process signed integer deltas in order. After EACH event clamp stock to zero if it would become negative. Return final stock. Clamping only the final sum is wrong: start=3, deltas=[-5,4] returns 4.",
            [contains("def reconcile"), contains("for "), contains("return")],
        ),
        case(
            "coding-input-validation",
            "coding",
            'Return {"code":"Python source"} containing only a function parse_limit(value, default=10, maximum=100). Use no imports or exception commentary.',
            "API specification: accept positive integers or strings containing only decimal digits after stripping whitespace. Reject booleans, floats, None, negative values, zero, and nondigit strings by returning default. Clamp accepted positive values to maximum. Examples: ' 12 ' -> 12; 999 -> 100; True -> 10; '1.5' -> 10. Current buggy implementation simply calls int(value).",
            [contains("def parse_limit"), contains("return")],
        ),
        case(
            "content-launch-email",
            "content generation",
            'Write launch email copy as {"subject":"...","body":"..."}. Body must be 80 to 120 whitespace-separated words. Mention the product, launch date, monthly price, and a clear CTA. Use only brief facts. Do not invent certifications.',
            "Product: Helio Notes. Launch: January 16, 2027. Price: $12 per month. Audience: independent consultants. Features: searchable meeting notes, offline editing, optional cloud sync. CTA must say Start your trial. Trial lasts 14 days. Brand voice: clear, warm, practical. No claim of SOC 2 certification is authorized.",
            [
                contains("Helio Notes"),
                contains("January 16, 2027"),
                contains("$12"),
                contains("Start your trial"),
                {"type": "not_contains", "terms": ["SOC 2 certified"]},
            ],
        ),
        case(
            "support-refund-policy",
            "customer support",
            "Return merchant, refund_days, currency, receipt_required, and shipping_hours as JSON. Preserve the spelling and distinguish the refund period from the shipping target.",
            "Café München permits refunds within 14 days with a receipt. Refunds are issued in EUR. A receipt is required. Shipping normally takes 72 hours; this does not change the refund period.",
            [
                equals(["merchant"], "Café München"),
                equals(["refund_days"], 14),
                equals(["currency"], "EUR"),
                equals(["receipt_required"], True),
                equals(["shipping_hours"], 72),
            ],
        ),
        case(
            "summarization-incident",
            "summarization",
            "Summarize the incident as JSON with service, root_cause, mitigation, and customer_data_lost. For root_cause use timeout_regression or database_migration. For mitigation use rollback or restart.",
            "Service: checkout. At 09:12 a release reduced upstream timeout from 2000 ms to 100 ms. At 09:14 checkout HTTP 500 errors rose. A suspected database migration was ruled out because no migration ran that day. At 09:22 operators rolled back the timeout change; error rates returned to normal. The confirmed root cause was timeout_regression. Mitigation was rollback. No customer data was lost.",
            [
                equals(["service"], "checkout"),
                equals(["root_cause"], "timeout_regression"),
                equals(["mitigation"], "rollback"),
                equals(["customer_data_lost"], False),
            ],
        ),
        case(
            "content-short-microcopy",
            "short content generation",
            'Return {"subject":"...","body":"..."} for a friendly trial reminder. Body must be 18 to 30 whitespace-separated words. Include Helio Notes, 3 days, and the exact CTA Choose your plan.',
            "The user's Helio Notes trial ends in 3 days. Use the CTA Choose your plan. No urgency tricks or invented discounts.",
            [contains("Helio Notes"), contains("3 days"), contains("Choose your plan")],
            long=False,
        ),
    ]


def coding_check(identifier, code):
    """Execute only restricted pure Python in a fresh, timed interpreter."""
    try:
        tree = ast.parse(code)
        blocked = (
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.Global,
            ast.Nonlocal,
            ast.With,
            ast.AsyncFunctionDef,
        )
        for node in ast.walk(tree):
            if isinstance(node, blocked):
                return False
            if isinstance(node, ast.Name) and "__" in node.id:
                return False
            if isinstance(node, ast.Attribute) and node.attr not in {
                "strip",
                "isdigit",
                "isdecimal",
            }:
                return False
        if any(not isinstance(n, ast.FunctionDef) for n in tree.body):
            return False
        tests = (
            "assert reconcile(3, [-5,4]) == 4\nassert reconcile(0, []) == 0\nassert reconcile(8, [-2,-9,3,-1]) == 2\nassert reconcile(10, [2,-3]) == 9"
            if identifier == "coding-stock-fix"
            else "assert parse_limit(' 12 ') == 12\nassert parse_limit(999) == 100\nassert parse_limit(True) == 10\nassert parse_limit(False) == 10\nassert parse_limit(None) == 10\nassert parse_limit(0) == 10\nassert parse_limit(-5) == 10\nassert parse_limit(2.5) == 10\nassert parse_limit('1.5') == 10\nassert parse_limit('abc') == 10\nassert parse_limit('55', maximum=20) == 20\nassert parse_limit('bad', default=7) == 7"
        )
        # No imports, files, network, reflection, or environment access in builtins.
        harness = "import json,sys\np=json.loads(sys.stdin.read())\nscope={'__builtins__':{k:__builtins__.__dict__[k] for k in ('int','str','bool','float','isinstance','type','min','max','sum','len','range','enumerate','zip','ValueError','TypeError')}}\nexec(p['code'],scope)\nexec(p['tests'],scope)"
        result = subprocess.run(
            [sys.executable, "-I", "-c", harness],
            input=json.dumps({"code": code, "tests": tests}),
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0
    except (SyntaxError, ValueError, TypeError, subprocess.TimeoutExpired):
        return False


def assess(case, record):
    content = ((record.get("response") or {}).get("choices") or [{}])[0].get(
        "message", {}
    ).get("content") or ""
    quality = grade(case, content, record["completed"])
    extra = []
    try:
        document = json.loads(content)
        if case.id.startswith("coding-"):
            extra.append(
                {
                    "name": "functional Python tests",
                    "passed": coding_check(case.id, document.get("code", "")),
                }
            )
        if case.id.startswith("content-"):
            count = len(document.get("body", "").split())
            low, high = (18, 30) if case.id.endswith("microcopy") else (80, 120)
            extra.append(
                {
                    "name": "body word count",
                    "passed": low <= count <= high,
                    "observed": count,
                }
            )
            extra.append(
                {
                    "name": "subject and body present",
                    "passed": isinstance(document.get("subject"), str)
                    and bool(document["subject"].strip())
                    and isinstance(document.get("body"), str),
                }
            )
    except (ValueError, TypeError, AttributeError):
        extra.append({"name": "valid requested JSON", "passed": False})
    checks = quality["checks"] + extra
    return {
        "passed": record["completed"] and all(c["passed"] for c in checks),
        "score": sum(c["passed"] for c in checks) / len(checks)
        if record["completed"]
        else 0,
        "checks": checks,
    }


def estimated_cost(record, prices):
    calls = ((record.get("trace") or {}).get("execution_usage") or {}).get("calls")
    if not calls:
        return None
    cost = 0
    for call in calls:
        usage = call.get("provider_usage")
        if not isinstance(usage, dict) or any(
            type(usage.get(k)) is not int
            for k in ("prompt_tokens", "completion_tokens")
        ):
            return None
        if call["model_id"] == "nemotron-super-reasoning":
            cost += (
                usage["prompt_tokens"] * prices["input"]
                + usage["completion_tokens"] * prices["output"]
            ) / 1_000_000
    return cost


def report(records, out, repeats):
    indexed = {}
    for record in records:
        indexed.setdefault((record["case_id"], record["repeat"]), {})[
            record["route"]
        ] = record
    pairs = [
        (pair["baseline"], pair["candidate"])
        for pair in indexed.values()
        if all(label in pair for label in ("baseline", "candidate"))
        and all(pair[label]["completed"] for label in ("baseline", "candidate"))
        and all(
            estimated_cost(pair[label], PRICES["astra_standard"]) is not None
            for label in ("baseline", "candidate")
        )
    ]
    lines = [
        "# Free preparation, premium-priced synthesis: a live Prism pilot",
        "",
        "Run date: October 4, 2026 (America/New_York). All model calls used OpenRouter `:free` Nemotron models. Actual configured model token cost: $0. The dollar figures below are hypothetical costs obtained by pricing Super's measured tokens like a premium model; Nano is priced at $0.",
        "",
        "## Method",
        "",
        f"Six synthetic tasks, {repeats} paired repetitions per task, sequential requests with alternating route order. Baseline: Super direct. Candidate: Nano plain-text preparation → Super final synthesis. Temperature 0; output cap 4096 tokens; no hidden retries. Five tasks include roughly 30 KB of synthetic office-history distractors; one is short microcopy. This deliberately examines sparse, long-context tasks and a small-input counterexample. It is a pilot, not a representative production workload or a benchmark of OpenAI/Anthropic model quality.",
        "",
        "Coding is graded by functional tests in a restricted, timed interpreter. Copy is checked for required facts, JSON shape, prohibited claims, and length. Support/summaries use exact reference fields. These checks do not measure aesthetic quality, full security, or general reasoning ability. Worker notes are unverified observations: reducing input can lose facts.",
        "",
        "## Hypothetical pricing",
        "",
        "| Scenario | Input / million | Output / million | Reference |",
        "|---|---:|---:|---|",
    ]
    for name, p in PRICES.items():
        lines.append(
            f"| {name} | ${p['input']:.2f} | ${p['output']:.2f} | [Official pricing]({p['source']}) |"
        )
    lines += [
        "",
        "Rates checked October 4, 2026. Standard uncached rates; no caching, batch discount, tool charges, local compute cost, or long-context premium. Nemotron tokenization, reasoning, quality, and speed are not equivalent to the priced reference models. Provider completion-token totals include reasoning tokens when reported; Prism logical byte counters are never used for these estimates.",
        "",
        "## Results",
        "",
        "Acceptance and latency include every attempt. Cost comparisons use only matched, completed pairs with known physical usage on both routes; a quality failure remains in the cost comparison if it completed. Failed or unknown-usage requests never count as zero cost. Premium costs below use the Astra scenario and are means per matched request.",
        "",
        "| Task | Pass baseline / mix | Mean latency baseline / mix | Cost pairs | Premium cost baseline / mix | Savings |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    aggregate = {}
    for scenario, prices in PRICES.items():
        aggregate[scenario] = {
            label: sum(estimated_cost(pair[index], prices) for pair in pairs)
            for index, label in enumerate(("baseline", "candidate"))
        }
    for case in cases():
        rows = {
            label: [
                r for r in records if r["case_id"] == case.id and r["route"] == label
            ]
            for label in ("baseline", "candidate")
        }
        passes = [sum(r["quality"]["passed"] for r in rows[label]) for label in rows]
        latency = [
            statistics.mean(r["latency_ms"] for r in rows[label]) / 1000
            for label in rows
        ]
        matched = [pair for pair in pairs if pair[0]["case_id"] == case.id]
        cost = [
            statistics.mean(
                estimated_cost(pair[index], PRICES["astra_standard"])
                for pair in matched
            )
            if matched
            else 0
            for index in (0, 1)
        ]
        saving = f"{(1 - cost[1] / cost[0]) * 100:.1f}%" if cost[0] else "unknown"
        cost_text = f"${cost[0]:.5f} / ${cost[1]:.5f}" if matched else "unknown"
        lines.append(
            f"| {case.id} | {passes[0]}/{repeats} / {passes[1]}/{repeats} | {latency[0]:.2f}s / {latency[1]:.2f}s | {len(matched)}/{repeats} | {cost_text} | {saving} |"
        )
    b = [r for r in records if r["route"] == "baseline"]
    c = [r for r in records if r["route"] == "candidate"]
    bp, cp = (
        sum(r["quality"]["passed"] for r in b),
        sum(r["quality"]["passed"] for r in c),
    )
    bm, cm = (
        statistics.mean(r["latency_ms"] for r in b) / 1000,
        statistics.mean(r["latency_ms"] for r in c) / 1000,
    )
    lines += [
        "",
        f"Task acceptance: baseline {bp}/{len(b)}, mixed {cp}/{len(c)} ({(cp / len(c) - bp / len(b)) * 100:+.1f} percentage points). Mean end-to-end latency across all attempts: {bm:.2f}s → {cm:.2f}s ({(cm / bm - 1) * 100:+.1f}%). Matched completed pairs with known usage: {len(pairs)}/{len(b)}. Scores below are observed checks, not confidence intervals.",
        "",
        "| Cost scenario | Baseline paired total | Mixed paired total | Estimated savings | At 100,000 completed requests with the observed matched mix |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in aggregate.items():
        baseline, candidate = values["baseline"], values["candidate"]
        saving = (1 - candidate / baseline) * 100 if baseline else 0
        lines.append(
            f"| {name} | ${baseline:.5f} | ${candidate:.5f} | {saving:.1f}% | ${(baseline - candidate) / max(1, len(pairs)) * 100000:,.2f} saved (linear illustration) |"
        )
    premium_usage = [
        [
            call["provider_usage"]
            for pair in pairs
            for call in pair[index]["trace"]["execution_usage"]["calls"]
            if call["model_id"] == "nemotron-super-reasoning"
        ]
        for index in (0, 1)
    ]
    premium_input = [sum(u["prompt_tokens"] for u in rows) for rows in premium_usage]
    premium_output = [
        sum(u["completion_tokens"] for u in rows) for rows in premium_usage
    ]
    input_saving = (
        (1 - premium_input[1] / premium_input[0]) * 100 if premium_input[0] else 0
    )
    lines += [
        "",
        "| Matched premium-model work | Super direct | Nano → Super |",
        "|---|---:|---:|",
        f"| Input tokens (same Super tokenizer) | {premium_input[0]:,} | {premium_input[1]:,} |",
        f"| Completion tokens, including reported reasoning | {premium_output[0]:,} | {premium_output[1]:,} |",
        f"| Physical calls, including the free worker | {sum(len(p[0]['trace']['execution_usage']['calls']) for p in pairs)} | {sum(len(p[1]['trace']['execution_usage']['calls']) for p in pairs)} |",
        "",
        f"Premium input fell {input_saving:.1f}% on the matched completed requests. The smaller reduction in total estimated cost reflects completion/reasoning work, priced at a higher rate. Nano's work adds a second call; free token pricing does not eliminate latency or compute. Different models' token counts are not combined into a purported universal token total.",
    ]
    missing = sum(estimated_cost(r, PRICES["astra_standard"]) is None for r in records)
    lines += [
        "",
        f"Unknown-usage requests: {missing}. The paired cost estimate excludes {len(b) - len(pairs)} incomplete/unknown pairs and does not estimate a whole-run bill. Excluding failed pairs changes the task mix; the 100,000-request projection applies only to the observed completed mix and is not a production forecast. Latency includes upstream load, reasoning, networking, and Prism overhead. An aggregate can conceal a slower route or a quality regression on an individual task.",
        "",
        "JSON-object mode guarantees neither a requested field layout nor its types. Some completed responses failed coding tests or returned mismatched JSON fields/types. Applications requiring a shape should request JSON Schema and retain task-level checks. Upstream failures are included as failed task attempts; no retries conceal them.",
        "",
        "## Reproduction and evidence",
        "",
        "Start `prism serve --config src/prism/resources/openrouter/prism.json --no-auth`, then run:",
        "",
        "```sh",
        f"python examples/standalone/openrouter_evaluation.py --no-auth --repeats {repeats} --out-dir docs/evaluations/NEW-RUN",
        "```",
        "",
        "`cases.jsonl` contains prompts and references; `requests.jsonl` retains every response, sanitized trace, grader result, physical token count and latency. `manifest.json` records versions, hashes, pricing assumptions and settings. No provider credentials are saved.",
    ]
    (out / "benchmark-results.md").write_text("\n".join(lines) + "\n")
    values = aggregate["astra_standard"]
    savings = (
        (1 - values["candidate"] / values["baseline"]) * 100
        if values["baseline"]
        else 0
    )
    marketing = [
        "# Marketing narrative: spend premium tokens on the answer",
        "",
        f"Across {len(pairs)} matched completed pairs in a six-task synthetic pilot, Prism's free Nano preparation followed by Super synthesis reduced the hypothetical premium-model token bill by {savings:.1f}%. All-attempt task acceptance was {cp}/{len(c)} for the mix and {bp}/{len(b)} for Super direct. All-attempt mean latency changed from {bm:.2f}s to {cm:.2f}s. Incomplete or unknown-usage pairs are excluded from the cost estimate, never priced at zero.",
        "",
        "The product story: a free model reads the background and prepares the useful context; a larger model spends its work on the final response. Developers keep one OpenAI-compatible endpoint, with explicit stage limits and structured-output routing.",
        "",
        f"The mechanism in this pilot: premium input tokens fell {premium_input[0]:,} → {premium_input[1]:,} ({input_saving:.1f}%), while premium completion/reasoning tokens changed {premium_output[0]:,} → {premium_output[1]:,}. Show both counters: input compression alone overstates total cost savings.",
        "",
        "Suggested campaign line: **Give the larger model a focused brief. Give your application a smaller token bill.**",
        "",
        "Useful demo: show the same code repair or launch email from a long brief, then display premium input tokens, total completion tokens, elapsed time, and the reference checks side by side. Include short microcopy and failed requests so the demo shows where an extra stage adds delay. The pilot's lower acceptance and slower responses make this a workload-selection story, not a launch claim of equivalent quality.",
        "",
        "Use the numeric claim only with the pilot qualification and the pricing assumption adjacent to it. The runs used free Nemotron models, not GPT-6 Astra or Claude; the estimate reuses their token rates without establishing frontier-model quality, latency, or actual invoices. Actual free-tier token charges were zero under the configured model prices. Free-tier quotas, failures, local serving costs, and missing usage still matter.",
        "",
        "Avoid claiming production-wide savings, identical quality, faster responses, or replacing a frontier model. This report measures a narrow mechanism and identifies workloads for a larger follow-up evaluation.",
        "",
        "[Detailed results](benchmark-results.md)",
    ]
    (out / "marketing-narrative.md").write_text("\n".join(marketing) + "\n")


async def run(args):
    import hashlib

    from prism import __version__

    config, models = load_config(ROOT / "src/prism/resources/openrouter/prism.json")
    if not all(m.name.endswith(":free") for m in models.values()):
        raise ValueError("evaluation only permits explicit free models")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    tasks = cases()
    (args.out_dir / "cases.jsonl").write_text(
        "".join(json_text(c.model_dump()) + "\n" for c in tasks)
    )
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "prism_version": __version__,
        "repeats": args.repeats,
        "actual_pricing": "all configured model input/output token prices are zero",
        "hypothetical_prices": PRICES,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/prism").glob("*.py"))
        },
        "config": config.model_dump(mode="json"),
        "models": {
            k: m.model_dump(mode="json", exclude={"api_key", "api_key_env"})
            for k, m in models.items()
        },
        "settings": {
            "temperature": 0,
            "output_tokens": 4096,
            "warmups": 0,
            "request_timeout_seconds": 300,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    headers = {}
    if not args.no_auth:
        import os

        headers["authorization"] = "Bearer " + os.environ["PRISM_API_KEY"]
    records = []
    async with httpx.AsyncClient(
        headers=headers, timeout=300, trust_env=False
    ) as client:
        with (args.out_dir / "requests.jsonl").open("w") as output:
            for repeat in range(args.repeats):
                for index, case in enumerate(tasks):
                    routes = [
                        ("baseline", "nemotron-super-reasoning"),
                        ("candidate", "prism-nano-synthesis"),
                    ]
                    if (repeat + index) % 2:
                        routes.reverse()
                    for label, alias in routes:
                        record = await sample(
                            client,
                            case,
                            alias=alias,
                            base_url=args.base_url.rstrip("/"),
                            output_tokens=4096,
                            temperature=0,
                            models=models,
                        )
                        record.update(route=label, repeat=repeat)
                        record["quality"] = assess(case, record)
                        record["hypothetical_cost_usd"] = {
                            name: estimated_cost(record, p)
                            for name, p in PRICES.items()
                        }
                        records.append(record)
                        output.write(json_text(record) + "\n")
                        output.flush()
                        print(
                            f"{repeat + 1}/{args.repeats} {case.id} {label}: status={record['status']} pass={record['quality']['passed']} time={record['latency_ms'] / 1000:.2f}s",
                            flush=True,
                        )
    report(records, args.out_dir, args.repeats)
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "complete"
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "docs/evaluations/2026-10-04-openrouter"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    asyncio.run(run(args))
