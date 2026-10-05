# Token cost since startup

Set input **and** output rates in each `models/MODEL_ID.json`, in USD per million tokens:

```json
{
  "input_cost_per_million": "$1/m",
  "output_cost_per_million": "$5/m"
}
```

This is a price fragment to add to an existing model definition. Numeric `1` and `5`, `"$0.25/1M"`, and `"5/million"` are also accepted. Negative values, non-finite values, and other units are rejected. Zero explicitly means free; omitted/null prices mean unknown. Prices come from your configuration, not a provider catalog. Review them when account pricing changes.

```sh
prism start --profile nano-openai --show-cost
```

In an interactive terminal, `--show-cost` opens a full-screen dashboard. It shows cumulative spend, estimated dollars and percent saved, comparison counts, reported input/output tokens, and a per-model usage/rate table. Uptime refreshes while idle; costs refresh as physical calls finish, including failures and capacity probes. Taller terminals also show recent server activity and the last request's policy/status/latency. Resizing adjusts the layout. Ctrl+C stops the server, restores the terminal, and prints a final summary. Restarting resets the counters.

The dashboard uses the terminal's alternate screen. With `--json` or redirected output in automatic mode, reports stay as one JSON object per line on stderr at startup, after requests/probes, and shutdown. Unsupported terminals use ordinary summaries. Reports do not print prompts, answers, or keys.

## Try OpenRouter with hypothetical prices

`prism` combines Gemma at `http://127.0.0.1:12434/engines/v1` and Spark Nemotron at `http://10.0.4.32:12434/engines/v1`, both with zero configured token rates. Hardware and electricity are outside this report. A zero-cost baseline cannot produce a meaningful savings percentage.

Use the separate `prism-mock-cost` profile to exercise savings with real OpenRouter calls. It prepares text with free Nemotron Super and synthesizes with free Nemotron Ultra:

| Model definition | Input USD / 1M | Output USD / 1M |
|---|---:|---:|
| `models/nemotron-super-mock-cost.json` (Super worker) | 0.10 | 0.50 |
| `models/nemotron-ultra-mock-cost.json` (Ultra final/baseline) | 5.00 | 25.00 |

These are **hypothetical rates**, not OpenRouter bills or advertised provider prices. The physical slugs end in `:free`; ordinary Super/Ultra definitions keep zero prices. Both model definitions set `cost_rates_are_hypothetical: true`. The terminal labels simulated dollars; the costs endpoint returns `pricing_mode: "hypothetical"`. Real inference and real reported input/output usage still determine the totals. Regular model definitions retain their real configured rates. `prism init --preset openrouter` copies the demo profile and its models; in a checkout they already exist. Set `OPENROUTER_API_KEY` before starting Prism; no local model servers are needed. Free upstream availability and quotas still apply.

```sh
prism start --profile prism-mock-cost --show-cost --no-auth
# In another terminal in this checkout:
python examples/standalone/cost_demo.py --no-auth
```

The included script sends a source with routine background plus three relevant policies, then reads `/v1/prism/costs`. To run a demonstration outside the checkout, use this copyable request after starting the server:

```sh
python - <<'PY'
import json, urllib.request
source = "\n".join(
    f"Team {n}: routine documentation updates completed; the existing release process remains unchanged."
    for n in range(40)
)
source += "\nProduction releases require two reviewers. Critical incidents need a response in 30 minutes. Logs are kept 14 days."
body = {
    "model": "prism-mock-cost",
    "messages": [{"role": "user", "content":
        'Summarize the release, support and retention rules in three short bullets.\n'
        '<prism-source id="review">' + source + '</prism-source>'}],
    "max_completion_tokens": 1024,
}
request = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
with urllib.request.urlopen("http://127.0.0.1:8080/v1/prism/costs") as response:
    print(json.dumps(json.load(response), indent=2))
PY
```

This example assumes `--no-auth`. For authenticated deployments, set `PRISM_API_KEY` and run the checkout script without `--no-auth`. The profile forces `text_synthesis`: Super condenses the explicit source into notes and Ultra answers from those notes. Explicit source boundaries allow input savings; source-free prompts retain their original instructions. Savings depend on actual notes/output lengths and can be negative, especially for short inputs. The profile does not generate mock responses or force positive savings.

## Spend calculation

For every started physical call:

```text
USD = (provider prompt_tokens × model input_cost_per_million
     + provider completion_tokens × model output_cost_per_million) / 1,000,000
```

Prism sums workers, reviews, reduction/repair calls, synthesis, direct calls, and server-side `/capacity` probes. Concurrent requests share one process total. CLI `capacity` and benchmarks run in separate processes and are not part of this server's counters. Laya runs locally and adds no provider token charge. Infrastructure cost is outside these token rates.

Spend uses reported physical usage, including reported reasoning completion tokens. Public response `usage` is Prism's byte-based logical accounting and is **not used for billing**. The resource ledger's conservative reservations are separate from this report.

If any started call lacks prices or usage, `total_cost_usd` is null. `known_cost_usd` is the subtotal of calls with both prices and reported usage. `unpriced_calls` and `unreported_usage_calls` show what's missing. For priced calls without usage, `unknown_usage_upper_estimate_usd` separately sums their conservative reservations. An upstream failure or disconnect is not treated as free. A reported total is an estimate at configured flat rates, without cached-input discounts, batch/service-tier rates, taxes, tool fees, or provider invoice reconciliation.

## Savings calculation

The default comparison model is the profile's `synthesizer`, falling back to `direct`. JSON requests first apply `structured_output_model` as their direct/final assignment. Optionally set `cost_baseline_model` to another model ID in the profile; that model file is loaded for comparison but does not become an execution stage. Startup does not require a comparison-only model's key; explicitly probing it does.

For identical one-call direct routes, baseline cost equals reported cost, so savings are exactly zero. For other routes, Prism estimates tokens in the **original** text request with the shared o200k tokenizer and uses the final call's reported completion tokens. It prices those counts at the baseline model's configured rates. This is a hypothetical direct answer of the same reported output-token length, not a second live baseline call or a claim about equivalent answers.

```text
estimated saved USD = sum(eligible direct baseline costs) − sum(eligible actual stage costs)
estimated saved %   = estimated saved USD / sum(eligible direct baseline costs) × 100
```

Only completed requests with complete stage prices/usage, a priced baseline, available token estimates, and an original request that fits the baseline's declared context/capabilities/output limit are comparable. Images, failures, interrupted delivery, missing usage/prices, and impossible direct baselines are excluded. A zero-cost baseline has no defined percentage. Negative savings remain negative: extra stages can cost more.

`compared_requests`, `excluded_requests`, and `compared_cost_usd` define the comparison cohort; lifetime spend also includes excluded calls and probes. Savings do not silently compare that larger spend with a smaller baseline cohort. Use a real paired benchmark and task-quality results for stronger marketing claims.

## Read machine-readable totals

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/prism/costs \
  -H "Authorization: Bearer $PRISM_API_KEY"
```

The endpoint is available with or without `--show-cost`, using the server's normal authentication. It returns `started_at`, `scope: "since_process_start"`, `pricing_mode`, physical call/request counts, per-model usage/rates, spend completeness, the comparison cohort, and `estimated_baseline_cost_usd`, `estimated_saved_usd`, and `estimated_saved_percent`. It is process-wide and resets on restart; it is not a persisted invoice ledger.
