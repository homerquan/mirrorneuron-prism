# Speed, quality, and cost benchmarks

Run the same source questions through a direct baseline and the bounded extraction/synthesis pipeline. The packaged suite has six synthetic cases covering deployment qualifications, incident/runbook synthesis, UTF-8 names and units, prohibitions, effective policy dates, and numeric connection settings. Each case has an independent reference rubric; response schemas constrain the shape but do not supply the correct field values.

Start the configured physical model server and Prism in one terminal:

```sh
export PRISM_API_KEY='your-configured-secret'
prism serve --config prism.json
```

In another terminal, use the same key and run:

```sh
export PRISM_API_KEY='your-configured-secret'
prism benchmark run --config prism.json
```

This defaults to `prism-direct` versus `prism-evidence`, three measured repetitions per case, one warmup pair, temperature zero, and a 2048-token public output budget. The six sources are short enough for both routes, so this comparison measures the overhead and reference quality of orchestration on the same inputs. It does not establish a long-context recall advantage. Pass `--candidate prism-batched`, `prism-verified`, or `prism-retrieve` to measure another fixed policy. Retrieval has focused coverage, so its reference score alone cannot establish complete coverage. For automatic Laya routing, pass `--candidate prism` and inspect each trace's chosen strategy. To test a stronger baseline, configure another public alias in the server JSON and pass it with `--baseline`.

Laya is required and loads before the server becomes ready. Fixed policies skip decision inference when only one policy is eligible. Automatic routing includes decision time in request latency and records `decision.elapsed_ms`; checkpoint download/startup time is excluded from this HTTP benchmark. Physical work/cost includes verification as well as extraction and synthesis. Token pricing does not measure Laya CPU operating costs.

The command prints progress to stderr and JSON results to stdout. It creates a fresh timestamped folder under `benchmark-results/`, or you can choose a new folder:

```sh
prism benchmark run --config prism.json \
  --baseline prism-direct --candidate prism-evidence \
  --repeats 3 --warmup 1 --output-tokens 2048 \
  --out-dir benchmark-results/gemma-run-a
```

Existing output folders are never overwritten. For a quick smoke check, use `--repeats 1 --warmup 0 --max-cases 2`. Use `--base-url` for another Prism deployment and `--api-key-env` for a different credential environment variable. Both routes receive the same per-case inference parameters and output budget; their execution order alternates to reduce order bias. Warmup pairs use the first cases in sequence and are retained separately. This is a sequential latency benchmark, not a concurrent saturation/load test.

## Saved artifacts

```text
benchmark-results/gemma-run-a/
  manifest.json   # version, environment, settings, hashes, redacted config
  cases.jsonl     # exact measured inputs and reference checks
  requests.jsonl  # measured outputs, latency, checks, errors, usage, traces
  warmup.jsonl    # warmup outputs and their work/cost, excluded from measured scores
  summary.json    # per-route aggregates and candidate/baseline ratios
  report.md       # readable results table and metric definitions
```

Each response is flushed to disk as it completes. Interrupted runs keep a partial summary and an `interrupted` manifest. A run marked `complete` means the scheduled attempts finished; inspect completion and quality rates to see how many answers succeeded. The folder preserves source inputs and answer text for review; credentials and raw-model API-key fields are excluded. Generated run folders are ignored by Git.

## What the metrics mean

| Dimension | Measurement |
|---|---|
| Speed | End-to-end generation HTTP latency, mean/p50/p95, successful-request mean, and sequential requests/second. Trace retrieval is excluded. |
| Quality | Equal-weight reference checks within each case, then mean score and fraction passing every check. Transport errors, invalid responses, content filtering, and truncated answers score zero. |
| Work | Provider-reported input/output tokens and physical calls across direct, extraction, and synthesis stages. Missing traces/usage are counted separately. |
| Cost | Configured per-model prices multiplied by reported physical tokens across the complete execution. Reservation upper estimates are reported separately. |

JSON equality checks verify expected values and preserve types, so `true` does not pass a reference of numeric `1`. Text checks normalize case and whitespace; they are transparent fixture checks rather than semantic judges. Schemas alone are not correctness references. No extra LLM judge or its unreported latency/cost is introduced. Small synthetic samples do not establish general answer quality or statistically reliable performance; repeat the same workload and add representative held-out cases before drawing workload conclusions.

Prism's public `usage` is logical UTF-8 accounting. Benchmark monetary estimates use physical usage in the authenticated execution trace instead, including workers and failed started calls. Unknown prices, usage, or traces leave total cost **unknown**, rather than treating that work as free. Known subtotals remain inspectable. `per_completed_request_usd` and `per_passing_request_usd` include the cost of failed measured attempts in their numerator.

Declare `input_cost_per_million` and `output_cost_per_million` for each physical model in your raw-model JSON. Supply actual rates for your deployment; the benchmark makes no external price lookup. Explicit zero rates represent zero API token charges, and do not measure hardware, electricity, or CPU/GPU operating cost. Restart Prism after changing its configuration so the server's reservation estimates use the updated settings. Pass that configuration with `--config` to preserve the client-side pricing snapshot. Without it, trace work and reservation upper estimates can still be recorded, but reported-token monetary estimates remain unknown.

## Compare saved runs

Run the benchmark again after changing models, profiles, Laya routing configuration, or code. Keep the same cases and public output budget, then compare:

```sh
prism benchmark compare \
  benchmark-results/gemma-run-a \
  benchmark-results/gemma-run-b \
  --out-dir benchmark-results/comparison-a-b
```

Comparison runs offline and requires no API key. It prints both summaries, successful latency ratios, quality-score deltas, and mean request cost deltas relative to the first run. The optional output folder contains `comparison.json` and a readable `report.md`. Different dataset/rubric hashes, different temperature/output budgets, or incomplete runs mark the workload non-comparable and suppress deltas. Repetition/warmup differences are flagged. The config, environment, model identities, and implementation hash remain available to explain intended changes and external differences. Zero-baseline ratios and unknown costs remain undefined rather than producing fabricated ratios.

## Add your own workload

Pass `--cases path/to/cases.jsonl`. Each nonempty line is one case:

```json
{
  "id": "timeout-policy",
  "description": "Read a numeric value with its units",
  "tags": ["units"],
  "request": {
    "messages": [{
      "role": "user",
      "content": "Return timeout_seconds as JSON. <prism-source>Connection timeout is 3 seconds.</prism-source>"
    }],
    "response_format": {"type": "json_object"}
  },
  "checks": [{"type": "json_equals", "path": ["timeout_seconds"], "value": 3}]
}
```

IDs must be unique. Cases cannot override `model`, streaming, or output limits; the runner controls those identically. Other supported inference parameters, such as temperature or response format, may be supplied per case and are preserved in the dataset hash. `json_equals` follows a path of object keys or array indices. `contains_any` accepts a `terms` list with alternative required phrases; `not_contains` fails if any listed phrase appears. Every case needs at least one check. For an evidence route, keep operative instructions outside explicit `<prism-source>` blocks and respect the implemented coverage contract.

## Verify the benchmark implementation

```sh
python -m pytest tests/standalone/test_benchmark.py -q
python -m pytest tests/standalone/test_benchmark.py -q -m integration -o addopts=''
```

Deterministic tests establish a known quality difference, exact physical token/call/cost totals across workers and synthesis, warmup exclusion, credential omission, durable interrupted runs, missing-cost behavior, comparison compatibility checks, and the installed CLI over real HTTP. They require no external model or weights.
