# Speed, quality, and cost benchmarks

For native-token-sized sources beyond an installed runtime's window, use the [native context qualification runner](../examples/standalone/docker-spark/native_context.py), described in [the Docker/Spark instructions](../examples/standalone/docker-spark/README.md#native-context-qualification). This is separate from the original byte-admission fixtures. It sizes seeded cases with the installed tokenizer/template, checks calibration against provider usage, exercises actual oversized direct controls, and reports answer correctness, lookup span recall, stage fit, and physical work. It uses the same physical model at every stage to isolate orchestration from model assignment.

See the [2026-10-03 native validation](native-context-validation-20261003.md) for measured results, implementation changes, physical work, and remaining limits.

[RULER (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654) is a related evaluation foundation covering multiple-needle retrieval, multi-hop tracing, aggregation, and question answering. Prism's current native qualification uses independently written fixtures and reports no RULER score. Its distributed-fact and chain cases exercise related behaviors; aggregation remains untested. See [attribution and provenance](native-context-validation-20261003.md#prior-work-and-attribution) for the paper, official benchmark repository, and algorithm/runtime references.

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

## Per-stage performance

New runs include a `stages` breakdown in each request record, in `summary.json`, and in the readable report. Extraction, verification, drafting, review, synthesis, and direct generation are identified from the selected execution graph. Each stage reports model assignments, call counts, provider-reported input/output tokens, unknown-usage calls, and backend elapsed time. Missing usage remains unknown; reported totals are known subtotals, including failed work.

Authenticated traces include `started_ms` and `finished_ms` for each started backend call, relative to creation of the request's accounting ledger. This allows phase wall time to be measured across overlapping calls. The sum of worker call times is backend work time, not request latency. A call's existing `elapsed_ms` includes semaphore queueing; its start timestamp is recorded after acquiring the model semaphore. Phase wall time spans the first admitted call through the last completed or cancelled call and includes gaps within that span. Older traces without timestamps retain unknown phase wall time.

The decision stage reports inference attempts, mean/sum elapsed time, dispositions, and Laya-reported input/output tokens separately. Single-policy execution skips inference. Laya is a classification decision model, so reported output tokens can be zero; this does not imply zero CPU work. Decision timing excludes checkpoint preparation at service startup. No decision-model dollar cost or hardware cost is inferred.

Quality remains the independent final-answer rubric. Backend call completion is not a quality score. Coverage validation, malformed evidence, incomplete intermediates, cancellation, and failed synthesis remain visible in the per-request trace. For large-request tests, inspect `ineligible_policies`, `decision`, `strategy`, `plan`, and `coverage` to establish why direct was rejected, whether Laya's recommendation was accepted, how many partitions were required, and whether the graph completed. Use explicit `<prism-source>` boundaries for lossless automatic partitioning; arbitrary conversation instructions are not automatically converted into source data. Configured context admission uses a conservative UTF-8 bound, so admission failure does not by itself establish that the backend's native tokenizer would overflow.

## Repeat local Docker and SSH Spark experiments

The [saved live runner](../examples/standalone/docker-spark/benchmark.py) discovers installed Gemma4 and Nemotron 3.5 Lightning models, creates a private temporary Prism service and SSH tunnel, generates small-context large-source fixtures, and writes detailed stage/routing/token reports. Run `.venv/bin/python examples/standalone/docker-spark/benchmark.py --spark spark --repeats 3`; use `--suites` to select comparisons. See [runner options](../examples/standalone/docker-spark/README.md) and the [2026-10-02 qualification results](live-benchmark-20261002.md), including visible failures and extraction recall limitations.

The runner now has a forced-policy matrix: `--suites policies` tests `evidence_map`, `batched_map`, `verified_map`, and `retrieve_read` with both model assignments/final models, plus `draft_review`. Use `--suites policies --max-cases 1 --repeats 1` for qualification or raise repetitions for repeated measurements. `--suites all` additionally includes direct baselines and automatic selection. Full-coverage automatic profiles admit verification as well as the two map strategies.

Fixed-policy profiles prohibit fallback to another policy. The aggregate report and `analysis.json` audit requested versus selected policy, required operator graphs, completed nodes, missing traces, and unexpected completed graphs. Per-case tables include verification coverage. Retrieval uses its own focused large-lookup fixture rather than claiming full coverage of the map fixture; draft/review uses source-free requests. All policies retain stage timing and physical-token cost proxies, including failures and unreached stages.


Structured reduction is now the live runner's default. Use `--reduction structured|rolling|none` to compare bounded state/tree/temporary Markdown lookup, rolling memory, or the original single synthesis. Per-stage metrics include `reduce` and `evidence_lookup`; the detailed report shows tree levels, state bounds, retrieval sizes and scratch cleanup. Completed-answer quality is reported separately from overall quality, which scores failed jobs zero. Intermediate artifact validation is separate from HTTP completion. Manifests now fingerprint all Prism source modules as well as benchmark metric implementations. [Configuration and limits](structured-reduction.md).
