# Local Gemma4 and Spark Nemotron live qualification — 2026-10-02

The repeatable runner is [examples/standalone/docker-spark/benchmark.py](../examples/standalone/docker-spark/benchmark.py), with [usage instructions](../examples/standalone/docker-spark/README.md). Run from the repository root:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py --spark spark --repeats 3
```

It discovers installed endpoints, opens its own SSH tunnel, creates isolated configurations and fixtures, starts Prism with resident Laya, runs sequential comparisons, saves detailed reports and raw traces, and closes its own services. Existing deployment configuration and Docker models are untouched. Each invocation creates a new folder under `benchmark-results/`; existing folders are never overwritten. `--suites direct`, `--suites large-gemma,large-nemotron`, and `--suites review` select subsets. Defaults are one measured repetition, one warmup pair for short-source suites, no warmup for large/review, and two review cases. Raise `--review-max-cases 6` for the full review suite.

## Models and measurement

Local Docker Model Runner: `ai/gemma4:E2B`, 4.65B, Q4_K_M. Spark through SSH: `docker.io/ai/nemotron-3.5-lightning:latest`, 32.91B, Q4_K_M. These are the installed models discovered live; no new models were downloaded. Laya runs on the local CPU using the pinned deployment checkpoint and a 0.7 option-confidence threshold. Its checkpoint emits a confidence-calibration warning; no threshold was lowered.

The six short synthetic cases have independent JSON reference checks. Each run below used one measured repetition per case, temperature zero, and an 8192-token public output budget; warmups are excluded. The review qualification uses two source-free versions of those cases, preserving their references, and no warmup. Samples are small and do not establish general quality or reliable statistical performance.

No operator dollar prices were supplied. Physical input/output tokens are the cost proxy, including failed started work. Missing usage makes reported tokens a known lower bound. Reasoning consumes backend output tokens. Laya tokenization and CPU work are recorded separately; hardware and electricity are unmeasured. These live profiles exercise automatic policy selection with configured stage assignments; cost/power assignment optimization remains disabled without operator ratings/prices.

## Measured outcomes

| Workload / route | Complete | All checks pass | Mean reference score | Mean latency | P95 latency | Physical input tokens | Physical output tokens | Calls | Unknown-usage calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short cases: local Gemma direct | 6/6 | 5/6 | 0.917 | 5.88 s | 7.19 s | 768 | 2807 | 6 | 0 |
| Short cases: Spark Nemotron direct | 6/6 | 6/6 | 1.000 | 11.75 s | 22.70 s | 758 | 4992 | 6 | 0 |
| Short cases: Gemma extraction → Nemotron synthesis | 2/6 | 2/6 | 0.333 | 22.48 s | 46.34 s | 4525 | 9551 | 14 | 3 |
| Large cases: clarified Nemotron extraction → Gemma synthesis | 2/3 | 1/3 | 0.556 | 130.77 s | 165.93 s | 14295 | 33544 | 11 | 0 |
| Review cases: Nemotron direct baseline | 2/2 | 2/2 | 1.000 | 9.36 s | 13.39 s | 217 | 1277 | 2 | 0 |
| Review cases: Gemma draft → Nemotron review → Nemotron synthesis | 2/2 | 2/2 | 1.000 | 69.65 s | 78.85 s | 1229 | 9722 | 6 | 0 |

Mean and P95 include failed attempts. The short mixed evidence route's successful-request mean was 43.25 s; fast early failures must not be interpreted as faster successful generation. The review route took 7.44× its direct baseline latency and 7.33× the physical tokens, with identical reference scores on two cases. There is no demonstrated speed or token-cost improvement from mixing on these fixtures.

## Automatic divide-and-conquer and stage performance

Three 12,786-byte requests place the same facts at the beginning, middle, and end of two documents containing unrelated telemetry. Admission contexts were deliberately constrained: Gemma worker 8K, Nemotron direct/worker 16K, and synthesis 32K. The compiler uses a conservative UTF-8 bound plus output reserve; rejecting direct admission does **not** establish overflow under the backend's native tokenizer. Source boundaries are explicit; splitting into six smaller partitions and grouping feasible worker calls happen automatically.

Direct was excluded with `context_length_exceeded` on each variant. Laya proposed `evidence_map` at confidence below 0.7 and abstained; the deterministic eligible-policy fallback selected `batched_map`. The clarified reverse assignment compiled six partitions into three extraction calls and one final synthesis. This is real automatic policy routing and splitting, not a manually fixed map policy or an accepted Laya recommendation.

| Stage, clarified large run | Attempts / calls | Mean call time | Mean phase wall time | Reported input tokens | Reported output tokens |
|---|---:|---:|---:|---:|---:|
| Laya decision, local CPU | 3 | 0.230 s | 0.230 s | 771 Laya tokens | 0 classifier output tokens |
| Nemotron extraction | 9 | 72.75 s | 123.42 s | 13308 | 32124 |
| Gemma synthesis, reached by two requests | 2 | 10.66 s | 10.66 s | 987 | 1420 |

Worker call times overlap and include queueing; their sum is not end-to-end elapsed time. Synthesis statistics include only the requests reaching that stage. The start-position variant covered six partitions but omitted manager approval, scoring 2/3. The middle-position variant covered six partitions and passed every check. The end-position variant failed visibly on incomplete evidence. Full validated coverage is not a guarantee of semantic recall.

| Stage, two-case draft/review run | Calls | Mean phase wall time | Reported input tokens | Reported output tokens |
|---|---:|---:|---:|---:|
| Gemma draft | 2 | 6.42 s | 273 | 742 |
| Nemotron structured review | 2 | 50.44 s | 456 | 7189 |
| Nemotron final synthesis | 2 | 12.77 s | 500 | 1791 |

Fixed review policies skip Laya inference; the decision stage is recorded as `rules_only`, not zero-duration inference.

## Failure investigation and artifacts

An initial 2048-token run was interrupted and preserved after Gemma incomplete-evidence failures and Nemotron reasoning-budget exhaustion. The completed short runs used an 8192 public budget. Initial large runs failed for both stage directions. A captured Nemotron batch artifact marked irrelevant partitions `incomplete` with empty `needs`. The extraction contract was clarified to explicitly return `complete` with empty records/needs after fully inspecting an irrelevant partition. The reverse worker reserve was also increased from 4096 to 8192. Validators were preserved. These changes are confounded; the rerun does not attribute gains to either independently.

Live artifacts are ignored by Git and retained in this workspace:

- `benchmark-results/spark-gemma-detailed-20261002/report.md`: comprehensive quality, speed, token-cost, routing, coverage, and per-node report, including failed runs.
- `benchmark-results/spark-gemma-direct-8192-20261002`: six-case direct comparison.
- `benchmark-results/spark-gemma-evidence-8192-20261002`: six-case mixed evidence comparison.
- `benchmark-results/spark-gemma-large-auto-20261002`: initial Gemma-worker automatic splitting.
- `benchmark-results/spark-gemma-large-reverse-20261002`: initial Nemotron-worker automatic splitting.
- `benchmark-results/spark-gemma-large-reverse-clarified-20261002`: clarified large-request qualification.
- `benchmark-results/spark-gemma-script-review-20261002`: end-to-end saved-runner verification, including temporary service logs and generated fixtures/configuration.

Each run retains exact cases, manifest/config snapshots, responses, errors, reference checks, and authenticated traces. The comprehensive analysis JSON also derives stage P50/P95 timings from recorded call durations. Earlier traces lack start/finish timestamps and retain unknown phase wall time. Service startup and model loading are excluded from request latency.

## Follow-up: complete policy matrix

The saved runner now provides eight forced map/retrieval suites (four policies × two model assignments/final models), plus the existing mixed draft/review suite. `--suites policies` runs this matrix; `--suites all` includes automatic and direct comparisons too. Automatic profiles now permit `verified_map`. Fixed profiles cannot silently choose `batched_map`. The report audits selected policies and full operator/node execution, and displays verification coverage separately from extraction coverage.

Deterministic execution tests completed every map/retrieval combination, including six verification calls for each verification assignment, exact model assignments, and full versus focused coverage. Fresh real-model retrieval qualifications each passed their independent check: local Gemma final took 8.14 seconds; Spark Nemotron final took 12.69 seconds. Both selected `retrieve_read`, executed one synthesis call, and had zero policy/graph audit mismatches. The context-limited direct baseline was rejected before dispatch. Artifacts are retained in `benchmark-results/spark-gemma-all-policies-retrieval-20261002/report.md`. This follow-up did not rerun the complete expensive live matrix; the prior live map and draft/review results remain recorded above.
