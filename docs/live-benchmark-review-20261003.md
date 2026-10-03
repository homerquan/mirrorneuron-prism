# Reliability review and bounded structured reduction — 2026-10-03 UTC

Prism now completes the previously failing divide-and-conquer requests in the native qualification runs, and has a structured alternative to the single large reducer. Completion remains separate from answer correctness. Every result below is retained with inputs, responses, configuration, readiness, traces, and per-stage measurements under `benchmark-results/`.

## What failed in the supplied report

The supplied [original report](../benchmark-results/docker-spark-20261002T233307Z/report.md) contained 78 candidate attempts: 29 completed, including retrieval and review. Across the six full-source map suites, only **5/54** completed. Gemma evidence/batched/verified mapping completed **0/27**. Both verified suites completed **0/18**. Errors were predominantly incomplete evidence, plus 14 generic upstream errors and verifier rejection. A zero quality score often meant no public answer, not a bad answer that a user could inspect.

The configured small direct baselines intentionally rejected the large requests before inference. Their near-zero latency and zero tokens measure admission rejection; they cannot be used as successful generation speed/cost baselines. Fixed mapping suites now compare against fitting full-context local Gemma. The automatic small-context suite retains an explicitly failing direct negative control.

Current inference/schema readiness checks succeeded on local Gemma, Spark Gemma, and Spark Nemotron. Both Docker engines serve models and load them on demand. There is no current evidence of an unready model causing these failures. This does not prove historical readiness for every old request: old generic upstream errors did not retain their provider reason.

A confirmed configuration mismatch contributed risk: Spark's running Nemotron context is **8192**, while earlier Prism metadata allowed a much larger context. Model maximum context and the running engine's configured slot limit differ. The runner now reads explicit Docker configuration, conservatively caps its generated model entries, and saves `runtime-configuration.json`. The old generic errors cannot all be conclusively attributed to context overflow retrospectively. Current reruns have no generic backend failures.

Raw probes also found project-level issues: Gemma sometimes answered the public fields instead of the intermediate evidence schema, irrelevant telemetry caused incomplete/overlong extraction, and verification treated a single partition as if it had to answer the entire final request. Nemotron even rejected a verbatim manager-approval fact while declaring that the surrounding telemetry contained no policy. Clarifying local entailment and source ID metadata fixed that reproduced false rejection without accepting unsupported verdicts.

## Implemented changes

- Constrain intermediate JSON with a backend schema where supported, then validate types, completion and exact quote provenance in Prism.
- Share physical concurrency across model aliases. Disable optional thinking in this test setup so intermediate tokens produce usable artifacts. Public direct reasoning controls retain precedence.
- Reserve one stronger extraction retry for invalid partitions; retain successful siblings. Verified mapping can re-extract a rejected interpretation once and re-check it. Original source bytes remain immutable; replacement IDs, digests, counts and failures are traced.
- Treat an empty record set as having no claims to verify, with an explicit trace method. This does **not** establish extraction recall.
- Keep review/synthesis at least as powerful as preceding stages in optimized profiles, with strictly higher worker-to-hard-stage ratings unless the worker is already at the highest feasible rating. The native examples use Spark Nemotron for review, verification, synthesis and lookup; Gemma performs narrow extraction/reduction.
- Add bounded structured state, recursive reduction and on-demand evidence lookup. Raw source blocks and extracted records live in generated temporary Markdown files, with no database. Lookup accepts known IDs or a lexical query, searches original source blocks as well as mapped records, returns bounded exact excerpts, and retains original UTF-8 byte provenance. Files are cleaned after prompt assembly, failure or cancellation.
- Split structured-mode verification into one-record packets, omit duplicate quote copies, and narrow surrounding context only when necessary. Full raw evidence remains accessible.
- Distinguish HTTP completion, artifact validity, public job completion and reference quality in reports. Measure recovery, compaction, reduction, lookup, synthesis and Laya separately. Source-module hashes are included in new manifests.

Configuration, limits and semantics are documented in [structured reduction](structured-reduction.md), [model optimization](model-optimization.md), and the [runner instructions](../examples/standalone/docker-spark/README.md). Root deployment profiles and model entries were not modified.

## Completion regression over all fact positions

The [completed regression](../benchmark-results/reliability-fixed-20261003/report.md) uses one repetition at each of start/middle/end, after evidence recovery and the local verification correction. It uses rolling compaction when synthesis overflows; the later structured runs follow below.

| Candidate pipeline | Complete | Mean reference quality | Mean elapsed seconds | Known input + output tokens, all three attempts |
|---|---:|---:|---:|---:|
| Local Gemma evidence → Nemotron | 3/3 | 0.556 | 55.91 | 47,736 |
| Local Gemma batched → Nemotron | 3/3 | 0.111 | 84.94 | 82,969 |
| Local Gemma verified → Nemotron | 3/3 | 0.667 | 54.33 | 48,960 |
| Nemotron verified → Nemotron | 3/3 | 0.333 | 5.66 | 22,317 |
| Spark Gemma verified → Nemotron | 3/3 | 0.778 | 38.68 | 47,046 |

All **15/15** candidates returned usable public answers. All **9/9** verified candidates completed their intended graphs. This fixes the no-answer problem on these cases; it does not claim that low-quality answers became correct. In particular, local batching is still poor on this telemetry-heavy workload.

This is a diagnostic before/after qualification, not a controlled statistical estimate: repetitions, intermediate output budgets, runtime caps, thinking settings and large direct baselines changed. The original had three repetitions per fact position; this regression had one. Compare saved manifests and use identical settings for future performance studies.

## Structured state/tree results

The [structured qualification](../benchmark-results/structured-bounded-20261003/report.md) completed all three candidates. The later [source-lookup qualification](../benchmark-results/structured-source-lookup-20261003/report.md) includes the final original-source search and exact excerpt-span implementation, completing both candidates.

| Candidate | Complete | Quality | Total seconds | Known backend input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| Automatic local Gemma → Nemotron, structured tree | 1/1 | 0.667 | 69.86 | 12,630 | 5,114 | 17,744 |
| Spark Gemma verified → Nemotron, structured tree | 1/1 | 1.000 | 45.77 | 16,434 | 4,271 | 20,705 |
| Local Gemma batched → Nemotron, structured tree | 1/1 | 0.667 | 102.11 | 16,259 | 7,584 | 23,843 |
| Automatic local Gemma → Nemotron, **original-source lookup** | 1/1 | **1.000** | 73.09 | 12,601 | 5,171 | 17,772 |
| Nemotron verified → Nemotron, original-source lookup | 1/1 | 0.667 | 11.98 | 7,932 | 252 | 8,184 |

These five attempts all completed, and none scored below 0.667. Samples are too small to infer a general quality gain. Some configurations still miss manager approval because extraction omits it and the final model does not request additional raw evidence. Making raw evidence available is insufficient if the model declines to inspect it.

The automatic case demonstrates the intended bounded tree: **13 observations → groups of 8 and 5 → 2 states → 1 state**. Three reducer calls produced a 398-byte final state, with every state at most the configured 512-byte cap. The batched case processed **34 observations → groups of 8/8/8/8/2 → 5 states → 1 state**, with six reducer calls and a 469-byte final state. Every source partition was mapped: automatic 6/6, batching 12/12. Temporary Markdown cleanup was confirmed for every structured candidate.

The first native implementation relied on the model honoring a byte-size instruction and failed when valid JSON exceeded it. The final implementation enforces the invariant in code: normalize/deduplicate, rank complete observations by relevance, retain those that fit, and record pruned counts plus an unresolved-detail marker. Raw evidence is never pruned. Invented references, malformed/truncated output and exhausted graph budgets still fail visibly.

In the final automatic case, the root retained just one evidence reference. Lookup recovered both requested source blocks and extracted records within a 1536-byte excerpt window; the final answer passed all three checks. A separate deterministic test proves that an unmapped UTF-8 fact can be found in raw source Markdown and that the returned excerpt resolves to the exact original bytes.

## Stage performance

Numbers below are measured mean phase-wall milliseconds converted to seconds. Calls include repair work. Phase spans can overlap—for example verification can trigger another extraction—so summing phase times need not equal request latency. Per-call mean/P50/P95, queue-inclusive elapsed time, model assignments, failures and token subtotals are in each detailed report and `analysis.json`.

| Native case | Stage | Calls | Phase seconds | Input tokens | Output tokens |
|---|---|---:|---:|---:|---:|
| Final automatic/source lookup | Laya CPU decision | 1 | 0.337 | 289 | 0 |
| Final automatic/source lookup | Extraction + recovery | 9 | 49.393 | 8,498 | 3,701 |
| Final automatic/source lookup | Structured reduction | 3 | 18.107 | 1,863 | 1,330 |
| Final automatic/source lookup | Evidence lookup | 2 | 3.661 | 1,312 | 106 |
| Final automatic/source lookup | Final synthesis | 1 | 1.539 | 928 | 34 |
| Spark Gemma verified/structured | Extraction + recovery | 11 | 39.292 | 10,427 | 3,689 |
| Spark Gemma verified/structured | Verification | 4 | 6.576 | 3,908 | 247 |
| Spark Gemma verified/structured | Structured reduction | 1 | 3.198 | 397 | 230 |
| Spark Gemma verified/structured | Evidence lookup | 2 | 1.825 | 1,067 | 71 |
| Spark Gemma verified/structured | Final synthesis | 1 | 0.592 | 635 | 34 |
| Local Gemma batched/structured | Extraction + recovery | 12 | 61.536 | 10,799 | 4,746 |
| Local Gemma batched/structured | Structured reduction | 6 | 38.092 | 3,823 | 2,743 |
| Local Gemma batched/structured | Evidence lookup | 2 | 1.429 | 1,026 | 62 |
| Local Gemma batched/structured | Final synthesis | 1 | 1.005 | 611 | 33 |
| Nemotron verified/source lookup | Extraction | 6 | 6.888 | 5,687 | 141 |
| Nemotron verified/source lookup | Verification | 1 | 1.485 | 1,020 | 32 |
| Nemotron verified/source lookup | Evidence lookup | 2 | 2.340 | 797 | 46 |
| Nemotron verified/source lookup | Final synthesis | 1 | 1.256 | 428 | 33 |

Fixed policies skip decision inference. In the automatic case, Laya proposed evidence mapping with confidence **0.3306**, below the configured 0.7 threshold, and abstained. The compiler's feasible fallback selected **batched_map**, direct was infeasible, and divide-and-conquer actually executed. This is not a claim of a confident correct Laya recommendation. Decision inference used about 0.46% of that request's elapsed time; extraction and reduction dominate this workload.

Dollar prices were unavailable and remain **unknown**. Tokens include all reported started work, including invalid artifacts and repair. Missing usage is counted and makes known totals lower bounds. Laya classifier tokens are separate from backend tokens; zero Laya output tokens does not mean zero CPU work. Different backend tokenizers, hardware and electricity prevent interpreting these totals as dollar costs or native-token-equivalent throughput.

## Validation and remaining limits

**Final validation: 236 unit tests and 12 local HTTP integration tests passed; Ruff and diff checks passed.** Unit checks cover multi-level reduction, hard input/output state bounds, reference validation, deterministic oversize pruning, deduplication with merged provenance, raw-source lookup without mapper records, UTF-8 excerpt spans, temporary-file integrity/lifetime, cancellation, helper model permissions, budget rejection before dispatch, stage-power ordering, direct SSE, cost/rating controls, and bounded re-extraction/re-verification. Local HTTP integration uses injected decisions and mock backends; native qualification uses resident Laya and the actual three Docker endpoints.

Structured state remains lossy. Ranking is lexical and heuristic; a small model can misclassify evidence or omit a qualifier. Lookup has finite rounds and bounded windows, and repeated requests can retrieve the same material. Call/deadline/input/output ceilings still apply; the tree is not an unlimited processing promise. The implementation covers explicit source-backed text, not arbitrary conversation compaction, tools, Responses or exhaustive counting. Final public formatting remains enforced.

The next workload improvements should target mapper relevance/recall and avoiding repeated lookups before fine-tuning Laya: classifier overhead is small, and its current low-confidence abstention preserved a valid execution. The current checkpoint has not been fine-tuned or calibrated against these policies.

## Repeat the experiment

From the repository root:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites reliability --reduction structured --repeats 3 --warmup 0
```

For a quick native qualification:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites large-gemma,large-verified-spark-gemma,large-batched-gemma \
  --reduction structured --max-cases 1 --repeats 1 --warmup 0
```

Use the identical command with `--reduction rolling` to compare the alternative. New folders are created for each run; no deployment settings, model pulls, Docker services or runtime context settings are changed. The runner cleans up only its own Prism process and SSH tunnel.
