# Native long-context validation — 2026-10-03

The updated structured-reduction path answered all six measured 32K task families and three additional 64K examples using one Nemotron model with an unchanged 8192-token runtime slot. The native backend rejected every unchanged 32K and 64K request. This demonstrates larger effective source processing through multiple bounded calls on these examples; it does not increase the model's native window or establish unlimited context.

## Implementation

The improvements apply to map profiles with `evidence_reduction` enabled. The original Docker/Spark runner already enables structured reduction by default; the repository's deployment configurations are not changed by this qualification.

- Original sources are indexed as overlapping passages. Lexical ranking downweights common distractor terms, finds separated locations in one source, and returns exact original UTF-8 spans.
- A deterministic search from the user's question seeds lookup before a model can stop searching an empty extraction. Later rounds retain earlier evidence in a bounded pool and share a bounded final prompt window.
- Final synthesis instructions consider applicable exceptions and effective revisions. When supported, an object-only schema enforces the caller's JSON object contract without supplying field names or answer values.
- Traces expose mapped, retrieved, and final evidence spans, so reference recall can be measured independently of answer correctness.

The [reduction documentation](structured-reduction.md) describes admission, graph budgets, provenance checks, and cleanup. The [new runner](../examples/standalone/docker-spark/native_context.py) adds native sizing and real oversized controls, while the original benchmark remains available for policy and model-assignment comparisons.

## Method

Every stage used `docker.io/ai/nemotron-3.5-lightning:latest` on the existing Spark Docker Model Runner. Native `/props` confirmed `n_ctx=8192`. The installed chat template and tokenizer sized the full original request, with thinking disabled. Two calibration prompts matched provider-reported prompt usage exactly. The model, source and runner fingerprints, configuration, cases, gold byte spans, responses, and traces are retained in each run folder. No weights, runtime contexts, or deployments were changed.

Six synthetic families exercise keyword distractors, distributed facts, exceptions, reference chains, effective revisions, and unspecified answers. Seeds vary entity IDs, approval roles, review intervals, and distractor metadata. Gold values and gold spans remain outside model packets. Multifact cases spread their facts across the archive; placement rotates their order. Grades require every requested field to match its independent reference, including JSON types and named null values. Failed jobs score zero.

Prism admission still uses conservative UTF-8 accounting. The Prism `direct` route therefore rejects some inputs that fit native context; those rejections are not a successful baseline. Separate native HTTP controls send the unchanged original request, check prompt counts and possible truncation, and retain actual backend errors. Adaptive synthesis's object-only schema is part of the candidate implementation; fitting-window quality differences cannot be attributed solely to partitioning.

## Final implementation results

| Native input target | Cases | Fully correct | Original native direct | Evidence recall in lookup / final window |
|---|---:|---:|---|---|
| 32768 | Six families, seed 47, end placement | 6/6 | 6/6 rejected with HTTP 400 context overflow | 100% / 100% |
| 65536 | Distributed facts, chain, unspecified answer; seed 47, middle placement | 3/3 | 3/3 rejected with HTTP 400 context overflow | 100% / 100% |

Measured 32K prompts contained 32775–32796 native tokens; 64K prompts contained 65547–65564. All observed stage prompt counts plus reserved output fit the 8192-token slot, with no missing stage usage. Complete gold spans were present in the final quoted excerpts, independently of retained state references. The 32K distributed case had zero mapped gold-span recall, yet lookup and the final evidence window contained both complete gold spans and the answer was correct.

The 32K run used 62–63 physical calls per answer, a median latency of 61.83 seconds, and a mean of 66348 reported input-plus-output tokens per correct answer. This was sequential inference with model concurrency one. Latency excludes fixture sizing, calibration, and native direct controls; tokens include the complete Prism graph. These are work measurements, not dollar or electricity estimates. Rejected direct requests provide no successful speed ratio.

The 64K run used 119 calls per answer, a median latency of 112.37 seconds, and a mean of 127504 reported input-plus-output tokens per correct answer. It checked only three families and is a scale spot check, rather than a complete six-family qualification at 64K. All nine final-run requests reported temporary evidence cleanup.

Artifacts: [32K report](../benchmark-results/native-context-32k-20261003/report.md), [responses and traces](../benchmark-results/native-context-32k-20261003/requests.jsonl), [native direct controls](../benchmark-results/native-context-32k-20261003/native-direct.jsonl), [manifest](../benchmark-results/native-context-32k-20261003/manifest.json).

64K artifacts: [report](../benchmark-results/native-context-64k-20261003/report.md), [responses and traces](../benchmark-results/native-context-64k-20261003/requests.jsonl), [native direct controls](../benchmark-results/native-context-64k-20261003/native-direct.jsonl), [manifest](../benchmark-results/native-context-64k-20261003/manifest.json). Run folders are retained in the testing workspace and ignored by Git; this document records their results for review.

## Development observations

The first passage-retrieval prototype passed 3/6 at 16K on seed 11. Question-driven seeding and explicit exception/revision instructions raised that development result to 5/6; JSON mode still returned bare `null` for the unspecified-answer case. A subsequent seed-29 middle-placement run passed 6/6 at 16K and 5/6 at 4K, where bare `null` recurred. This prompted the final object-contract fix, which passed the 32K unspecified-answer check.

These are development runs with changing implementations, rather than a controlled comparison against the original Git revision. Their failures remain available: [first prototype](../benchmark-results/native-context-16k-20261003/report.md), [seeded-search update](../benchmark-results/native-context-16k-v2-20261003/report.md), and [seed-29 run before the object-contract fix](../benchmark-results/native-context-final-20261003/report.md). The seed-29 native 4K controls were accepted with exact prompt counts; its native 16K controls were rejected with actual context overflow.

## Prior work and attribution

The implementation uses established decomposition and retrieval ideas. The references below identify conceptual precedents, related evaluation, and the runtime APIs actually used. The Prism changes and synthetic fixtures were written within this repository; no paper-specific implementation or external benchmark generator was ported for these runs.

- **Related evaluation foundation:** Cheng-Ping Hsieh et al. (2024), [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), COLM 2024; [official benchmark repository](https://github.com/hsiehjackson/RULER). RULER tests multiple-needle retrieval, multi-hop tracing, aggregation, and question answering at configurable lengths and complexities. Its retrieval/tracing categories parallel Prism's custom distributed-fact/chain checks. Current Prism results use independently written fixtures, with no official RULER code, generators, or datasets imported and no RULER score reported. Aggregation has not been implemented or run here. Extended qualification should use versioned official generators and independent references, including aggregation.
- **Decomposition precedent:** Jeffrey Dean and Sanjay Ghemawat (2004), [MapReduce: Simplified Data Processing on Large Clusters](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/), OSDI 2004. Partitioned extraction followed by combination follows the established map/reduce pattern. Prism's context-bounded prompts, recursive reasoning-state reduction, evidence provenance, and lookup pool are repository-specific applications of that general pattern.
- **Retrieval weighting precedent:** Karen Spärck Jones (1972), [A statistical interpretation of term specificity and its application in retrieval](https://doi.org/10.1108/eb026526), *Journal of Documentation*, 28(1), 11–21. Weighting rarer matching terms more strongly follows the inverse-document-frequency principle. Prism uses an adapted presence-based score: sum `log(1 + N / (1 + df(t)))` over matching unique query terms, where indexed units are source passages and mapped records. The smoothing and passage/window selection are implementation choices in this repository.
- **Runtime APIs used:** The [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) describes `/apply-template`, `/tokenize`, and `/props`. The installed Docker Model Runner's llama.cpp backend supplied those APIs for native prompt sizing and runtime-window checks. Prism's benchmark transport and calibration logic call that existing implementation; native tokenization is not a Prism tokenizer.

## Limits and verification

This is a small seeded qualification, with one example per family in the final 32K run. It does not establish general accuracy, confidence intervals, semantic entailment, exhaustive aggregation, or arbitrary conversation-history support. Lexical search can miss dependencies without shared terms, and tight evidence/state budgets can still discard qualifiers. Recursive reduction remains lossy. Dependency repair and semantic final-answer verification are future work. Larger source requests also consume more calls, time, and tokens; fixed graph budgets remain enforceable limits.

Regression tests cover missed extraction, early keyword distractors, separated passages, evidence retention across rounds, Unicode byte provenance, bounded excerpts, source tampering, scratch cleanup, cancellation, schema capability fallback, caller schema preservation, native sizing/calibration, actual runtime mismatch, and silent direct truncation detection. Verification passed: 255 unit tests and 12 local HTTP integration tests, lint, formatting, package build, and distribution metadata checks.

Repeat the native qualification using the [Docker/Spark instructions](../examples/standalone/docker-spark/README.md#native-context-qualification). Add representative held-out workload cases and more seeds/placements before choosing production limits.
