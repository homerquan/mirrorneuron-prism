# Execution policies and required Laya routing

Install with `python -m pip install .` from the checkout, or `python -m pip install mirrorneuron-prism` after publication. Laya installs as a required dependency. Keep physical model names, URLs, limits, capabilities, and prices in the raw-model JSON; individual policy profiles belong in `profiles/`. The multi-alias examples below use the bundled `src/prism/resources/prism.json` configuration.

```sh
export PRISM_API_KEY='your-configured-secret'
prism validate --config src/prism/resources/prism.json
prism policies
prism serve --config src/prism/resources/prism.json
```

The service prepares `convaiinnovations/laya-typed-decisions` on CPU before accepting requests. Its first startup may download weights. Set `decision.model` to a prepared local checkpoint for offline use, or pin `decision.revision` and optionally `expected_sha256` file hashes for reproducibility. Installation/configuration commands do not load weights.

An LLM call follows this flow:

```text
Client: POST /v1/chat/completions, model="prism"
                         |
               Authenticate + validate
                         |
      Preserve original messages and source bytes
                         |
       Compile finite plans; remove those exceeding
        capabilities, context, coverage, or budgets
                         |
           Several eligible? --- no ---> sole plan
                         |                  |
                        yes                 |
                         |                  |
              Resident CPU Laya             |
        bounded instructions + plan metadata |
                         |                  |
       eligible, confident, untruncated?     |
                /                \          |
               yes                no        |
         proposed plan      rules fallback  |
                \                /          |
                 +--------------+-----------+
                                |
              Reserve the entire chosen graph
                                |
          +---------------------+-----------------------+
          |                     |                       |
        DIRECT              MAP POLICIES          RETRIEVE_READ
          |                     |                       |
   original messages     source partitions      lexical span selection
          |                     |                (focused coverage only)
          |              +------+---------+             |
          |              |                |             |
          |       evidence/verified     batched          |
          |        extract per part  extract per group   |
          |              +--------+-------+             |
          |                       |                     |
          |           validate every partition          |
          |             and original quote span         |
          |                       |                     |
          |         verified only: independent checks   |
          |                       |                     |
          |             synthesize ALL records          |
          |                       |                     |
          +-----------------------+---------------------+
                                  |
                  Validate final response/schema
                                  |
                  One assistant response + trace
```

`N` is the number of lossless source partitions; `M` is the number of context-fitting groups after batching. Calls below count physical LLM requests, excluding the local CPU decision.

| Policy / fixed alias | Physical calls | Suitable workload | Coverage |
|---|---:|---|---|
| `direct` / `prism-direct` | 1 | Self-contained requests that fit; caller tools and direct-only parameters | Full original request |
| `evidence_map` / `prism-evidence` | N + 1 | Parallel evidence extraction across large explicit sources | Every required partition |
| `batched_map` / `prism-batched` | M + 1 | Several short documents; reduces extractor call overhead | Every required partition |
| `verified_map` / `prism-verified` | 2N + 1 | Interpretations needing an independent source check | Every required partition and record |
| `retrieve_read` / `prism-retrieve` | 1 | Focused lookup with strong lexical matches | Selected original partitions |
| `draft_review` / operator-configured alias | 3 | Source-free tasks benefiting from drafting and critique | Full original request |
| `vision_synthesis` / operator-configured alias | 2 | Image-capable worker → text/reasoning final | Unverified visual observations |
| `text_synthesis` / operator-configured alias | 2 | Plain-text worker prepares explicit source context → final | Unverified text observations |

Use `model: "prism"` to let Laya select a feasible policy; use a fixed alias for controlled comparisons. `X-Prism-Policy` reports the selection, `X-Prism-Coverage` identifies focused/full coverage, and the authenticated trace records eligible/rejected plans, physical calls, partition coverage, decision probability, and timing. Fixed aliases still require startup preparation, but skip decision inference when there is only one eligible policy.

Configure allowed plans and stage models explicitly:

```json
{
  "profiles": {
    "prism": {
      "direct": "small",
      "worker": "small",
      "synthesizer": "small",
      "verifier": "small",
      "strategy": "auto",
      "allowed_policies": ["direct", "evidence_map", "batched_map", "verified_map"],
      "batch_max_partitions": 4,
      "coverage": "exhaustive"
    },
    "prism-lookup": {
      "direct": "small",
      "strategy": "retrieve_read",
      "coverage": "focused",
      "retrieval_top_k": 4
    }
  },
  "decision": {
    "mode": "route",
    "model": "convaiinnovations/laya-typed-decisions",
    "min_option_confidence": 0.7
  }
}
```

Merge these fields into the complete generated configuration and reference IDs in your raw-model JSON. `verifier` defaults to the worker, then direct model. `shadow` can record Laya proposals while retaining rules selection; Laya remains required, and there is no `off` mode.

All map policies validate each partition's completion, needs, and exact original quote spans. Batching cannot omit a partition or borrow its neighbor's quote. Verification checks all record IDs and fails on unsupported facts, issues, missing IDs, or truncation; it is a model judgment and does not prove correctness or recall. Without optional reduction/compaction, final synthesis keeps every validated record or fails visibly if they exceed context. Configured structured reduction carries bounded observations and pointers into synthesis and retrieves raw evidence from temporary Markdown files. It is lossy and does not prove exhaustive recall. Retrieval deliberately selects a subset, never promises exhaustive coverage, and rejects a query with no lexical matches. Counts and exhaustive inventories require a fitting direct route.

Laya receives bounded instruction samples and structural metadata, without source text. Its option probability must pass the configured threshold, and truncated decisions cannot select a policy. Rejected choices and inference failures use the feasible rules fallback; model-loading failure prevents startup. Probabilities have not been calibrated for these policies, so evaluate routing on your workload before interpreting them as quality evidence.

Profiles with `optimization` evaluate physical model assignments as well as policies. They give Laya the best assignment per policy and apply its recommendation as a 10% contribution to power scoring, then select deterministically according to `context_management: [{"prism_cost_priority": 0.8}]`. Calls can narrow permitted models/policies and tighten budgets. Direct-only requests still select direct models. See [model optimization](model-optimization.md) for JSON metadata, controls, exact scoring, and trace details. The flow diagram above describes legacy policy routing; optimized profiles insert cost/power ranking after the Laya recommendation.

Explicitly permit `draft_review` to use a three-node draft → review → synthesis graph for source-free text. It uses the worker, verifier, and synthesizer assignments; all may share a backend. Original messages are retained. Reviews return typed `issues`/`suggestions` arrays; internal artifacts are untrusted, byte bounded, and never streamed to the caller. Both internal stages use `worker_output_tokens`; `intermediate_max_bytes` defaults to 4096 including JSON artifact serialization. Preflight reserves every stage before dispatch. Invalid, truncated, or failed intermediate results abort synthesis. Existing direct-only and exact-count requirements still apply.

`vision_synthesis` and `text_synthesis` accept bounded plain-text worker observations, so a worker without JSON support can participate. Caller JSON requirements go to the final model; `structured_output_model` can explicitly select that model. The vision worker receives original images; final synthesis receives original text instructions and image placeholders plus observations. Text preparation removes explicit `<prism-source>` blocks from final input and replaces them with the worker's notes; source-free prompts retain original text. These policies check completion and limits, but do not validate facts against source quotes. Both stages are reserved before dispatch, and worker failure, refusal, truncation, empty output, or excessive intermediate size aborts the graph. Response coverage is `visual_observations` or `text_observations`, and streams are buffered until final validation. Direct-only features other than explicitly permitted image handoff still require a fitting direct route.

Compare policies with the same saved reference suite:

```sh
prism benchmark run --config src/prism/resources/prism.json --candidate prism-batched \
  --out-dir benchmark-results/batched-a
prism benchmark run --config src/prism/resources/prism.json --candidate prism \
  --out-dir benchmark-results/laya-auto-a
prism benchmark compare benchmark-results/batched-a benchmark-results/laya-auto-a \
  --out-dir benchmark-results/batched-vs-auto
```

See [benchmark metrics](benchmarking.md) and [tested curl examples](flagship-curl-cases.md). A smaller graph is a potential overhead saving; actual speed, reference quality, and declared cost depend on model behavior and must be measured.


Optional `worker_fallback` adds one reserved extraction retry per invalid partition. Verified mapping may re-extract a rejected interpretation once and re-check it independently. Replacement record IDs, counts and digests are traced; the original source remains immutable. Empty record sets need no model judgment and are explicitly labeled, without asserting recall. JSON-capable backends receive constrained intermediate schemas. Aliases for the same physical endpoint/model share a concurrency semaphore. Operator `enable_thinking` controls backend template thinking unless a caller supplies direct reasoning controls.

[Structured reduction](structured-reduction.md) replaces the single large reducer with a bounded tree and internal raw-evidence lookup. When enabled, verification uses one-record packets, with exact quote provenance checked by Prism and no duplicate quote copy. Oversized surrounding context can narrow to a labeled quote window; the original bytes remain available. All possible helper calls are reserved upfront and included in benchmark stage timing/cost.
