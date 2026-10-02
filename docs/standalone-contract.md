# Standalone Prism 0.2 contract

The architectural source is `prism_standalone_proxy_design.md`. The release implements direct execution, bounded map variants, focused lexical retrieval, and constrained Laya policy selection. Dynamic repair/joins and exact exhaustive reduction remain future work.

| Design boundary | Implementation |
| --- | --- |
| API ownership | Authenticated FastAPI service; `/v1/chat/completions`, `/v1/models`, metadata traces; one choice, text only |
| Backend adapter | Direct HTTPX OpenAI-compatible backend; explicit limits/capabilities, no automatic retries or redirect following |
| Public identity | Virtual alias returned in completions/chunks; physical model IDs in traces |
| Compatibility gate | Known parameter allowlist; untransformed direct-only tools/logprobs/bias/seed/reasoning; unsupported features rejected |
| Source preservation | Immutable request-local content, SHA-256 identity, UTF-8 half-open spans, explicit user source blocks |
| Plan compiler | Five finite policy templates; capability/context/coverage/resource preflight, checked vocabulary, dependencies, depth/node bounds |
| Evidence | Typed fact/quote artifacts; quote uniqueness and provenance checks; partition completion/truncation/needs checks |
| Finalization | All validated map artifacts included in one synthesis or visible failure; retrieval explicitly uses focused original spans |
| Resource ledger | Atomic reservations for call/input/output/cost bounds; pre-reserved finalization, deadline and cancellation accounting |
| Scheduling | Per-request bounded worker pool, per-backend semaphores, server admission backpressure; strict cross-request fairness is not claimed |
| CPU decisions | Required pip-installed Laya in a spawned CPU process; chooses only preflighted plans, bounded instruction sample, revision/question fingerprint, cancellation/reaping, rules fallback |
| Streaming | Direct SSE forwarding; adaptive and schema-constrained output buffered before emission; stable public identity and usage-only final chunks |
| Privacy | Metadata-only in-memory traces; no prompt/source logging, no cross-request artifact cache, credential-scoped access |
| Evaluation | Paired HTTP benchmark folders retaining failures, reference quality, latency, physical work/cost, and offline comparisons |

Exact exhaustive reduction, dynamic joins/repair, persisted artifact caches, concurrent saturation profiling, Responses, multimodality, and production tokenizer-specific adapters are not implemented. Exact count/exhaustive keywords in adaptive requests are conservatively rejected. This is a guard, not a general task-understanding classifier. A semantically exhaustive task without those keywords still requires operator/workload validation; map policies do not promise exhaustive extraction.

`auto` compiles every allowed feasible policy before Laya chooses among them. The fallback prefers direct when the original fits; otherwise batching for multiple documents, then evidence mapping. Focused profiles prefer retrieval when available. Only explicit sources and transformable text requests qualify for alternative plans. Fixed strategies select one guarded plan even for short sources and skip decision inference. Features requiring untransformed execution remain direct, and a profile excluding direct rejects them. Conversation histories involving assistant/tool messages are always direct. No route guesses source boundaries, silently truncates instructions, revives omitted conversation state, executes caller tools, or switches to an unconfigured endpoint.

`direct` uses one unchanged call. `evidence_map` uses one extractor per partition and one synthesis. `batched_map` groups partitions within context and configured batch limits, requires a complete typed result for each, and uses one synthesis. `verified_map` adds a verification call for every partition; every record ID must be checked without issues before synthesis. Rejected verification fails visibly and never rewrites/drops records. `retrieve_read` selects lexically relevant original partitions within the final context and answers in one call; it requires `coverage: "focused"`, exposes selected/unselected spans, and forbids inferring global absence from the subset. Zero lexical matches fail visibly. See [execution policies](execution-policies.md).

Laya is required at installation and prepares at startup; model-loading failure prevents readiness. `route` is the default and `shadow` records proposals without changing execution. An untruncated eligible choice needs a finite option probability at least `min_option_confidence`. That threshold is not calibrated answer correctness. Inference errors, rejected proposals, and abstention preserve the preflighted rules fallback. Deadline/disconnect cancellation kills the CPU worker, and subsequent decisions fall back until restart. Laya cannot change the graph, endpoints, budgets, coverage, or evidence checks.

Source preservation lasts until request execution completes. The release does not expose a persisted corpus store. The caller supplies all content on every request. A filesystem path in a message does not grant file access. JSON Schema validation accepts local pointer references only and does not fetch remote schemas. Artifacts and HTTP/SSE buffers are size bounded.

Workers may miss facts or misinterpret valid quotes. Provenance and processed-partition coverage are separate from answer correctness. Incomplete worker outputs/needs abort the request; this release does not invent dependency repairs. Adaptive output is useful for bounded evidence-backed summarization/QA after workload evaluation, not arbitrary proof or guaranteed global reasoning.

All prices and context/token bounds are declared by the operator. The ledger refuses hard dollar caps with missing prices; over-bound reported usage fails visibly. Unreported started work consumes conservative reservations and is traced as unknown. Local cancellation does not establish remote cancellation or zero provider cost. Admission semaphores limit backend dispatch but are not GPU-memory measurements.

The physical output cap includes backend reasoning tokens if the backend includes them in completion usage. Logical `usage` fields are byte-based diagnostic counts, fixed across routes. They must not be compared to billed model tokens or interpreted as token savings. There is no aggregate physical-token total that pretends different tokenizers are interchangeable.
