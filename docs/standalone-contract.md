# Standalone Prism 0.2 contract

The architectural source is `prism_standalone_proxy_design.md`. The implemented release is a conservative vertical slice of milestones A–C; D/E describe subsequent work, not advertised features.

| Design boundary | Implementation |
| --- | --- |
| API ownership | Authenticated FastAPI service; `/v1/chat/completions`, `/v1/models`, metadata traces; one choice, text only |
| Backend adapter | Direct HTTPX OpenAI-compatible backend; explicit limits/capabilities, no automatic retries or redirect following |
| Public identity | Virtual alias returned in completions/chunks; physical model IDs in traces |
| Compatibility gate | Known parameter allowlist; untransformed direct-only tools/logprobs/bias/seed/reasoning; unsupported features rejected |
| Source preservation | Immutable request-local content, SHA-256 identity, UTF-8 half-open spans, explicit user source blocks |
| Plan compiler | Fixed direct and evidence-map templates; checked operator vocabulary, dependencies, graph depth/node bounds |
| Evidence | Typed fact/quote artifacts; quote uniqueness and provenance checks; partition completion/truncation/needs checks |
| Finalization | All validated artifacts included in one synthesis, or visible failure; no hidden compression/drop |
| Resource ledger | Atomic reservations for call/input/output/cost bounds; pre-reserved finalization, deadline and cancellation accounting |
| Scheduling | Per-request bounded worker pool, per-backend semaphores, server admission backpressure; strict cross-request fairness is not claimed |
| CPU decisions | Optional pip-installed Laya in a resident spawned CPU process; bounded state, model revision/question fingerprint, cancellation/reaping, abstention without gate authority |
| Streaming | Direct SSE forwarding; adaptive and schema-constrained output buffered before emission; stable public identity and usage-only final chunks |
| Privacy | Metadata-only in-memory traces; no prompt/source logging, no cross-request artifact cache, credential-scoped access |
| Evaluation | Paired JSONL HTTP runner retaining failures and physical traces; no empirical performance claims |

Exact exhaustive reduction, lexical retrieval, dynamic joins/repair, select/verify, enforced learned policies, persisted artifact caches, full performance profiling, Responses, multimodality, and production tokenizer-specific adapters are not implemented. Exact count/exhaustive keywords in oversized adaptive requests are conservatively rejected. This is a guard, not a general task-understanding classifier. A semantically exhaustive task without those keywords still requires operator/workload validation; the evidence-map route does not promise exhaustive extraction.

`auto` selects direct when the original request fits the direct model's configured byte/token bound. Otherwise it uses evidence-map only for explicit sources and transformable text requests. `evidence_map` opts into the same guarded plan even for small explicit sources; features requiring an untransformed route remain direct. Conversation histories involving assistant/tool messages are always direct. No route guesses source boundaries, silently truncates instructions, revives omitted conversation state, executes caller tools, or switches to an unconfigured endpoint.

Source preservation lasts until request execution completes. The release does not expose a persisted corpus store. The caller supplies all content on every request. A filesystem path in a message does not grant file access. JSON Schema validation accepts local pointer references only and does not fetch remote schemas. Artifacts and HTTP/SSE buffers are size bounded.

Workers may miss facts or misinterpret valid quotes. Provenance and processed-partition coverage are separate from answer correctness. Incomplete worker outputs/needs abort the request; this release does not invent dependency repairs. Adaptive output is useful for bounded evidence-backed summarization/QA after workload evaluation, not arbitrary proof or guaranteed global reasoning.

All prices and context/token bounds are declared by the operator. The ledger refuses hard dollar caps with missing prices; over-bound reported usage fails visibly. Unreported started work consumes conservative reservations and is traced as unknown. Local cancellation does not establish remote cancellation or zero provider cost. Admission semaphores limit backend dispatch but are not GPU-memory measurements.

The physical output cap includes backend reasoning tokens if the backend includes them in completion usage. Logical `usage` fields are byte-based diagnostic counts, fixed across routes. They must not be compared to billed model tokens or interpreted as token savings. There is no aggregate physical-token total that pretends different tokenizers are interchangeable.
