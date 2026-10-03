# Bounded structured reduction and evidence lookup

Add this operator-owned profile configuration to an evidence, batched, or verified mapping profile:

```json
{
  "worker": "small",
  "worker_fallback": "strong",
  "verifier": "strong",
  "synthesizer": "strong",
  "evidence_reduction": {
    "model": "small",
    "max_input_tokens": 8000,
    "output_tokens": 1024,
    "state_max_tokens": 1024,
    "fanout": 8,
    "max_calls": 16,
    "lookup_rounds": 2,
    "evidence_max_tokens": 1024,
    "verification_max_extra_calls": 32
  },
  "limits": { "max_calls": 128, "max_output_tokens": 262144 }
}
```

Use registered model IDs and workload-appropriate limits. The default deployment files are unchanged. This is an internal Prism map/reduce extension, independent of the public `context_management` cost/power controls and unsupported OpenAI conversation-compaction entries. An optimized request's model subset must include its configured reducer. Set `lookup_rounds` to zero to disable internal evidence requests. `evidence_reduction` takes precedence over legacy `evidence_compaction` on the same profile.

Mappings still return typed source-backed `quote`/`fact` observations. Prism validates completion and exact original spans, stores raw sources and records in a request-local temporary directory containing only Markdown files, and converts observations into:

```json
{
  "facts": [{"text": "Standard deployments require manager approval.", "evidence_refs": ["partition-0-record-0"]}],
  "events": [],
  "claims": [],
  "hypotheses": [],
  "conflicts": [],
  "open_questions": [],
  "evidence_refs": ["partition-0-record-0"]
}
```

Every reduce input/output uses this same state shape. Equivalent observations are deduplicated while merging references. Overlong initial observations retain a bounded excerpt, a raw-evidence pointer, and an explicit unresolved question. Reducers are instructed to preserve task-relevant facts, qualifiers, conflicting interpretations, and uncertainty while ranking background details lower. Actual model outputs must be untruncated, typed, and reference only their child states' evidence IDs. If valid structured output exceeds the state cap, Prism deterministically ranks observations by request relevance, prioritizes facts/conflicts, retains complete observations that fit, and records pruned counts plus an unresolved detail marker. Raw evidence remains intact and searchable. Grounded facts/events/claims need references. Hypotheses remain distinct from facts. These checks validate structure and provenance, not semantic correctness.

`state_max_tokens` caps the conservative serialized UTF-8 state bound, with a maximum of 4096. It is tighter than native tokenizer counts and excludes separately budgeted message/schema framing. Preflight may tighten the effective byte cap to ensure that two maximum-size child states fit the reducer input. Both the reducer and final model's declared token bounds govern this cap. The effective size is traced. Small 8K profiles may need a 512-byte state; raising the nominal setting cannot bypass the model context limit.

Prism groups states by the complete prompt's conservative token bound and `fanout`. It reduces independent groups concurrently, then repeats until one state remains. A singleton passes through without an extra call. Each successful level must reduce the number of states; an inability to merge two states fails before making an oversized call. `max_input_tokens` includes framing and the structured-output schema. `output_tokens` reserves each reducer generation. The maximum reducer-call pool, extra verification packets, lookup rounds, and final synthesis are reserved before any dispatch and included in cost/power admission. Exhausting a finite graph budget remains a visible failure. This removes the giant final reducer; it does not promise unlimited processing under fixed call/deadline limits.

The relatively stronger final model sees the bounded state and can request known evidence IDs or a short lexical query. Prism resolves only generated IDs and searches overlapping passages of the original source blocks plus mapped records in its request-local Markdown evidence. Passage ranking uses lexical overlap weighted by inverse document frequency; it can find several separated locations in one source. Whole-source handles expand to matching passages. Excerpts begin at the strongest matching line, with subsequent context included within the budget, rather than the first matching word in a whole document. References cannot open arbitrary files, URLs, or tools. Each excerpt includes original UTF-8 byte provenance and a generated Markdown line pointer; limited windows are explicitly marked.

When lookup is enabled, Prism first searches passages using terms from the original user request, so empty extraction cannot stop evidence recovery before any source lookup. This deterministic search uses the already-reserved evidence window and no additional model call. Setting `lookup_rounds` to zero disables it along with model-requested lookup.

Lookup rounds accumulate a request-local pool of distinct source spans. Repeated overlapping search hits do not crowd out fresh evidence. The next lookup and final synthesis receive a bounded working set drawn from that pool, with a representative for each nonempty lookup query when possible. Excerpts share the configured byte budget and at most eight enter a prompt. The pool is bounded by the pre-reserved lookup rounds, and original evidence remains intact in the temporary store. A tight working-set budget can still omit or shorten evidence; retention is not a guarantee of recall or semantic correctness. Traces report new/pooled excerpt counts, initial and subsequent working-set source spans, mapped quote spans, and final state/excerpt evidence spans without quote text or queries. Final synthesis explicitly considers applicable exceptions and policy replacements and preserves requested JSON object fields when their values are unknown. After the configured rounds, the final model answers in the caller's requested format. No user-facing tool calls or intermediate streaming are introduced.

For a `json_object` request, adaptive final synthesis uses an object-only JSON Schema when the final backend supports `json_schema`. This prevents JSON-mode backends from returning bare `null` instead of the requested object. The schema does not specify field names or answer values, and its framing is included in preflight context and resource bounds. Explicit caller schemas are preserved. Backends without JSON Schema support retain JSON mode and the object instruction; normal output validation still rejects an invalid result.

Temporary files are removed once the final prompt is assembled, or on failure/cancellation. No persisted corpus, database, or external storage is needed. Authenticated traces contain state sizes/digests, child counts, levels, generated record IDs, lookup sizes, and cleanup status; they do not contain source text, observations, queries, filesystem paths, or model intermediate contents. Benchmarks explicitly retain their own input/output artifacts as before.

With this mode enabled, verification is local to each returned observation. Exact quote provenance is checked by Prism first, and the verifier sees source text once rather than duplicating the quote in the record. A packet that still exceeds context narrows only surrounding text, retaining the full exact quote and labeling the limited source window. Large sets use additional pre-reserved one-record verification calls, instead of one large record-list prompt. Empty sets are explicitly labeled, and unsupported/malformed/truncated verification still fails or invokes the configured single re-extraction/re-verification attempt.

Reduction is lossy. References and lazy lookup recover retained or lexically matching evidence; they cannot guarantee that a mapper found every important fact or that a reducer retained every conflict. Exact counts and exhaustive inventories retain their direct-route restrictions. Tools, conversation histories, and direct-only parameters keep their existing execution requirements. This implementation targets source-backed text requests, not arbitrary long agent histories.

Repeat a native comparison with the [Docker/Spark runner](../examples/standalone/docker-spark/benchmark.py):

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites reliability --reduction structured --max-cases 1 --repeats 1 --warmup 0
```

Use the same suites and budgets with `--reduction rolling` for a controlled alternative. The report records separate reduction and evidence-lookup stage timing, calls and physical tokens, tree depth, maximum state bounds, evidence retrieval, completion, completed-answer quality, and cleanup. Unknown dollar prices stay unknown; tokens are a proxy for work, not hardware or electricity cost.

## Algorithmic precedents

Partitioned extraction and combination use the general map/reduce pattern described by [Dean and Ghemawat (2004)](https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/). Passage ranking applies the term-specificity/inverse-document-frequency principle of [Spärck Jones (1972)](https://doi.org/10.1108/eb026526), with a repository-specific smoothed score and bounded excerpt selection. These references credit established ideas; the context-bounded state, provenance, retention, and prompt logic are implemented in Prism. See [detailed attribution and evaluation provenance](native-context-validation-20261003.md#prior-work-and-attribution), including RULER and the llama.cpp APIs used by native qualification.
