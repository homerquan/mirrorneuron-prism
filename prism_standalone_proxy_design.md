# Prism: a standalone adaptive, long-context LLM proxy

**Design proposal — October 2, 2026**

Status: architecture and implementation specification, not implemented behavior or a performance claim. This design has no dependency on MirrorNeuron or OtterDesk and does not preserve their interfaces. Configuration examples and proposed CLI commands below describe the target implementation.

## 1. Product contract

Prism exposes an ordinary OpenAI-compatible LLM interface. Internally it compiles a request into a bounded inference plan that combines deterministic processing, a structured CPU decision model, small generative workers, and an optional stronger synthesizer/verifier. The client submits messages and receives one normal assistant message or tool-call response. The client does not submit a workflow or manage workers.

The objective is **the least expensive execution that meets the workload's quality and completeness requirements within a latency/resource envelope**, not maximum call count or maximum token compression.

Separate two goals:

- **Effective input capacity:** operate on a source corpus larger than any individual model's working context, through external storage and repeated access.
- **Reasoning quality:** allocate sufficiently capable models to subtasks, integration, and verification. More small-model calls do not establish equivalence to a stronger model on every problem.

A virtual context is not a larger native attention window. Prism does not enlarge model weights, positional capacity, output limits, or arbitrary global reasoning ability. It provides a different computation over the submitted context, with its own measured limits.

## 2. Repository baseline and reuse

Inspected Prism revision: `27f42dcf6e826e2c93f6f2d1e13d6895c29c94af`.

The repository's implementation-status document identifies a delivered CPU measurement slice and unfinished generative execution. `ExecutionEngine.run()` raises `NotImplementedError`; `ProxyRouterBackend.complete()` is also unimplemented. The existing `BudgetTracker` only limits a call counter. The classifier policy already separates scoring from execution authority and supports abstention and calibration-domain checks. [S1–S5]

Retain and adapt the useful classifier isolation, artifact provenance, calibration, configuration validation, measurement, and CLI facilities. Build the generative runtime as a new vertical slice. Replace the simple counter with a multi-resource execution ledger. Move backend-specific response objects behind an adapter rather than making them the core type system.

The status document's tests and measurements are repository-reported; they were not re-executed for this proposal. This document does not establish current end-to-end accuracy, speed, or full API compatibility.

## 3. Architectural decisions

### 3.1 Prism owns its API and execution policy

Use a thin ASGI HTTP service with a backend-neutral Python core. Implement a tested Chat Completions surface first, not a new general-purpose provider gateway. Initially support OpenAI-compatible backends directly. LiteLLM or TensorZero can be optional downstream adapters/gateways for provider diversity.

Only Prism expands the public request into a plan. Child calls go straight to physical backends, never back into Prism's public virtual model. Detect self-referential backend configuration and fail at startup. Backend retries must be visible to the execution ledger; do not stack invisible retry loops.

### 3.2 The model ID denotes a virtual model

Serve aliases such as `prism`, `prism-fast`, and `prism-careful`. An alias resolves to a server-owned policy, permitted backend pool, model capabilities, limits, tokenizer/accounting contract, and egress policy. Profiles are validated policies, not promises that every request becomes faster or more accurate.

Return the public alias as the model identity. Physical model IDs belong in the trace. Do not masquerade as a named upstream model while silently changing its inference semantics.

### 3.3 Evidence first; free-form summaries second

Keep the submitted source immutable and addressable. Workers return small typed artifacts with source references, not just prose summaries. Summaries are derived navigation views and are allowed when the task tolerates loss. They are never the only surviving representation of source evidence.

### 3.4 Finite operators before autonomous recursion

Start with a small set of validated plan templates. A planner may fill in questions and dependencies, but cannot invent executable operators, arbitrary tools, unbounded loops, or unrestricted code. Dynamic plans must compile to a checked, bounded graph.

### 3.5 CPU decisions are proposals, not authority

Use the existing structured-decision machinery behind an interface. A classifier predicts task features or proposes a next action. Deterministic policy checks enforce feasibility, budgets, completeness, scope, and calibration support. Unknown or out-of-domain cases abstain.

## 4. Runtime components

The request path is:

`API -> compatibility gate -> request contract/source arena -> strategy selection -> plan compiler -> bounded executor -> evidence reconciliation -> synthesis/verification -> response serializer`.

Shared services include a backend registry, admission/resource scheduler, usage ledger, tenant-scoped cache, and trace sink.

Suggested package structure:

```text
src/prism/
  api/             # Chat Completions, model discovery, SSE, errors, auth
  contracts/       # RequestContract, SourceRef, Plan, NodeResult, Usage
  decision/        # Existing CPU classifier adapted; rules and calibration
  context/         # Immutable arena, parsing, segmentation, span resolution
  planning/        # Operators, templates, compiler, dependency validation
  runtime/         # Scheduler, cancellation, reservations, execution state
  evidence/        # Typed records, coverage, joins, source checks
  backends/        # Direct OpenAI-compatible HTTP; optional adapters
  strategies/      # Direct, evidence-map, exhaustive-reduce, join, select
  telemetry/       # Trace records, metrics, privacy controls
  evaluation/      # Paired experiments and protocol conformance
  cli/             # Serve, validate, doctor, profile, trace, evaluate
```

These are boundaries, not a requirement for one class or process per folder. Keep deployment to one Prism service plus whatever model servers the operator chooses.

## 5. Public compatibility contract

### 5.1 Initial endpoints

Implement `POST /v1/chat/completions` and `GET /v1/models`. Support non-streaming and SSE, authenticated access, stable response/chunk IDs, finish reasons, errors, and one choice initially. Add `/v1/responses` separately only after implementing and testing its own contract. A JSON-shaped response is not sufficient to claim compatibility. [S8–S10]

Target client usage:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key=os.environ["PRISM_API_KEY"],
)

response = client.chat.completions.create(
    model="prism",
    messages=[
        {"role": "system", "content": "Answer using the supplied material."},
        {"role": "user", "content": "Find policy exceptions in the following corpus:\n..."},
    ],
    max_completion_tokens=2000,
)
print(response.choices[0].message.content)
```

No special client library, job submission, polling, or orchestration arguments are required. Optional debug/administrative endpoints remain outside the inference response contract.

### 5.2 Preserve instructions and message structure

Preserve role priority and message ordering. Pin system/developer instructions, the operative user request, output constraints, and valid tool-call/result relationships. Do not flatten a conversation into an undifferentiated string or promote extracted evidence to instruction authority.

Document-versus-instruction separation in arbitrary user text is intrinsically ambiguous. Prefer explicit content boundaries when available. When a request cannot be transformed without changing its operative instructions, choose an untransformed compatible route or reject it if no such route fits. Never silently truncate instructions. Inspect task-relevant instructions at the end as well as the beginning of a long user message.

Every child receives only the instructions required for its typed role. Child source material is untrusted data. Preserve source trust metadata through joins, retrieval, caches, and final synthesis.

### 5.3 Parameter behavior

The compatibility gate resolves each parameter as supported, direct-route-only, or rejected. Never accept and ignore a field.

- `response_format` and supported JSON Schema constraints apply to the public result; internal nodes have their own schemas.
- Output limits govern the public completion according to the published virtual-model contract. Separate internal budgets govern all child completions and backend reasoning tokens.
- Sampling parameters apply to the public generation stage; internal extraction/judging may use fixed settings under the alias's documented policy.
- Initially route `logprobs`, token-ID `logit_bias`, unusual modalities, and other unimplemented features unchanged to a capable backend, or return a clear unsupported-feature error. Token IDs and probabilities cannot be casually reinterpreted across models/tokenizers.
- Do not promise seed-equivalence to an upstream model after changing prompts, models, or call graphs.
- `n > 1` means multiple public choices, not the internal candidate count. Support independently and budget explicitly; initially direct-route or reject.
- Validate both maximum HTTP payload size and maximum admitted virtual input, independently of individual model context limits.

### 5.4 Tool calls

Client-provided tools belong to the caller. Prism returns valid selected `tool_calls` with correct IDs, names, argument strings, indexes, and finish reason. The caller executes them and sends tool results in its next request. Preserve `tool_choice`, named-tool constraints, and `parallel_tool_calls` semantics. [S10]

Do not execute application tools during candidate evaluation. Internal workers must not send email, update databases, or run a caller's tools. Internal source reads and deterministic reducers are separately allowlisted runtime operations. A model-proposed tool call is not permission to execute it.

### 5.5 Stateless conversation semantics

A Chat Completions request is self-contained. Caches can accelerate repeated prefixes and unchanged source blocks, but correctness cannot depend on a hidden server conversation state. Never bring back a user-removed message through a cache. A path mentioned in text does not give Prism access to that filesystem path; source content must actually be present or resolved through an explicitly supported input mechanism.

## 6. Request understanding and the CPU decision layer

Classify the required operation, not only the content domain. Two questions about the same logs can require different algorithms: locating one event, counting every event, reconstructing a causal chain, or proposing a fix.

A proposed decision record:

```json
{
  "operation": "compare",
  "coverage_requirement": "exhaustive",
  "precision_requirement": "source_spans",
  "dependency_shape": "cross_document",
  "evidence_density": "unknown",
  "suggested_strategy": "evidence_join",
  "disposition": "abstain",
  "calibration_id": null
}
```

Useful inputs are request features, source manifest, bounded structural samples, workload family, observed worker results, unresolved obligations, current queueing, and remaining budget. A tiny controller need not read the entire corpus just to choose a plan. But an inference from partial samples must not authorize irreversible deletion of unsampled material.

Separate three concepts:

1. Option score: how strongly the classifier favors an option.
2. Calibrated selective risk: how often accepted decisions were wrong in a supported held-out regime.
3. Answer correctness: whether the end-to-end result actually satisfies the task.

None is automatically interchangeable. Preserve the existing abstention and calibration-domain safeguards. The repository's current status does not establish a validated low-risk execution gate. [S1, S4]

Initial routing should be rules-first with the CPU model in shadow mode. Evaluate alternate strategies on the same training/validation requests to obtain outcome-grounded route labels. Select the cheapest non-dominated adequate route, rather than training only to imitate a large model's opinion about routing.

Later, re-evaluate at useful decision points: after a map wave, after a failed source check, or when a join reveals a missing dependency. Legal next actions may include `ACCEPT`, `READ_MORE`, `EXPAND_NEIGHBORS`, `RETRY_NODE`, `VERIFY`, `ESCALATE`, and `FAIL`. A deterministic policy filters the set before a decision is applied.

Escalation must remain within configured model, network, residency, and credential boundaries. No automatic cloud call merely because local evidence is uncertain.

## 7. Virtual context and source preservation

### 7.1 Immutable source arena

For every submitted content block, retain original text plus message index, role, part index, content hash, trust scope, and structural metadata. Store decoded text without discarding its relationship to the submitted representation. Use an explicit offset convention; UTF-8 byte ranges are one workable choice. Maintain mapping when parsing or normalization creates derived text.

A source reference can be represented as:

```json
{
  "source_id": "msg-3-part-0",
  "source_sha256": "<actual-content-hash>",
  "byte_start": 1820,
  "byte_end": 1948
}
```

Ranges are half-open and must resolve against the exact immutable source version. Model-generated offsets are untrusted until checked. A matching quote proves provenance, not the truth of the interpretation.

### 7.2 Structure-aware partitioning

Partition documents at sections/paragraphs, emails at message/thread boundaries, code at files/symbols where available, tables at rows with headers, and logs at event boundaries. Include task-relevant local context such as section scope, negation, dates, units, definitions, and adjacent records. Very large units need their own subpartitioning rule.

For every backend call enforce:

`instructions + schema + source view + dependency artifacts + reserved completion <= usable model context`.

Count with that backend's tokenizer and chat template where available; otherwise use a conservative explicit bound. Leave safety margin. An output reserve includes backend reasoning tokens when applicable. Never rely on a model server to silently crop an oversized request.

### 7.3 Retrieval is one access path

Begin with exact/lexical retrieval and structural indexes. Add embeddings only when evaluation demonstrates incremental value. A graph may store explicit references, entities, or extracted relationships for join tasks; it need not be built for every request.

Retrieval can find a positive witness cheaply. Failure to retrieve a record does not prove it is absent. Questions containing all, every, none, never, counts, exceptions, or global comparison need suitable coverage obligations, not a fixed top-k cutoff.

### 7.4 Keep large intermediate state out of prompts

Workers write artifacts to the arena and return handles plus compact manifests. A reducer or synthesizer reads only the next required view. Do not concatenate every leaf response into a new oversized prompt.

When essential evidence itself exceeds the synthesis window, partition the answer obligations, create exact intermediate aggregates where valid, or use a bounded read-expand loop. Do not silently drop essential evidence to hit a fixed compression ratio. If no valid plan fits the request budget, fail visibly.

## 8. Evidence and task contracts

A task contract should include the answer obligations, required coverage, required precision, permitted approximation, output schema, and forbidden actions.

A worker receives one narrow question, source refs, dependency refs, role-specific instructions, and an output schema. It must be able to return an explicit unknown/incomplete result.

A representative result:

```json
{
  "node_id": "extract-17",
  "status": "complete",
  "records": [
    {
      "record_id": "r-17-3",
      "kind": "policy_exception",
      "fields": {"subject": "deployment approval", "approval_status": "unknown"},
      "source_refs": ["validated-source-span-reference"],
      "qualification": "approval was requested; granting it was not established"
    }
  ],
  "coverage": {
    "processed_partition_ids": ["partition-17"],
    "unprocessed_partition_ids": [],
    "output_truncated": false
  },
  "needs": ["Locate the response to this approval request"]
}
```

The runtime, not the worker's self-report alone, records whether a partition was submitted, completed without truncation, parsed, and validated. If extraction reaches its output cap, paginate or subdivide; do not treat the partial record list as complete.

Track source coverage, extraction recall, provenance validity, and answer correctness separately. Processing every partition is not proof that a model found every relevant fact.

## 9. Strategy library

| Strategy | Suitable request | Execution | Main guard |
|---|---|---|---|
| DIRECT | Small input, low expected benefit from orchestration, or direct-only feature | One capable backend call | Capability and context checks |
| RETRIEVE_READ | Sparse lookup or narrow QA | Search, read original spans, answer, expand on uncertainty | Search miss is not proof of absence |
| EXHAUSTIVE_REDUCE | Counts, extraction, categorization, complete inventories | Process every required partition, emit records, deterministic aggregation | Coverage, row identity, extraction overflow |
| EVIDENCE_JOIN | Cross-document comparison, code/spec checks, dependent reasoning | Extract facts/dependencies, join, inspect gaps, synthesize | Missing definitions and cross-partition relationships |
| HIERARCHICAL_SYNTHESIS | Summarization and broad thematic analysis | Task-aware local summaries/evidence, bounded hierarchical synthesis | Explicitly lossy; originals remain recoverable |
| SELECT_VERIFY | A difficult local inference where alternatives can be checked | Small bounded candidate set, verifier, select or escalate | Correlated mistakes and weak judges |

Do not implement six engines. These strategies compose a common operator vocabulary: parse, partition, search, map, extract, group, join, deterministic reduce, generate, select, verify, and render.

### 9.1 Correctness of decomposition

A good partition minimizes information that must cross partition boundaries. Independent record labeling is naturally partitionable. A proof, a repository-wide semantic change, or a contradiction involving distant definitions may not be.

Make dependencies explicit. Each subtask declares required inputs, produced artifacts, assumptions, and unresolved questions. Preserve ordering where it changes meaning. For code, include referenced interfaces or symbol bodies rather than asking a worker to infer absent behavior. Detect invalid graphs, cycles, unknown references, unsupported operators, and impossible budgets before dispatch.

A stronger planner/integrator can be valuable even when almost all scanning uses small models. Do not use a tiny classifier as a general-purpose decomposer of novel problems.

### 9.2 Deterministic reducers

Use code for exact arithmetic, sorting, grouping, deduplicating overlapping source spans, or checking schema constraints. Preserve real multiplicity: two equal-looking events are not necessarily duplicates. Deduplicate by stable record/source identity, not only text content.

Use valid sufficient statistics. A global average requires sum and count, not an unweighted average of partition averages. Deterministically combining model classifications does not eliminate classification errors.

### 9.3 Bounded adaptive execution

Execute dependency-ready nodes in limited waves. New nodes may be added only after validation and budget reservation. Set hard graph depth, node count, round count, and per-node output limits. Refine only failed or unresolved regions rather than rerunning the full corpus.

Stop when the task's obligations and validation checks are met, not when a model says it feels confident. If they cannot be met within the budget, choose a permitted fallback or return an explicit failure/qualified answer according to the request contract.

## 10. What to borrow from TensorZero and RLMs

TensorZero's best-of-N generates alternatives and chooses one with an evaluator; mixture-of-N combines alternatives with a fuser. Its function/variant and evaluation concepts are useful for separating an operation from competing implementations and measuring outcomes. These techniques do not by themselves guarantee long-context coverage or safe task decomposition. [S6]

Use candidate selection locally, for example for a disputed classification or two alternative integrations. Do not send the entire long corpus to N models by default. Prefer independently checkable results over unconstrained debate.

Recursive Language Models provide a closer conceptual reference for context handling: the prompt lives in an external environment that the model examines through programmatic access and subcalls. Prism can adopt that external-context idea while replacing unrestricted program generation with a finite checked operator runtime. The RLM paper also flags guardrails and exploding subcall costs as unresolved implementation concerns. [S7]

This combination is not a claim of an unprecedented research idea. Prism's useful differentiation would be an ordinary endpoint, conservative semantics, measured small-model economics, evidence-preserving processing, and bounded operational behavior.

## 11. Illustrative long-context execution

Suppose a request submits approximately 240,000 source tokens and asks which policy exceptions occurred and what evidence supports them. Assume an 8,000-source-token partition target and worker capacity that also fits instructions, schema, and output reserve. These are design examples, not measured optimal settings.

1. Parse the task contract, retaining the original instructions and all source material.
2. Produce roughly 30 structure-aware partitions; exact count changes with boundaries and overlap.
3. Run narrow extraction tasks on small workers. Each task returns records, source pointers, unknowns, and overflow status, not an essay.
4. Validate references and join records by subject, time, thread, or explicit identifiers.
5. Search/read additional original material for missing approval or exception dependencies.
6. Use a stronger model only for unresolved ambiguity and/or final integration when justified.
7. Build a bounded final evidence view and return the requested answer/schema.

The worker stage still processes roughly the original source volume plus prompt/overlap overhead. A smaller final prompt is not automatically lower total token usage. The potential gain is replacing expensive long-context reasoning/prefill with cheaper narrow processing while preserving enough information for integration.

The final model can be the same model as the workers. Supporting a stronger synthesizer improves flexibility; making one mandatory for every request undermines simple requests and all-small deployments. No quality-equivalence claim is justified until this configuration is evaluated.

## 12. Performance and scheduling

### 12.1 Model the actual graph

For approximately homogeneous map tasks, a useful planning approximation is:

`T ≈ T_ingest + T_decision + T_plan + ceil(partitions / effective_parallelism) * T_leaf + T_join + T_synthesis + T_verification`.

In production use queue-aware critical-path estimates and measured node distributions. Effective parallelism depends on physical capacity, batching, context length, KV memory, and contention. Eight asynchronous HTTP requests are not eight independent GPUs.

Track latency and cost separately. A strategy can reduce monetary cost while increasing latency or total tokens. Compare the quality/latency/cost Pareto frontier instead of collapsing everything into one unsupported speedup number.

### 12.2 Optimize work before concurrency

Use deterministic parsers for machine-readable data. Bypass orchestration for simple requests. Limit internal output to typed facts and necessary qualifications. Re-read only changed/uncertain regions. Reserve a small selective-verification path rather than judging every record with another model.

Tune partition sizes and concurrency jointly per workload and backend. Smaller chunks improve local focus but increase duplicated instructions and boundary/join work. Larger chunks reduce overhead but may increase prefill, KV use, and extraction failures. Fixed universal thresholds are not justified.

### 12.3 Caching and resident models

Cache segmentation and source indexes by immutable source hash and parser version. Cache task-dependent artifacts only when the task, instructions, schema, prompt, model revision, inference settings, source scope, and tenant match. A previous query's summary is not a general reusable interpretation of the corpus.

Put genuinely shared prompt material before varying partition content when doing so preserves instructions and improves backend prefix reuse. vLLM's prefix cache reuses common-prefix KV computation; it accelerates prefill rather than generated-token decoding. Do not assume cross-model KV reuse. [S11]

On a constrained machine, prefer a resident worker model over repeated model swaps. Use measured per-backend concurrency and memory admission. CPU classification still competes for CPU/RAM/bandwidth; measure end-to-end overlap rather than inferring savings from spare CPU utilization.

### 12.4 Fair admission

Use both per-request and per-backend limits. Schedule dependency-ready nodes fairly across requests; one huge map must not monopolize the service. Reserve capacity for finalization and respect end-to-end deadlines. Apply backpressure before allocating an enormous graph or copying a large source corpus repeatedly.

## 13. Budgets, cancellation, and failures

Use one request-scoped ledger, backed by atomic reservation operations. A reservation covers expected input, capped output, call count, deadline allocation, and estimated monetary cost where prices are known. Reserve finalization and fallback capacity before speculative work.

Maintain, per enforced dimension:

`consumed + outstanding_reserved <= request_limit`.

All retries, planners, verifiers, synthesis calls, and abandoned speculation count. Once a backend call starts, cancellation is not a reason to assume it consumed zero resources. Refund only unused/unstarted reservations according to the ledger's verified reconciliation rules.

Hard cost guarantees require trustworthy price and usage bounds. When usage/pricing is unknown, keep that uncertainty explicit and enforce conservative limits or reject a request requiring a hard dollar ceiling. Do not report missing usage as zero.

Propagate disconnect and deadline cancellation to every child and backend. Record cancellation confirmation separately from cancellation requested; remote work can continue despite local cancellation. Reject stale results from obsolete plan states while still accounting for their cost.

Use a failure matrix:

| Failure | Behavior |
|---|---|
| Unsupported request feature | Direct compatible route or 400 before work |
| Input exceeds admissible plan/resources | Clear resource/context error; no truncation |
| Leaf timeout or invalid JSON | Bounded retry/subdivision or permitted escalation |
| Missing exhaustive partition | Do not return a complete inventory/count |
| Incomplete evidence | Read/expand if budget permits; otherwise qualify or fail |
| Upstream authentication error | Surface sanitized error; do not silently switch scope |
| Client disconnect | Cancel children and retain accounting metadata |
| Failure after response commit | Terminate as failure; no replacement answer masquerading as success |

## 14. Verification and streaming

### 14.1 Verification priorities

Use schema checks, source-span validation, deterministic arithmetic, completeness checks, and domain-specific verifiers before generic LLM judging. Stronger model review is appropriate for ambiguous entailment/integration, but remains an imperfect check.

Independent evidence matters more than repeated agreement. Two models seeing the same incomplete extraction can agree on the same false conclusion. A correct synthesis cannot recover an essential fact permanently discarded upstream.

### 14.2 The commit point

Treat response emission as a commitment. In direct mode, forward a compatible stream. In adaptive mode, perform evidence preparation before final generation and stream only the final answer. Do not stream workers' provisional answers or internal reasoning into ordinary assistant content.

For requests requiring verified full output, buffer, validate/repair within budget, then emit the verified result as a compatible stream. This is buffered delivery, not low-latency token generation. For other requests, pre-validate evidence and stream final synthesis; do not claim the completed prose was fully post-validated before it was sent.

Never fake progress tokens to improve time-to-first-token. Measure first answer content, not an empty role chunk or SSE keepalive. Once content has been sent, do not silently replace it with a fallback answer. If final validation is mandatory, choose buffering before beginning the stream.

Respect usage-only final chunks when requested and the final completion marker. An interrupted stream may lack final usage; missing statistics are not evidence of zero usage. [S9]

### 14.3 Two ledgers, not misleading token savings

Publish a stable virtual-model accounting contract. Recommended default: normal `usage` fields describe the logical request and public completion under the alias's declared tokenizer/template. These are not the sum of backend bills and must not be marketed as physical compute usage.

An authenticated execution trace separately exposes actual per-call provider-reported input/output/cached/reasoning usage, retries, model identities, timing, and monetary cost. Preserve unknown/estimated status. Keep tokenizer-specific records; a cross-model token sum is a work-volume diagnostic, not a universal cost or context metric.

If a deployment instead exposes aggregate physical usage in standard fields, document that mode explicitly and test clients that use prompt usage for context management. Never switch accounting modes silently. Never expose only the final child's usage while claiming it is the whole execution cost.

## 15. Traceability and security

Trace the selected strategy, policy/calibration revision, plan/node IDs, dependency links, source coverage, validation failures, stop reason, backend IDs, token accounting, and stage timings. Log bounded structured decision reasons, not unrestricted hidden model reasoning.

Return a request/trace ID in a response header. Keep debug data outside standard completion bodies. Source and prompt logging should be separately opt-in with retention controls; metadata-only traces should be useful. Scope caches and traces by tenant and authorization boundary.

Bind locally by default or require properly configured authentication/TLS for network exposure. Model endpoints come from operator configuration, not URLs inside user messages. Keep provider credentials out of child prompts. Disable arbitrary file/network/shell access in the core runtime. Any optional execution verifier must run in a separately configured sandbox with explicit limits and no implicit access to the host environment.

Protect against malicious source instructions, tool-call pollution, cross-tenant cache reuse, enormous model-produced artifacts, pathological partition counts, and recursive endpoint misconfiguration. Treat workers' proposed source refs, graph edges, and actions as untrusted inputs.

## 16. Evaluation design

### 16.1 Baselines

Compare paired requests across direct small model, direct stronger model when the complete input fits, lexical retrieval plus synthesis, fixed evidence map/reduce, fixed best-of-N where relevant, and adaptive Prism. Truncated-input baselines must be labeled as truncated; never present them as full-context baselines.

Ablate rules versus CPU routing, retrieval versus exhaustive processing, summaries versus source-backed artifacts, deterministic versus LLM reducers, selective verification, caching, and concurrency. Evaluate the same source/answer obligations, output budget, and resource constraints. Do not compare a many-call expensive run only against a one-call impoverished baseline.

### 16.2 Benchmark families

Use RULER for controlled retrieval, multi-hop, and aggregation diagnostics. Use OOLONG for long-context classification/aggregation that requires more than finding one passage. Use a pinned LongBench v2 subset for realistic multi-document, code, and dialogue reasoning. Add task-specific code/spec and exact-count fixtures with executable or human-reviewed ground truth. [S12–S14]

Include answer-absent tasks, contradictory statements, late instructions, exceptions, chunk-boundary facts, cross-document dependencies, repeated-but-distinct records, unit changes, malformed inputs, and prompt-injection documents. Sweep source length and evidence density, not just one maximum-length example.

### 16.3 Metrics

Primary: end-to-end task success under the required completeness/precision contract.

Also report evidence recall, provenance accuracy, source coverage, false absence rate, structured-output validity, p50/p95/p99 total latency, first answer token latency, throughput under load, all child calls/tokens/cost, peak memory, controller overhead, retries, and cancellation waste.

Report uncertainty with paired confidence intervals and group related requests by source corpus to avoid pseudo-independent samples. Include failures and timeouts in the denominator. Pin dataset splits, models/revisions, quantization, tokenizer, server settings, prompts, operator versions, policy, calibration, and hardware.

Choose quality non-inferiority margins and performance targets before test evaluation. A proposed criterion could be no more than a two-percentage-point task-success loss at a one-sided 95% confidence bound while achieving a preregistered cost or latency improvement. This is an example acceptance rule, not a demonstrated result or a claim that the necessary sample size is small.

### 16.4 Protocol conformance

Exercise the ordinary supported SDK against the actual Prism server. Cover plain text, multi-turn messages, developer/system roles, JSON Schema, tool calls/results, streaming deltas, finish reasons, usage chunks, `n`, unsupported parameters, timeouts, cancellation, upstream failures, auth, and partial-stream errors. Unit tests that bypass Prism's HTTP request path do not establish gateway compatibility.

## 17. Delivery sequence

### Milestone A — trustworthy direct proxy

Implement backend-neutral contracts, one real backend adapter, auth, `/v1/models`, non-streaming/streaming Chat Completions, errors, usage accounting, cancellation, and direct-only parameter handling. Prove real requests reach a real backend and SDK tests pass. This is also the baseline route.

### Milestone B — evidence-preserving large-input execution

Implement source arena, deterministic segmentation, bounded worker extraction, source checks, final context assembly, one synthesis stage, exhaustive coverage handling, and resource reservations. Start with a text-only evidence-map template. Include an exact aggregation template for count/inventory requests or keep those explicitly unsupported in the adaptive path.

Acceptance: process a documented source size beyond the worker window without any oversized child request or silent omissions, and demonstrate actual task results against transparent baselines. Do not advertise a million-token virtual context before admission, accuracy, and resource tests justify it.

### Milestone C — CPU decision layer in shadow, then enforcement

Reuse the existing classifier infrastructure. Record decisions without changing execution. Collect paired strategy outcomes, calibrate on held-out supported workloads, compare against rules, and enable only gates supported by evidence. Include the controller's runtime overhead in comparisons.

### Milestone D — bounded decomposition and selective verification

Add checked dependency joins, `READ_MORE`/repair decisions, local best-of-N, and permitted stronger-model escalation. Train/tune policies from measured failure patterns, not arbitrary agent roles.

### Milestone E — optimize and expand API support

Profile chunking, batching, prefix reuse, cache reuse, hardware overlap, fairness, and load. Add Responses API/multimodal features only with explicit adapters and conformance tests. Online bandits or learned scheduling come after reliable labels, safe exploration controls, and rollback.

Proposed CLI:

```text
prism serve --config prism.yaml
prism validate --config prism.yaml
prism doctor --config prism.yaml --probe-backends
prism profile --config prism.yaml --suite worker-shapes
prism trace show <request-id>
prism eval run --suite long-context --policy balanced
prism eval compare <baseline-run> <candidate-run>
```

## 18. The core recommendation

Build an **inference compiler behind an ordinary LLM endpoint**. The CPU model chooses among safe decisions; small models process narrow views; deterministic operators preserve structure and aggregate exact results; a capable synthesizer resolves the parts that genuinely require stronger reasoning. Keep the complete source recoverable, measure the complete computation, and use additional calls only when they earn their cost.

The first milestone that proves the idea is not a complex autonomous agent. It is a reliable proxy where direct execution and a source-preserving long-context plan are both real, measurable, and interchangeable through the same client API.

## Sources inspected

[S1] Prism, `docs/implementation-status.md`, revision `27f42dcf6e826e2c93f6f2d1e13d6895c29c94af`.

[S2] Prism, `src/litellm_multicall/engine.py`, same revision.

[S3] Prism, `src/litellm_multicall/backend.py`, same revision.

[S4] Prism, `src/litellm_multicall/classifier/policy.py`, same revision.

[S5] Prism, `src/litellm_multicall/budgets.py`, same revision.

[S6] TensorZero, `docs/gateway/guides/inference-time-optimizations.mdx`, revision `62eb8f63e8ec62018d70420dbf1a8c5d1c026315`; project README.

[S7] Zhang, Kraska, and Khattab, *Recursive Language Models*, arXiv:2512.24601, HTML version 3, including limitations.

[S8] OpenAI, *Chat API Reference*, inspected October 2, 2026.

[S9] OpenAI, *Chat Completions streaming events* and *Reviewing API usage and costs*, inspected October 2, 2026.

[S10] OpenAI, *Function calling*, inspected October 2, 2026.

[S11] vLLM, *Automatic Prefix Caching*, inspected October 2, 2026.

[S12] Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, arXiv:2404.06654.

[S13] Bertsch et al., *Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities*, arXiv:2511.02817; official Oolong benchmark site.

[S14] Bai et al., *LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks*, arXiv:2412.15204v2.

Source locator URLs:

```text
https://github.com/homerquan/mirrorneuron-prism/tree/27f42dcf6e826e2c93f6f2d1e13d6895c29c94af
https://github.com/tensorzero/tensorzero/blob/62eb8f63e8ec62018d70420dbf1a8c5d1c026315/docs/gateway/guides/inference-time-optimizations.mdx
https://arxiv.org/html/2512.24601v3
https://developers.openai.com/api/reference/resources/chat
https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events/
https://help.openai.com/en/articles/10478918-reviewing-api-usage-and-costs
https://developers.openai.com/api/docs/guides/function-calling
https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
https://arxiv.org/abs/2404.06654
https://oolongbench.github.io/
https://arxiv.org/abs/2412.15204
```
