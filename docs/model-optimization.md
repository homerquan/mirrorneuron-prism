# Model selection by cost and power

Prism can select physical models for each stage of a feasible plan. Enable this explicitly in a profile; profiles without `optimization` retain existing model assignments and Laya routing. The [example configuration](../examples/standalone/optimization/prism.json) uses illustrative ratings, prices, model names, and localhost endpoints. Replace them with operator-validated values before use. Existing deployment configuration is unchanged.

## Model JSON and profile configuration

Each allowlisted model needs `power_rating`, an integer from 1–10, and both `input_cost_per_million` and `output_cost_per_million` in USD per million tokens. Explicit zero prices are supported. Missing ratings/prices, unknown IDs, and duplicate optimization IDs fail configuration validation. Context, output, capabilities, conservative tokenizer bounds, and concurrency retain their existing meanings.

```json
{
  "id": "small",
  "name": "replace-with-small-model",
  "base_url": "http://127.0.0.1:8000/v1",
  "context_window": 32768,
  "max_output_tokens": 4096,
  "power_rating": 3,
  "input_cost_per_million": 0.10,
  "output_cost_per_million": 0.20
}
```

Add the following to a profile that already has valid stage references:

```json
{
  "direct": "small",
  "worker": "small",
  "synthesizer": "strong",
  "verifier": "strong",
  "strategy": "auto",
  "allowed_policies": ["direct", "evidence_map", "batched_map", "verified_map", "draft_review"],
  "optimization": {
    "model_ids": ["small", "strong"],
    "default_cost_priority": 0.5
  },
  "intermediate_max_bytes": 4096
}
```

The optimization allowlist governs every stage. Stage references must still name registered models, but optimized requests choose their actual assignments from the allowlist and check each model's output cap against that call's requested output size. A smaller legacy stage model therefore does not prevent a larger allowlisted model from serving a larger output request. Different stages may share a model. `draft_review` must be explicitly permitted; it is absent from the default policy allowlist. Retrieval still requires explicitly focused coverage. A fixed strategy restricts optimization to that policy; incompatible direct-only requests fail visibly.

## Per-call controls

Prism accepts the following Chat Completions request extension:

```javascript
context_management: [{ prism_cost_priority: 0.8 }]
```

`0.8` assigns 80% weight to cost and 20% to power, `0` favors power, `1` favors cost, and `0.5` balances them. This follows the array pattern of [OpenAI's Responses compaction setting](https://developers.openai.com/api/docs/guides/compaction); these `prism_*` keys are a Prism extension. The Responses endpoint and OpenAI conversation compaction are not implemented. Optional Prism intermediate evidence reduction is configured separately. A `type: "compaction"` entry returns `unsupported_feature`.

| Key | Values and effect |
|---|---|
| `prism_cost_priority` | Finite number from 0–1; booleans and numeric strings are rejected |
| `prism_model_ids` | Nonempty, unique subset of the profile's optimization model IDs |
| `prism_allowed_policies` | Nonempty, unique subset of the profile's allowed policies; cannot exclude a fixed strategy |
| `prism_max_cost_usd` | Positive dollar ceiling; cannot exceed an existing profile ceiling |
| `prism_max_calls` | Positive integer call ceiling; cannot exceed the profile ceiling |

Keys may share an object or be distributed across objects. Duplicate keys, unknown keys, malformed entries, and permission/limit expansions fail before dispatch. An empty array supplies no overrides. A profile without optimization rejects nonempty Prism controls. Prices, ratings, credentials, endpoints, coverage, and other profile limits remain operator-owned.

```python
response = client.chat.completions.create(
    model="prism-optimized",
    messages=[{"role": "user", "content": "Explain the tradeoffs of this design."}],
    max_completion_tokens=200,
    extra_body={
        "context_management": [
            {"prism_cost_priority": 0.8},
            {"prism_model_ids": ["small", "strong"]},
            {"prism_allowed_policies": ["direct", "draft_review"]},
            {"prism_max_calls": 3, "prism_max_cost_usd": 0.02},
        ]
    },
)
```

The controls affect only this request. Prism consumes them locally, omits them from upstream parameters, and excludes them from public logical usage. With no overrides, an optimized profile uses `default_cost_priority`, defaulting to `0.5`.

## Feasibility and deterministic ranking

Prism evaluates model assignments for every permitted policy, preserving capability, context, coverage, output, token, call, cost, and deadline checks. `optimization.enforce_stage_power_order` defaults to true: review/synthesis cannot use a lower rating than preceding stages; equal worker and harder-stage ratings are permitted only at the highest feasible final-stage rating. This admits all-strong or single-strength pools. Direct models are still ranked independently. Operators may explicitly disable this heuristic with false. Admission caches are request-local and include model identity; partitioning is computed separately for each worker. Cost is the sum of configured prices applied to conservative graph reservations. In particular, evidence synthesis and verification reserve their available context bounds; the score is not a forecast of actual prompt length or billing.

Model power uses these role weights:

| Policy | Model-power weights |
|---|---|
| Direct / retrieval | Final model: 100% |
| Evidence / batched mapping | Worker: 50%; synthesizer: 50% |
| Verified mapping / draft review | Worker: 25%; verifier/reviewer: 25%; synthesizer: 50% |

Normalize model power as `(model_power - 1) / 9`. Normalize cost with min/max scaling across all feasible assignments; equal costs normalize to zero. Keep the best assignment per policy before adding task fit. Laya receives those complete candidates, their stage model IDs/ratings, coverage, call counts, cost bounds, and the priority, alongside bounded instruction samples. Source text and credentials are excluded.

A confident, eligible, untruncated Laya recommendation in `route` mode gives its policy `task_fit = 1`; other policies receive zero. Rank with:

```text
power_score = 0.9 * normalized_model_power + 0.1 * task_fit
score = (1 - prism_cost_priority) * power_score
        - prism_cost_priority * normalized_cost
```

Select the highest score, then lower cost, higher model power, fewer calls, and stable policy/model IDs. At priority 1, task fit cannot override the cheapest feasible plan. Abstention, inference failure, invalid/truncated/low-confidence proposals, and `shadow` mode supply no fit bonus. A single eligible policy skips decision inference. Existing Laya startup, probability thresholds, deadline handling, and worker cancellation remain in effect.

Ratings, stage weights, and task fit are heuristics, not calibrated correctness or a claim that combining models increases measured quality. Declared cost bounds are operator assumptions, not provider billing guarantees.

## Draft, review, and synthesis

`draft_review` makes exactly three sequential calls for source-free, transformable text requests: the worker drafts, the verifier reviews, and the synthesizer answers. Each stage retains the original messages. The review must be typed JSON with `issues` and `suggestions`, both string arrays. Drafts and critiques are supplied as untrusted data, and critique is not treated as proof of correctness.

Both internal stages use `worker_output_tokens`. `intermediate_max_bytes` defaults to 4096 and caps both returned UTF-8 content and each artifact's complete JSON representation, including string escaping. Preflight includes the worst-case downstream message serialization bounds. The entire graph is reserved before drafting. Truncation, tools, refusals, malformed critique, oversized artifacts, failed calls, and context violations fail visibly without a replacement answer.

Tools, conversation histories, seeds, logprobs, token bias, reasoning controls, and existing exact-count/inventory restrictions retain direct execution requirements. Direct SSE uses the selected backend. Multi-stage SSE is buffered and publishes only the validated final answer; the caller's JSON/schema constraints apply to final synthesis.

Authenticated traces expose effective controls, feasible assignment count, cost range, best candidates, selected stage IDs, score components, Laya's recommendation, selection reason, and physical ledger calls. They omit prompts, draft/review content, sources, quotes, and credentials. Use `X-Request-ID` with `prism trace show` to inspect a request.


Configured `worker_fallback`, `evidence_compaction`, and `evidence_reduction` helpers are also governed by the request model subset. Bounded extraction repair and possible compaction/reduction/lookup calls are priced and reserved before dispatch. They can make a plan infeasible under tight request budgets. Evidence reduction requires its helper to remain permitted; rolling compaction/recovery are unavailable when excluded. See [structured reduction](structured-reduction.md).
