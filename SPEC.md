# SPEC.md — mirrorneuron-prism

**Status:** Implementation specification, v0.2  
**Date:** 2026-09-15  
**Project type:** A pip-installable Python package extending LiteLLM Proxy  
**Proposed distribution:** `mirrorneuron-prism`  
**Python import:** `mirrorneuron_prism`  
**LiteLLM provider prefix:** `multicall/`

> One ordinary API request enters LiteLLM Proxy. This extension turns it into several bounded calls through LiteLLM, selects or combines their results, and returns one ordinary completion.

This specification replaces the earlier standalone-gateway design. **Do not build another HTTP server, provider SDK layer, model registry, authentication system, database, or workflow engine.** The deliverable is a Python package, not a Rust service or a fork of LiteLLM.

The package name is proposed; this document does not claim that it is already published or available on PyPI. Installation examples describe the project to implement.

---

## 1. Product contract

The application's existing OpenAI-compatible client continues to call **the same LiteLLM Proxy**:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="<existing-litellm-virtual-key>",
)

response = client.chat.completions.create(
    model="smart-local",
    messages=[{"role": "user", "content": "Review this function for bugs..."}],
    max_completion_tokens=1500,
)

print(response.choices[0].message.content)
```

Only deployment configuration and, when necessary, the logical model name change. No new client SDK or request fields are required. An operator may retain an existing public model alias and change its backend to `multicall/<policy>`.

Example execution:

```text
Client: POST /v1/chat/completions, model="smart-local"
                         |
              Existing LiteLLM Proxy
        authentication, request policy, model routing
                         |
         CustomLLM provider: multicall/default
                         |
              MultiCall execution engine
                  /      |      \
           candidate A   B       C ...
                  \      |      /
            verifier / judge / optional fuser
                         |
              one assistant response
                         |
              Existing LiteLLM Proxy
                         |
                       Client

All candidate, judge, fuser, and fallback model calls:
     existing LiteLLM Router.acompletion(...)
                    -> LiteLLM provider adapter
                    -> configured local/cloud backend
```

“Transparent” means compatible request/response shapes and unchanged application control flow. It does **not** mean identical latency, token consumption, sampling distribution, or capability. Multi-call execution is not guaranteed to match a larger model.

## 2. Ownership boundaries

| Concern | Owner |
|---|---|
| HTTP routes, OpenAI response envelopes, SSE transport | LiteLLM Proxy |
| Provider APIs, credentials, provider-specific translation | LiteLLM |
| Physical model definitions and deployment pools | Existing LiteLLM configuration |
| Backend routing, cooldowns, deployment limits | Existing LiteLLM Router |
| Authentication, virtual keys, ordinary proxy policies | LiteLLM Proxy |
| Candidate generation plan, selection, fusion, stopping | This package |
| Per-logical-request expansion budgets and cancellation | This package |
| Child-call context and accounting integration | This package's small, version-tested LiteLLM adapter |
| Application tool execution | Calling application, never this package |

Do not duplicate LiteLLM capabilities merely to simplify implementation. Internal calls must reuse the existing router, not create a second router from copied configuration.

**Important integration boundary:** invoking `Router.acompletion()` in process is not equivalent to sending another authenticated HTTP request through the proxy. Do not assume that proxy-only authorization, guardrails, rate limits, or spend attribution automatically run again. The adapter requirements below explicitly cover that boundary. LiteLLM documents Router calls and proxy-only hooks as distinct interfaces. [2][4]

## 3. Integration decision: CustomLLM, not a replacement proxy

### 3.1 Required extension point

Implement a subclass of `litellm.CustomLLM`, registered using LiteLLM's documented `litellm_settings.custom_provider_map` configuration. Implement the asynchronous provider methods used for ordinary and streaming Chat Completions. LiteLLM documents both the registration mechanism and `acompletion`/`astreaming` interfaces. [1][3]

Ship importable instances:

```text
litellm_multicall.provider.multicall_provider
litellm_multicall.hooks.multicall_hooks
```

The provider owns expansion and response construction. The `CustomLogger` hook instance supplies authenticated request context, integration checks, and accounting/observability support. Hooks must not independently run a second copy of the inference plan.

Do not implement expansion as an ordinary success-logging callback. Do not assume a pre-call hook can return an arbitrary completed response and short-circuit every LiteLLM endpoint. Use the provider execution contract for this purpose.

### 3.2 Package loading

Configuration must refer to installed package modules, not require users to copy Python files next to `config.yaml`. LiteLLM's inspected loader resolves a dotted module/instance reference and falls back to normal Python module import when no corresponding local file exists. Verify this behavior against every supported LiteLLM release. [5]

No automatic monkey-patching on `import litellm_multicall`. Importing the package must not replace `litellm.completion`, `litellm.acompletion`, callback lists, or router methods.

### 3.3 Accessing the existing proxy router

Create `compat/proxy_runtime.py` as the only module allowed to resolve proxy runtime internals.

It must:

1. Lazily resolve the running worker's initialized router.
2. Reuse that exact router instance for every child call.
3. Fail clearly if no initialized router is available.
4. Revalidate references after an operator changes model configuration.
5. Support explicit router injection for tests and optional embedded use.

The inspected proxy source uses `llm_router` as runtime state, but this is **not a documented stable dependency-injection API**. A narrowly isolated, version-tested adapter may resolve it lazily from `litellm.proxy.proxy_server`. Do not import and capture its initial value before proxy startup. [6]

Illustrative adapter shape, not a substitute for version testing:

```python
def resolve_proxy_router():
    from litellm.proxy import proxy_server

    router = getattr(proxy_server, "llm_router", None)
    if router is None:
        raise IntegrationUnavailable("LiteLLM Proxy router is not initialized")
    return router
```

Prefer a documented router-injection interface if one becomes available. Do not silently fall back to constructing a fresh `Router` or calling raw provider APIs.

Worker initialization must be process-safe. LiteLLM's documented worker-startup hooks run **before** configuration loading; they cannot be assumed to provide an initialized router. They are optional for initializing package-local state, not required for the basic installation. [7]

## 4. Project and packaging requirements

### 4.1 Required deliverables

The repository must contain:

- A standard `pyproject.toml` project using a `src/` layout.
- A pure-Python wheel and source distribution.
- Editable installation for development and ordinary installation from a wheel.
- Packaged default prompts, policy schema, and `py.typed`.
- A small CLI for configuration validation and diagnostics, not serving HTTP.
- Unit tests plus tests against a real LiteLLM Proxy process.
- Example LiteLLM and MultiCall configuration files.
- README installation instructions and an explicit compatibility matrix.

Use standard Python build metadata and wheel/sdist tooling. [8]

### 4.2 Proposed project metadata

The following is the initial project skeleton. The broad LiteLLM dependency range is for bootstrapping only: **replace it with tested bounds before publishing**, and ship exact CI/deployment constraints. Do not imply that every LiteLLM 1.x version is compatible.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mirrorneuron-prism"
version = "0.1.0"
description = "Bounded multi-call inference as a LiteLLM Proxy extension"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "litellm>=1,<2", # Replace with compatibility-tested bounds before release.
  "pydantic>=2,<3",
  "PyYAML>=6,<7",
  "jsonschema>=4,<5",
]

[project.optional-dependencies]
proxy = ["litellm[proxy]>=1,<2"] # Use the same tested bounds as above.
dev = [
  "build",
  "twine",
  "pytest",
  "pytest-asyncio",
  "ruff",
  "mypy",
  "httpx",
  "openai",
]

[project.scripts]
mirrorneuron-prism = "litellm_multicall.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/litellm_multicall"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Add project ownership and the maintainer-selected license before release. Do not invent maintainer identities or upload credentials.

The base package must work when installed into an existing supported LiteLLM environment. The `proxy` extra installs LiteLLM's proxy dependencies for a fresh environment. Do not require Torch, Transformers, an agent framework, Redis, or a database just to import this extension.

### 4.3 Development and build workflow

From the project repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[proxy,dev]"

python -m pytest
python -m build
python -m twine check dist/*
```

After publication, the intended operator experience is:

```bash
python -m pip install "mirrorneuron-prism[proxy]"
```

Before publication, install the local wheel instead. Test a wheel in a clean environment **outside the repository**; an editable installation alone is insufficient evidence that package loading and resource inclusion work.

For production, install with the project's published, compatibility-tested constraints file. Never automatically upgrade the user's existing LiteLLM installation to an untested version.

### 4.4 CLI

Implement only these commands initially:

```bash
mirrorneuron-prism validate \
  --policy-config ./multicall.yaml \
  --litellm-config ./litellm.yaml

mirrorneuron-prism doctor \
  --policy-config ./multicall.yaml \
  --litellm-config ./litellm.yaml

mirrorneuron-prism --version
```

`validate` performs local syntax, references, cycles, budget, and compatibility checks. It makes no model calls.

`doctor` additionally checks import paths and reports the LiteLLM/Python versions and supported integration profile. A backend smoke test requires an explicit `--probe` flag because it sends data and may incur cost. Neither command starts another proxy.

## 5. Operator configuration

Use two files:

- `litellm.yaml`: existing physical models plus the custom-provider registration.
- `multicall.yaml`: package-specific execution policies referring only to LiteLLM model groups.

Provider names, API URLs, credentials, and physical-model pricing remain in LiteLLM configuration. Do not create a parallel provider registry in `multicall.yaml`.

### 5.1 LiteLLM configuration

The example below is a **trusted, local-only development profile**, not a ready-made paid/multi-tenant billing configuration. Replace the illustrative backend model ID with the ID actually served by the local endpoint.

```yaml
model_list:
  # An ordinary LiteLLM model group; keep existing definitions when available.
  - model_name: local-small
    litellm_params:
      model: openai/local-small-model
      api_base: os.environ/LOCAL_LLM_BASE_URL
      api_key: os.environ/LOCAL_LLM_API_KEY
      max_parallel_requests: 2
      num_retries: 0
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0

  # A logical model implemented by this package.
  - model_name: smart-local
    litellm_params:
      model: multicall/default
      num_retries: 0
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0

litellm_settings:
  custom_provider_map:
    - provider: multicall
      custom_handler: litellm_multicall.provider.multicall_provider
  callbacks:
    - litellm_multicall.hooks.multicall_hooks

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Merge the callback/provider entries with existing lists; do not replace the operator's other callbacks or providers.

**Zero-price warning:** LiteLLM documents that explicitly zero-priced models bypass monetary budget checks. The zero-price example is therefore restricted to the local development profile. Do not copy this wrapper-pricing pattern into a paid deployment without the child authorization, budget, and accounting bridge in section 15. [9]

### 5.2 MultiCall configuration

This is **this package's proposed configuration schema**, not an existing LiteLLM schema.

```yaml
schema_version: 1

runtime:
  backend: proxy_router
  integration_profile: trusted_local
  max_inflight_children_per_worker: 4
  max_inflight_children_per_request: 2
  child_cache: disabled
  outer_cache: disabled
  log_content: false

policies:
  default:
    strategy: best_of_n
    allowed_model_groups: [local-small]
    privacy: local_only

    candidates:
      - model: local-small
        samples: 4
        temperature: 0.7
        max_completion_tokens: 1500
        prompt_variants: [direct, independent_check]

    selection:
      type: llm_judge
      model: local-small
      max_completion_tokens: 160
      temperature: 0
      allow_abstain: true

    budgets:
      max_model_calls: 5
      max_backend_attempts: 5
      max_total_tokens: 16000
      deadline_ms: 120000
      child_timeout_ms: 45000

    on_unresolved: error
    api:
      unsupported: reject
      passthrough_model: local-small

prompt_variants:
  direct: |
    Produce a complete answer to the user's task.
  independent_check: |
    Check assumptions and likely edge cases before answering.
    Still return a complete answer, not a critique of an unseen answer.
```

Here, four candidate calls plus one judge call consume at most five model invocations before any explicitly budgeted retries. Candidate prompt variants cycle deterministically through the configured list.

`max_inflight_children_per_request: 2` deliberately avoids assuming that a local GPU can efficiently serve all candidates at once. Parallelism is a concurrency limit, not a promise of linear speedup.

### 5.3 Startup

After installing the project:

```bash
export MULTICALL_CONFIG="$(pwd)/multicall.yaml"
export LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_API_KEY="dummy"  # Only for a backend that does not require auth.
export LITELLM_MASTER_KEY="<set-a-real-proxy-secret>"

mirrorneuron-prism validate \
  --policy-config "$MULTICALL_CONFIG" \
  --litellm-config ./litellm.yaml

litellm --config ./litellm.yaml --port 4000
```

Use the existing proxy's normal virtual keys for clients. Install the extension inside the same Python environment or container that runs LiteLLM Proxy. The package must not introduce a second listener or mandatory launcher.

## 6. Execution engine and LiteLLM backend contract

### 6.1 Thin internal interface

The strategy engine depends on one injected backend:

```python
from typing import Any, Protocol
from litellm import ModelResponse

class CompletionBackend(Protocol):
    async def complete(
        self,
        *,
        model_group: str,
        messages: list[dict[str, Any]],
        parameters: dict[str, Any],
        context: "ChildContext",
    ) -> ModelResponse: ...
```

Production implementation: `ProxyRouterBackend`.

Core dispatch, after authorization, budgeting, and parameter preparation:

```python
response = await router.acompletion(
    model=model_group,
    messages=messages,
    stream=False,
    **safe_child_parameters,
)
```

`model_group` is an existing LiteLLM `model_name`, not a raw provider model ID. This preserves LiteLLM's configured deployment selection. Router model-group calls are a documented interface. [2]

Every model operation must use this backend, including judge, fuser, repair, optional classifier, and stronger-model fallback. The production engine must not instantiate `OpenAI`, call provider endpoints with `httpx`, or use a second `litellm.Router`.

Use an in-memory fake backend in unit tests. HTTP clients are permitted in integration tests and optional verifier adapters, not for implementing a competing provider layer.

### 6.2 Request and execution data

Define typed internal records:

```text
RequestContext
  logical_request_id, requested_public_model, policy_id, policy_version
  authenticated_principal_ref, effective_access, privacy_constraints
  deadline, validated_api_request

ChildContext
  logical_request_id, child_call_id, stage, candidate_id
  expansion_depth, trusted_attribution, reservation_id

Candidate
  candidate_id, model_group, actual_deployment_if_known
  complete_assistant_message, finish_reason, response_reference
  usage, duration_ms, validation_results, status

ExecutionResult
  final_assistant_message, finish_reason
  selected_candidate_id_or_fusion_id
  aggregate_usage, usage_completeness, aggregate_cost_if_known
  stop_reason, child_summaries
```

Do not copy opaque LiteLLM response/logging objects across child requests. Each call gets fresh IDs, a fresh parameter dictionary, and a deep copy of mutable message data.

The provider singleton stores immutable policy state and per-worker limits, not a mutable global “current request.” Execution state belongs to the request.

### 6.3 Trusted request context

The proxy hook obtains identity from LiteLLM's authenticated context, not from client-supplied `metadata`, `user`, headers, or tool arguments. The adapter must propagate an opaque reference into the provider execution path without exposing credentials.

Use explicitly passed context for child tasks. A `ContextVar` may assist correlation, but must not be the only security boundary: callback scheduling can cross task boundaries. Test real hook-to-provider propagation and cleanup.

Delete request registry entries on success, error, timeout, cancellation, and streaming disconnect. Do not store raw bearer tokens in logs, prompts, trace metadata, or policy files.

## 7. Execution strategies

### 7.1 MVP: `best_of_n`

1. Validate the incoming request and resolve the policy.
2. Reserve enough budget for the configured selection step before launching candidates.
3. Generate independent candidates with bounded concurrency.
4. Exclude transport failures, incomplete outputs when prohibited, malformed tool calls, and schema-invalid outputs.
5. Run deterministic task verification when explicitly configured.
6. If a conclusive semantic verifier passes, select a passing candidate according to policy.
7. Otherwise call the configured judge once over the eligible candidates.
8. Validate the judge's selection, then return the selected message **without rewriting it**.
9. If selection is unresolved, apply the configured fallback or return a standard error.

The default sample count is four, not a claimed optimum. Candidate independence means no candidate sees another candidate's output during generation.

If only one eligible candidate remains, a policy may allow `single_valid` degraded selection. Default to requiring the configured selector or returning an unresolved error; never silently equate “only surviving response” with “correct.”

### 7.2 MVP: `adaptive`

Provide a bounded, declarative sequence rather than a general DAG engine:

```text
1 initial candidate
       |
explicit semantic verifier passes? -- yes --> return
       |
       no / unknown / no verifier
       |
3 additional candidates (4 total)
       |
judge or task-specific consensus
       |
select / abstain
       |
optional authorized fallback; otherwise error
```

Example replacement policy body:

```yaml
strategy: adaptive
allowed_model_groups: [local-small, local-strong]
privacy: local_only

candidates:
  - model: local-small
    samples: 4
    temperature: 0.7
    max_completion_tokens: 1500

adaptive:
  initial_samples: 1
  expansion_batches: [3]  # Additional samples, not cumulative totals.
  early_stop: semantic_verifier_pass

selection:
  type: llm_judge
  model: local-small
  max_completion_tokens: 160
  allow_abstain: true

fallback:
  model: local-strong
  max_completion_tokens: 1500

budgets:
  max_model_calls: 6  # 1 initial + 3 extra + 1 judge + 1 fallback.
  max_backend_attempts: 6
  max_total_tokens: 22000
  deadline_ms: 180000
  child_timeout_ms: 45000

on_unresolved: fallback
```

`local-strong` must already exist in LiteLLM's model list. Fallback is optional and need not use a cloud model.

**No fabricated confidence:** a model saying “confidence = 0.95” is not a calibrated 95% correctness probability. JSON validity and an HTTP 200 do not establish task correctness. Without a semantic verifier or separately validated acceptance rule, the initial response must not automatically qualify for quality-based early stopping.

### 7.3 Next version: `consensus`

Use only with a configured answer extractor and normalizer appropriate to the task. Examples include a label, a number with specified tolerance, or canonical JSON.

Define agreement denominator as all valid generated candidates, require a configured minimum sample count, and specify tie behavior. Invalid outputs are not votes. Correlated agreement is not proof of correctness.

Do not normalize free-form architecture reviews into a majority answer with string matching. Tool-call normalization must preserve meaningful differences in arguments and ordering.

### 7.4 Next version: `mixture_of_n`

Generate candidates through the same backend, then call a fuser through LiteLLM to create a new final answer. Budget for `N + 1` calls before generation.

The fuser must preserve the original output requirements and source evidence. Its result is a new candidate and must be validated again. Do not merge incompatible tool-call plans; route tool-bearing requests to selection or explicit pass-through instead.

No recursive agent debates, autonomous tool loops, learned routers, or arbitrary graph execution in the first release.

## 8. Judge, validation, and verifier contracts

### 8.1 Judge behavior

The judge receives the original task, relevant original instructions, bounded candidate answers, and optional verifier summaries. Candidate IDs are opaque; omit model brands. Shuffle display order using a recorded seed and map IDs back correctly.

Request a compact structured result:

```json
{
  "winner_id": "c2",
  "abstain": false,
  "reason_code": "best_supported"
}
```

`winner_id` must reference a supplied eligible candidate. For abstention it must be null. Reject contradictory combinations, invented IDs, malformed JSON, and excessive output.

Judge prompts must explicitly treat candidate contents as untrusted data, not instructions to the judge. The extension must never obey a candidate's request to change tools, policies, models, keys, or budgets. Delimiting text is a defense layer, not a guarantee against prompt injection.

An invalid judge response must not default to the first candidate. An optional judge repair is a separate, budgeted call; the default policy reserves no such repair.

### 8.2 Three different kinds of checks

| Check | What it establishes | May trigger semantic early stop? |
|---|---|---|
| Response/schema/tool-argument validation | Output is structurally usable | No |
| Configured task verifier | The specified task checks passed | Only if marked conclusive for that task |
| LLM judge | A model prefers or rejects a candidate | Not a correctness guarantee |

MVP built-ins: JSON parsing/schema validation and tool-argument validation.

MVP task-verifier interface: an operator-installed Python callable, referenced by import path in trusted configuration. A verifier returns `pass`, `fail`, or `unknown`, plus a reason code and an explicit `conclusive` flag governed by its configuration.

The gateway must not execute user-generated Python, shell, SQL writes, or application tools in its own process. Executable testing belongs in a separate sandbox or externally managed verifier. An optional HTTP-verifier extra may be added later with endpoint allowlisting and timeouts.

Passing a test suite means those tests passed, not that all possible behaviors are correct. The operator is responsible for choosing checks that justify early acceptance.

## 9. API compatibility and parameter handling

### 9.1 Scope

Expansion applies to `POST /v1/chat/completions` and LiteLLM's equivalent Chat Completions route only.

Existing non-MultiCall models and unrelated routes must remain unchanged. `/v1/models`, health checks, authentication, and admin routes remain owned by LiteLLM. Do not add a second implementation of them.

For `multicall/*`, the first release supports text messages, normal multi-turn histories, one returned choice, JSON output, and function-tool proposals. Other APIs such as Responses, embeddings, realtime, audio, and image generation are not claimed as expanded by this package.

### 9.2 Explicit parameter matrix

| Field or behavior | Required handling |
|---|---|
| `messages` | Preserve roles, ordering, tool results, and original meaning. |
| `stream=false` | Return a normal LiteLLM `ModelResponse`. |
| `stream=true` | Use the buffered-final strategy in section 10. |
| `tools`, `tool_choice`, `parallel_tool_calls` | Preserve constraints; never execute client tools. |
| `response_format` | Enforce on candidates and any fused/fallback output. |
| `max_completion_tokens` / legacy `max_tokens` | Bound final candidate output separately from aggregate execution budget. |
| `temperature`, `top_p`, `stop`, penalties | Forward explicit client values where supported; policy defaults fill omissions. |
| `seed` | Derive recorded per-candidate seeds; document reproducibility limits. |
| `n=1` or omitted | One public choice; internal sample count comes from policy. |
| `n>1` | Explicit pass-through or 400; never reinterpret as internal Best-of-N. |
| `logprobs`, `top_logprobs`, `logit_bias` | Initially pass through or reject; do not invent cross-model token semantics. |
| Unsupported modality/provider-specific parameters | Explicit pass-through or 400 before any expansion. |

`api.unsupported` accepts `reject` or `passthrough`; the latter uses `api.passthrough_model` and records that no expansion happened. A pass-through still goes through LiteLLM and preserves the original request semantics, including streaming and `n`.

A client must not supply raw provider URLs, credentials, extra model groups, an arbitrary policy path, or a higher internal budget. Reject reserved package metadata and routing overrides. Do not forward proxy bookkeeping, credentials, pricing overrides, logging objects, or the parent's call ID into child provider parameters.

Do not globally enable parameter dropping. Request-critical constraints must not disappear because one candidate backend lacks a feature.

### 9.3 Messages and prompt variants

Keep the caller's system/developer instructions intact. Apply configured variants only as compatible supplemental instructions; never promote candidate text or retrieved documents into a privileged instruction role.

Each generated candidate must attempt the user's complete requested output. A “critic-only” response is not a comparable full answer in Best-of-N. Separate critique can be added as an explicitly different later-stage operation.

Build judge input separately. For example, the caller's code-oriented `stop` sequence or forced application `tool_choice` must not be blindly applied to the judge's JSON control output.

### 9.4 Tools and structured output

Select one complete assistant message. Preserve its tool names, argument JSON, tool-call IDs, and matching `finish_reason`. Validate named/required tool choices. Never combine tool calls from competing candidates into a larger, potentially side-effecting action.

On the next client request, accept the selected tool result and continue normally. Unselected tool calls must never have escaped to the client or run server-side.

Keep refusals and content-filter outcomes distinct from malformed output. Do not configure retries whose purpose is to evade safety policy. Mandatory guardrail rejection is terminal according to the operator's policy.

## 10. Streaming

### 10.1 Buffered-final semantics

For an expanded streaming request:

1. Generate and select internally without sending candidate text to the client.
2. Construct the final validated response.
3. Emit only that response through LiteLLM's custom-provider streaming interface.
4. Let LiteLLM own SSE serialization and stream termination.

Implement `CustomLLM.astreaming` using the streaming chunk contract supported by the pinned LiteLLM version, rather than yielding arbitrary dictionaries or manually writing SSE strings. LiteLLM exposes `GenericStreamingChunk` for this extension path. [1][3]

No fake progress tokens, internal reasoning, draft answers, or speculative output followed by a replacement answer. No artificial delays to simulate typing. Time to first output will include the hidden selection work.

### 10.2 Required stream tests

Verify role/content ordering, one final finish reason, stable public ID/model, Unicode, tool-call argument deltas, empty content, refusal handling, and `stream_options.include_usage` with the real OpenAI client.

If usage is requested, emit one terminal usage payload, not a copy on every chunk. The proxy must produce its normal end marker exactly once.

Before output commitment, failures use the normal HTTP error path where LiteLLM permits it. After the stream is committed, follow the tested LiteLLM error behavior; do not claim a new HTTP status can replace bytes already sent.

On cancellation or disconnect, cancel pending child tasks and release request state in `finally`. Treat cancellation of backend work as best effort; a provider may continue computation or charge for an already-started request.

## 11. Budgeting, retries, and local concurrency

### 11.1 Counters

Maintain distinct counters for:

```text
logical request             1 client request
model invocation            candidate, judge, fuser, repair, or fallback
backend attempt             an actual deployment attempt, including retries
verifier invocation         non-LLM check, separately metered
```

`max_model_calls` includes judges and fallbacks. `max_backend_attempts` additionally includes retries and router fallback attempts. A retry is not a new independent candidate.

### 11.2 Reservations

Budget checks must be atomic within a request. Before dispatch, reserve call slots and conservative prompt-plus-output tokens; reserve selector/fallback capacity before filling the candidate pool. Set an actual output limit on every child call.

Include repeated prompt tokens and judge input containing candidate answers. A 1,500-token public output limit does not mean the entire ensemble consumes only 1,500 tokens.

A deadline includes semaphore waiting, LiteLLM routing/retries, verification, and judging. Never launch a stage that cannot fit its configured reservation. Stop launching work after cancellation or budget exhaustion.

Unknown usage must be represented as unknown or estimated, never silently zero. Strict monetary caps require known pricing and adequate reservation support; otherwise reject the strict profile rather than claim a hard cap.

### 11.3 Retry policy

MVP default: disable automatic retry/fallback amplification for the virtual wrapper and use zero child retries. LiteLLM supports `num_retries`; the documented precedence includes client-provided values, so the package must enforce its trusted policy rather than rely only on a deployment default. [2]

Do not mutate the global router retry settings per request. Apply tested per-call controls and reject conflicting wrapper settings. Inspect reachable router fallback paths: none may recurse into the same virtual policy or leave the allowed model/privacy set.

A later opt-in retry profile may reuse LiteLLM retries with per-attempt hooks and the shared attempt budget. LiteLLM documents separate per-deployment hooks for attempts; ordinary request-success logging is not an adequate retry counter. [10]

Outer proxy retries must not restart a complete five-call plan under a fresh invisible budget. A wrapper retry either reuses the same execution ledger under a tested contract or is disabled.

### 11.4 Concurrency and cleanup

Use `asyncio` tasks with bounded semaphores. Never call synchronous `litellm.completion()` on the proxy event loop. Do not call `asyncio.run()` from an active async request.

The first release targets the async proxy path. A synchronous provider method may be explicitly unsupported; this does not prevent a synchronous OpenAI HTTP client from calling the async proxy.

Enforce per-request and per-worker child limits, alongside LiteLLM's physical-deployment limits. Process-local semaphores are not cluster-wide GPU admission control. Document the scope, and rely on existing LiteLLM/backend capacity management for shared devices.

A parent may occupy a virtual-model routing slot, but must not hold a physical-model permit while waiting to launch its children. Test against nested-semaphore deadlocks.

Always await task cleanup with exceptions collected. Never cancel unrelated requests or close shared LiteLLM clients.

## 12. Recursion and configuration safety

Use both static validation and runtime protection.

Static checks must reject:

- Direct policy self-reference or cycles through aliases/fallbacks.
- A child group that can route to `multicall/*` in MVP.
- Unknown candidate/judge/fallback groups.
- Missing selector budget, negative counts, or unbounded stages.
- Client-controlled configuration paths or provider endpoints.
- A `local_only` policy with a reachable non-local deployment or verifier.

Runtime protection must use trusted execution context and an expansion-depth limit of one for MVP. Client-supplied depth/bypass metadata must never disable protection. Confirm the resolved deployment at dispatch time, not just a superficial alias prefix.

Configuration is administrator-controlled and validated once per version. Policies are immutable during a request. Changes require a proxy restart initially; hot reload is out of scope. A request records the policy content hash for reproducibility.

## 13. Caching and diversity

Disable response-cache reads/writes for internal sampling by default using the tested LiteLLM per-request cache controls. Do not toggle a global cache flag. LiteLLM provides per-request cache controls; the adapter must test the exact supported invocation. [11]

Otherwise repeated calls can return the same cached response, falsely appearing to be independent samples. Semantic-cache reuse is also inappropriate for this purpose.

Disable caching of the outer virtual model by default. A future logical-result cache must include tenant/privacy boundary, original request, policy hash, model configuration identity, and sampling parameters.

Backend prefix/KV caching is different: it can share prompt computation without reusing a completed answer. The extension must not disable it merely because response caching is disabled.

Do not promise that repeated calls will produce different answers. Greedy generation, identical seeds, or correlated models may yield duplicates. Record duplicates; do not invent diversity.

## 14. Usage, tracing, and response normalization

### 14.1 Public response

Return one ordinary completion with a new logical completion ID and the requested public model alias. For selection, preserve the winner's full assistant message and finish reason. For fusion, use the validated fused message.

Public `usage` reports aggregate known token consumption from all child calls belonging to this execution, not merely the selected output. This includes judge/fuser and billable failed attempts when known. The numbers describe compute consumed, not the token length of the visible answer.

Do not combine incompatible token-detail structures into misleading values. Aggregate basic counters; retain model-specific details per child. Mark incomplete usage through trace data and an optional response header instead of adding nonstandard required body fields.

The final public object must not inherit a misleading physical-model fingerprint or hidden child cost fields. Normalize it through the compatibility adapter.

### 14.2 Tracing

Use existing LiteLLM callbacks/logging integrations plus optional package-local standard logging. No mandatory new database or observability service.

Record:

```text
logical_request_id, child_call_id, candidate_id, stage
public_model, policy_id, policy_hash, actual_model/deployment when known
latency, queue_time, prompt_tokens, completion_tokens
cost_if_known, usage_completeness, attempt_number
validation_status, selection, stop_reason, cache_hit
```

Default to metadata-only traces. Do not log prompts, candidate contents, credentials, or detailed judge explanations unless explicitly enabled by the operator.

Optional headers, injected through tested LiteLLM hooks:

```text
X-Multicall-Request-Id
X-Multicall-Model-Calls
X-Multicall-Strategy
X-Multicall-Usage-Complete
```

The response remains useful without these headers. Hook support must be tested for both streaming and non-streaming paths. [4]

## 15. Authorization, guardrails, and cost accounting

This section is a required integration contract, not a claim that nested SDK calls automatically inherit every Proxy feature.

### 15.1 Profiles

`trusted_local` is the initial development profile. It requires administrator-controlled local endpoints, no cloud fallbacks, bounded calls/tokens, and no claim of multi-tenant monetary enforcement. “Local” is an operator-verified allowlist and deployment property, not merely a model name or a private-looking IP address.

`governed` is the production profile for paid or multi-tenant operation. It must remain unavailable on a LiteLLM version until its adapter passes the authorization, guardrail, accounting, and budget tests below. Fail closed; do not silently downgrade to trusted-local behavior.

### 15.2 Authorization and guardrails

Default production access is the intersection of:

```text
models allowed by the authenticated LiteLLM principal
AND models explicitly permitted by the execution policy
AND deployments satisfying the request's privacy/capability constraints
```

Permission to call a virtual alias does not, by itself, authorize arbitrary hidden backends. Reuse LiteLLM's effective authorization logic through the version adapter rather than trusting user metadata or implementing a simplified parallel ACL system. BYOK/team-specific deployments require explicit support and tests.

Reject incoming raw `api_base`, `api_key`, bypass flags, and arbitrary target overrides for virtual models. Do not use a master key to make internal calls on a user's behalf.

The governed adapter must preserve required ingress and final-output guardrails and apply any required child-level rules before content reaches a backend. Never assume that a `Router` call runs proxy-only hooks. Candidate/judge prompt transformations must not undo redaction or privacy controls.

Mandatory policy violations must not be treated as ordinary candidate failures that encourage bypass retries.

### 15.3 One accounting owner

Use **leaf-call accounting** for governed deployments:

- Each physical child call is attributed and charged exactly once through LiteLLM.
- The virtual wrapper is an orchestration span, not an additional billable inference.
- The wrapper's public usage is an aggregate for the client, not a second charge.
- Logical and physical request counters are separate.

Do not calculate dollars by multiplying aggregate tokens by one arbitrary model's price. Sum actual child costs with each child's model, cache, and pricing information.

The adapter must prove that trusted key/team attribution is retained for child spend events and that parent logging does not add the cost again. Never mutate private `response_cost` fields speculatively and assume all accounting consumers will respect them.

**Budget pitfall:** a zero-priced wrapper may bypass LiteLLM's ordinary budget checks. A governed deployment must perform the proper authenticated budget/quota checks for the underlying work and reserve expansion capacity before dispatch. Wrapper pricing set to zero is not a complete integration. [9]

Token dashboards and exports must distinguish aggregate parent usage from leaf usage; summing both doubles consumption. Integration documentation must identify the supported metric views instead of claiming every existing dashboard is automatically correct.

When a provider times out without final usage, record uncertain consumption and retain conservative budget reservations according to policy. Do not report an exact all-in cost or promise refunds for canceled calls.

### 15.4 Required production tests

A real-proxy test suite must prove:

1. An unauthorized child/fallback is never contacted.
2. Paid children cannot be reached through a zero-priced wrapper after budget exhaustion.
3. A five-call request produces exactly five child charges and no sixth aggregate charge.
4. Model-level physical quotas see the child load; logical request limits have documented semantics.
5. Mandatory guardrails still apply, including after candidate selection/fusion.
6. Failure, cancellation, and streaming paths preserve attribution and cleanup.

Use existing LiteLLM budget/cache/spend facilities where supported. Do not build a new general billing service. If the necessary integration is unavailable, keep `governed` disabled and document the supported local profile precisely.

## 16. Failure behavior

| Condition | Required behavior |
|---|---|
| Some candidates fail | Continue with eligible candidates if the policy permits. |
| All candidates fail | Standard LiteLLM/OpenAI-compatible upstream error. |
| All outputs invalid | Explicit validation/unresolved error, or authorized configured fallback. |
| Judge abstains or returns invalid selection | No fabricated winner; fallback or error. |
| Context too large for candidates/judge | Reject or use an explicit compatible fallback; never silently truncate. |
| Deadline/budget exhausted | Stop dispatch; return a policy-accepted result only if one exists, otherwise error. |
| Missing router/context/version support | Fail closed with a clear integration error. |
| Client disconnect | Cancel queued/running child work best-effort and clean up. |

An optional `on_unresolved: best_available` mode is permitted only when the operator deliberately chooses best-effort answers. It must still exclude structurally invalid outputs and must record degraded selection. Default is `error` or an explicitly configured fallback.

Use LiteLLM's supported exception/error mapping via `compat/errors.py`. Do not return HTTP 200 with an error string pretending to be the assistant's answer. Do not swallow `CancelledError`.

## 17. Implementation layout

```text
mirrorneuron-prism/
├── pyproject.toml
├── README.md
├── SPEC.md
├── LICENSE
├── src/
│   └── litellm_multicall/
│       ├── __init__.py
│       ├── py.typed
│       ├── provider.py           # CustomLLM subclass + exported instance.
│       ├── hooks.py              # CustomLogger instance; context/integration.
│       ├── config.py             # Strict schema, loading, reference checks.
│       ├── types.py              # Request, candidate, execution records.
│       ├── engine.py             # Bounded per-request execution.
│       ├── backend.py            # CompletionBackend + ProxyRouterBackend.
│       ├── budgets.py            # Reservations, deadlines, call limits.
│       ├── selection.py          # Judge and deterministic selection.
│       ├── validation.py         # JSON/tool validation and verifier protocol.
│       ├── streaming.py          # Final-response chunk conversion.
│       ├── telemetry.py          # Parent/child summaries; no new database.
│       ├── cli.py                # validate / doctor / version.
│       ├── compat/
│       │   ├── proxy_runtime.py  # Isolated runtime access.
│       │   ├── request_context.py
│       │   ├── accounting.py
│       │   ├── errors.py
│       │   └── versions.py
│       └── resources/
│           ├── judge.txt
│           ├── fusion.txt
│           └── policy.schema.json
├── examples/
│   ├── litellm.yaml
│   ├── multicall.yaml
│   └── client.py
├── constraints/
│   └── <tested-environment>.txt
├── tests/
│   ├── unit/
│   ├── integration/
│   └── packaging/
└── benchmarks/
    ├── run.py
    └── README.md
```

Use `importlib.resources` for packaged prompts/schemas. Do not rely on the current working directory or a source checkout.

## 18. Test and benchmark requirements

### 18.1 Unit and property tests

Test candidate counts, independent message copies, selector IDs, invalid judge output, token reservations, concurrency limits, cancellation, recursion, unsupported parameters, and policy schema errors.

Use fake outcomes to prove adaptive execution makes one call only after a conclusive verifier pass, five calls for the normal four-candidate-plus-judge path, and at most six when an explicitly configured fallback runs.

Include prompt-injection strings in candidate text and reserved-metadata injection in incoming requests. Check that neither can alter model routing or execution limits.

### 18.2 Real-proxy integration tests

Start a normal LiteLLM Proxy subprocess with this package installed and instrumented local test backends. Use the real OpenAI client over HTTP.

Verify:

- Ordinary non-MultiCall models behave exactly as before.
- `model="smart-local"` enters the installed `CustomLLM` handler.
- All children invoke the existing router; no extra router/server exists.
- Four samples plus one judge produce one public completion.
- JSON and multi-turn tool calls work in both response modes.
- Response IDs, model alias, finish reason, and aggregate usage are correct.
- Cached responses cannot masquerade as fresh candidate samples.
- Timeouts, process workers, callback composition, and disconnects clean up.
- Model changes do not preserve stale or unauthorized policy resolution.

Pin tested LiteLLM releases and record exact versions in CI outputs. Do not certify compatibility using mocks alone.

### 18.3 Packaging tests

Build wheel and sdist; install each into a clean environment. Start LiteLLM outside the repository and resolve the configured package instances without a copied handler file.

Verify resources, CLI entry points, `py.typed`, and dependency extras. Importing the package must not start a server, send traffic, or overwrite another plugin registration.

### 18.4 Benchmark harness

Compare on the same task set and validation rules:

```text
1 × small model, direct through LiteLLM
4 × small model + judge, through this package
adaptive small-model policy
1 × stronger model, direct through LiteLLM
```

Report task success, returned-answer accuracy, candidate oracle coverage separately, total input/output tokens, actual model/attempt counts, cost when known, wall-clock latency, time to first output, and failure rates.

Record model versions, quantization, backend/hardware, context lengths, concurrency, cache settings, prompts, and judge/verifier choices. Use held-out tasks and report uncertainty when sample sizes permit.

The package must not claim an improvement merely because at least one hidden candidate was correct. The **returned** answer is the quality metric. Compare quality at matched cost and latency budgets, not only against one cheap attempt.

## 19. Delivery sequence and acceptance criteria

### Milestone A — Working pip extension

Deliver the Python package, wheel/sdist, real custom-provider registration, existing-router dispatch, CLI validation, and a non-streaming Best-of-N example. No extra service and no provider HTTP code.

### Milestone B — Usable local MVP

Add adaptive execution, JSON/tool validation, buffered-final streaming, cancellation, budgets, recursion protection, metadata-only tracing, and the real-proxy/packaging tests. Ship the `trusted_local` profile with clear limits.

### Milestone C — Governed production compatibility

Certify child access, guardrails, budgets, accounting, quotas, retries, and tenant isolation against pinned LiteLLM versions. Only then enable paid/multi-tenant profiles.

### Later, only after benchmark evidence

Add consensus, fusion, sandbox-verifier adapters, or learned strategy selection. Do not expand into an agent platform or generic workflow system.

The project is complete for the local MVP when an operator can:

```text
install one Python package
+ add a custom provider and hook to existing LiteLLM configuration
+ write a small policy referencing existing model groups
+ call the existing proxy using an ordinary OpenAI client
= transparently execute one logical request as several bounded LLM calls
```

## 20. Engineering constraints to keep the project small

- No LiteLLM fork or editing installed LiteLLM source files.
- No monkey-patched global completion functions.
- No second HTTP gateway, separate daemon, mandatory database, or web UI.
- No duplicate model credentials, provider implementations, or routing pools.
- No unbounded agent loops, hidden cloud fallback, or application tool execution.
- No invented confidence guarantees or unmeasured claims about matching frontier models.
- Keep version-sensitive integration in `compat/`, strategy logic independently testable, and the installation path usable with the standard `litellm` command.

**Core implementation rule:**

> LiteLLM decides how to call a model. This package decides how many calls to make, how to compare their outputs, and when to stop.

---

## References and implementation-verification notes

These are primary-source references checked while preparing this specification. They establish available LiteLLM/Python extension surfaces, not that the proposed package has been implemented or that all deployment combinations are supported. Source branches and documentation can change; record immutable source/version references when implementing the compatibility adapter.

[1] LiteLLM — Custom API Server / CustomLLM registration and streaming:
https://docs.litellm.ai/docs/providers/custom_llm_server

[2] LiteLLM — Router model-group calls, retries, deployment concurrency, and caching:
https://docs.litellm.ai/docs/routing

[3] LiteLLM upstream — CustomLLM method and streaming contracts:
https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/llms/custom_llm.py

[4] LiteLLM — Proxy request, response, and streaming hooks:
https://docs.litellm.ai/docs/proxy/call_hooks

[5] LiteLLM upstream — Importable instance loader:
https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/proxy/types_utils/utils.py

[6] LiteLLM upstream — Proxy runtime/router state:
https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/proxy/proxy_server.py

[7] LiteLLM — Worker startup hooks and initialization order:
https://docs.litellm.ai/docs/proxy/worker_startup_hooks

[8] Python Packaging User Guide — Packaging Python projects:
https://packaging.python.org/en/latest/tutorials/packaging-projects/

[9] LiteLLM — Custom pricing and the budget-check behavior of zero-cost models:
https://docs.litellm.ai/docs/proxy/custom_pricing

[10] LiteLLM — Custom logging and per-deployment attempt hooks:
https://docs.litellm.ai/docs/observability/custom_callback

[11] LiteLLM — Per-request response-cache controls:
https://docs.litellm.ai/docs/caching/all_caches
