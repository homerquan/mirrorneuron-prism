# Prism

Prism is a standalone OpenAI-compatible LLM proxy with six bounded execution policies. A required CPU Laya model chooses among feasible plans: direct execution, independent evidence mapping, batched mapping, verified mapping, focused retrieval, and draft/review/synthesis. Optional cost/power optimization selects stage models from an operator allowlist using JSON ratings and prices plus Laya's task recommendation. Workers extract source-backed facts; Prism checks their quotes against immutable source bytes. The client receives one ordinary assistant response.

The distribution is **`mirrorneuron-prism`**, the Python package is **`prism`**, and the CLI is **`prism`**. The service connects directly to OpenAI-compatible model servers. A stronger synthesizer is optional; all stages can share one local model. No dependency on MirrorNeuron, OtterDesk, or the local Laya source checkout is required.

This is an alpha implementation with constrained learned routing and several finite execution graphs. It does not claim stronger-model equivalence, a million-token context, complete extraction recall, or measured speed/cost improvements. See [the implemented contract](docs/standalone-contract.md), [execution policies](docs/execution-policies.md), and [design specification](prism_standalone_proxy_design.md).

## Install and run

Requires Python 3.11 or later. From this checkout:

The repository's `prism.json` already points to the existing `models/muse-gemma-mix.json` registry. Its default `prism` route uses Gemma for direct/worker/synthesis stages; `prism-careful` permits the configured Muse backend. Verify capabilities/context settings and endpoint availability for your deployment. Use `prism init` in a new directory to generate a clean configuration.

```sh
python -m pip install .

# For this repository, use the existing prism.json and models/ registry.
# In a new deployment directory, run prism init first.
# Edit the raw-model JSON: physical names, endpoints, capabilities and limits.
export PRISM_API_KEY='choose-a-long-random-secret'
prism validate --config prism.json
prism doctor --config prism.json --probe-backends
prism serve --config prism.json
```

After publication, install with `python -m pip install mirrorneuron-prism`. Laya and its ML dependencies install automatically. The first server startup prepares its checkpoint on CPU and may download weights. The server binds to `127.0.0.1:8080` by default and requires bearer authentication. Put TLS at a reverse proxy before exposing it over a network.

Configure physical models in **JSON**, separately from virtual policies. Existing files shaped like `models/muse-gemma-mix.json` still work: `id` defaults to `name`. Set the `models_file` path and profile references accordingly. Relative paths resolve against the config file, not the working directory.

```json
{
  "models": [
    {
      "id": "small",
      "name": "your-physical-model",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key_env": "BACKEND_API_KEY",
      "context_window": 32768,
      "max_output_tokens": 4096,
      "concurrency": 2,
      "capabilities": ["text", "stream", "tools", "json_object", "json_schema"]
    }
  ]
}
```

`api_key` and `api_key_env` are alternatives; an unset configured credential fails visibly. Omit both for a backend without authentication. Capabilities are operator declarations and must match the actual server. There are no hidden retries, implicit external fallbacks, or calls to URLs from prompts.

The generated `prism.json` serves `prism` for automatic routing and fixed aliases `prism-direct`, `prism-evidence`, `prism-batched`, `prism-verified`, and `prism-retrieve`. Profiles select physical model IDs for `direct`, `worker`, `synthesizer`, and an optional separate `verifier`; all stages may share one small model. Set `strategy` to `auto` or one of the policies listed by `prism policies`. `allowed_policies` limits automatic choices. Each profile caps calls, input/output work, deadline, partitions, and parallel workers. An optional dollar ceiling requires configured input/output prices for every participating backend.

See [the flagship curl cases](docs/flagship-curl-cases.md) for copyable direct, JSON Schema, evidence-map, SSE, tool-call, and error examples. Tests read the exact request payloads from that document and run its curl commands against local HTTP servers.

## Ordinary OpenAI client

```sh
python -m pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.environ["PRISM_API_KEY"],
)
response = client.chat.completions.create(
    model="prism",
    messages=[{"role": "user", "content": "Explain virtual context briefly."}],
    max_completion_tokens=200,
)
print(response.choices[0].message.content)
```

Text conversations, system/developer roles, JSON output, function tools, and SSE are supported on the declared backend capabilities. Caller tools are returned to the caller for execution. Unsupported fields, modalities, `n > 1`, malformed tool relationships, and oversized instructions fail before dispatch. Tool calls, seeds, logprobs, token-ID bias, and reasoning controls require an untransformed direct route. Responses API is not implemented.

## Automatic model selection by cost and power

Add `power_rating` (integer 1–10) and both existing token prices to the model JSON, then opt a profile into `optimization`:

```json
"optimization": {
  "model_ids": ["small", "strong"],
  "default_cost_priority": 0.5
}
```

Each request can change the balance through `context_management`. A priority of `0.8` weights cost 80% and power 20%; `0` favors power and `1` selects the cheapest feasible plan. Use the Python SDK's `extra_body`:

```python
response = client.chat.completions.create(
    model="prism-optimized",
    messages=[{"role": "user", "content": "Explain the tradeoffs of this design."}],
    extra_body={"context_management": [{"prism_cost_priority": 0.8}]},
)
```

Prism evaluates direct and permitted multi-stage plans, checks their capabilities/context/budgets, and chooses models for each stage. Laya sees the best assignment for each eligible policy and supplies a bounded task-fit bonus. Calls can also narrow allowed models/policies or tighten cost/call limits. List `draft_review` in `allowed_policies` to enable three-stage drafting, structured critique, and final synthesis for ordinary source-free prompts. Tools and other direct-only features still select among eligible direct models.

Ratings and task fit are heuristics; costs use conservative reservations, not predicted provider bills. Legacy profiles keep their existing routing. Current deployment models are left without invented ratings or prices. See [configuration, scoring, and request controls](docs/model-optimization.md) and the [illustrative JSON configuration](examples/standalone/optimization/prism.json). This is a Prism Chat Completions extension; it does not implement OpenAI compaction or Responses.

## Large source input

Explicit boundaries identify data without guessing which part of a long message is the user's instruction. Instructions before and after the blocks retain their roles and order. Multiple blocks and text content parts are supported.

```python
response = client.chat.completions.create(
    model="prism",
    messages=[{
        "role": "user",
        "content": "Summarize policy exceptions and qualifications.\n"
                   "<prism-source id=\"policies\">\n"
                   + corpus_text + "</prism-source>\n"
                   "Cite the supplied evidence and state uncertainty.",
    }],
    max_completion_tokens=1000,
)
```

Prism stores original content unchanged for the request lifetime and partitions source blocks losslessly at paragraph/record boundaries where possible. It reserves finalization before starting workers. Every worker must finish without truncation, return valid typed JSON, report no unresolved needs, and provide quotes that resolve uniquely against original UTF-8 spans. Every required partition must pass these checks. Equal-looking records from distinct spans retain their multiplicity.

The map policies include all validated evidence in final synthesis. Optional [structured reduction](docs/structured-reduction.md) uses bounded observations, a recursive reduce tree, and raw-evidence lookup from temporary Markdown files to avoid an oversized reducer. Without reduction, oversized synthesis fails visibly. `retrieve_read` instead answers from selected original spans and requires explicit `coverage: "focused"`; it cannot satisfy exhaustive coverage. A matching quote establishes provenance, not the truth of the worker's interpretation, and full partition submission is not proof of extraction recall. Verification adds an independent model check but does not guarantee correctness. Cross-partition dependency repair and exact aggregation are future work. Count/exhaustive-inventory requests require direct execution when the original request fits.

## Laya decisions

**`laya>=0.3.23,<0.4` is a required pip dependency**, without vendoring or importing a sibling checkout. The default decision mode is `route`, using `convaiinnovations/laya-typed-decisions`. Prism first compiles plans against capabilities, context, coverage, and resource limits. Laya sees only eligible choices, bounded instruction samples, source sizes, and call bounds. A valid, untruncated choice with `answer_confidence >= min_option_confidence` (default 0.7) selects an existing plan. Abstention, low confidence, or inference failure uses the rules-selected feasible plan. With only one eligible policy, Prism skips decision inference.

The checkpoint is prepared once at startup in a spawned CPU process. For an offline/reproducible deployment, set `model` to a prepared local checkpoint, or configure an immutable Hub `revision` and `expected_sha256` file hashes. Model loading failure prevents startup. Deadline/disconnect cancellation kills and reaps the process; subsequent decisions fall back to rules until restart. `shadow` remains available for evaluating proposals without changing execution, and still requires Laya. There is no `off` mode. Help/configuration commands keep ML imports out of the parent process. The option probability threshold is not a calibrated answer-quality guarantee.

## Streaming, accounting, and traces

Direct text requests forward the final backend's SSE deltas with one stable public ID and alias. Adaptive and JSON-constrained streams are **buffered delivery** after evidence/schema validation, indicated by `X-Prism-Stream-Mode`. Internal worker output is never emitted as assistant progress. Failures after streaming begins terminate with an SSE error without a success marker or replacement answer.

`usage` follows the fixed **`prism-utf8-v1` logical accounting** contract: UTF-8 bytes of compact serialized messages plus supplied inference parameters, and public content/tool-call bytes. This deliberately conservative logical measure is not a physical tokenizer count or a provider bill. Output-limit parameters cap the selected public backend's generation-token budget, including reasoning where reported by that backend. Backend context admission separately uses a declared conservative byte-to-token bound and chat overhead. Validate those settings against the backend's tokenizer and template.

The `X-Request-ID` header identifies an authenticated, metadata-only trace:

```sh
prism trace show prism-REQUEST_ID
```

Traces expose eligible policies, the selected graph, Laya proposals and timing, source hashes, coverage, backend identities, provider-reported usage, unknown usage, cancellations, and cost upper estimates. Optimized traces also include effective controls, candidate stage models/ratings, score components, and the selection reason. `X-Prism-Policy` and `X-Prism-Coverage` identify execution on the response. Traces omit source text, prompts, evidence quotes, and credentials. Storage is in memory, scoped to the authenticated credential, with configurable capacity and TTL. Rotating the credential invalidates access to prior traces. Consumed work with missing usage is conservatively charged to resource reservations, never reported as zero provider usage. Price/usage bounds are operator assumptions; unknown billing is labeled explicitly.

## Evaluation and release

Use the packaged six-case benchmark to compare direct execution with evidence extraction/synthesis. Each run saves inputs, answers, traces, latency percentiles, reference quality scores, physical work, and pricing estimates in a new folder:

```sh
# With Prism running and PRISM_API_KEY set:
prism benchmark run --config prism.json --out-dir benchmark-results/run-a
# Repeat after changing a model/profile, using a new folder:
prism benchmark run --config prism.json --out-dir benchmark-results/run-b
prism benchmark compare benchmark-results/run-a benchmark-results/run-b \
  --out-dir benchmark-results/comparison-a-b
```

The default compares `prism-direct` with `prism-evidence`, using three repetitions, one warmup pair, and the same answer budget. Pass `--candidate prism-batched`, `prism-verified`, or `prism-retrieve` to compare another fixed policy; `--candidate prism` measures active Laya routing. Cost stays unknown until physical usage and configured model prices are available. Read [benchmark instructions and metric definitions](docs/benchmarking.md) before interpreting speed, quality, or cost differences.

For repeatable local/Spark Docker Model Runner Gemma4 + Spark Nemotron tests, run `.venv/bin/python examples/standalone/docker-spark/benchmark.py --repeats 3`. The [runner instructions](examples/standalone/docker-spark/README.md) cover direct, mixed evidence, automatic small-context partitioning, and draft/review suites, with per-stage timing, Laya decisions, coverage, completion/quality, and physical-token cost proxies. The runner defaults to structured reduction; use `--reduction rolling` for a controlled alternative.

```sh
python -m pip install '.[dev]'
ruff check src tests examples
python -m pytest -q
python -m pytest tests/standalone/test_http.py -m integration -o addopts=''
python -m pytest tests/standalone/test_flagship_curl.py -m integration -o addopts=''
python -m pytest tests/standalone/test_benchmark.py -m integration -o addopts=''
python -m build
python -m twine check dist/*

# Paired requests against two aliases on a running deployment:
prism eval run --cases examples/standalone/cases.jsonl \
  --baseline prism-direct --candidate prism --out paired.jsonl
prism eval compare paired.jsonl
```

The evaluation harness records failures in the denominator, logical usage, physical traces, and request latency. `expected_contains` is a transparent fixture check, not a general semantic correctness evaluator. It makes no non-inferiority or performance claim. Test coverage includes SDK requests over HTTP/ASGI, evidence validation, bounds, auth, streaming/tool behavior, cancellation, and release installation.

The wheel ships only the `prism` package, its JSON defaults, typing marker, and MIT license. The source distribution also includes the documentation and tests. CI checks Python 3.11–3.13, including the curl cases from the document. The manual release workflow uses a `pypi` GitHub environment and PyPI Trusted Publishing; configure repository/environment protections and the trusted publisher before running it. See [release instructions](docs/releasing.md). Nothing is automatically uploaded by installation or builds.

MIT license.
