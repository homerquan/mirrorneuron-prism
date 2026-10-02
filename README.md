# Prism

Prism is a standalone OpenAI-compatible LLM proxy that routes small requests directly and processes explicitly bounded large sources through a finite evidence-map plan. Workers extract source-backed facts; Prism checks their quotes against immutable source bytes, then sends the complete validated evidence view to a synthesizer. The client receives one ordinary assistant response.

The distribution is **`mirrorneuron-prism`**, the Python package is **`prism`**, and the CLI is **`prism`**. Version 0.2 replaces the LiteLLM-extension product with its own ASGI service. LiteLLM and a stronger synthesizer are optional. No dependency on MirrorNeuron, OtterDesk, or the local Laya source checkout is required.

This is an alpha implementation of the design's direct proxy and bounded evidence-map milestones, with optional Laya shadow decisions. It does not claim stronger-model equivalence, a million-token context, complete extraction recall, or measured speed/cost improvements. See [the implemented contract](docs/standalone-contract.md) and [design specification](prism_standalone_proxy_design.md).

## Install and run

Requires Python 3.11 or later. From this checkout:

The repository's `prism.json` already points to the existing `models/muse-gemma-mix.json` registry. Its default `prism` route uses Gemma for direct/worker/synthesis stages; `prism-careful` permits the configured Muse backend. Verify capabilities/context settings and endpoint availability for your deployment. Use `prism init` in a new directory to generate a clean configuration.

```sh
python -m pip install .
# Optional local CPU decision model, installed from PyPI:
python -m pip install '.[laya]'

# For this repository, use the existing prism.json and models/ registry.
# In a new deployment directory, run prism init first.
# Edit the raw-model JSON: physical names, endpoints, capabilities and limits.
export PRISM_API_KEY='choose-a-long-random-secret'
prism validate --config prism.json
prism doctor --config prism.json --probe-backends
prism serve --config prism.json
```

After publication, install with `python -m pip install mirrorneuron-prism` or `python -m pip install 'mirrorneuron-prism[laya]'`. The server binds to `127.0.0.1:8080` by default and requires bearer authentication. Put TLS at a reverse proxy before exposing it over a network.

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

The generated `prism.json` serves `prism` and `prism-direct`. Its `profiles` select physical model IDs for `direct`, `worker`, and `synthesizer`; all three may reference the same small model. Set `strategy` to `auto`, `direct`, or `evidence_map`. Each profile caps calls, input/output work, deadline, partitions, and parallel workers. An optional dollar ceiling requires configured input/output prices for every participating backend.

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

All validated evidence must fit the final backend context. If it does not fit, Prism returns a resource/context error. It never silently selects a subset to fit. A matching quote establishes provenance, not the truth of the worker's interpretation, and full partition submission is not proof of extraction recall. Cross-partition dependency repair and exact aggregation are future work. Oversized count/exhaustive-inventory requests are rejected in the adaptive route; direct execution remains available when the original request fits.

## Laya decisions

The optional extra uses **`laya>=0.3.23,<0.4` from pip**, without vendoring or importing a sibling checkout. Enable `"decision": {"mode": "shadow", "model": "typed-decisions"}` in `prism.json`. The checkpoint is prepared once at startup on CPU, and Laya proposes a strategy from a bounded structural/instruction sample. It never changes the rules-selected plan. Model confidence is not a correctness or calibrated-risk guarantee.

Startup can download the checkpoint through Laya. For an offline/reproducible deployment, set `model` to a prepared local checkpoint, or configure an immutable Hub `revision` and `expected_sha256` file hashes. Model loading failure prevents startup. The resident CPU model runs in a spawned process; deadline/disconnect cancellation kills and reaps it. Subsequent shadow decisions abstain until a service restart, avoiding request-time downloads or reloads. `off` is the default, so normal installation and help commands do not import Torch. Existing JEV artifact preparation/calibration utilities remain in the optional legacy component.

## Streaming, accounting, and traces

Direct text requests forward the final backend's SSE deltas with one stable public ID and alias. Adaptive and JSON-constrained streams are **buffered delivery** after evidence/schema validation, indicated by `X-Prism-Stream-Mode`. Internal worker output is never emitted as assistant progress. Failures after streaming begins terminate with an SSE error without a success marker or replacement answer.

`usage` follows the fixed **`prism-utf8-v1` logical accounting** contract: UTF-8 bytes of compact serialized messages plus supplied inference parameters, and public content/tool-call bytes. This deliberately conservative logical measure is not a physical tokenizer count or a provider bill. Output-limit parameters cap the selected public backend's generation-token budget, including reasoning where reported by that backend. Backend context admission separately uses a declared conservative byte-to-token bound and chat overhead. Validate those settings against the backend's tokenizer and template.

The `X-Request-ID` header identifies an authenticated, metadata-only trace:

```sh
prism trace show prism-REQUEST_ID
```

Traces expose plan nodes, source hashes, coverage, shadow proposals, backend identities, provider-reported usage, unknown usage, cancellations, and cost upper estimates. They omit source text, prompts, evidence quotes, and credentials. Storage is in memory, scoped to the authenticated credential, with configurable capacity and TTL. Rotating the credential invalidates access to prior traces. Consumed work with missing usage is conservatively charged to resource reservations, never reported as zero provider usage. Price/usage bounds are operator assumptions; unknown billing is labeled explicitly.

## Evaluation and release

```sh
python -m pip install '.[dev,legacy]'
ruff check src/prism tests/standalone
python -m pytest -q
python -m pytest tests/standalone/test_http.py -m integration -o addopts=''
python -m build
python -m twine check dist/*

# Paired requests against two aliases on a running deployment:
prism eval run --cases examples/standalone/cases.jsonl \
  --baseline prism-direct --candidate prism --out paired.jsonl
prism eval compare paired.jsonl
```

The evaluation harness records failures in the denominator, logical usage, physical traces, and request latency. `expected_contains` is a transparent fixture check, not a general semantic correctness evaluator. It makes no non-inferiority or performance claim. Test coverage includes SDK requests over HTTP/ASGI, evidence validation, bounds, auth, streaming/tool behavior, cancellation, and release installation.

Wheel and source distribution packaging includes JSON defaults, type markers, schemas, and third-party license notices. CI checks Python 3.11–3.13. The manual release workflow uses a `pypi` GitHub environment and PyPI Trusted Publishing; configure repository/environment protections and the trusted publisher before running it. See [release instructions](docs/releasing.md). Nothing is automatically uploaded by installation or builds.

Legacy `mn_prism` classifier/benchmark tooling is retained for migration and requires `mirrorneuron-prism[legacy]`; its LiteLLM extension is not the new server. Old design notes and examples describe that legacy interface.

MIT license. The vendored legacy Semif component retains its own included license.
