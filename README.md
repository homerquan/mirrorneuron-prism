# Prism

**One OpenAI-compatible API. Small models prepare; stronger models finish.**

Prism combines text, reasoning, and vision models in explicit, bounded workflows. Keep your usual OpenAI client, select a virtual model alias, and inspect which physical models did the work. Use a free OpenRouter mix, native cloud APIs, or local OpenAI-compatible servers.

Distribution: **`mirrorneuron-prism`** · import: **`prism`** · command: **`prism`** · Python **3.11+** · MIT

- **Model combinations:** vision → text/reasoning, draft → review → synthesis, free text preparation → final model, and source-backed evidence workflows.
- **JSON-aware routing:** explicitly choose a capable final model when JSON or JSON Schema is required.
- **Measured capacity:** challenge actual JSON, image, and reasoning behavior on physical models and complete profiles.
- **Bounded execution:** context, calls, concurrency, output, and deadlines are checked before dispatch. Traces show physical usage and unknown costs.
- **A useful CLI:** Rich tables and help, status indicators, secret-free inventories, and JSON for automation.
- **Portable deployment:** pip packaging, bundled presets, and a non-root Docker image with CPU Laya routing.

Prism is alpha software. Observation-based preparation can lose information; source-backed evidence policies validate quote provenance, which does not prove correctness or recall. Qualify models on your own workload.

## Start with free OpenRouter models

Install from this checkout now; use `python -m pip install mirrorneuron-prism` after publication. Release preparation does not upload anything.

```sh
python -m pip install .
mkdir prism-demo && cd prism-demo
prism init --preset openrouter
export OPENROUTER_API_KEY='your-openrouter-key'
export PRISM_API_KEY='your-prism-client-secret'
prism validate
prism profiles
prism serve
```

First startup may download the required Laya checkpoint and prepares it on CPU. The server defaults to `http://127.0.0.1:8080`. The OpenRouter preset exclusively uses free Nano Omni, Super, and Ultra models; upstream credentials and free-tier quotas still apply.

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $PRISM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"prism-balanced","messages":[{"role":"user","content":"Explain decorators in Python briefly."}],"max_completion_tokens":4096}'
```

For explicit anonymous serving, use `prism serve --no-auth`. Upstream credentials remain required. Clients share anonymous trace access in this mode; use it on a trusted network.

## Pick a workflow

| Free-preset alias | Behavior |
|---|---|
| `prism-balanced` | Nano text; Super when JSON is required |
| `prism-vision-llm` / `prism-omni-llm` | Nano interprets images, Ultra writes the answer |
| `prism-vision-reasoning` | Nano interprets images, Super writes the answer |
| `prism-reasoning-image` / `prism-reasoning-omni` | Super drafts/reviews, Nano finishes; Super finishes JSON |
| `prism-llm-omni` | Ultra drafts, Super reviews, Nano finishes; Super finishes JSON |
| `prism-nano-synthesis` | Nano prepares text context, Super finishes |
| `prism` / `prism-evidence` | Automatic or fixed source-backed evidence routing |

Vision means image understanding with text output. Additional omni modalities and image generation are not implemented. Nano's direct alias deliberately rejects JSON requirements; profiles use an explicit `structured_output_model` instead.

For an image plus JSON output, choose a vision-synthesis alias such as `prism-vision-reasoning` so pixels reach Nano and structured final output comes from Super.

[Free-model setup and live results](docs/openrouter-nemotron-mix.md) · [Native OpenAI / Claude / Gemini examples](examples/standalone/providers/README.md)

## Keep your OpenAI client

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key=os.environ["PRISM_API_KEY"])
answer = client.chat.completions.create(
    model="prism-balanced",
    messages=[{"role": "user", "content": "Return JSON with ok=true."}],
    response_format={"type": "json_object"},
    max_completion_tokens=4096,
)
print(answer.choices[0].message.content)
```

Chat Completions supports ordinary text, image inputs, tools on direct routes, and SSE. Multi-stage and JSON-constrained streams are delivered after validation. Responses API is not implemented. For a guaranteed requested shape, use JSON Schema and retain task-level checks.

## Inspect and qualify

```sh
prism --help
prism models
prism profiles --json > profiles.json
prism doctor --probe-backends
prism capacity --model prism-vision-reasoning
prism trace show prism-REQUEST_ID
```

Terminal output uses tables; redirected results are JSON. Use `--json`, `--output table`, or `NO_COLOR=1` explicitly. Inventory capabilities are declarations; capacity results come from live challenges and distinguish failures from inconclusive upstream errors.

The [live six-task pilot](docs/evaluations/2026-10-04-openrouter/benchmark-results.md) measured coding, copywriting, support, and summaries. Nine matched completed pairs showed **56.3% lower hypothetical premium token cost**, using free Nemotron tokens priced like GPT-6 Astra / Claude Opus. Across all attempts, acceptance fell **9/12 → 6/12** and mean latency rose **4.83s → 19.96s**. These are workload tradeoffs, not a production savings or frontier-model quality claim. [Qualified marketing narrative](docs/evaluations/2026-10-04-openrouter/marketing-narrative.md).

## Run with Docker

```sh
docker build -t mirrorneuron-prism:0.3.0 .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENROUTER_API_KEY -e PRISM_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:0.3.0
```

Or run `docker compose up --build`. [Custom config, checkpoint caching, and anonymous Docker serving](docs/usage.md#docker).

## Documentation and development

| Guide | Covers |
|---|---|
| [Usage](docs/usage.md) | Installation, auth, CLI, JSON, Docker, local models, streaming, and traces |
| [Model configuration and capacity](docs/model-config-capacity.md) | LiteLLM transports, shared registries, and live probes |
| [Execution policies](docs/execution-policies.md) | Stage graphs, limits, and Laya routing |
| [Copyable curl examples](docs/flagship-curl-cases.md) | Requests exercised by integration tests |
| [Benchmarking](docs/benchmarking.md) | Reproducible runs and metric limitations |
| [Release instructions](docs/releasing.md) | Build, validate, install, and manually publish |
| [Implemented contract](docs/standalone-contract.md) | Guarantees and boundaries |

```sh
python -m pip install '.[dev]'
ruff check src tests examples
python -m pytest --cov=prism --cov-report=term-missing -q
python -m pytest tests/standalone -q -m integration -o addopts=''
python -m build
python -m twine check dist/mirrorneuron_prism-0.3.0*
```

CI exercises Python 3.11–3.13. Live provider qualification is separate from deterministic tests. The release workflow is manual; the prepared distribution ships bundled presets, benchmark fixtures, typing metadata, and the MIT license.
