# Prism

[![PyPI](https://img.shields.io/pypi/v/mirrorneuron-prism.svg)](https://pypi.org/project/mirrorneuron-prism/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/homerquan/mirrorneuron-prism/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://github.com/homerquan/mirrorneuron-prism/blob/main/pyproject.toml)
[![CI](https://github.com/homerquan/mirrorneuron-prism/actions/workflows/ci.yml/badge.svg)](https://github.com/homerquan/mirrorneuron-prism/actions/workflows/ci.yml)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange.svg)](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/standalone-contract.md)

**Turn local models and cloud LLMs into one AI API.**

![A glass prism splitting white light into a rainbow spectrum](https://raw.githubusercontent.com/homerquan/mirrorneuron-prism/main/docs/assets/prism.jpeg)

[Why Prism](https://github.com/homerquan/mirrorneuron-prism#why-prism) · [How it works](https://github.com/homerquan/mirrorneuron-prism#how-it-works) · [Quick start](https://github.com/homerquan/mirrorneuron-prism#start-with-free-openrouter-models) · [Documentation](https://github.com/homerquan/mirrorneuron-prism#documentation-and-development) · [Contributing](https://github.com/homerquan/mirrorneuron-prism/blob/main/CONTRIBUTING.md)

Prism lets several models work together behind one OpenAI-compatible endpoint. Connect your application once, then choose a profile that combines the models you need: a small model to prepare context, a vision model to read an image, or a larger model to review and finish the answer. Your application receives one assistant response.

Prism is a component of [MirrorNeuron](https://www.mirrorneuron.io), built to make AI workflows useful on infrastructure you control. It helps solve local AI by composing available models into a service your applications can use. You can also run Prism independently: combine local servers with OpenRouter, OpenAI, Claude, Gemini, and other APIs supported by LiteLLM, using local models, cloud models, or a mix of both.

Distribution: **`mirrorneuron-prism`** · import: **`prism`** · command: **`prism`** · Python **3.11+** · MIT

## Why Prism

AI applications need different capabilities for different jobs. A coding assistant may benefit from drafting and review; a support tool needs structured answers; an image workflow needs vision before reasoning. Prism gives you a place to compose those capabilities while your application keeps the same API.

- **Keep your application simple.** Use your existing OpenAI client and switch workflows by model alias. Define physical models once and reuse them across profiles.
- **Put each model to useful work.** Let a small or free model prepare context, a vision model interpret pixels, and a selected final model write the answer. Measure cost, latency, and task quality to find a combination that fits your workload.
- **Bring your own compute and providers.** Start with local inference, cloud APIs, or both. Run Prism as a Python package or Docker service; MirrorNeuron is optional for standalone use.
- **Make model choices visible.** Test actual JSON, image, and reasoning behavior, route structured output to a capable final model, and inspect stage usage in traces. Set limits on calls, context, output, concurrency, and deadlines.

## What Prism does

Prism is a model-composition proxy that serves the OpenAI Chat Completions API. A profile defines the physical models and workflow behind a public alias. You can choose a direct call, vision → text/reasoning, draft → review → synthesis, plain-text preparation → final answer, or source-backed evidence extraction → synthesis.

For example, `prism-balanced` uses free Nano for ordinary text and Super when JSON is required. `prism-vision-reasoning` uses Nano to inspect an image, then Super to answer from its observations. Your client selects the alias; Prism executes the configured stages and returns the result through the same endpoint.

Prism is alpha software. Observation-based preparation can lose information; source-backed evidence policies validate quote provenance, which does not prove correctness or recall. Qualify models on your own workload.

## How it works

**One call from your application. One or more model calls inside Prism. One response back.** Prism acts as a transparent proxy at the Chat Completions interface: your client selects a public model alias while Prism handles the configured stages. Models can run locally, behind cloud APIs, or across both.

![Prism routes one API request through direct execution, preparation and synthesis, or drafting, review and synthesis](https://raw.githubusercontent.com/homerquan/mirrorneuron-prism/main/docs/assets/prism-flow.png)

[Diagram source](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/assets/prism-flow.mmd)

The diagram shows three representative paths. Source-backed evidence policies can fan out across multiple workers, then synthesize their results. Every path stays within the profile's declared limits, and traces expose which physical models ran. The proxy keeps the client interface consistent; the answer, latency, and cost depend on the selected workflow.

### Request classification with Laya

Prism uses [Laya](https://github.com/NandhaKishorM/laya), a local typed-decision engine, to classify requests for routing and choose among eligible execution policies. The default checkpoint is `convaiinnovations/laya-typed-decisions`, prepared on CPU at server startup. Laya receives a bounded instruction sample and plan metadata; source documents remain outside its decision input.

Prism checks feasible plans before asking Laya to choose. A confident, valid choice selects an existing policy; abstention, low confidence, or decision-inference failure uses the feasible rules fallback. Fixed profiles and requests with a single eligible policy skip decision inference. Laya's confidence is a routing signal, not an answer-quality guarantee. See [execution policies](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/execution-policies.md) for the full decision flow.

### Choose your cost–quality tradeoff

| Workflow | Cost and latency | Quality consideration |
|---|---|---|
| Direct | One model call, without preparation/review overhead | The chosen model handles the original request itself |
| Small/free preparation → final model | Can reduce premium input when source context is condensed; adds a worker call | Notes can omit facts or qualifications; validate task results |
| Vision preparation → final model | Adds image interpretation before text/reasoning synthesis | Enables a text-only final model to use images through potentially lossy observations |
| Draft → review → synthesis | Adds drafting and review calls | Review can catch problems, but improvements need workload evidence |

Use profiles to decide where to spend model work, then benchmark the result against a direct baseline. Cheaper input, stronger final models, and extra review each change the tradeoff; none establishes equivalent quality by itself. Optional [cost/power selection](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/model-optimization.md) ranks feasible assignments using configured prices and operator ratings. The [live pilot](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/evaluations/2026-10-04-openrouter/benchmark-results.md) shows measured savings alongside quality and latency regressions.

## Start with free OpenRouter models

Try the included free-model profiles with an OpenRouter key. Install [mirrorneuron-prism from PyPI](https://pypi.org/project/mirrorneuron-prism/) with Python 3.11 or later.

```sh
python -m pip install mirrorneuron-prism
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

[Free-model setup and live results](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/openrouter-nemotron-mix.md) · [Native OpenAI / Claude / Gemini examples](https://github.com/homerquan/mirrorneuron-prism/blob/main/examples/standalone/providers/README.md)

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

The [live six-task pilot](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/evaluations/2026-10-04-openrouter/benchmark-results.md) measured coding, copywriting, support, and summaries. Nine matched completed pairs showed **56.3% lower hypothetical premium token cost**, using free Nemotron tokens priced like GPT-6 Astra / Claude Opus. Across all attempts, acceptance fell **9/12 → 6/12** and mean latency rose **4.83s → 19.96s**. These are workload tradeoffs, not a production savings or frontier-model quality claim. [Qualified marketing narrative](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/evaluations/2026-10-04-openrouter/marketing-narrative.md).

## Run with Docker

```sh
docker build -t mirrorneuron-prism:0.3.1 .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENROUTER_API_KEY -e PRISM_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:0.3.1
```

Or run `docker compose up --build`. [Custom config, checkpoint caching, and anonymous Docker serving](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/usage.md#docker).

## Documentation and development

| Guide | Covers |
|---|---|
| [Usage](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/usage.md) | Installation, auth, CLI, JSON, Docker, local models, streaming, and traces |
| [Model configuration and capacity](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/model-config-capacity.md) | LiteLLM transports, shared registries, and live probes |
| [Execution policies](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/execution-policies.md) | Stage graphs, limits, and Laya routing |
| [Copyable curl examples](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/flagship-curl-cases.md) | Requests exercised by integration tests |
| [Benchmarking](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/benchmarking.md) | Reproducible runs and metric limitations |
| [Release instructions](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/releasing.md) | Build, validate, install, and manually publish |
| [Implemented contract](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/standalone-contract.md) | Guarantees and boundaries |

```sh
python -m pip install '.[dev]'
ruff check src tests examples
python -m pytest --cov=prism --cov-report=term-missing -q
python -m pytest tests/standalone -q -m integration -o addopts=''
python -m build
python -m twine check dist/mirrorneuron_prism-0.3.1*
```

CI exercises Python 3.11–3.13. Live provider qualification is separate from deterministic tests. The release workflow is manual; the published distribution ships bundled presets, benchmark fixtures, typing metadata, and the MIT license.

## Contributing and support

Bug reports, documentation improvements, provider fixtures, and focused pull requests are welcome. Read the [contribution guide](https://github.com/homerquan/mirrorneuron-prism/blob/main/CONTRIBUTING.md) for setup, validation, and review expectations. For bugs or feature requests, [open an issue](https://github.com/homerquan/mirrorneuron-prism/issues) with a minimal reproducible example. See the [security policy](https://github.com/homerquan/mirrorneuron-prism/blob/main/SECURITY.md) for reporting vulnerabilities privately.

## License

Prism is open source under the [MIT License](https://github.com/homerquan/mirrorneuron-prism/blob/main/LICENSE). Copyright © 2026 mirrorneuron-prism.
