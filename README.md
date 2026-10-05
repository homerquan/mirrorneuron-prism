# Prism

[![PyPI](https://img.shields.io/pypi/v/mirrorneuron-prism.svg)](https://pypi.org/project/mirrorneuron-prism/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/homerquan/mirrorneuron-prism/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://github.com/homerquan/mirrorneuron-prism/blob/main/pyproject.toml)

**Combine local and cloud models behind one OpenAI-compatible API.** Define connections once in `models/`, combine them in `profiles/`, and start a profile. Your application receives one assistant response.

Prism can send a small model's notes to a stronger model, have a vision model interpret an image before text synthesis, or draft and review an answer. Use the same OpenAI client for every workflow. Prism is part of [MirrorNeuron](https://www.mirrorneuron.io) and also runs independently.

## Quick start: get your first answer

These profile commands are new in this checkout. Install from source with Python 3.11+; the published 0.3.1 package has the older `serve --config` interface.

```sh
git clone https://github.com/homerquan/mirrorneuron-prism.git
cd mirrorneuron-prism
python -m pip install .
export OPENROUTER_API_KEY='your-openrouter-key'
export PRISM_API_KEY='your-prism-client-secret'
prism start --profile prism-balanced
```

The repository already includes the model and profile files. First startup prepares Laya's CPU routing checkpoint and may download weights. The server is ready at `http://127.0.0.1:8080` after startup completes. This profile uses OpenRouter models with explicit `:free` IDs; upstream availability and quotas apply.

In another terminal, set the same `PRISM_API_KEY` and send a request:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $PRISM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"prism-balanced","messages":[{"role":"user","content":"Explain Python decorators in three sentences."}],"max_completion_tokens":4096}'
```

To create a separate deployment directory after installation:

```sh
mkdir prism-demo && cd prism-demo
prism init --preset openrouter
prism start --profile prism-balanced
```

`init` writes `models/` and `profiles/` and never overwrites existing files. Use `--preset openai`, `claude`, `gemini`, `local`, or `providers` for other samples. Cloud keys are read from environment variables.

## Choose your provider and workflow

Set `PRISM_API_KEY` plus the upstream key, then start one of these included profiles:

| Profile | Work performed | Upstream key |
|---|---|---|
| `prism-openai-direct` | One OpenAI Luna call | `OPENAI_API_KEY` |
| `prism-openai` | OpenAI Luna extracts evidence; Astra finishes when needed | `OPENAI_API_KEY` |
| `prism-claude-direct` / `prism-claude` | Haiku alone / Haiku evidence → Opus | `ANTHROPIC_API_KEY` |
| `prism-gemini-direct` / `prism-gemini` | Flash-Lite alone / Flash-Lite evidence → Flash | `GEMINI_API_KEY` |
| `nano-openai` | Free Nano text preparation → OpenAI Astra | `OPENROUTER_API_KEY` + `OPENAI_API_KEY` |
| `prism-openai-reviewed` | Luna draft → Astra review → Astra answer | `OPENAI_API_KEY` |
| `prism-vision-reasoning` | Free Nano image observations → free Super answer | `OPENROUTER_API_KEY` |
| `prism-local-direct` | Docker Model Runner Gemma4 at localhost:12434 | None upstream |
| `prism` | Local Gemma preparation → Spark Nemotron when needed; zero token rates | None upstream |
| `prism-openrouter` | Free Super evidence → free Ultra when needed | `OPENROUTER_API_KEY` |

```sh
export OPENAI_API_KEY='your-openai-key'
prism start --profile prism-openai --show-cost
```

The request's `model` must match the started profile's `id`. Only that profile is served, and only its referenced models are loaded. Other providers' keys are unnecessary. `auto` prefers a fitting direct request; evidence routing needs explicit source blocks. See [usage and sources](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/usage.md).

Native provider examples use real SDK transports and documented model IDs. They have configuration and local protocol-fixture coverage; live paid-provider qualification remains account specific. [Provider samples and sources](https://github.com/homerquan/mirrorneuron-prism/blob/main/examples/standalone/providers/README.md).

## Define once, combine freely

Each `models/MODEL_ID.json` holds its physical name, provider, endpoint, environment key, timeouts, inference defaults, capabilities, token bounds, concurrency, and optional prices. Each `profiles/NAME.json` holds model references and workflow limits. Connection settings stay in the model file.

```text
models/
  openai-small.json
  openai-strong.json
  claude-small.json
  gemini-small.json
profiles/
  prism-openai-direct.json
  prism-openai.json
  prism-openai-reviewed.json
```

Copy a profile, change its `id` and stage model IDs, then start it by name or path:

```sh
prism profiles
prism models --profile prism-openai
prism validate --profile profiles/prism-openai.json
prism start --profile profiles/prism-openai.json
```

[Copyable model and profile definitions](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/model-config-capacity.md). Existing combined configurations still work through `prism serve --config PATH`.

## See token spend and estimated savings

In a model definition, set both prices in USD per million tokens:

```json
"input_cost_per_million": "$1/m",
"output_cost_per_million": "$5/m"
```

Numbers such as `1` and `5` also work. `prism start --profile NAME --show-cost` opens a **full-screen terminal dashboard** with cumulative spend, estimated savings in dollars and percent, input/output tokens, and usage per model. It refreshes while requests run; Ctrl+C stops the server, restores the terminal, and prints a final summary. `--json` or redirected output keeps ordinary JSON reports. Spend uses each physical call's provider-reported **input and output** tokens and that model's configured rates, including all mixing stages and server-side capacity probes. Missing prices or usage keep the total visibly incomplete.

Savings compare completed, fully priced requests with the profile's final/direct model (or `cost_baseline_model`). Mixed routes estimate the original text input with a shared tokenizer and reuse the final reported output count. Identical direct routes show zero savings; higher mixing cost shows negative savings. Image requests and requests that cannot fit the baseline are excluded from estimates. Free profiles retain zero rates. These are token-cost estimates, without provider discounts or a claim of equal answer quality. [Calculation and API details](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/cost-tracking.md).

To try the dashboard with real OpenRouter calls and hypothetical prices:

```sh
prism start --profile prism-mock-cost --show-cost --no-auth
# In another terminal, from this checkout:
python examples/standalone/cost_demo.py --no-auth
```

Set `OPENROUTER_API_KEY` before starting Prism. The demo uses OpenRouter's free Nemotron Super worker and Ultra final model with separate, explicitly hypothetical rates. Only the prices are simulated; responses and reported tokens are real. Regular OpenRouter model definitions retain zero rates, and `prism` remains the local Gemma/Spark profile. The demo ships with `prism init --preset openrouter`; [rates and a copyable request](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/cost-tracking.md#try-openrouter-with-hypothetical-prices).

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

Install the client with `python -m pip install openai`. Prism supports Chat Completions, text, image understanding, direct tool calls, JSON validation, and SSE. Multi-stage and JSON-constrained streams are buffered until validation. Responses, image generation, audio, and video are not implemented. Preparation can lose information; evaluate your workload before relying on a combination.

## Run with Docker

```sh
docker build -t mirrorneuron-prism:dev .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENROUTER_API_KEY -e PRISM_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:dev
```

Or use `docker compose up --build`. The image starts `prism-balanced`. [Custom profiles, caching, and auth](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/usage.md#docker).

## Inspect and develop

```sh
prism doctor --profile prism-balanced --probe-backends
prism capacity --profile prism-balanced
prism trace show prism-REQUEST_ID
```

Doctor/capacity make real upstream calls. Capabilities in model JSON are declarations; capacity reports live observations. Terminal output uses tables; redirected output is JSON. Use `--json` or `NO_COLOR=1` explicitly.

[Usage](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/usage.md) · [Execution policies](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/execution-policies.md) · [Cost tracking](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/cost-tracking.md) · [Benchmarking](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/benchmarking.md) · [Implemented contract](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/standalone-contract.md) · [Contributing](https://github.com/homerquan/mirrorneuron-prism/blob/main/CONTRIBUTING.md)

```sh
python -m pip install '.[dev]'
ruff check src tests examples
python -m pytest -q
python -m pytest tests/standalone -q -m integration -o addopts=''
python -m build
python -m twine check dist/*
```

Deterministic HTTP and decision fixtures are confined to tests. Production backends call the configured provider; upstream failures remain visible. See the [live pilot](https://github.com/homerquan/mirrorneuron-prism/blob/main/docs/evaluations/2026-10-04-openrouter/benchmark-results.md) for measured quality, cost, and latency tradeoffs. Prism is alpha software, licensed under [MIT](https://github.com/homerquan/mirrorneuron-prism/blob/main/LICENSE).
