# Using Prism

Define physical connections in `models/`, model combinations in `profiles/`, then start one profile. Prism calls local OpenAI-compatible servers and native OpenAI, Anthropic, Gemini, and OpenRouter through LiteLLM. Your client uses Chat Completions and selects the started profile's `id`.

## First answer

Install this checkout with Python 3.11+ using `python -m pip install .`. The new profile CLI is not part of the published 0.3.1 interface. In the checkout, the files already exist:

```sh
export OPENROUTER_API_KEY='your-openrouter-key'
export PRISM_API_KEY='your-prism-client-secret'
prism start --profile prism-balanced
```

First startup prepares Laya on CPU and may download its checkpoint. Wait for server startup, then send a request from another terminal with the same client secret:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $PRISM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"prism-balanced","messages":[{"role":"user","content":"Explain Python decorators briefly."}],"max_completion_tokens":4096}'
```

For a fresh deployment directory after installing:

```sh
mkdir prism-demo && cd prism-demo
prism init --preset openrouter
prism start --profile prism-balanced
```

`init` writes individual model/profile files, prints the required environment variables and start command, and never overwrites existing destinations. Presets are `openrouter` (default), `openai`, `claude`, `gemini`, `local`, and `providers`. Single-provider presets include direct, automatic evidence, review, and vision workflows. `providers` additionally includes free Nano → native final combinations. Only the selected profile's model files are loaded.

## Select a profile

```sh
prism profiles
prism models --profile prism-openai
prism validate --profile prism-openai
export OPENAI_API_KEY='your-openai-key'
prism start --profile prism-openai --show-cost
# The same profile can be selected by an explicit path:
prism start --profile /path/to/deployment/profiles/prism-openai.json
```

`--profile NAME` resolves `profiles/NAME.json` in the working directory. An explicit JSON path also works. `models_dir` resolves relative to the profile file, regardless of your working directory. `prism profiles` lists available files; `prism models` lists the physical models referenced by those files. Both accept `--profile` to inspect one selection. Validation is offline and needs no keys. Startup checks the selected upstream keys before loading the server.

Each `models/MODEL_ID.json` is a single model object whose `id` matches its filename; use letters, digits, dots, underscores, and hyphens. `name` is the physical provider model name and can contain provider namespaces. Keep endpoint/provider/credentials/timeouts/defaults in this file. Standalone model files reject inline `api_key`; use `api_key_env`. For an unauthenticated local endpoint, omit the key variable or use null. Model stages, policies, output limits, and optional server/decision/capacity settings belong in the profile. See [copyable definitions](model-config-capacity.md).

## Native providers and local inference

| Preset | First profile | Upstream credential |
|---|---|---|
| `openai` | `prism-openai-direct` | `OPENAI_API_KEY` |
| `claude` | `prism-claude-direct` | `ANTHROPIC_API_KEY` |
| `gemini` | `prism-gemini-direct` | `GEMINI_API_KEY` |
| `openrouter` | `prism-balanced` | `OPENROUTER_API_KEY` |
| `local` | `prism-local-direct` | None for the included localhost endpoint |

The local model sample targets Docker Model Runner's `ai/gemma4:E2B` at `http://127.0.0.1:12434/engines/v1`; start that model server or edit `models/local.json` to match your backend. Native samples use documented model IDs and real SDK transports; paid-cloud examples have local fixture coverage, without live paid-provider qualification. [Provider details](../examples/standalone/providers/README.md).

The `local` preset also includes `prism`, which combines that Gemma worker with Spark's `docker.io/ai/nemotron-3.5-lightning:latest` at `http://10.0.4.32:12434/engines/v1`. Both have explicit zero token prices, excluding hardware costs. Edit `models/local-strong.json` for your Spark endpoint. The automatic OpenRouter workflow is named `prism-openrouter` in standalone profiles; its retained multi-alias config still uses the historical `prism` alias.

## Track token cost

Set `input_cost_per_million` and `output_cost_per_million` in model JSON to numeric USD rates or strings such as `"$1/m"` and `"$5/m"`. Then use `prism start --profile NAME --show-cost` for a live full-screen dashboard with spend, savings, input/output tokens, model rates, uptime, and server activity. Ctrl+C restores the terminal and prints a final summary. Redirected output and `--json` produce JSON reports. This sums **provider input and output usage** across all physical stages since startup. Unpriced/unreported calls are visible and leave total cost incomplete. Estimated savings show dollars and percent for comparable completed requests; negative savings are retained. Configure `cost_baseline_model` in a profile to choose another priced comparison model. [Full calculation and authenticated totals endpoint](cost-tracking.md).

For an OpenRouter demonstration, set `OPENROUTER_API_KEY`, start `prism start --profile prism-mock-cost --show-cost --no-auth`, then run `python examples/standalone/cost_demo.py --no-auth` from another terminal in this checkout. This profile always prepares text with free Nemotron Super and synthesizes with free Ultra using **hypothetical** rates. It needs no local model servers. It uses real responses and usage; the dashboard and costs endpoint label the simulated prices. The example uses explicit source boundaries so the final model receives compact notes rather than the original source. [Copyable request and rates](cost-tracking.md#try-openrouter-with-hypothetical-prices).

## Output, health, and authentication

```sh
prism --help
prism profiles --json
prism models --profile prism-balanced --output table
prism doctor --profile prism-balanced --probe-backends
prism capacity --profile prism-balanced --json
```

Terminal output uses tables; redirected command results are JSON. `--json` works before or after the subcommand; `--output table` forces terminal presentation and `NO_COLOR=1` disables colors. Inventory capabilities are declarations. `capacity --profile NAME` evaluates that actual profile graph unless `--model ID` selects a physical model or loaded alias. Doctor and capacity probes make real upstream calls and may be billed.

Doctor without `--probe-backends` deliberately reports unprobed backends and exits 3. Exit codes are 0 for command success, 2 for command/configuration errors, and 3 for doctor not ready. Capacity command success means the evaluation ran; inspect each result.

`PRISM_API_KEY` protects client requests by default. `prism start --profile NAME --no-auth` explicitly disables Prism authentication, while upstream credentials remain required. Anonymous callers share trace/cost access. CLI `capacity` calls providers directly and needs upstream keys rather than a Prism client key. An OpenAI client still requires a nonempty `api_key` argument; use `"unused"` when connecting to an explicitly anonymous server.

## JSON and images

`prism-balanced` calls Nano for ordinary text and Super when `response_format` requests JSON. Nano deliberately has no JSON capability declaration. `json_object` validates parseable JSON; `json_schema` validates the requested shape. JSON instructions alone do not request this routing change.

For image inputs plus JSON, start `prism-vision-reasoning`. It sends pixels to Nano and text observations to Super for the final structured answer. `prism-balanced` cannot handle image+JSON through its text-only JSON final assignment. Vision/text preparation is lossy; source-backed evidence mapping instead validates quote provenance. Neither proves answer correctness.

## Docker

```sh
docker build -t mirrorneuron-prism:dev .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENROUTER_API_KEY -e PRISM_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:dev
```

The image contains the free `prism-balanced` profile, CPU torch, a non-root user, and a health check. The volume caches Laya. `docker compose up --build` requires both environment variables. To show costs or explicitly disable client auth, replace the complete command:

```sh
docker run --rm -p 127.0.0.1:8080:8080 -e OPENROUTER_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:dev start --profile /app/profiles/prism-balanced.json \
  --host 0.0.0.0 --no-auth --show-cost
```

For a custom deployment, mount its directory read-only at `/config`, pass the selected upstream keys, and use `start --profile /config/profiles/NAME.json --host 0.0.0.0`. The directory includes both `models/` and `profiles/`; model paths resolve against the profile. The Docker build excludes local secrets and evaluation artifacts.

## Legacy combined configuration

Existing combined configurations and registries remain supported by `prism serve --config PATH`. They can expose multiple profiles plus raw-model aliases, unlike standalone `start`. For multi-alias experiments from this checkout, the bundled local and OpenRouter configs are `src/prism/resources/prism.json` and `src/prism/resources/openrouter/prism.json`; provider examples remain under `examples/standalone/providers/`. `--config` and `--profile` are mutually exclusive.

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
    model="prism-balanced",
    messages=[{"role": "user", "content": "Explain virtual context briefly."}],
    max_completion_tokens=200,
)
print(response.choices[0].message.content)
```

Text conversations, system/developer roles, JSON output, image inputs on direct and explicitly permitted vision-synthesis routes, function tools, and SSE are supported on the declared backend capabilities. Caller tools are returned to the caller for execution. Unsupported fields, modalities, `n > 1`, malformed tool relationships, and oversized instructions fail before dispatch. Tool calls, seeds, logprobs, token-ID bias, and reasoning controls require an untransformed direct route. Responses API is not implemented.

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

Ratings and task fit are heuristics; costs use conservative reservations, not predicted provider bills. Legacy profiles keep their existing routing. Current deployment models are left without invented ratings or prices. See [configuration, scoring, and request controls](model-optimization.md) and the [illustrative JSON configuration](../examples/standalone/optimization/prism.json). This is a Prism Chat Completions extension; it does not implement OpenAI compaction or Responses.

## Large source input

Start an evidence-capable profile such as `prism` (`prism start --profile prism`) and use its ID for the following request. The direct/balanced quick-start profile does not enable evidence mapping.

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

The map policies include all validated evidence in final synthesis. Optional [structured reduction](structured-reduction.md) uses bounded observations, a recursive reduce tree, and raw-evidence lookup from temporary Markdown files to avoid an oversized reducer. Without reduction, oversized synthesis fails visibly. `retrieve_read` instead answers from selected original spans and requires explicit `coverage: "focused"`; it cannot satisfy exhaustive coverage. A matching quote establishes provenance, not the truth of the worker's interpretation, and full partition submission is not proof of extraction recall. Verification adds an independent model check but does not guarantee correctness. Cross-partition dependency repair and exact aggregation are future work. Count/exhaustive-inventory requests require direct execution when the original request fits.

## Laya decisions

**`laya>=0.3.23,<0.4` is a required pip dependency**, without vendoring or importing a sibling checkout. The default decision mode is `route`, using `convaiinnovations/laya-typed-decisions`. Prism first compiles plans against capabilities, context, coverage, and resource limits. Laya sees only eligible choices, bounded instruction samples, source sizes, and call bounds. A valid, untruncated choice with `answer_confidence >= min_option_confidence` (default 0.7) selects an existing plan. Abstention, low confidence, or inference failure uses the rules-selected feasible plan. With only one eligible policy, Prism skips decision inference.

The checkpoint is prepared once at startup in a spawned CPU process. For an offline/reproducible deployment, set `model` to a prepared local checkpoint, or configure an immutable Hub `revision` and `expected_sha256` file hashes. Model loading failure prevents startup. Deadline/disconnect cancellation kills and reaps the process; subsequent decisions fall back to rules until restart. `shadow` remains available for evaluating proposals without changing execution, and still requires Laya. There is no `off` mode. Help/configuration commands keep ML imports out of the parent process. The option probability threshold is not a calibrated answer-quality guarantee.

## Streaming, accounting, and traces

Direct text requests forward the final backend's SSE deltas with one stable public ID and alias. Adaptive and JSON-constrained streams are **buffered delivery** after evidence/schema validation, indicated by `X-Prism-Stream-Mode`. Internal worker output is never emitted as assistant progress. Failures after streaming begins terminate with an SSE error without a success marker or replacement answer.

`usage` follows the fixed **`prism-utf8-v1` logical accounting** contract: UTF-8 bytes of compact serialized messages plus supplied inference parameters, and public content/tool-call bytes. This deliberately conservative logical measure is not a physical tokenizer count or a provider bill. Output-limit parameters cap the selected public backend's generation-token budget, including reasoning where reported by that backend. Backend context admission separately uses a declared conservative byte-to-token bound and chat overhead. Validate those settings against the backend's tokenizer and template.

The `X-Request-ID` header identifies a metadata-only trace:

```sh
prism trace show prism-REQUEST_ID
```

Traces expose eligible policies, the selected graph, Laya proposals and timing, source hashes, coverage, backend identities, provider-reported usage, unknown usage, cancellations, and cost upper estimates. Optimized traces also include effective controls, candidate stage models/ratings, score components, and the selection reason. `X-Prism-Policy` and `X-Prism-Coverage` identify execution on the response. Traces omit source text, prompts, evidence quotes, and credentials. Storage is in memory, scoped to the authenticated credential (or a shared anonymous namespace under `--no-auth`), with configurable capacity and TTL. Rotating the credential invalidates access to prior traces. Consumed work with missing usage is conservatively charged to resource reservations, never reported as zero provider usage. Price/usage bounds are operator assumptions; unknown billing is labeled explicitly.

## Evaluation and release

The paired benchmark examples below use the bundled **multi-alias** `src/prism/resources/prism.json` configuration, started with `prism serve --config src/prism/resources/prism.json`. A standalone `start --profile` process exposes one alias; it does not serve the benchmark defaults together.

Use the packaged six-case benchmark to compare direct execution with evidence extraction/synthesis. Each run saves inputs, answers, traces, latency percentiles, reference quality scores, physical work, and pricing estimates in a new folder:

```sh
# With Prism running and PRISM_API_KEY set:
prism benchmark run --config src/prism/resources/prism.json --out-dir benchmark-results/run-a
# Repeat after changing a model/profile, using a new folder:
prism benchmark run --config src/prism/resources/prism.json --out-dir benchmark-results/run-b
prism benchmark compare benchmark-results/run-a benchmark-results/run-b \
  --out-dir benchmark-results/comparison-a-b
```

The default compares `prism-direct` with `prism-evidence`, using three repetitions, one warmup pair, and the same answer budget. Pass `--candidate prism-batched`, `prism-verified`, or `prism-retrieve` to compare another fixed policy; `--candidate prism` measures active Laya routing. Cost stays unknown until physical usage and configured model prices are available. Read [benchmark instructions and metric definitions](benchmarking.md) before interpreting speed, quality, or cost differences.

For repeatable local/Spark Docker Model Runner Gemma4 + Spark Nemotron tests, run `.venv/bin/python examples/standalone/docker-spark/benchmark.py --repeats 3`. The [runner instructions](../examples/standalone/docker-spark/README.md) cover direct, mixed evidence, automatic small-context partitioning, and draft/review suites, with per-stage timing, Laya decisions, coverage, completion/quality, and physical-token cost proxies. The runner defaults to structured reduction; use `--reduction rolling` for a controlled alternative.

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

The wheel ships only the `prism` package, its JSON defaults, typing marker, and MIT license. The source distribution also includes the documentation and tests. CI checks Python 3.11–3.13, including the curl cases from the document. The manual release workflow uses a `pypi` GitHub environment and PyPI Trusted Publishing; configure repository/environment protections and the trusted publisher before running it. See [release instructions](releasing.md). Nothing is automatically uploaded by installation or builds.

MIT license.
