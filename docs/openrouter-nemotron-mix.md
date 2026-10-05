# Free OpenRouter combinations

The checkout's `prism-openrouter.json` and `models/openrouter-nemotron-mix.json` are mirrored in the installed `prism init --preset openrouter` resources. The existing local `prism.json` remains available. All three upstream slugs explicitly end in `:free`; input/output token prices are zero, with no invented power ratings.

## Physical models

| Registry ID | Native LiteLLM model | Context / output | Admission |
|---|---|---:|---|
| `nemotron-ultra` | `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` | 262144 / 16384 | Text, JSON, JSON Schema; no image |
| `nemotron-super-reasoning` | `openrouter/nvidia/nemotron-3-super-120b-a12b:free` | 262144 / 16384 | Text, JSON, JSON Schema; no image |
| `nemotron-nano-reasoning` | `openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 256000 / 16384 | Text and image; JSON modes deliberately excluded |

Each uses `OPENROUTER_API_KEY`, concurrency 1, and a 300-second backend timeout. The larger output budget accommodates reasoning as well as visible content. Limits are conservative operator settings; they do not certify the provider's full context window. Transport support in [LiteLLM's OpenRouter docs](https://docs.litellm.ai/docs/providers/openrouter) does not imply every model supports JSON. [JSON mode documentation](https://docs.litellm.ai/docs/completion/json_mode) recommends a JSON instruction and model-specific support checks; Prism also evaluates actual outputs.

## Profiles

| Alias | Workflow | JSON final |
|---|---|---|
| `prism`, `prism-evidence` | Super typed evidence → Ultra synthesis; `prism` can select direct | Ultra |
| `prism-direct` | Ultra direct | Ultra |
| `prism-vision-llm`, `prism-omni-llm` | Image: Nano observations → Ultra; text: Ultra direct | Ultra |
| `prism-vision-reasoning` | Image: Nano observations → Super; text: Super direct | Super |
| `prism-reasoning-image`, `prism-reasoning-omni` | Super draft/review → Nano; image/tools take Nano direct | Super |
| `prism-llm-omni` | Ultra draft → Super review → Nano; image/tools take Nano direct | Super |
| `prism-balanced` | Nano direct | Super |
| `prism-nano-synthesis` | Nano plain-text preparation → Super | Super |

Every physical registry ID also gets a direct public alias. Nano's direct alias rejects `response_format` JSON requirements before dispatch. `structured_output_model` switches the profile's direct/final assignment when JSON is required; it does not silently change to an arbitrary provider. Profiles using draft/review still need a JSON-capable reviewer. Request output limits and graph limits remain enforced.

For image inputs combined with JSON requirements, use `prism-vision-reasoning`, `prism-vision-llm`, or `prism-omni-llm`: their image worker prepares text for a JSON-capable final. `prism-balanced` and the draft/review aliases switch their direct model to text-only Super for JSON, so combined image+JSON requests on those aliases fail admission. The server does not silently switch to a different profile.

`vision_synthesis` sends original pixels only to the image worker, then sends bounded observations plus the original text instructions to the final model. `text_synthesis` condenses explicit `<prism-source>` blocks into worker notes; source-free prompts retain their original text. Both are lossy, unverified preparation. Headers label `visual_observations` or `text_observations`; these routes do not promise exhaustive evidence coverage. Evidence-map policies use Super rather than Nano for typed worker output.

## Run and test

```sh
export OPENROUTER_API_KEY='your-openrouter-key'
export PRISM_API_KEY='your-prism-client-secret'
prism validate --config prism-openrouter.json
prism profiles --config prism-openrouter.json
prism doctor --config prism-openrouter.json --probe-backends
prism capacity --config prism-openrouter.json --model nemotron-nano-reasoning
prism capacity --config prism-openrouter.json --model prism-vision-reasoning
prism serve --config prism-openrouter.json
```

To explicitly skip client auth, add `--no-auth` to `serve`, `doctor`, and remote trace/benchmark commands. CLI capacity calls upstream directly. With a running server:

```sh
python examples/standalone/openrouter_smoke.py --capacity --out openrouter-smoke.json
python examples/standalone/openrouter_evaluation.py --repeats 2 --out-dir benchmark-results/openrouter-pilot
```

The smoke test creates a randomized image whose correct answer appears only in pixels, checks the final model's answer, verifies JSON routing by trace, and verifies Nano rejects direct JSON. Failed calls remain in the saved run.

## Live evidence, October 4, 2026

Nano passed an image challenge and small reasoning tasks; an initial JSON-object challenge failed. Its JSON declarations remain excluded as requested. Super and Ultra passed JSON-object, JSON-schema, and small reasoning challenges. Results are transient behavior samples, not constrained-decoding or general quality certification.

The Nano → Super image route passed end to end. Its profile capacity checks passed JSON object, JSON Schema, image, and reasoning through the actual graph. The Nano → Ultra smoke route encountered an upstream error at Nano's worker stage on both attempts; this does not establish failure in Ultra. Both the [initial smoke run](evaluations/2026-10-04-openrouter/smoke.json) and [explicit rerun](evaluations/2026-10-04-openrouter/smoke-rerun.json) retain the failures. [Physical capacity observations](evaluations/2026-10-04-openrouter/physical-capacity.json) preserve the model-level results. The [Docker smoke check](evaluations/2026-10-04-openrouter/docker-smoke.json) also passed real Super JSON Schema output under explicit anonymous serving. Tests with HTTP protocol fixtures verify image handoff, JSON finalization, truncation rejection, reservations, and buffered streaming. See the [benchmark report](evaluations/2026-10-04-openrouter/benchmark-results.md) for every measured task attempt and limitations.

Free-tier overload, quotas, and rate limits can make calls fail or probes inconclusive. Capacity labels authentication, connectivity, rate-limit, timeout, refusal, and truncation outcomes unknown; it does not turn them into unsupported model claims. No hidden inference retry or catalog-based pass substitutes for the measurements.
