# Model JSON and live capacity

Prism calls `litellm.acompletion()` inside its backend adapter. It does not start a LiteLLM proxy or Router. Prism retains authentication, routing, context admission, ledgers, output validation, streaming, and cancellation. SDK retries and automatic parameter dropping are disabled. OpenAI-compatible endpoints receive the original messages and configured model name, including vendor namespaces; native transports use LiteLLM's vendor translation. CLI help and configuration validation keep SDK imports lazy. The SDK uses its bundled cost map, but Prism's capacity results never come from that map.

## Configure each model once

`models_file` accepts a JSON file or a directory of `*.json` files, sorted by filename. Paths resolve relative to `prism.json`; absolute paths also work. The existing `{ "models": [...] }` registry remains supported. A single raw model object is also accepted. Duplicate IDs are rejected.

The provider JSON format used by GomokuBench is supported directly:

```json
{
  "models_file": "../GomokuBench/models",
  "server": {"port": 18080}
}
```

No `profiles` section is required. Every registry combination gets a direct public alias. Provider JSON IDs are `provider-key/model-alias`, which distinguishes configurations with different parameters or providers. The physical `model` defaults to the model alias; the display `name` is not sent upstream. For example, `nvidia/mistral-medium-3.5-128b-nvidia-api-fast` and `nvidia/mistral-medium-3.5-128b-nvidia-api` remain separate configurations even if they call the same physical model. Every profile references existing IDs; endpoints, credentials, and physical definitions stay exclusively in model JSON.

Provider `options.baseURL`, `apiKey`, `apiKeyEnv`, and `apiVersion` map to the corresponding backend settings. Model `extra_body` becomes inference defaults. `timeout_seconds`, `rate_limit_rpm`, `limit.context`, and `limit.output` are honored. `tools: false` excludes caller tool execution. `prism` inside a model entry accepts Prism context, concurrency, price, power, and admission settings. Unknown wire parameters are passed in `extra_body`; execution controls, credentials, identity, and token limits cannot be overridden there. A caller's explicit sampling and output settings take precedence within the configured bounds. Caller `stream` controls delivery.

`@ai-sdk/openai-compatible` explicitly selects LiteLLM's OpenAI transport, even for OpenRouter, NVIDIA, or Gemini-compatible endpoints. A vendor namespace in a model name does not change that transport. Native npm mappings include `@ai-sdk/openai`, `@ai-sdk/anthropic`, `@ai-sdk/google`, `@ai-sdk/azure`, `@ai-sdk/amazon-bedrock`, and `@ai-sdk/google-vertex`. Set provider `litellm_provider` explicitly for another SDK provider or npm transport.

Native SDK definitions can use prefixes without a base URL:

```json
{
  "models": [
    {"id": "claude", "name": "anthropic/YOUR_MODEL_ID", "api_key_env": "ANTHROPIC_API_KEY"},
    {"id": "gemini", "name": "gemini/YOUR_MODEL_ID", "api_key_env": "GEMINI_API_KEY"},
    {"id": "azure", "name": "YOUR_DEPLOYMENT", "provider": "azure", "base_url": "https://YOUR_RESOURCE.openai.azure.com", "api_version": "YOUR_API_VERSION", "api_key_env": "AZURE_API_KEY"},
    {"id": "bedrock", "name": "bedrock/YOUR_MODEL_ID", "provider_options": {"aws_region_name": "us-east-1"}},
    {"id": "vertex", "name": "vertex_ai/YOUR_MODEL_ID", "provider_options": {"vertex_project": "YOUR_PROJECT", "vertex_location": "us-central1"}}
  ]
}
```

Fill in model IDs and provider settings, and validate context/output bounds. API keys may be supplied with `api_key_env`; an explicitly configured missing variable fails visibly. Native providers can also use their SDK's conventional environment credentials or cloud identity. Dependencies required by a particular cloud transport remain provider specific.

The [local/Spark example](../examples/standalone/litellm/prism.json) defines models once in two individual provider JSON files, contacts Spark directly at `10.0.4.32`, and requires no SSH forwarding or CUDA argument:

```sh
export PRISM_API_KEY='your-prism-secret'
prism validate --config examples/standalone/litellm/prism.json
prism serve --config examples/standalone/litellm/prism.json
```

Use `docker-local/gemma` or `docker-spark/nemotron` in Chat Completions. To retain adaptive large-source routing, add a profile with `direct`, `worker`, and `synthesizer` references to those IDs plus the desired stage and graph limits. Existing profiles and raw IDs remain compatible.

## Query measured behavior

Authenticated `GET /capacity` and `POST /capacity` take query parameters `model` and `refresh`. POST also accepts a JSON body such as `{ "model": "spark", "refresh": true }`; do not specify a field twice. Select a registry combination ID or a public profile. Omitting `model` evaluates all registered combinations. Each combination runs four bounded inference calls; providers may bill those calls. Evaluation is on demand, rather than at startup.

```sh
curl --silent --show-error --fail-with-body --get \
  http://127.0.0.1:18080/capacity \
  -H "Authorization: Bearer $PRISM_API_KEY" \
  --data-urlencode 'model=docker-spark/nemotron' \
  --data-urlencode 'refresh=true'
```

For the existing local/Spark deployment, use `model=spark` or `model=local`. The same evaluation is available without starting Prism:

```sh
prism capacity --config examples/standalone/litellm/prism.json \
  --model docker-spark/nemotron
```

The API returns `data[].capabilities` with these independent observations:

| Field | Live check |
| --- | --- |
| `json_object` | Response parses as JSON and has exactly the requested randomized tag and integer value. |
| `json_schema` | Response validates against a strict schema with enum values, required fields, types, and no extra properties, despite conflicting extra-field/type instructions. |
| `image` | Model identifies a randomized four-quadrant PNG's colors in the correct spatial order; the answer appears only in pixels. |
| `reasoning` | Final answers correctly solve randomized multi-operation inventory and constrained ordering tasks; returns `passed`/`total` and whether reasoning metadata was observed. |

Each observation has `supported` (`true`, `false`, or `null`), `status`, scores, latency, a per-feature observation timestamp, and a sanitized error code when relevant. Wrong answers and explicit feature rejection are false. Authentication failures, rate limits, connection failures, insufficient context, timeouts, refusals, and truncated outputs are unknown/inconclusive. Probe output bodies, image answers, reasoning traces, and credentials are not returned. Reasoning measures these small tasks, independently of whether a provider exposes a thinking mode or trace.

Results include `checked_at`, `expires_at`, the evaluation revision, and `cached`. Defaults are a 60-second TTL, 30-second timeout per call, 512 output tokens, and two concurrent probe calls. Configure these under `capacity` in `prism.json`; use `refresh=true` for new observations. Configuration and explicit credential changes invalidate cached results, and simultaneous refreshes share work. Results expire and disappear when the service restarts. An in-flight request can be cancelled on client disconnect. For profiles, `profile.evaluation` executes the challenges through the actual graph; `profile.supported` uses those end-to-end scores. `data` separately reports physical-model observations. Profile stages can produce more than four physical calls, and unavailable/invalid graph paths remain visible. This prevents a capable worker from masking a failed final answer. CLI `capacity --model` accepts either a physical ID or a profile alias. Authentication is required unless the server was explicitly started with `--no-auth`.

These are small behavior checks for the configured model, endpoint, defaults, and current runtime. A successful schema sample does not certify constrained decoding, and a reasoning score does not establish general reasoning ability. A failed image sample records a failed probe, not proof that a vendor has no vision models. `/capacity` does not change operator admission declarations or silently enable/disable routes. Prism continues to validate every requested final JSON structure and propagates upstream failures.

User messages also accept OpenAI `image_url` content parts (inline image data or explicit HTTP(S) image URLs) on direct routes or profiles explicitly permitting `vision_synthesis`. Vision synthesis sends pixels to the image worker and bounded observations to the final text model. Images cannot be partitioned into evidence-map text workflows. Accounting includes serialized bytes and a per-image `image_token_reserve` (default 4096); operators must validate that reserve against their vision backend and image sizes. The local example uses 1024 for its small-image qualification fixture. Other modalities remain unsupported.

Profiles can set `structured_output_model` to a validated registry ID. JSON-required requests use it for the effective direct/final assignment; other requests retain their ordinary assignment. This allows a non-JSON worker such as free Nano to prepare plain text for a JSON-capable final model in `text_synthesis` or `vision_synthesis`. Typed evidence extraction and draft review still require their respective internal capabilities. [Free combinations](openrouter-nemotron-mix.md) and [native samples](../examples/standalone/providers/README.md) illustrate these choices. Live probes observe behavior without modifying admission declarations.

## Qualification in this checkout

The unit suite exercises the real LiteLLM SDK with HTTP protocol fixtures for Anthropic, Gemini, Azure, OpenRouter, DeepSeek, and Groq, including vendor request conversion and normalized usage. Other tests cover directory loading, inference defaults and output limits, missing usage, cancellation, scoring errors, credential/configuration cache invalidation, concurrent refreshes, authenticated GET/POST, and image-only direct routing. These vendor fixture checks are not live cloud qualification.

Live curl checks used local Docker Gemma and Spark Nemotron at `10.0.4.32`, with required Laya startup on a temporary Prism server. Both passed JSON-object and JSON-schema samples. Gemma passed the randomized image samples; Spark image requests returned an upstream error and were reported as unknown. Reasoning scores varied across randomized reruns, including failures; the endpoint preserves those observations rather than substituting catalog declarations. Auth, refresh/cache, Spark JSON Schema completion, and streamed completion all passed. The existing large policy fixture returned the four expected values through `evidence_map`, with 34/34 partitions validated and 38 physical calls. This is a functional regression check, without a general accuracy claim.
