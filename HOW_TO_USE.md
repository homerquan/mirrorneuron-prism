# How to Use mn_prism

`mn_prism` is a **transparent** LiteLLM Proxy extension. From the client point of view you call a normal OpenAI-compatible endpoint. The multi-call expansion, selection, budgeting and aggregation happen inside the proxy.

## Client view – nothing changes

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="your-litellm-key",
)

response = client.chat.completions.create(
    model="smart-local",  # logical model name
    messages=[{"role": "user", "content": "Explain DAGs in one paragraph."}],
    temperature=0.7,
    max_completion_tokens=256,
)

print(response.choices[0].message.content)
```

No special headers, no extra fields, no SDK changes. The same request works with streaming, tools, JSON mode, etc.

## Operator view – two config files

### 1. LiteLLM config `litellm.yaml`

Register the custom provider and hook:

```yaml
model_list:
  - model_name: smart-local
    litellm_params:
      model: multicall/default
      num_retries: 0

litellm_settings:
  custom_provider_map:
    - provider: multicall
      custom_handler: litellm_multicall.provider.multicall_provider
  callbacks:
    - litellm_multicall.hooks.multicall_hooks
```

### 2. MultiCall policy `multicall.yaml`

Define *how* a logical model is expanded:

```yaml
schema_version: 1
runtime:
  backend: proxy_router
  integration_profile: trusted_local

policies:
  default:
    strategy: best_of_n
    allowed_model_groups: [local-small]
    candidates:
      - model: local-small
        samples: 4
        temperature: 0.7
        max_completion_tokens: 1500
    selection:
      type: llm_judge
      model: local-small
      max_completion_tokens: 160
      temperature: 0
    budgets:
      max_model_calls: 5
      max_total_tokens: 16000
      deadline_ms: 120000
```

## What happens under the hood

```
Client -> LiteLLM Proxy -> multicall/* provider
                              |
                              +-> Expand request per policy
                              +-> Generate N candidates via Router.acompletion()
                              +-> Run verifier / judge
                              +-> Select / fuse one response
                              +-> Return single OpenAI-compatible completion
```

* The client sees **one** model, one request, one response.
* The proxy makes several bounded child calls through the existing LiteLLM Router.
* No new HTTP server, no extra credentials, no client-side logic.

## Common patterns

### Best-of-N
Generate several independent samples and pick the best via LLM judge or semantic verifier.

### Adaptive
Start with 1 sample, expand only if a verifier fails.

### Decompose & aggregate
Big task → decompose into sub-tasks → call each sub-task → fuse results.

All patterns are expressed declaratively in `multicall.yaml`. The proxy remains the single entry point.

## Validation

Validate configs without making model calls:

```bash
mn_prism validate --policy-config multicall.yaml --litellm-config litellm.yaml
mn_prism doctor --policy-config multicall.yaml --litellm-config litellm.yaml
```

## Metering & cost estimate

Each logical request is metered per physical model group. `litellm_multicall.meter.RequestMeter` records calls, prompt/completion tokens and can produce a cost estimate using LiteLLM price tables.

Enable a cost estimate by sending:

```python
client.chat.completions.create(
    model="smart-local",
    messages=[...],
    extra_body={"return_cost_estimate": True}
)
```

The response includes an `x-multicall-usage` header and, when requested, a cost estimate:

```json
{
  "model_usage": {
    "local-small": {"calls": 4, "prompt_tokens": 1200, "completion_tokens": 800},
    "local-strong": {"calls": 1, "prompt_tokens": 400, "completion_tokens": 200}
  },
  "cost_estimate_usd": 0.00042
}
```

The meter is offline – no new billing is created, it only aggregates usage from child calls using LiteLLM's pricing table.

## Notes

* All child calls use the existing LiteLLM Router, so auth, rate limits, and accounting stay in place.
* The wrapper is transparent: response shape, streaming, tool calls, and usage aggregation are normalized.
* No changes are required on the client side.
