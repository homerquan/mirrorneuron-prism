# Flagship curl cases

These nine requests are runnable examples and the source of request payloads for the test suite. Run Prism with the repository configuration, or generate a new configuration with `prism init` and edit its raw-model JSON. Fixed aliases select each bounded policy even when sample sources are short. Laya is required and prepares on CPU at startup; automatic routing uses the `prism` alias.

```sh
export PRISM_BASE_URL='http://127.0.0.1:8080/v1'
export PRISM_API_KEY='your-configured-secret'
prism serve --config prism.json
```

Run the curl commands in a second terminal with the same environment variables. `--include` shows response headers, including `X-Request-ID` and the accounting/streaming mode. Answer wording can vary by model. The JSON contracts following each command are read by tests; they describe required HTTP behavior and fixture checks rather than model-quality guarantees. The curl blocks themselves supply the exact tested request bodies.

Output budgets include backend reasoning tokens. A live model can return `finish_reason: length` before producing its answer; increase the request budget within the configured profile and backend caps if needed. The evidence example uses the default public cap of 2048 tokens.

<!-- prism-case: direct-approval -->
## 1. A direct answer that preserves policy qualifications

Ask about a short source while preserving the developer instruction and the ordinary Chat Completions interface.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "prism-direct",
    "messages": [
      {"role": "developer", "content": "Answer using the provided material and retain policy qualifications."},
      {"role": "user", "content": "What deployment approvals are required? Standard deployments require manager approval recorded in a change ticket. Emergency deployments can proceed with incident commander authorization, followed by retrospective review within one business day."}
    ],
    "max_completion_tokens": 512
  }'
```

Expected: HTTP 200, one assistant answer under the public alias, one physical call, and a request ID. The answer should describe approval requirements with the emergency qualification.

```json prism-expect
{
  "status": 200,
  "kind": "completion",
  "model": "prism-direct",
  "strategy": "direct",
  "backend_calls": 1,
  "finish_reason": "stop",
  "content_contains": ["approval", "retrospective"]
}
```

<!-- prism-case: structured-incident -->
## 2. Normalize an incident into validated JSON

Use a strict public schema to turn explicit incident facts into a machine-readable result. Prism validates the public output against this schema before returning it.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "prism-direct",
    "messages": [
      {"role": "user", "content": "Normalize this incident into JSON. The service is checkout, severity is high, and the new release causes HTTP 500 errors. Recommend a rollback to the last known good release."}
    ],
    "max_completion_tokens": 1024,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "incident_triage",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "service": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "recommendation": {"type": "string"}
          },
          "required": ["service", "severity", "recommendation"],
          "additionalProperties": false
        }
      }
    }
  }'
```

Expected: HTTP 200 and assistant content containing valid JSON with the required fields. A malformed or schema-violating backend answer produces an error, rather than unchecked JSON.

```json prism-expect
{
  "status": 200,
  "kind": "completion",
  "model": "prism-direct",
  "strategy": "direct",
  "backend_calls": 1,
  "finish_reason": "stop",
  "json_values": {"service": "checkout", "severity": "high"}
}
```

<!-- prism-case: evidence-policies -->
## 3. Source-backed policy synthesis across three documents

The `prism-evidence` alias forces three extraction calls and one synthesis call. The short documents make the complete request easy to copy; a larger corpus uses the same source-boundary convention, subject to configured limits. This example demonstrates the execution path, not a native context-window or recall benchmark.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "prism-evidence",
    "messages": [
      {"role": "user", "content": "Summarize deployment requirements and their qualifications.\n<prism-source id=\"standard\">\nStandard deployments require manager approval recorded in the change ticket. The release owner must retain the approval record.</prism-source>\n<prism-source id=\"emergency\">\nEmergency deployments may proceed before approval when an incident commander authorizes the change. A retrospective review is required within one business day.</prism-source>\n<prism-source id=\"rollback\">\nA rollback to the last known good version is allowed during an active outage. The operator must record the outage identifier and notify the incident commander.</prism-source>\nAnswer in English. Cite the supplied evidence and keep each condition attached to its requirement."}
    ],
    "max_completion_tokens": 2048
  }'
```

Expected: HTTP 200, validated evidence from three required partitions, and four recorded backend calls. Instructions after the sources must survive transformation. Source coverage is visible in the authenticated trace; valid provenance does not prove that a model interpreted every fact correctly.

```json prism-expect
{
  "status": 200,
  "kind": "completion",
  "model": "prism-evidence",
  "strategy": "evidence_map",
  "backend_calls": 4,
  "finish_reason": "stop",
  "validated_partitions": 3,
  "content_contains": ["approval", "retrospective", "rollback"]
}
```

<!-- prism-case: stream-explanation -->
## 4. Stream a final answer with a usage-only final chunk

`--no-buffer` displays SSE as it arrives. The direct route forwards final-generation deltas, using a stable public ID and model alias. No worker output appears as assistant progress.

```sh
curl --silent --show-error --include --no-buffer \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "prism-direct",
    "messages": [
      {"role": "user", "content": "Briefly explain how keeping original source content and validating evidence quotes helps a proxy answer document questions."}
    ],
    "max_completion_tokens": 1024,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Expected: HTTP 200 with `Content-Type: text/event-stream` and `X-Prism-Stream-Mode: forwarded`, stable chunk IDs, a finish reason, a final chunk with empty `choices` and logical usage, then `data: [DONE]`. Usage counts UTF-8 bytes under `prism-utf8-v1`; it is not a provider bill. Adaptive and schema-constrained streaming use buffered delivery instead.

```json prism-expect
{
  "status": 200,
  "kind": "stream",
  "model": "prism-direct",
  "strategy": "direct",
  "backend_calls": 1,
  "finish_reason": "stop",
  "stream_mode": "forwarded",
  "content_contains": ["evidence"]
}
```

<!-- prism-case: caller-tool -->
## 5. Return a named tool call for the caller to execute

The caller supplies the tool and chooses its name. Prism returns the selected call with an ID and argument string. The caller executes it, then submits a tool-result message in its next request; Prism does not look up the ticket or execute application tools.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "prism-direct",
    "messages": [{"role": "user", "content": "Look up deployment ticket CHG-42."}],
    "max_completion_tokens": 512,
    "tools": [{
      "type": "function",
      "function": {
        "name": "lookup_deployment_ticket",
        "description": "Read a deployment ticket by its ID.",
        "parameters": {
          "type": "object",
          "properties": {"ticket_id": {"type": "string"}},
          "required": ["ticket_id"],
          "additionalProperties": false
        }
      }
    }],
    "tool_choice": {"type": "function", "function": {"name": "lookup_deployment_ticket"}},
    "parallel_tool_calls": false
  }'
```

Expected: HTTP 200, `finish_reason: tool_calls`, one function call named `lookup_deployment_ticket`, and JSON arguments containing `ticket_id: CHG-42`. Only one physical generation call is made; no application tool is executed.

```json prism-expect
{
  "status": 200,
  "kind": "tool",
  "model": "prism-direct",
  "strategy": "direct",
  "backend_calls": 1,
  "finish_reason": "tool_calls",
  "tool_name": "lookup_deployment_ticket",
  "tool_arguments": {"ticket_id": "CHG-42"}
}
```

<!-- prism-case: unsupported-choices -->
## 6. Reject an unsupported feature before spending backend work

The current public interface supports one choice. `n: 2` is rejected rather than interpreted as an internal candidate count.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "prism-direct",
    "messages": [{"role": "user", "content": "Explain virtual context."}],
    "n": 2,
    "max_completion_tokens": 512
  }'
```

Expected: HTTP 400 with `error.code: unsupported_feature`, `error.param: n`, and zero physical calls. Curl intentionally omits `--fail` so this expected error can be inspected; the tests assert the HTTP status explicitly.

```json prism-expect
{
  "status": 400,
  "kind": "error",
  "backend_calls": 0,
  "error_code": "unsupported_feature",
  "error_param": "n"
}
```

## Run the document as tests

```sh
python -m pip install '.[dev]'
# Document payloads sent through the ASGI request path, with a deterministic backend:
python -m pytest tests/standalone/test_flagship_curl.py -q
# The exact curl argv sent to real local Uvicorn servers and a deterministic HTTP backend:
python -m pytest tests/standalone/test_flagship_curl.py -q -m integration -o addopts=''
```

These tests do not need model weights or an external endpoint. They verify request handling, source checks, trace coverage/accounting, stream shape, and tool semantics. Generative quality still requires separate real-model evaluation. To inspect a trace from a manual request, use the returned `X-Request-ID` with `prism trace show prism-REQUEST_ID`.

<!-- prism-case: batched-policies -->
## 7. Batch short sources into fewer extraction calls

One extractor handles the three short partitions, followed by one synthesis. Each partition needs its own complete typed result and original quote validation.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
  "model": "prism-batched",
  "messages": [
    {
      "role": "user",
      "content": "Summarize deployment qualifications.\n<prism-source id=\"standard\">Standard deployments require manager approval.</prism-source>\n<prism-source id=\"emergency\">Emergency deployments require retrospective review within one business day.</prism-source>\n<prism-source id=\"rollback\">Rollback during an outage requires incident commander notification.</prism-source>\nAnswer in English."
    }
  ],
  "max_completion_tokens": 2048
}'
```

The deterministic test contract is:

```json prism-expect
{
  "status": 200,
  "kind": "completion",
  "model": "prism-batched",
  "strategy": "batched_map",
  "backend_calls": 2,
  "finish_reason": "stop",
  "validated_partitions": 3,
  "content_contains": [
    "approval",
    "retrospective",
    "rollback"
  ]
}
```

<!-- prism-case: verified-policies -->
## 8. Independently check extracted interpretations

Three extraction calls, three independent verification calls, and one synthesis. A rejected or incomplete verification returns an error; no unchecked synthesis is substituted.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
  "model": "prism-verified",
  "messages": [
    {
      "role": "user",
      "content": "Summarize deployment qualifications.\n<prism-source id=\"standard\">Standard deployments require manager approval.</prism-source>\n<prism-source id=\"emergency\">Emergency deployments require retrospective review within one business day.</prism-source>\n<prism-source id=\"rollback\">Rollback during an outage requires incident commander notification.</prism-source>\nAnswer in English."
    }
  ],
  "max_completion_tokens": 2048
}'
```

The deterministic test contract is:

```json prism-expect
{
  "status": 200,
  "kind": "completion",
  "model": "prism-verified",
  "strategy": "verified_map",
  "backend_calls": 7,
  "finish_reason": "stop",
  "validated_partitions": 3,
  "content_contains": [
    "approval",
    "retrospective",
    "rollback"
  ]
}
```

<!-- prism-case: focused-review -->
## 9. Read original spans for a focused lookup

Lexical selection reads the matching emergency partition in one final call. X-Prism-Coverage is focused. This request cannot establish exhaustive coverage or absence elsewhere in the sources.

```sh
curl --silent --show-error --include \
  --request POST "${PRISM_BASE_URL}/chat/completions" \
  --header "Authorization: Bearer ${PRISM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary '{
  "model": "prism-retrieve",
  "messages": [
    {
      "role": "user",
      "content": "When is retrospective review required?\n<prism-source id=\"standard\">Standard deployments require manager approval.</prism-source>\n<prism-source id=\"emergency\">Emergency deployments require retrospective review within one business day.</prism-source>\n<prism-source id=\"rollback\">Rollback during an outage requires incident commander notification.</prism-source>\nAnswer in English."
    }
  ],
  "max_completion_tokens": 2048
}'
```

The deterministic test contract is:

```json prism-expect
{
  "status": 200,
  "kind": "completion",
  "model": "prism-retrieve",
  "strategy": "retrieve_read",
  "backend_calls": 1,
  "finish_reason": "stop",
  "coverage": "focused",
  "selected_partitions": 1,
  "content_contains": [
    "retrospective",
    "one business day"
  ]
}
```
