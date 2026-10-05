"""Read model JSON registries without consulting provider capability catalogs."""

import json
from pathlib import Path

# Inference defaults, not arbitrary LiteLLM execution options. Unknown provider
# request fields belong inside extra_body, which is passed as wire JSON only.
INFERENCE_PARAMETERS = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "reasoning_effort",
    "thinking",
    "reasoning",
    "verbosity",
    "chat_template_kwargs",
    "extra_body",
    "stream",
}
PROVIDER_OPTIONS = {
    "aws_region_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_session_name",
    "vertex_project",
    "vertex_location",
    "vertex_credentials",
    "azure_ad_token",
    "organization",
    "extra_headers",
}
PROTECTED = {
    "model",
    "messages",
    "stream",
    "n",
    "api_key",
    "api_base",
    "base_url",
    "api_version",
    "max_tokens",
    "max_completion_tokens",
    "timeout",
    "client",
    "custom_llm_provider",
    "mock_response",
    "num_retries",
    "max_retries",
    "drop_params",
    "fallbacks",
    "context_window_fallback_dict",
    "success_callback",
    "failure_callback",
    "callbacks",
    "caching",
}


def validate_parameters(parameters, options):
    if set(parameters) - INFERENCE_PARAMETERS:
        raise ValueError("unsupported model parameter; use extra_body for wire fields")
    if set(options) - PROVIDER_OPTIONS:
        raise ValueError("unsupported provider option")
    if "extra_body" in parameters:
        extra = parameters["extra_body"]
        if not isinstance(extra, dict) or PROTECTED & extra.keys():
            raise ValueError("extra_body cannot override model identity or call limits")
    if {"max_tokens", "max_completion_tokens"} <= parameters.keys():
        raise ValueError("model defaults must use only one output limit spelling")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in parameters and (
            type(parameters[key]) is not int or parameters[key] < 1
        ):
            raise ValueError("model output default must be a positive integer")


def _defaults(extra):
    known = {k: v for k, v in extra.items() if k in INFERENCE_PARAMETERS}
    unknown = {k: v for k, v in extra.items() if k not in INFERENCE_PARAMETERS}
    if unknown:
        known["extra_body"] = {**known.get("extra_body", {}), **unknown}
    return known


def normalize(data):
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return [{"id": m.get("id", m.get("name")), **m} for m in data["models"]]
    if (
        isinstance(data, dict)
        and "provider" in data
        and isinstance(data["provider"], dict)
    ):
        result = []
        transports = {
            "@ai-sdk/openai-compatible": "openai",
            "@ai-sdk/openai": "openai",
            "@ai-sdk/anthropic": "anthropic",
            "@ai-sdk/google": "gemini",
            "@ai-sdk/google-vertex": "vertex_ai",
            "@ai-sdk/amazon-bedrock": "bedrock",
            "@ai-sdk/azure": "azure",
        }
        for provider_id, provider in data["provider"].items():
            options = provider.get("options", {})
            transport = provider.get(
                "litellm_provider", transports.get(provider.get("npm"))
            )
            if not transport:
                if provider.get("npm"):
                    raise ValueError(
                        "unknown provider npm transport; set litellm_provider"
                    )
                transport = provider_id
            for alias, definition in provider.get("models", {}).items():
                parameters = _defaults(
                    {
                        **options.get("extra_body", {}),
                        **definition.get("extra_body", {}),
                    }
                )
                limit = definition.get("limit", {})
                raw = {
                    "id": f"{provider_id}/{alias}",
                    "name": definition.get("model", alias),
                    "provider": transport,
                    "base_url": options.get("baseURL"),
                    "api_key": options.get("apiKey"),
                    "api_key_env": options.get("apiKeyEnv"),
                    "api_version": options.get("apiVersion"),
                    "parameters": parameters,
                    "provider_options": options.get("provider_options", {}),
                    "context_window": limit.get("context", 32768),
                    "max_output_tokens": limit.get(
                        "output",
                        parameters.get(
                            "max_tokens", parameters.get("max_completion_tokens", 4096)
                        ),
                    ),
                    "timeout_seconds": definition.get("timeout_seconds", 120),
                    "rate_limit_rpm": definition.get("rate_limit_rpm"),
                    **definition.get("prism", {}),
                }
                if definition.get("tools") is False:
                    raw["capabilities"] = raw.get(
                        "capabilities",
                        [
                            "text",
                            "stream",
                            "json_object",
                            "json_schema",
                            "image",
                            "reasoning_effort",
                        ],
                    )
                    raw["capabilities"] = [
                        c for c in raw["capabilities"] if c != "tools"
                    ]
                result.append(raw)
        return result
    if isinstance(data, dict) and "name" in data:
        return [{"id": data.get("id", data["name"]), **data}]
    raise ValueError("expected a models registry, model definition, or provider JSON")


def read_models(path):
    path = Path(path)
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("model directory contains no JSON files")
    return [
        model for file in paths for model in normalize(json.loads(file.read_text()))
    ]
