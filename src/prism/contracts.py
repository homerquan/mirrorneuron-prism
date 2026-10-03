"""Compatibility gate and the published logical accounting contract."""

import json
import math

from .errors import PrismError
from .policies import POLICIES

CONTEXT_CONTROLS = {
    "prism_cost_priority",
    "prism_model_ids",
    "prism_allowed_policies",
    "prism_max_cost_usd",
    "prism_max_calls",
}


def context_controls(value):
    """Parse proxy-only controls; never include them in upstream inference parameters."""

    def invalid(message, code="invalid_request"):
        raise PrismError(message, code, param="context_management")

    if not isinstance(value, list):
        invalid("context_management must be an array of objects")
    controls = {}
    for entry in value:
        if not isinstance(entry, dict) or not entry:
            invalid("context_management entries must be nonempty objects")
        if entry.get("type") == "compaction":
            invalid("Prism does not implement compaction", "unsupported_feature")
        if set(entry) - CONTEXT_CONTROLS:
            invalid("unsupported context_management control", "unsupported_parameter")
        if controls.keys() & entry.keys():
            invalid("duplicate context_management control")
        controls.update(entry)
    for key in ("prism_cost_priority", "prism_max_cost_usd"):
        if key in controls:
            number = controls[key]
            if type(number) not in {int, float}:
                invalid("context_management control must be a finite number")
            try:
                finite = math.isfinite(number)
            except OverflowError:
                finite = False
            if not finite:
                invalid("context_management control must be a finite number")
            if key == "prism_cost_priority" and not 0 <= number <= 1:
                invalid("prism_cost_priority must be between 0 and 1")
            if key == "prism_max_cost_usd" and number <= 0:
                invalid("prism_max_cost_usd must be positive")
    if "prism_max_calls" in controls and (
        type(controls["prism_max_calls"]) is not int or controls["prism_max_calls"] < 1
    ):
        invalid("prism_max_calls must be a positive integer")
    for key in ("prism_model_ids", "prism_allowed_policies"):
        if key in controls:
            refs = controls[key]
            if (
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or not ref for ref in refs)
                or len(set(refs)) != len(refs)
            ):
                invalid(
                    "model and policy subsets must be nonempty unique string arrays"
                )
            if key == "prism_allowed_policies" and not set(refs) <= POLICIES.keys():
                invalid("unknown Prism policy")
    return controls


PUBLIC_PARAMETERS = {
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "seed",
    "reasoning_effort",
}
DIRECT_ONLY = {
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "seed",
    "reasoning_effort",
}


def parse_json(value):
    def reject_constant(constant):
        raise ValueError("non-finite JSON number")

    return json.loads(value, parse_constant=reject_constant)


def byte_tokens(value):
    """prism-utf8-v1: one token per serialized UTF-8 byte. This is logical usage."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return len(value.encode("utf-8"))


def prompt_bound(messages, parameters, model):
    if model.enable_thinking is not None:
        parameters = {
            **parameters,
            "chat_template_kwargs": {"enable_thinking": model.enable_thinking},
        }
    return (
        math.ceil(
            byte_tokens({"messages": messages, **parameters})
            * model.tokens_per_byte_bound
        )
        + (len(messages) + 1) * model.chat_overhead_tokens
    )


def check_context(messages, parameters, output, model):
    bound = prompt_bound(messages, parameters, model)
    if (
        output > model.max_output_tokens
        or bound + output + model.safety_margin > model.context_window
    ):
        raise PrismError(
            "request does not fit this backend's context/output bound",
            "context_length_exceeded",
        )
    return bound


def validate_request(body):
    if not isinstance(body, dict):
        raise PrismError("request must be a JSON object")
    known = PUBLIC_PARAMETERS | {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "context_management",
    }
    unknown = set(body) - known
    if unknown:
        raise PrismError(
            "unsupported request parameter",
            "unsupported_parameter",
            param=sorted(unknown)[0],
        )
    if "context_management" in body:
        context_controls(body["context_management"])
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise PrismError("model must be a virtual model alias", param="model")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 1024:
        raise PrismError("messages must contain 1 to 1024 entries", param="messages")
    pending = set()
    seen_calls = set()
    total_parts = 0
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message.get("role")
            not in {
                "system",
                "developer",
                "user",
                "assistant",
                "tool",
            }
        ):
            raise PrismError("invalid message role", param="messages")
        if set(message) - {"role", "content", "name", "tool_calls", "tool_call_id"}:
            raise PrismError("unsupported message field", "unsupported_parameter")
        content = message.get("content")
        total_parts += len(content) if isinstance(content, list) else 1
        if total_parts > 1024:
            raise PrismError(
                "request exceeds 1024 content parts", "resource_limit", 413
            )
        if isinstance(content, list):
            if not content or any(
                not isinstance(part, dict)
                or set(part) != {"type", "text"}
                or part["type"] != "text"
                or not isinstance(part["text"], str)
                for part in content
            ):
                raise PrismError(
                    "only text content parts are supported", "unsupported_feature"
                )
        elif not isinstance(content, str) and not (
            message["role"] == "assistant"
            and content is None
            and message.get("tool_calls")
        ):
            raise PrismError("message content must be text", param="messages")
        if "name" in message and not isinstance(message["name"], str):
            raise PrismError("message name must be a string")
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise PrismError("tool result must reference a pending tool call")
            pending.remove(call_id)
        else:
            if pending:
                raise PrismError("pending tool calls require tool results in order")
            if "tool_call_id" in message:
                raise PrismError("tool_call_id is only valid on tool messages")
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if (
                message["role"] != "assistant"
                or not isinstance(calls, list)
                or not calls
            ):
                raise PrismError("tool_calls must be a nonempty assistant call list")
            for call in calls:
                if (
                    not isinstance(call, dict)
                    or set(call) != {"id", "type", "function"}
                    or not isinstance(call["id"], str)
                    or not call["id"]
                    or call["id"] in seen_calls
                    or call["type"] != "function"
                    or not isinstance(call["function"], dict)
                    or set(call["function"]) != {"name", "arguments"}
                    or not isinstance(call["function"]["name"], str)
                    or not isinstance(call["function"]["arguments"], str)
                ):
                    raise PrismError("invalid tool call relationship")
                seen_calls.add(call["id"])
                pending.add(call["id"])
    if pending:
        raise PrismError("all assistant tool calls require tool results")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise PrismError(
            "Prism currently supports n=1", "unsupported_feature", param="n"
        )
    if type(body.get("stream", False)) is not bool:
        raise PrismError("stream must be a boolean", param="stream")
    if "stream_options" in body:
        options = body["stream_options"]
        if (
            not body.get("stream")
            or not isinstance(options, dict)
            or set(options) - {"include_usage"}
            or type(options.get("include_usage", False)) is not bool
        ):
            raise PrismError("unsupported stream_options", param="stream_options")
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise PrismError("use only one output limit")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in body and (type(body[key]) is not int or body[key] < 1):
            raise PrismError("output limit must be a positive integer", param=key)
    for key, low, high in (
        ("temperature", 0, 2),
        ("top_p", 0, 1),
        ("frequency_penalty", -2, 2),
        ("presence_penalty", -2, 2),
    ):
        if key in body and (
            type(body[key]) not in {int, float}
            or not math.isfinite(body[key])
            or not low <= body[key] <= high
        ):
            raise PrismError("sampling parameter out of range", param=key)
    if "stop" in body and not (
        isinstance(body["stop"], str)
        or isinstance(body["stop"], list)
        and 1 <= len(body["stop"]) <= 4
        and all(isinstance(s, str) for s in body["stop"])
    ):
        raise PrismError("stop must be a string or up to four strings", param="stop")
    for key in ("parallel_tool_calls", "logprobs"):
        if key in body and type(body[key]) is not bool:
            raise PrismError("parameter must be a boolean", param=key)
    if "seed" in body and type(body["seed"]) is not int:
        raise PrismError("seed must be an integer", param="seed")
    if "reasoning_effort" in body and (
        not isinstance(body["reasoning_effort"], str)
        or body["reasoning_effort"]
        not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        }
    ):
        raise PrismError("unsupported reasoning_effort", param="reasoning_effort")
    if "top_logprobs" in body and (
        type(body["top_logprobs"]) is not int
        or not 0 <= body["top_logprobs"] <= 20
        or body.get("logprobs") is not True
    ):
        raise PrismError("top_logprobs requires logprobs=true and a value from 0 to 20")
    if "logit_bias" in body:
        bias = body["logit_bias"]
        if not isinstance(bias, dict) or any(
            not isinstance(k, str)
            or not k.isdigit()
            or type(v) not in {int, float}
            or not math.isfinite(v)
            or not -100 <= v <= 100
            for k, v in bias.items()
        ):
            raise PrismError("invalid logit_bias", param="logit_bias")
    if "tools" in body:
        tools = body["tools"]
        if not isinstance(tools, list) or not 1 <= len(tools) <= 128:
            raise PrismError("tools must be a nonempty function list", param="tools")
        names = set()
        for tool in tools:
            if (
                not isinstance(tool, dict)
                or set(tool) != {"type", "function"}
                or tool["type"] != "function"
                or not isinstance(tool["function"], dict)
            ):
                raise PrismError(
                    "only function tools are supported", "unsupported_feature"
                )
            function = tool["function"]
            if (
                set(function) - {"name", "description", "parameters", "strict"}
                or not isinstance(function.get("name"), str)
                or not function["name"]
                or function["name"] in names
                or (
                    "parameters" in function
                    and not isinstance(function["parameters"], dict)
                )
                or (
                    "description" in function
                    and not isinstance(function["description"], str)
                )
                or ("strict" in function and type(function["strict"]) is not bool)
            ):
                raise PrismError("invalid function tool definition", param="tools")
            names.add(function["name"])
    if "tool_choice" in body:
        choice = body["tool_choice"]
        if isinstance(choice, str):
            if choice not in {"none", "auto", "required"}:
                raise PrismError("unsupported tool_choice", param="tool_choice")
            if choice != "none" and not body.get("tools"):
                raise PrismError("tool_choice requires tools")
        elif (
            not isinstance(choice, dict)
            or set(choice) != {"type", "function"}
            or choice["type"] != "function"
            or not isinstance(choice["function"], dict)
            or set(choice["function"]) != {"name"}
            or choice["function"]["name"]
            not in {t["function"]["name"] for t in body.get("tools", [])}
        ):
            raise PrismError("named tool_choice must reference a supplied function")
    if "parallel_tool_calls" in body and not body.get("tools"):
        raise PrismError("parallel_tool_calls requires tools")
    fmt = body.get("response_format")
    if "response_format" in body:
        if (
            not isinstance(fmt, dict)
            or not isinstance(fmt.get("type"), str)
            or fmt.get("type")
            not in {
                "text",
                "json_object",
                "json_schema",
            }
        ):
            raise PrismError("unsupported response_format", param="response_format")
        if fmt["type"] == "json_schema":
            import jsonschema

            schema = fmt.get("json_schema")
            if (
                set(fmt) != {"type", "json_schema"}
                or not isinstance(schema, dict)
                or set(schema) - {"name", "description", "strict", "schema"}
                or not isinstance(schema.get("name"), str)
                or not isinstance(schema.get("schema"), dict)
                or ("strict" in schema and type(schema["strict"]) is not bool)
            ):
                raise PrismError("invalid JSON Schema response_format")
            try:
                jsonschema.Draft202012Validator.check_schema(schema["schema"])
            except jsonschema.SchemaError as exc:
                raise PrismError("invalid output JSON Schema") from exc
            _reject_remote_refs(schema["schema"])
        elif set(fmt) != {"type"}:
            raise PrismError("unsupported response_format field")
    return body


def _reject_remote_refs(value):
    if isinstance(value, dict):
        for key in ("$ref", "$dynamicRef", "$recursiveRef"):
            if key in value and (
                not isinstance(value[key], str) or not value[key].startswith("#/")
            ):
                raise PrismError("only local JSON Schema references are supported")
        for item in value.values():
            _reject_remote_refs(item)
    elif isinstance(value, list):
        for item in value:
            _reject_remote_refs(item)


def required_capabilities(body):
    result = {"text"}
    if body.get("stream"):
        result.add("stream")
    if "tools" in body or any(
        m.get("tool_calls") or m["role"] == "tool" for m in body["messages"]
    ):
        result.add("tools")
    fmt = body.get("response_format", {}).get("type", "text")
    if fmt != "text":
        result.add(fmt)
    for key in {"logprobs", "logit_bias", "seed", "reasoning_effort"}:
        if key in body:
            result.add(key)
    return result


def validate_output(message, fmt):
    if not fmt or fmt["type"] == "text":
        return
    import jsonschema

    try:
        value = parse_json(message.get("content", ""))
        if not isinstance(value, dict) and fmt["type"] == "json_object":
            raise ValueError()
        if fmt["type"] == "json_schema":
            jsonschema.Draft202012Validator(fmt["json_schema"]["schema"]).validate(
                value
            )
    except (
        ValueError,
        TypeError,
        jsonschema.ValidationError,
        jsonschema.exceptions._RefResolutionError,
    ) as exc:
        raise PrismError(
            "backend result failed the public output schema",
            "invalid_backend_output",
            502,
        ) from exc
