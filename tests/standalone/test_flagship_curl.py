"""Exercise exact document payloads through ASGI and the actual curl executable."""

import json
import shutil
import subprocess

import httpx
import jsonschema
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from prism.api import create_app
from prism.backends import OpenAIBackend
from prism.config import PrismConfig, RawModel
from prism.contracts import PUBLIC_PARAMETERS, check_context

from .decision_stub import RulesAgent
from .flagship_cases import load_cases
from .test_http import serve

CASES = load_cases()
API_KEY = "flagship-test-key"
DIRECT_ANSWER = "Standard deployments require manager approval; emergencies need incident commander authorization and retrospective review within one business day."
SYNTHESIS_ANSWER = (
    DIRECT_ANSWER
    + " A rollback is allowed during an active outage; record the outage identifier and notify the incident commander."
)
STREAM_ANSWER = "Prism keeps original sources immutable and validates evidence quotes against their source spans."


def completion(content=None, finish="stop", **fields):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **fields},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    }


class DocumentBackend:
    """Deterministic physical backend fixture; it never executes an application tool."""

    def __init__(self):
        self.calls = []

    def respond(self, body):
        self.calls.append(body)
        if body["model"] == "physical-synth":
            if "prism_retrieved_sources" in body["messages"][-1]["content"]:
                return completion(
                    "Retrospective review is required within one business day."
                )
            return completion(SYNTHESIS_ANSWER)
        if body.get("tools"):
            return completion(
                finish="tool_calls",
                tool_calls=[
                    {
                        "id": "call_ticket_42",
                        "type": "function",
                        "function": {
                            "name": body["tool_choice"]["function"]["name"],
                            "arguments": json.dumps({"ticket_id": "CHG-42"}),
                        },
                    }
                ],
            )
        fmt = body.get("response_format", {})
        intermediate_schema = (
            fmt.get("json_schema", {}).get("name", "").startswith("prism_")
        )
        if fmt.get("type") == "json_schema" and not intermediate_schema:
            return completion(
                json.dumps(
                    {
                        "service": "checkout",
                        "severity": "high",
                        "recommendation": "Rollback to the last known good release.",
                    }
                )
            )
        if fmt.get("type") == "json_object" or intermediate_schema:
            partition = json.loads(body["messages"][-1]["content"])
            if "records" in partition:
                return completion(
                    json.dumps(
                        {
                            "supported": True,
                            "checked_record_ids": [
                                record["record_id"] for record in partition["records"]
                            ],
                            "issues": [],
                        }
                    )
                )
            if "partitions" in partition:
                return completion(
                    json.dumps(
                        {
                            "partitions": [
                                {
                                    "partition_id": part["partition_id"],
                                    "status": "complete",
                                    "needs": [],
                                    "records": [
                                        {
                                            "quote": part["source"],
                                            "fact": part["source"],
                                        }
                                    ],
                                }
                                for part in partition["partitions"]
                            ]
                        }
                    )
                )
            return completion(
                json.dumps(
                    {
                        "status": "complete",
                        "records": [
                            {"quote": partition["source"], "fact": partition["source"]}
                        ],
                        "needs": [],
                    }
                )
            )
        if body["stream"]:
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": None},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": STREAM_ANSWER},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 20},
                },
            ]
            return (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            )
        return completion(DIRECT_ANSWER)


def configuration(backend_url="http://fixture"):
    models = {
        key: RawModel(id=key, name="physical-" + key, base_url=backend_url + "/v1")
        for key in ("worker", "synth")
    }
    profiles = {
        "prism": {"direct": "worker"},
        "prism-direct": {"direct": "worker", "strategy": "direct"},
        "prism-evidence": {
            "direct": "worker",
            "worker": "worker",
            "synthesizer": "synth",
            "strategy": "evidence_map",
        },
    }
    for alias, strategy in [
        ("prism-batched", "batched_map"),
        ("prism-verified", "verified_map"),
        ("prism-retrieve", "retrieve_read"),
    ]:
        profiles[alias] = {**profiles["prism-evidence"], "strategy": strategy}
    profiles["prism-retrieve"].update(coverage="focused")
    config = PrismConfig(profiles=profiles)
    return config, models


def assert_response(case, status, headers, text):
    expected = case.expected
    assert status == expected["status"], text
    if expected["kind"] == "error":
        error = json.loads(text)["error"]
        assert error["code"] == expected["error_code"]
        assert error["param"] == expected["error_param"]
        return
    assert headers["x-request-id"].startswith("prism-")
    assert headers["x-prism-accounting"] == "prism-utf8-v1"
    assert headers["x-prism-policy"] == expected["strategy"]
    assert headers["x-prism-coverage"] == expected.get("coverage", "full")
    if expected["kind"] == "stream":
        assert headers["content-type"].startswith("text/event-stream")
        assert headers["x-prism-stream-mode"] == expected["stream_mode"]
        assert text.rstrip().endswith("data: [DONE]")
        chunks = [
            json.loads(line[6:])
            for line in text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        assert len({c["id"] for c in chunks}) == 1
        assert len({c["created"] for c in chunks}) == 1
        assert all(c["model"] == expected["model"] for c in chunks)
        assert (
            not chunks[-1]["choices"] and chunks[-1]["usage"]["completion_tokens"] > 0
        )
        assert chunks[-2]["choices"][0]["finish_reason"] == expected["finish_reason"]
        content = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c["choices"]
        )
    else:
        data = json.loads(text)
        assert data["model"] == expected["model"]
        assert data["object"] == "chat.completion" and data["id"].startswith(
            "chatcmpl-"
        )
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0 and choice["message"]["role"] == "assistant"
        assert choice["finish_reason"] == expected["finish_reason"]
        assert data["usage"]["completion_tokens"] > 0
        content = choice["message"].get("content") or ""
        if expected["kind"] == "tool":
            calls = choice["message"]["tool_calls"]
            assert len(calls) == 1 and calls[0]["id"]
            assert calls[0]["function"]["name"] == expected["tool_name"]
            assert (
                json.loads(calls[0]["function"]["arguments"])
                == expected["tool_arguments"]
            )
        if "json_values" in expected:
            value = json.loads(content)
            jsonschema.Draft202012Validator(
                case.body["response_format"]["json_schema"]["schema"]
            ).validate(value)
            assert all(
                value[key] == wanted for key, wanted in expected["json_values"].items()
            )
    for required in expected.get("content_contains", []):
        assert required.casefold() in content.casefold()


def assert_execution(case, trace, backend, models):
    expected = case.expected
    assert len(backend.calls) == expected["backend_calls"]
    if expected["kind"] == "error":
        assert trace is None
        return
    assert (
        trace["strategy"] == expected["strategy"] and trace["stop_reason"] == "complete"
    )
    usage = trace["execution_usage"]
    assert len(usage["calls"]) == expected["backend_calls"]
    assert (
        usage["consumed_or_conservatively_charged"]["calls"]
        == expected["backend_calls"]
    )
    assert all(call["usage_status"] == "reported" for call in usage["calls"])
    assert not usage["outstanding_nodes"]
    for call in backend.calls:
        model = next(model for model in models.values() if model.name == call["model"])
        parameters = {
            key: value for key, value in call.items() if key in PUBLIC_PARAMETERS
        }
        check_context(
            call["messages"], parameters, call["max_completion_tokens"], model
        )
    if "validated_partitions" in expected:
        coverage = trace["coverage"]
        assert len(coverage["validated_partitions"]) == expected["validated_partitions"]
        assert set(coverage["required_partitions"]) == set(
            coverage["validated_partitions"]
        )
        assert all(
            "Answer in English" in json.dumps(call["messages"])
            for call in backend.calls
        )
        assert backend.calls[-1]["model"] == "physical-synth"
    elif "selected_partitions" in expected:
        assert trace["coverage"]["scope"] == "focused"
        assert (
            len(trace["retrieval"]["selected_partitions"])
            == expected["selected_partitions"]
        )
        assert "prism_retrieved_sources" in backend.calls[0]["messages"][-1]["content"]
    elif backend.calls:
        assert backend.calls[0]["messages"] == case.body["messages"]
        for parameter in (
            "response_format",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ):
            if parameter in case.body:
                assert backend.calls[0][parameter] == case.body[parameter]


def test_document_has_the_flagship_contracts():
    assert {case.id for case in CASES} >= {
        "direct-approval",
        "structured-incident",
        "evidence-policies",
        "stream-explanation",
        "caller-tool",
        "unsupported-choices",
        "batched-policies",
        "verified-policies",
        "focused-review",
    }
    for case in CASES:
        argv = case.curl_args("http://127.0.0.1:8080/v1", API_KEY)
        assert all("${" not in argument for argument in argv)
        assert argv[0] == "curl"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
async def test_document_request_through_asgi(case, monkeypatch):
    monkeypatch.setenv("PRISM_API_KEY", API_KEY)
    config, models = configuration()
    backend = DocumentBackend()

    def upstream(request):
        assert (
            "authorization" not in request.headers
        )  # Do not forward the caller credential.
        result = backend.respond(json.loads(request.content))
        return (
            httpx.Response(200, text=result)
            if isinstance(result, str)
            else httpx.Response(200, json=result)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as physical:
        app = create_app(config, models=models, backend=OpenAIBackend(models, physical))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://prism"
        ) as client:
            response = await client.post(
                case.path, content=case.payload, headers=case.request_headers(API_KEY)
            )
            assert_response(case, response.status_code, response.headers, response.text)
            trace = None
            if "x-request-id" in response.headers:
                trace = (
                    await client.get(
                        "/v1/prism/traces/" + response.headers["x-request-id"],
                        headers=case.request_headers(API_KEY),
                    )
                ).json()
            assert_execution(case, trace, backend, models)


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_document_curl_over_real_http(case, monkeypatch):
    if shutil.which("curl") is None:
        pytest.skip("curl executable is required for document socket tests")
    monkeypatch.setenv("PRISM_API_KEY", API_KEY)
    backend = DocumentBackend()
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def physical(request: Request):
        assert "authorization" not in request.headers
        result = backend.respond(await request.json())
        if isinstance(result, str):
            return StreamingResponse(iter([result]), media_type="text/event-stream")
        return result

    with serve(upstream) as upstream_url:
        config, models = configuration(upstream_url)
        app = create_app(config, models=models, decision_agent=RulesAgent())
        with serve(app) as prism_url:
            result = subprocess.run(
                case.curl_args(prism_url + "/v1", API_KEY),
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
            head, text = result.stdout.split("\n\n", 1)
            lines = head.splitlines()
            status = int(lines[0].split()[1])
            headers = dict(
                (name.lower(), value.strip())
                for name, value in (line.split(":", 1) for line in lines[1:])
            )
            assert_response(case, status, headers, text)
            trace = None
            if "x-request-id" in headers:
                trace = httpx.get(
                    prism_url + "/v1/prism/traces/" + headers["x-request-id"],
                    headers=case.request_headers(API_KEY),
                    trust_env=False,
                ).json()
            assert_execution(case, trace, backend, models)
