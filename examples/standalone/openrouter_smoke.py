"""Behavior checks against a running Prism server; no provider catalog assertions."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from prism.capacity import image_challenge


async def run(args):
    headers = {}
    if not args.no_auth:
        import os

        headers["authorization"] = "Bearer " + os.environ["PRISM_API_KEY"]
    results = []
    async with httpx.AsyncClient(
        headers=headers, timeout=300, trust_env=False
    ) as client:
        for alias in ("prism-vision-llm", "prism-vision-reasoning"):
            messages, colors = image_challenge()
            response = await client.post(
                args.base_url + "/v1/chat/completions",
                json={
                    "model": alias,
                    "messages": messages,
                    "max_completion_tokens": 4096,
                },
            )
            data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get(
                "content"
            ) or ""
            passed = (
                response.status_code == 200
                and [s.strip().lower() for s in content.strip().split(",")] == colors
                and response.headers.get("x-prism-policy") == "vision_synthesis"
            )
            trace = (
                (
                    await client.get(
                        args.base_url
                        + "/v1/prism/traces/"
                        + response.headers["x-request-id"]
                    )
                ).json()
                if response.headers.get("x-request-id")
                else None
            )
            results.append(
                {
                    "model": alias,
                    "passed": passed,
                    "status": response.status_code,
                    "expected": colors,
                    "response": data,
                    "trace": trace,
                }
            )
            print(alias, "PASS" if passed else "FAIL", flush=True)
        response = await client.post(
            args.base_url + "/v1/chat/completions",
            json={
                "model": "prism-balanced",
                "messages": [
                    {"role": "user", "content": 'Return {"ok":true} as JSON.'}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "smoke",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean", "enum": [True]}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                },
                "max_completion_tokens": 4096,
            },
        )
        data = response.json()
        trace = (
            (
                await client.get(
                    args.base_url
                    + "/v1/prism/traces/"
                    + response.headers["x-request-id"]
                )
            ).json()
            if response.headers.get("x-request-id")
            else {}
        )
        calls = trace.get("execution_usage", {}).get("calls", [])
        passed = (
            response.status_code == 200
            and json.loads(data["choices"][0]["message"]["content"]) == {"ok": True}
            and [c["model_id"] for c in calls] == ["nemotron-super-reasoning"]
        )
        results.append(
            {
                "model": "prism-balanced",
                "check": "JSON uses Super",
                "passed": passed,
                "status": response.status_code,
                "response": data,
                "trace": trace,
            }
        )
        print("structured-output routing", "PASS" if passed else "FAIL", flush=True)
        response = await client.post(
            args.base_url + "/v1/chat/completions",
            json={
                "model": "nemotron-nano-reasoning",
                "messages": [{"role": "user", "content": "Return JSON."}],
                "response_format": {"type": "json_object"},
            },
        )
        passed = (
            response.status_code == 400
            and response.json()["error"]["code"] == "unsupported_feature"
        )
        results.append(
            {
                "model": "nemotron-nano-reasoning",
                "check": "JSON rejected before dispatch",
                "passed": passed,
                "status": response.status_code,
            }
        )
        print("Nano JSON exclusion", "PASS" if passed else "FAIL", flush=True)
        if args.capacity:
            capacity = await client.get(
                args.base_url + "/capacity",
                params={"model": "prism-vision-reasoning", "refresh": True},
            )
            results.append(
                {
                    "check": "end-to-end capacity",
                    "status": capacity.status_code,
                    "result": capacity.json(),
                }
            )
            print("capacity", capacity.status_code, flush=True)
    args.out.write_text(json.dumps(results, indent=2))
    return all(r.get("passed", True) for r in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("openrouter-smoke.json"))
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
