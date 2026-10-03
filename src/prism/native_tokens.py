"""Benchmark-only native llama.cpp prompt counting; no admission shortcuts."""

import asyncio
import json

import httpx


class NativeCounter:
    """Use an explicit HTTP endpoint or argv transport to the installed tokenizer.

    The command is a curl-compatible argv prefix, never a shell expression.
    It receives JSON on stdin and endpoint arguments, including -d @-.
    """

    def __init__(self, *, base_url=None, command=None, timeout=30):
        if bool(base_url) == bool(command):
            raise ValueError("Supply exactly one native tokenizer URL or command")
        self.base_url = (base_url or "http://localhost").rstrip("/")
        self.command = command
        self.timeout = timeout

    async def post(self, endpoint, body, method="POST"):
        if self.command:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                "--fail-with-body",
                "--silent",
                "--show-error",
                "--max-time",
                str(self.timeout),
                "-X",
                method,
                "-H",
                "Content-Type:application/json",
                "-d",
                "@-",
                self.base_url + "/" + endpoint,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(json.dumps(body).encode()), self.timeout + 5
                )
            except BaseException:
                if process.returncode is None:
                    process.kill()
                await process.wait()
                raise
            if process.returncode:
                raise RuntimeError(
                    "Native tokenizer command failed; no byte estimate substituted"
                )
            data = json.loads(stdout)
        else:
            async with httpx.AsyncClient(
                timeout=self.timeout, trust_env=False
            ) as client:
                response = await client.request(
                    method, self.base_url + "/" + endpoint, json=body
                )
                response.raise_for_status()
                data = response.json()
        return data

    async def properties(self):
        return await self.post("props", {}, method="GET")

    async def count(self, messages):
        rendered = await self.post(
            "apply-template",
            {
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Native template endpoint did not return a prompt")
        data = await self.post(
            "tokenize", {"content": prompt, "add_special": True, "parse_special": True}
        )
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or any(
            type(token) is not int for token in tokens
        ):
            raise ValueError("Native tokenizer endpoint did not return token IDs")
        return len(tokens)


async def size_case(build, target, count):
    """Find a fixture at/above target native prompt tokens, without generation."""
    if target < 1:
        raise ValueError("Native target must be positive")
    low, high = 0, 32
    case = build(low)
    measured = await count(case["request"]["messages"])
    if measured > target:
        raise ValueError("Native target is smaller than the fixture's minimum prompt")
    while True:
        case = build(high)
        measured = await count(case["request"]["messages"])
        if measured >= target:
            break
        low, high = high, high * 2
        if high > 131072:
            raise ValueError("Native fixture sizing exceeded its finite limit")
    while high - low > 1:
        middle = (low + high) // 2
        case = build(middle)
        measured = await count(case["request"]["messages"])
        if measured >= target:
            high = middle
        else:
            low = middle
    case = build(high)
    measured = await count(case["request"]["messages"])
    return case, {
        "target_prompt_tokens": target,
        "native_prompt_tokens": measured,
        "filler_records": high,
    }
