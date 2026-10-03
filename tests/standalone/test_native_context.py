"""Native-count validity, seeded gold references and benchmark qualification."""

import argparse
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from prism.benchmark import Case, grade
from prism.context import SourceArena, SourceRef
from prism.long_context_cases import FAMILIES, build_case, evidence_recall
from prism.native_tokens import NativeCounter, size_case


@pytest.mark.asyncio
async def test_native_counter_counts_rendered_special_tokens_and_rejects_bad_usage(
    monkeypatch,
):
    requests = []
    bad = False

    def handler(req):
        requests.append((req.url.path, json.loads(req.content)))
        if req.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<user>José<assistant>"})
        assert requests[-1][1] == {
            "content": "<user>José<assistant>",
            "add_special": True,
            "parse_special": True,
        }
        return httpx.Response(200, json={"tokens": [True] if bad else [1, 2, 3, 4]})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        "prism.native_tokens.httpx.AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handler), **kw),
    )
    counter = NativeCounter(base_url="http://native")
    assert await counter.count([{"role": "user", "content": "José"}]) == 4
    assert requests[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    bad = True
    with pytest.raises(ValueError, match="token IDs"):
        await counter.count([{"role": "user", "content": "José"}])
    with pytest.raises(ValueError, match="exactly one"):
        NativeCounter()


@pytest.mark.asyncio
async def test_native_sizing_uses_counter_and_never_substitutes_bytes():
    def build(count):
        return {"request": {"messages": [{"role": "user", "content": "é" * count}]}}

    async def count(messages):
        return len(messages[0]["content"]) * 7 + 23

    case, sizing = await size_case(build, 1024, count)
    assert sizing["native_prompt_tokens"] >= 1024
    assert sizing["native_prompt_tokens"] < 1031
    assert (
        len(case["request"]["messages"][0]["content"].encode())
        != sizing["native_prompt_tokens"]
    )

    async def unavailable(messages):
        raise RuntimeError("tokenizer unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        await size_case(build, 1024, unavailable)


@pytest.mark.asyncio
async def test_native_command_transports_source_on_stdin_and_reaps_cancellation(
    monkeypatch,
):
    import asyncio

    commands = []
    entered = asyncio.Event()

    class Process:
        returncode = None
        killed = False
        waited = False

        async def communicate(self, payload):
            assert json.loads(payload) == {"content": "private benchmark source"}
            entered.set()
            await asyncio.Event().wait()

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            self.waited = True

    process = Process()

    async def spawn(*args, **kwargs):
        commands.append(args)
        return process

    monkeypatch.setattr("prism.native_tokens.asyncio.create_subprocess_exec", spawn)
    counter = NativeCounter(command=["curl"])
    task = asyncio.create_task(
        counter.post("tokenize", {"content": "private benchmark source"})
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed and process.waited
    assert "private benchmark source" not in str(commands)
    assert "Content-Type:application/json" in commands[0]


@pytest.mark.parametrize("family", FAMILIES)
def test_seeded_fixture_gold_resolves_and_rubric_is_independent(family):
    case, gold = build_case(family, 11, "end", 300)
    assert Case.model_validate(case)
    assert (case, gold) == build_case(family, 11, "end", 300)
    assert case != build_case(family, 29, "end", 300)[0]
    arena = SourceArena(case["request"]["messages"])
    for ref in gold:
        assert arena.resolve(SourceRef(**ref))
    assert evidence_recall(gold, gold) == 1
    assert evidence_recall(gold, []) == 0
    partial = {**gold[0], "byte_end": gold[0]["byte_end"] - 1}
    assert evidence_recall(gold[:1], [partial]) == 0
    expected = {check["path"][0]: check["value"] for check in case["checks"]}
    assert grade(Case.model_validate(case), json.dumps(expected), True)["passed"]
    if family != "unanswerable":
        assert not grade(Case.model_validate(case), "{}", True)["passed"]
    assert "gold_spans" not in json.dumps(case["request"])


@pytest.fixture
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/standalone/docker-spark/native_context.py"
    )
    spec = importlib.util.spec_from_file_location("native_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_native_direct_detects_silent_prompt_truncation(runner):
    from prism.config import RawModel

    case, _ = build_case("distractor", 11, "end", 10)
    expected = {c["path"][0]: c["value"] for c in case["checks"]}

    def handler(req):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(expected),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    model = RawModel(id="n", name="native", base_url="http://native")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await runner.native_direct(client, model, case, 1000, 256)
    assert result["possible_truncation"]
    assert not result["quality"]["passed"]


@pytest.mark.asyncio
async def test_mismatched_native_template_stops_before_measured_jobs(
    runner, monkeypatch, tmp_path
):
    async def count(self, messages):
        return 99

    async def direct(*args):
        return {"provider_usage": {"prompt_tokens": 98}, "completed": True}

    monkeypatch.setattr(runner.NativeCounter, "count", count)
    monkeypatch.setattr(runner, "native_direct", direct)
    args = argparse.Namespace(
        out_dir=str(tmp_path / "run"),
        model="native",
        base_url="http://native",
        context_window=8192,
        tokenizer_url="http://tokenizer",
        tokenizer_command=None,
        timeout=10,
        lookup_rounds=2,
        sizes=[16384],
        families=["distractor"],
        seeds=[11],
        positions=["end"],
        repeats=1,
        policies=["direct", "evidence_map"],
        output_tokens=256,
        native_direct=True,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        await runner.execute(args)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["measured_requests"] == 0
    assert (tmp_path / "run/report.md").exists()
    with pytest.raises(FileExistsError):
        await runner.execute(args)


@pytest.mark.asyncio
async def test_declared_context_must_match_native_runtime(
    runner, monkeypatch, tmp_path
):
    async def count(self, messages):
        return 99

    async def direct(*args):
        return {"provider_usage": {"prompt_tokens": 99}, "completed": True}

    async def properties(self):
        return {"default_generation_settings": {"n_ctx": 4096}}

    monkeypatch.setattr(runner.NativeCounter, "count", count)
    monkeypatch.setattr(runner.NativeCounter, "properties", properties)
    monkeypatch.setattr(runner, "native_direct", direct)
    args = argparse.Namespace(
        out_dir=str(tmp_path / "mismatch"),
        model="native",
        base_url="http://native",
        context_window=8192,
        tokenizer_url="http://tokenizer",
        tokenizer_command=None,
        timeout=10,
        lookup_rounds=2,
        sizes=[16384],
        families=["distractor"],
        seeds=[11],
        positions=["end"],
        repeats=1,
        policies=["evidence_map"],
        output_tokens=256,
        native_direct=True,
    )
    with pytest.raises(RuntimeError, match="Declared runtime context"):
        await runner.execute(args)
    manifest = json.loads((tmp_path / "mismatch/manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["measured_requests"] == 0
