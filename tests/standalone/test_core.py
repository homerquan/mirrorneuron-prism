import asyncio
import json
import os
import subprocess
import sys

import pytest

from prism.cli import main
from prism.config import DecisionConfig, Limits, RawModel, load_config
from prism.context import SourceArena, SourceRef
from prism.decision import ShadowDecision
from prism.errors import PrismError
from prism.runtime import Ledger, Plan, PlanNode


def test_init_json_validate_existing_registry_and_lightweight_help(tmp_path, capsys):
    assert main(["init", "--out-dir", str(tmp_path)]) == 0
    assert main(["validate", "--config", str(tmp_path / "prism.json")]) == 0
    before = (tmp_path / "models.json").read_text()
    assert main(["init", "--out-dir", str(tmp_path)]) == 2
    assert (tmp_path / "models.json").read_text() == before
    raw = json.loads(before)
    del raw["models"][0]["id"]
    raw["models"][0]["name"] = "local"
    (tmp_path / "models.json").write_text(json.dumps(raw))
    assert load_config(tmp_path / "prism.json")[1]["local"].name == "local"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import prism; from prism.cli import parser; parser().format_help(); assert 'torch' not in sys.modules; assert 'litellm' not in sys.modules",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_self_reference_invalid_config_and_secret_redaction(tmp_path, capsys):
    main(["init", "--out-dir", str(tmp_path)])
    capsys.readouterr()
    path = tmp_path / "models.json"
    models = json.loads(path.read_text())
    models["models"][0]["base_url"] = "http://localhost:8080/v1"
    path.write_text(json.dumps(models))
    with pytest.raises(ValueError, match="self-referential"):
        load_config(tmp_path / "prism.json")
    models["models"][0]["base_url"] = "http://bob:secret@host/v1?key=secret"
    path.write_text(json.dumps(models))
    assert main(["validate", "--config", str(tmp_path / "prism.json")]) == 2
    assert "secret" not in capsys.readouterr().out


def test_utf8_arena_lossless_spans_trailing_instructions_and_immutability():
    text = "α😀records\n" * 100
    arena = SourceArena(
        [
            {"role": "system", "content": "Do not guess"},
            {
                "role": "user",
                "content": "Summarize <prism-source>"
                + text
                + "</prism-source> answer briefly",
            },
        ]
    )
    partitions = arena.partitions(128, 100)
    assert "".join(p.text for p in partitions) == text
    assert all(len(p.text.encode()) <= 128 for p in partitions)
    assert "answer briefly" in arena.instructions[-1]["content"]
    assert arena.sources["msg-1-part-0"].text.endswith("answer briefly")
    assert arena.resolve(partitions[0].ref) == partitions[0].text
    with pytest.raises(PrismError):
        arena.resolve(SourceRef("msg-1-part-0", "wrong hash", 0, 10))
    with pytest.raises(PrismError, match="ambiguous"):
        arena.quote_ref(partitions[0], "records")


@pytest.mark.asyncio
async def test_atomic_ledger_concurrency_unknown_cost_and_cancelled_usage():
    model = RawModel(
        id="local",
        name="local",
        base_url="http://backend/v1",
        input_cost_per_million=1,
        output_cost_per_million=2,
    )
    ledger = Ledger(Limits(max_calls=2, max_input_tokens=100, max_output_tokens=40))
    results = await asyncio.gather(
        *(ledger.reserve(str(i), model, 40, 20) for i in range(4)),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, Exception) for r in results) == 2
    reservation = next(r for r in results if not isinstance(r, Exception))
    await ledger.start(reservation)
    await ledger.finish(reservation, status="cancelled")
    assert ledger.consumed["input_tokens"] == 40
    assert ledger.consumed["output_tokens"] == 20
    assert ledger.usage[0]["provider_usage"] is None
    assert ledger.usage[0]["remote_cancellation_confirmed"] is False
    model.input_cost_per_million = None
    with pytest.raises(PrismError, match="prices"):
        await Ledger(Limits(max_cost_usd=1)).reserve("x", model, 1, 1)


@pytest.mark.parametrize(
    "nodes",
    [
        (PlanNode("a", "generate", ("missing",)),),
        (PlanNode("a", "generate", ("b",)), PlanNode("b", "extract", ("a",))),
        (PlanNode("a", "shell"),),
        (PlanNode("a", "extract"), PlanNode("a", "generate")),
    ],
)
def test_checked_graph_rejects_cycles_references_and_operators(nodes):
    with pytest.raises(PrismError):
        Plan("direct", nodes).validate(10)


@pytest.mark.asyncio
async def test_laya_api_shadow_only_and_question_shape():
    class Agent:
        def system_one(self, state, questions, **kwargs):
            assert questions["strategy"]["type"] == "choice"
            assert "source_bytes" in state
            return {
                "answers": {"strategy": {"choice": "evidence_map", "confidence": 0.99}},
                "usage": {"truncated": False},
            }

    arena = SourceArena([{"role": "user", "content": "hi"}])
    result = await ShadowDecision(DecisionConfig(mode="shadow"), Agent()).propose(
        arena, "direct"
    )
    assert result["proposal"] == "evidence_map"
    assert result["disposition"] == "abstain"
    assert result["reason"] == "no_validated_execution_gate"


def test_resident_decision_process_and_cancellation_without_reload(tmp_path):
    # Stub only the optional model, exercising the actual spawn/pipe/kill lifecycle.
    (tmp_path / "laya.py").write_text("""
import time
class Agent:
    def system_one(self, state, questions, **kwargs):
        if "slow" in state["instruction_sample"]:
            time.sleep(5)
        return {"answers": {"strategy": {"choice": "direct", "confidence": 0.5}}}
def load(*args, **kwargs):
    return Agent()
""")
    script = """
import asyncio
from prism.config import DecisionConfig
from prism.context import SourceArena
from prism.decision import ShadowDecision
async def run():
    decision = ShadowDecision(DecisionConfig(mode="shadow"))
    decision.prepare()
    assert decision.process.is_alive()
    result = await decision.propose(SourceArena([{"role":"user","content":"hi"}]), "direct")
    assert result["proposal"] == "direct" and result["disposition"] == "abstain"
    try:
        async with asyncio.timeout(0.05):
            await decision.propose(SourceArena([{"role":"user","content":"slow"}]), "direct")
        raise AssertionError("deadline should cancel the process")
    except TimeoutError:
        pass
    assert decision.process is None and decision.connection is None
    result = await decision.propose(SourceArena([{"role":"user","content":"hi"}]), "direct")
    assert result["reason"] == "model_not_prepared"
    decision.close()
if __name__ == "__main__":
    asyncio.run(run())
"""
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
