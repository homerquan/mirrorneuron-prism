"""Local Laya proposals stay isolated from execution authority."""

import asyncio
import hashlib
import json
import multiprocessing
import time

from .contracts import byte_tokens

QUESTIONS = {
    "strategy": {
        "type": "choice",
        "instructions": "Which execution strategy fits this request?",
        "criteria": {
            "direct": "Answer a small self-contained request using one model call",
            "evidence_map": "Read explicit large sources in partitions and synthesize source-backed evidence",
            "abstain": "Unknown task, exact aggregation, or insufficient context to choose safely",
        },
    }
}


class ShadowDecision:
    def __init__(self, config, agent=None):
        self.config = config
        self.agent = agent
        self.lock = asyncio.Lock()
        self.process = None
        self.connection = None

    def prepare(self):
        """Explicit startup preparation; never load/download a model on a request."""
        if self.config.mode == "shadow" and self.agent is None:
            from .decision_worker import serve

            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            self.process = context.Process(
                target=serve, args=(child, self.config.model_dump()), daemon=True
            )
            self.connection = parent
            self.process.start()
            child.close()
            try:
                if not parent.poll(300) or not parent.recv().get("ready"):
                    raise ValueError(
                        "Laya preparation failed; verify the optional package and checkpoint"
                    )
            except BaseException:
                self.close()
                raise

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    async def _predict(self, state):
        if self.agent is not None:  # Injection for tests and embedded applications.
            return await asyncio.to_thread(
                self.agent.system_one,
                state,
                QUESTIONS,
                max_len=self.config.max_len,
                head_max_len=256,
            )
        self.connection.send({"state": state, "questions": QUESTIONS})
        while not await asyncio.to_thread(self.connection.poll, 0.05):
            if not self.process.is_alive():
                raise RuntimeError("decision worker exited")
        message = self.connection.recv()
        if "result" not in message:
            raise RuntimeError("decision unavailable")
        return message["result"]

    async def propose(self, arena, selected_strategy):
        if self.config.mode == "off":
            return {"mode": "off", "disposition": "abstain"}
        if self.agent is None and self.process is None:
            return {
                "mode": "shadow",
                "disposition": "abstain",
                "reason": "model_not_prepared",
            }
        # First and last instruction samples; sources remain outside controller context.
        text = json.dumps(arena.instructions, ensure_ascii=False)
        raw = text.encode("utf-8")
        half = self.config.max_state_bytes // 2
        state = {
            "instruction_sample": (
                raw[:half].decode("utf-8", errors="ignore")
                + "\n...\n"
                + raw[-half:].decode("utf-8", errors="ignore")
            ),
            "source_bytes": sum(r.byte_end - r.byte_start for r in arena.documents),
            "source_count": len(arena.documents),
            "rules_strategy": selected_strategy,
        }
        started = time.monotonic()
        try:
            async with self.lock:
                try:
                    result = await self._predict(state)
                except asyncio.CancelledError:
                    # Kill/reap the resident worker; no request-time reload/download.
                    # Later shadow requests abstain until explicit service restart.
                    self.close()
                    raise
            answer = result["answers"]["strategy"]
            choice = answer.get("choice")
            if choice not in QUESTIONS["strategy"]["criteria"]:
                raise ValueError("invalid decision")
            return {
                "mode": "shadow",
                "disposition": "abstain",
                "reason": "no_validated_execution_gate",
                "proposal": choice,
                "confidence": answer.get("confidence"),
                "usage": result.get("usage"),
                "model": self.config.model,
                "revision": self.config.revision,
                "question_sha256": hashlib.sha256(
                    json.dumps(QUESTIONS, sort_keys=True).encode()
                ).hexdigest(),
                "state_sample_bytes": byte_tokens(state),
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
        except Exception:
            return {
                "mode": "shadow",
                "disposition": "abstain",
                "reason": "decision_unavailable",
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
