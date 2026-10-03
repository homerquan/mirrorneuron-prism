"""Required resident Laya routing over runtime-validated policy choices."""

import asyncio
import hashlib
import json
import math
import multiprocessing
import time

from .contracts import byte_tokens
from .policies import POLICIES

QUESTIONS = {
    "strategy": {
        "type": "choice",
        "instructions": "Choose the cheapest adequate policy for this task from the eligible policies. Use verification for disputed interpretations, batching for short documents, and retrieval only for focused lookup.",
        "criteria": {
            **POLICIES,
            "abstain": "Unknown task, exact aggregation, or insufficient context to choose safely",
        },
    }
}


class LayaDecision:
    def __init__(self, config, agent=None):
        self.config = config
        self.agent = agent
        self.lock = asyncio.Lock()
        self.process = None
        self.connection = None

    def prepare(self):
        """Explicit startup preparation; never load/download a model on a request."""
        if self.agent is None:
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
                        "Laya preparation failed; verify the installed package and checkpoint"
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

    async def _predict(self, state, questions):
        if self.agent is not None:  # Injection for tests and embedded applications.
            return await asyncio.to_thread(
                self.agent.system_one,
                state,
                questions,
                max_len=self.config.max_len,
                head_max_len=256,
            )
        self.connection.send({"state": state, "questions": questions})
        while not await asyncio.to_thread(self.connection.poll, 0.05):
            if not self.process.is_alive():
                raise RuntimeError("decision worker exited")
        message = self.connection.recv()
        if "result" not in message:
            raise RuntimeError("decision unavailable")
        return message["result"]

    async def propose(self, arena, selected_strategy, eligible=None, features=None):
        eligible = list(POLICIES) if eligible is None else eligible
        if len(eligible) == 1:
            return {
                "mode": self.config.mode,
                "disposition": "rules_only",
                "reason": "single_eligible_policy",
                "proposal": eligible[0],
            }
        if self.agent is None and self.process is None:
            return {
                "mode": self.config.mode,
                "disposition": "abstain",
                "reason": "model_not_prepared",
            }
        # First and last instruction samples; sources remain outside controller context.
        text = json.dumps(arena.instructions, ensure_ascii=False)
        raw = text.encode("utf-8")
        half = self.config.max_state_bytes // 2
        sample = (
            text
            if len(raw) <= self.config.max_state_bytes
            else raw[:half].decode("utf-8", errors="ignore")
            + "\n...\n"
            + raw[-half:].decode("utf-8", errors="ignore")
        )
        state = {
            "instruction_sample": sample,
            "source_bytes": sum(r.byte_end - r.byte_start for r in arena.documents),
            "source_count": len(arena.documents),
            "rules_strategy": selected_strategy,
            "eligible_policies": eligible,
            **(features or {}),
        }
        questions = {
            "strategy": {
                **QUESTIONS["strategy"],
                "criteria": {
                    key: value
                    for key, value in QUESTIONS["strategy"]["criteria"].items()
                    if key in eligible or key == "abstain"
                },
            }
        }
        if features and "candidate_plans" in features:
            questions["strategy"]["instructions"] = (
                "Recommend the most adequate eligible plan that can finish the task, considering its assigned "
                "models, ratings, conservative total cost, coverage, and prism_cost_priority "
                "(0 favors power, 1 favors cost). Use draft_review for tasks benefiting from "
                "critique, verified_map for disputed source interpretations, and direct for "
                "self-contained tasks. Prism applies deterministic cost/power ranking; your "
                "recommendation contributes a bounded task-fit bonus. Review and synthesis "
                "are harder than partition extraction; consider the relative stage model "
                "ratings. Abstain if uncertain."
            )
        started = time.monotonic()
        try:
            async with self.lock:
                try:
                    result = await self._predict(state, questions)
                except asyncio.CancelledError:
                    # Kill/reap the resident worker; no request-time reload/download.
                    # Later decisions abstain until explicit service restart.
                    self.close()
                    raise
            answer = result["answers"]["strategy"]
            choice = answer.get("choice")
            if choice not in questions["strategy"]["criteria"]:
                raise ValueError("invalid decision")
            confidence = answer.get("answer_confidence")
            usable = (
                type(confidence) in {int, float}
                and math.isfinite(confidence)
                and 0 <= confidence <= 1
            )
            truncated = result.get("usage", {}).get("truncated", False)
            accepted = (
                self.config.mode == "route"
                and choice in eligible
                and usable
                and confidence >= self.config.min_option_confidence
                and not truncated
            )
            reason = (
                "shadow_mode"
                if self.config.mode == "shadow"
                else "model_abstained"
                if choice == "abstain"
                else "decision_input_truncated"
                if truncated
                else "low_or_missing_option_confidence"
                if not accepted
                else "validated_policy_choice"
            )
            entropy_confidence = answer.get("confidence")
            if type(entropy_confidence) not in {int, float} or not math.isfinite(
                entropy_confidence
            ):
                entropy_confidence = None
            return {
                "mode": self.config.mode,
                "disposition": "accept" if accepted else "abstain",
                "reason": reason,
                "proposal": choice,
                "confidence": entropy_confidence,
                "answer_confidence": confidence if usable else None,
                "quality_calibrated": False,
                "usage": result.get("usage"),
                "model": self.config.model,
                "revision": self.config.revision,
                "question_sha256": hashlib.sha256(
                    json.dumps(questions, sort_keys=True).encode()
                ).hexdigest(),
                "state_sample_bytes": byte_tokens(state),
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
        except Exception:
            return {
                "mode": self.config.mode,
                "disposition": "abstain",
                "reason": "decision_unavailable",
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
