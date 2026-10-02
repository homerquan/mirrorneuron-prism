from typing import Protocol

from .types import DecisionBatch, DecisionRequest, DecisionResult


class QuickClassifierBackend(Protocol):
    def load(self) -> None: ...
    def score(self, request: DecisionRequest) -> DecisionBatch: ...
    def score_many(self, requests: list[DecisionRequest]) -> list[DecisionBatch]: ...
    def capabilities(self) -> dict: ...
    def close(self) -> None: ...


class RulesBackend:
    """Narrow structured-state rules; abstain when no deterministic rule applies."""

    metadata = {"backend": "rules", "version": "structured-retry-v1", "load_seconds": 0}

    def load(self):
        pass

    def capabilities(self):
        return {"mode": "rules", "authority": "none", "unstructured_text": "abstain"}

    def score(self, request):
        import time

        started, cpu_start = time.perf_counter(), time.process_time()
        results = []
        for c in request.criteria:
            winner = None
            state = request.state
            if c.primitive == "retry" and isinstance(state, dict):
                if (
                    state.get("operation") == "read_repository_file"
                    and state.get("result") == "temporary timeout"
                    and type(state.get("attempts")) is int
                    and 0 <= state["attempts"] < 2
                ):
                    if "retry_read" in {o.id for o in c.options}:
                        winner = "retry_read"
            results.append(
                DecisionResult(
                    request_id=request.request_id,
                    logical_request_id=request.logical_request_id,
                    state_version=request.state_version,
                    criterion_id=c.id,
                    primitive=c.primitive,
                    backend="rules",
                    winner_option_id=winner,
                    option_ids=[o.id for o in c.options],
                    scoring_mode="rules",
                    model_forward_passes=0,
                    probability_status="not_applicable",
                    disposition="scored" if winner else "abstain",
                    reason="deterministic_rule" if winner else "no_applicable_rule",
                )
            )
        return DecisionBatch(
            request_id=request.request_id,
            results=results,
            measurements={
                "model_forward_passes": 0,
                "semantic_decisions": len(results),
                "generated_output_tokens": 0,
                "logical_input_tokens": 0,
                "backend_total_ms": (time.perf_counter() - started) * 1000,
                "process_cpu_seconds": time.process_time() - cpu_start,
            },
            backend_metadata=self.metadata,
        )

    def score_many(self, requests):
        return [self.score(r) for r in requests]

    def close(self):
        pass


class MockBackend(RulesBackend):
    """Deterministic test double, never registered as a production backend."""

    def score(self, request):
        batch = super().score(request)
        for result in batch.results:
            n = len(result.option_ids)
            result.option_logits = [1.0] + [0.0] * (n - 1)
            from .._vendor.semif.core import softmax

            result.option_probabilities = softmax(result.option_logits)
            result.winner_option_id = result.option_ids[0]
            result.disposition = "scored"
        return batch
