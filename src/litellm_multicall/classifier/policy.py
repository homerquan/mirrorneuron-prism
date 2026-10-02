from .._vendor.semif.core import softmax
from .calibration import verify


class DecisionPolicy:
    """Standalone scoring cannot authorize execution. Gates only admit option IDs."""

    def __init__(self, calibration=None):
        self.calibration = calibration

    def unavailable(self, request, error, profile):
        """Explicit service fallback policy; CLI commands surface preparation errors."""
        if profile.on_unavailable == "error":
            raise error
        from .types import DecisionBatch, DecisionResult

        return DecisionBatch(
            request_id=request.request_id,
            results=[
                DecisionResult(
                    request_id=request.request_id,
                    logical_request_id=request.logical_request_id,
                    state_version=request.state_version,
                    criterion_id=c.id,
                    primitive=c.primitive,
                    backend=profile.backend,
                    option_ids=[o.id for o in c.options],
                    disposition="abstain",
                    reason=error.kind,
                    probability_status="unavailable",
                    input_tokens=None,
                    model_forward_passes=None,
                )
                for c in request.criteria
            ],
            measurements={"usage_complete": False, "failure": error.kind},
        )

    def apply(self, batch, *, task_families=None):
        if self.calibration is None:
            return batch
        verify(self.calibration, batch.backend_metadata)
        for result in batch.results:
            if result.disposition in ("abstain", "reject"):
                continue
            result.accepted_option_id = None
            result.disposition = "abstain"
            if not task_families or task_families.get(
                result.criterion_id
            ) not in self.calibration.get("supported_task_families", []):
                result.reason = "unsupported_calibration_domain"
                continue
            if (
                result.primitive not in self.calibration["supported_primitives"]
                or len(result.option_ids) not in self.calibration["option_counts"]
            ):
                result.reason = "unsupported_calibration_regime"
                continue
            probs = softmax(
                [v / self.calibration["temperature"] for v in result.option_logits]
            )
            result.confidence = max(probs)
            result.reason = "insufficient_selective_risk_evidence"
            threshold = self.calibration["threshold"]
            if threshold is not None and result.confidence >= threshold:
                result.disposition, result.reason = (
                    "accept",
                    "compatible_validated_gate",
                )
                result.accepted_option_id = result.winner_option_id
            elif threshold is not None:
                result.reason = "below_validated_threshold"
        batch.measurements["calibration_sha256"] = self.calibration["artifact_sha256"]
        return batch
