"""Explicit local-model integration; default pytest never loads/downloads weights."""

import os
from pathlib import Path

import pytest


@pytest.mark.real_model
def test_pinned_cpu_direct_shared_and_compact_parity():
    from litellm_multicall._vendor.semif.direct import score
    from litellm_multicall.classifier.artifacts import describe_snapshot
    from litellm_multicall.classifier.jev_cpu import JevCPUBackend, assert_cpu
    from litellm_multicall.classifier.types import DecisionRequest
    from litellm_multicall.project_config import Controller

    snapshot = os.environ.get("PRISM_TEST_MODEL_SNAPSHOT")
    revision = os.environ.get("PRISM_TEST_MODEL_REVISION")
    if not snapshot or not revision:
        pytest.skip(
            "set PRISM_TEST_MODEL_SNAPSHOT and immutable PRISM_TEST_MODEL_REVISION"
        )
    artifact = describe_snapshot(
        Path(snapshot), "Qwen/Qwen3-0.6B", revision, "upstream model card"
    )
    backend = JevCPUBackend(Controller(max_input_tokens=1024), artifact)
    request = DecisionRequest.model_validate(
        {
            "request_id": "parity",
            "state_version": "one",
            "state": "I was charged twice. Please refund the duplicate payment.",
            "criteria": [
                {
                    "id": "route",
                    "primitive": "route",
                    "question": "Which team should handle this ticket?",
                    "options": [
                        {"id": "billing", "description": "Billing and payments"},
                        {"id": "tech", "description": "Technical support"},
                    ],
                },
                {
                    "id": "relevance",
                    "primitive": "relevance",
                    "question": "Is this related to billing?",
                    "options": [
                        {"id": "yes", "description": "Related to billing"},
                        {"id": "no", "description": "Unrelated to billing"},
                    ],
                },
            ],
        }
    )
    try:
        backend.load()
        assert_cpu(backend.model)
        direct = backend.score(request)
        for row, result in zip(request.rows(), direct.results):
            raw = score(
                backend.model, backend.tokenizer, row, backend.metadata, max_tokens=1024
            )
            assert result.option_logits == pytest.approx(raw["option_logits"], abs=1e-6)
            assert result.option_probabilities == pytest.approx(
                raw["probabilities"], abs=1e-7
            )
        backend.profile.scoring_mode = "shared"
        shared = backend.score(request)
        for a, b in zip(direct.results, shared.results):
            assert a.option_ids == b.option_ids
            # Different float32 prefix/batch layouts introduce roundoff. Recorded
            # native fixtures reached 1.49e-4 logits and 2.44e-5 probabilities.
            assert a.option_logits == pytest.approx(b.option_logits, abs=2e-4, rel=0)
            assert a.option_probabilities == pytest.approx(
                b.option_probabilities, abs=3e-5, rel=0
            )
        assert shared.measurements["model_forward_passes"] == 2
        assert (
            shared.measurements["padded_suffix_tokens"]
            >= shared.measurements["true_suffix_tokens"]
        )
        backend.profile.scoring_mode = "compact_generation"
        compact = backend.score(request)
        assert compact.measurements["generated_output_tokens"] == 2
        assert [r.winner_option_id for r in direct.results] == [
            r.winner_option_id for r in compact.results
        ]
        backend.profile.max_input_tokens = 2
        with pytest.raises(ValueError, match="no truncation"):
            backend.score(request)
    finally:
        backend.close()
