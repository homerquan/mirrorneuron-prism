import math

import pytest
from pydantic import ValidationError

from litellm_multicall._vendor.semif.core import direct_messages
from litellm_multicall.classifier.base import MockBackend, RulesBackend
from litellm_multicall.classifier.types import DecisionRequest, DecisionResult


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(state=float("nan")),
        lambda r: r.update(state={"a": float("inf")}),
        lambda r: r.update(state=""),
        lambda r: r.update(request_id=" "),
        lambda r: r.update(criteria=r["criteria"] * 2),
        lambda r: r["criteria"][0].update(options=r["criteria"][0]["options"][:1]),
        lambda r: r["criteria"][0].update(options=r["criteria"][0]["options"] * 9),
        lambda r: r["criteria"][0]["options"][1].update(id="retry_read"),
        lambda r: r["criteria"][0]["options"][0].update(description=" "),
        lambda r: r.update(label=0),
    ],
)
def test_invalid_requests(decision_request, mutate):
    data = decision_request.model_dump()
    mutate(data)
    with pytest.raises((ValidationError, ValueError)):
        DecisionRequest.model_validate(data)


def test_option_id_mapping_and_no_authority(decision_request):
    backend = MockBackend()
    a = backend.score(decision_request).results[0]
    assert a.winner_option_id == decision_request.criteria[0].options[0].id
    assert a.confidence is None and a.accepted_option_id is None
    reversed_request = decision_request.model_copy(deep=True)
    reversed_request.criteria[0].options.reverse()
    b = backend.score(reversed_request).results[0]
    assert b.winner_option_id == "abstain"
    assert b.option_ids == ["abstain", "retry_read"]


def test_rules_abstain_on_ambiguous(decision_request):
    decision_request.state = "There may be a timeout."
    result = RulesBackend().score(decision_request).results[0]
    assert result.disposition == "abstain" and result.winner_option_id is None
    assert result.option_probabilities is None


def test_gold_never_in_model_payload(decision_request):
    rows = decision_request.rows()
    assert set(rows[0]) == {"id", "state", "question", "options"}
    prompt = direct_messages(rows[0])
    assert len(prompt) == 2
    assert "logical_request_id" not in prompt[-1]["content"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"option_probabilities": [math.nan, 0]},
        {"option_probabilities": [0.2, 0.2]},
        {"option_logits": [1]},
        {"winner_option_id": "unknown"},
        {"timing_ms": {"forward": -1}},
    ],
)
def test_invalid_results(decision_request, kwargs):
    data = MockBackend().score(decision_request).results[0].model_dump()
    data.update(kwargs)
    with pytest.raises(ValidationError):
        DecisionResult.model_validate(data)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def encode(self, text, **kwargs):
        if text in ("A", "B"):
            return [10 + ord(text)]
        if text == "prompt":
            return [1, 2, 3]
        if text in ("promptA", "promptB"):
            return [1, 2, 3, 10 + ord(text[-1])]
        return []

    def decode(self, ids):
        return chr(ids[0] - 10)


def test_token_limit_and_answer_boundary(decision_request):
    from litellm_multicall._vendor.semif.direct import encode_prompt

    row = decision_request.rows()[0]
    assert encode_prompt(Tokenizer(), row, 3)[0] == [1, 2, 3]
    with pytest.raises(ValueError, match="no truncation"):
        encode_prompt(Tokenizer(), row, 2)

    class Bad(Tokenizer):
        def encode(self, text, **kwargs):
            return [1, 2] if text == "A" else super().encode(text, **kwargs)

    with pytest.raises(ValueError, match="one exact"):
        encode_prompt(Bad(), row, 10)


def test_cpu_device_contract():
    from types import SimpleNamespace

    from litellm_multicall.classifier.jev_cpu import assert_cpu
    from litellm_multicall.errors import PrismError

    class Model:
        def parameters(self):
            return [SimpleNamespace(device=SimpleNamespace(type="cuda"))]

        def buffers(self):
            return []

    with pytest.raises(PrismError, match="non-CPU"):
        assert_cpu(Model())


def test_unavailable_service_policy(decision_request):
    from litellm_multicall.classifier.policy import DecisionPolicy
    from litellm_multicall.errors import PrismError
    from litellm_multicall.project_config import Controller

    error = PrismError("not prepared", 3, "unprepared_model")
    batch = DecisionPolicy().unavailable(decision_request, error, Controller())
    assert (
        batch.results[0].disposition == "abstain"
        and batch.results[0].accepted_option_id is None
    )
    assert not batch.measurements["usage_complete"]
    with pytest.raises(PrismError):
        DecisionPolicy().unavailable(
            decision_request, error, Controller(on_unavailable="error")
        )
