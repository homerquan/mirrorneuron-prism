import asyncio
import time

import pytest

from litellm_multicall.classifier.base import MockBackend
from litellm_multicall.classifier.executor import DecisionExecutor
from litellm_multicall.errors import PrismError
from litellm_multicall.project_config import Controller


class Slow(MockBackend):
    def score(self, decision_request):
        time.sleep(5)
        return super().score(decision_request)


class SlowStart(MockBackend):
    def load(self):
        time.sleep(5)


def mock_factory(profile, artifact):
    return MockBackend()


def slow_factory(profile, artifact):
    return Slow()


def slow_start_factory(profile, artifact):
    return SlowStart()


@pytest.mark.asyncio
async def test_load_once_and_state_staleness(decision_request):
    executor = DecisionExecutor(Controller(backend="rules"), factory=mock_factory)
    try:
        one, two = await executor.decide_many([decision_request, decision_request])
        assert executor.starts == 1 and one.request_id == two.request_id
        stale = await executor.decide(
            decision_request, current_state_version=lambda: "new"
        )
        assert stale.results[0].reason == "stale_state_version"
        assert stale.measurements["wasted_work"]
    finally:
        await executor.close()
    assert executor.process is None


@pytest.mark.asyncio
async def test_timeout_kills_reaps_and_bounded_restart(decision_request):
    profile = Controller(backend="rules", inference_timeout_ms=40, max_restarts=0)
    executor = DecisionExecutor(profile, factory=slow_factory)
    try:
        with pytest.raises(PrismError) as error:
            await executor.decide(decision_request)
        assert error.value.code == 6 and executor.process is None
        assert executor.wasted_work[0]["usage_complete"] is False
        with pytest.raises(PrismError, match="restart budget"):
            await executor.decide(decision_request)
        assert executor.starts == 1
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_cancel_and_saturation(decision_request):
    executor = DecisionExecutor(
        Controller(backend="rules", queue_capacity=0), factory=slow_factory
    )
    task = asyncio.create_task(executor.decide(decision_request))
    await asyncio.sleep(0.02)
    with pytest.raises(PrismError, match="saturated"):
        await executor.decide(decision_request)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert executor.process is None and executor.pending == 0
    await executor.close()


@pytest.mark.asyncio
async def test_queue_timeout_does_not_kill_active_worker(decision_request):
    executor = DecisionExecutor(
        Controller(backend="rules", queue_capacity=1), factory=slow_factory
    )
    task = asyncio.create_task(executor.decide(decision_request))
    await asyncio.sleep(0.02)
    with pytest.raises(PrismError, match="queue deadline"):
        await executor.decide(decision_request, timeout=0.01)
    assert executor.process is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await executor.close()


@pytest.mark.asyncio
async def test_startup_timeout(decision_request):
    executor = DecisionExecutor(
        Controller(backend="rules", startup_timeout_ms=40), factory=slow_start_factory
    )
    with pytest.raises(PrismError) as error:
        await executor.decide(decision_request)
    assert error.value.code == 6 and executor.process is None
    await executor.close()


@pytest.mark.asyncio
async def test_byte_and_criterion_admission_before_spawn(decision_request):
    executor = DecisionExecutor(Controller(backend="rules", max_request_bytes=10))
    with pytest.raises(PrismError, match="byte limit"):
        await executor.decide(decision_request)
    assert executor.starts == 0
    await executor.close()


@pytest.mark.asyncio
async def test_close_with_active_and_queued_work(decision_request):
    executor = DecisionExecutor(
        Controller(backend="rules", queue_capacity=1), factory=slow_factory
    )
    active = asyncio.create_task(executor.decide(decision_request))
    await asyncio.sleep(0.01)
    queued = asyncio.create_task(executor.decide(decision_request))
    await asyncio.sleep(0.01)
    await executor.close()
    outcomes = await asyncio.gather(active, queued, return_exceptions=True)
    assert all(isinstance(outcome, PrismError) for outcome in outcomes)
    assert executor.process is None and executor.pending == 0
