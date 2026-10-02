"""Bounded spawned worker. Hard timeout/cancellation kills and reaps compute."""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
import time

from ..errors import PrismError
from ..project_config import Controller
from ..storage import canonical
from .base import RulesBackend
from .types import DecisionBatch, DecisionRequest


def _worker(connection, profile_data, artifact, factory):
    sys.stdout = sys.stderr
    backend = None
    try:
        profile = Controller.model_validate(profile_data)
        if factory:
            backend = factory(profile, artifact)
        elif profile.backend == "rules":
            backend = RulesBackend()
        else:
            from .jev_cpu import JevCPUBackend

            backend = JevCPUBackend(profile, artifact)
        backend.load()
        connection.send({"ready": True, "metadata": backend.metadata})
        while True:
            data = connection.recv()
            if data is None:
                break
            try:
                request = DecisionRequest.model_validate(data)
                connection.send(
                    {"batch": backend.score(request).model_dump(mode="json")}
                )
            except PrismError as exc:
                connection.send(
                    {"error": {"message": str(exc), "code": exc.code, "kind": exc.kind}}
                )
            except ValueError:
                connection.send(
                    {
                        "error": {
                            "message": "scoring rejected input/token boundary or invalid scores; no truncation performed",
                            "code": 2,
                            "kind": "scoring_contract",
                        }
                    }
                )
            except Exception:
                connection.send(
                    {
                        "error": {
                            "message": "classifier backend capability/runtime failed",
                            "code": 3,
                            "kind": "backend_failure",
                        }
                    }
                )
    except PrismError as exc:
        connection.send(
            {"error": {"message": str(exc), "code": exc.code, "kind": exc.kind}}
        )
    except Exception:
        connection.send(
            {
                "error": {
                    "message": "classifier startup failed; verify dependency constraints and local snapshot",
                    "code": 3,
                    "kind": "startup_failure",
                }
            }
        )
    finally:
        if backend:
            backend.close()
        connection.close()


class DecisionExecutor:
    def __init__(self, profile, artifact=None, *, factory=None):
        self.profile, self.artifact, self.factory = profile, artifact, factory
        self.process = self.connection = None
        self.lock = asyncio.Lock()
        self.pending = 0
        self.starts = 0
        self.metadata = {}
        self.closed = False
        self.wasted_work = []

    async def _receive(self, deadline):
        while True:
            if self.connection is None or self.process is None:
                raise PrismError("classifier worker was closed", 3, "unavailable")
            if self.connection.poll():
                break
            if self.process is not None and not self.process.is_alive():
                raise PrismError(
                    "classifier worker exited unexpectedly", 5, "worker_exit"
                )
            if time.monotonic() >= deadline:
                raise PrismError("classifier deadline exhausted", 6, "deadline")
            await asyncio.sleep(0.005)
        try:
            message = self.connection.recv()
        except EOFError as exc:
            raise PrismError(
                "classifier IPC closed unexpectedly", 5, "worker_exit"
            ) from exc
        if "error" in message:
            error = message["error"]
            raise PrismError(error["message"], error["code"], error["kind"])
        return message

    async def _start(self, deadline):
        if self.process is not None and self.process.is_alive():
            return
        self._stop()
        if self.starts > self.profile.max_restarts:
            raise PrismError("classifier restart budget exhausted", 6, "restart_budget")
        parent, child = multiprocessing.get_context("spawn").Pipe()
        process = multiprocessing.get_context("spawn").Process(
            target=_worker,
            args=(child, self.profile.model_dump(), self.artifact, self.factory),
            daemon=True,
        )
        process.start()
        child.close()
        self.connection, self.process = parent, process
        self.starts += 1
        self.metadata = (
            await self._receive(
                min(deadline, time.monotonic() + self.profile.startup_timeout_ms / 1000)
            )
        )["metadata"]

    def _stop(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
        if self.connection:
            self.connection.close()
        self.process = self.connection = None

    def validate(self, request):
        if (
            len(canonical(request.model_dump()).encode())
            > self.profile.max_request_bytes
        ):
            raise PrismError("request byte limit exceeded", 6, "resource_budget")
        if len(request.criteria) > self.profile.max_criteria_per_batch:
            raise PrismError("criterion batch limit exceeded", 6, "resource_budget")
        if any(
            len(canonical(c.model_dump()).encode()) > self.profile.max_criterion_bytes
            for c in request.criteria
        ):
            raise PrismError("criterion byte limit exceeded", 6, "resource_budget")

    async def decide(self, request, *, timeout=None, current_state_version=None):
        self.validate(request)
        if self.closed:
            raise PrismError("executor is closed", 3, "unavailable")
        if self.pending >= 1 + self.profile.queue_capacity:
            raise PrismError("classifier queue saturated", 6, "queue_saturated")
        self.pending += 1
        admitted = time.monotonic()
        deadline = admitted + (
            timeout
            if timeout is not None
            else (self.profile.startup_timeout_ms + self.profile.inference_timeout_ms)
            / 1000
        )
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self.lock.acquire(), timeout=max(0, deadline - time.monotonic())
                )
            except TimeoutError as exc:
                raise PrismError("queue deadline exhausted", 6, "deadline") from exc
            acquired = True
            if self.closed:
                raise PrismError("executor is closed", 3, "unavailable")
            queue_ms = (time.monotonic() - admitted) * 1000
            startup = time.monotonic()
            prior_starts = self.starts
            await self._start(deadline)
            startup_ms = (time.monotonic() - startup) * 1000
            work_start = time.monotonic()
            self.connection.send(request.model_dump(mode="json"))
            message = await self._receive(
                min(deadline, work_start + self.profile.inference_timeout_ms / 1000)
            )
            batch = DecisionBatch.model_validate(message["batch"])
            if batch.request_id != request.request_id or len(batch.results) != len(
                request.criteria
            ):
                raise PrismError(
                    "worker response does not match request", 7, "identity_mismatch"
                )
            for criterion, result in zip(request.criteria, batch.results):
                if (
                    result.criterion_id != criterion.id
                    or result.state_version != request.state_version
                    or result.option_ids != [o.id for o in criterion.options]
                ):
                    raise PrismError(
                        "worker response binding changed", 7, "identity_mismatch"
                    )
            elapsed = (time.monotonic() - admitted) * 1000
            batch.measurements.update(
                queue_ms=queue_ms,
                startup_ms=startup_ms,
                total_ms=elapsed,
                startup_process_cpu_seconds=self.metadata.get(
                    "load_process_cpu_seconds", 0
                )
                if prior_starts != self.starts
                else 0,
                ipc_and_coordination_ms=max(
                    0,
                    (time.monotonic() - work_start) * 1000
                    - batch.measurements.get("backend_total_ms", 0),
                ),
            )
            stale = (
                current_state_version
                and current_state_version() != request.state_version
            )
            if stale:
                batch.measurements["wasted_work"] = True
                for result in batch.results:
                    result.disposition, result.reason, result.accepted_option_id = (
                        "abstain",
                        "stale_state_version",
                        None,
                    )
            return batch
        except (PrismError, asyncio.CancelledError) as exc:
            if acquired:
                self.wasted_work.append(
                    {
                        "request_id": request.request_id,
                        "elapsed_ms": (time.monotonic() - admitted) * 1000,
                        "usage_complete": False,
                    }
                )
                if not isinstance(exc, PrismError) or exc.code != 2:
                    self._stop()
            raise
        finally:
            self.pending -= 1
            if acquired:
                self.lock.release()

    async def decide_many(self, requests, **kwargs):
        return [await self.decide(r, **kwargs) for r in requests]

    async def close(self):
        self.closed = True
        if self.lock.locked():
            self._stop()
            return
        if self.process and self.process.is_alive():
            try:
                self.connection.send(None)
                deadline = time.monotonic() + 2
                while self.process.is_alive() and time.monotonic() < deadline:
                    await asyncio.sleep(0.005)
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._stop()
