"""Atomic multi-resource reservations and bounded, dependency-checked plans."""

import asyncio
import time
from dataclasses import dataclass, field

from .errors import PrismError


@dataclass
class Reservation:
    node_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    started: bool = False
    finished: bool = False


class Ledger:
    def __init__(self, limits):
        self.limits = limits
        self.deadline = time.monotonic() + limits.deadline_seconds
        self.lock = asyncio.Lock()
        self.reservations = {}
        self.consumed = dict(calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)
        self.usage = []

    def remaining_seconds(self):
        return max(0, self.deadline - time.monotonic())

    async def reserve(self, node_id, model, input_tokens, output_tokens):
        async with self.lock:
            if node_id in self.reservations:
                raise PrismError("duplicate plan reservation", "invalid_plan")
            if not self.remaining_seconds():
                raise PrismError("request deadline exceeded", "deadline_exceeded", 504)
            known_price = (
                model.input_cost_per_million is not None
                and model.output_cost_per_million is not None
            )
            if self.limits.max_cost_usd is not None and not known_price:
                raise PrismError(
                    "hard cost budget requires both backend prices", "unknown_price"
                )
            cost = (
                (
                    input_tokens * model.input_cost_per_million
                    + output_tokens * model.output_cost_per_million
                )
                / 1_000_000
                if known_price
                else None
            )
            outstanding = [r for r in self.reservations.values() if not r.finished]
            proposals = {
                "calls": 1 + sum(1 for r in outstanding),
                "input_tokens": input_tokens + sum(r.input_tokens for r in outstanding),
                "output_tokens": output_tokens
                + sum(r.output_tokens for r in outstanding),
                "cost_usd": (cost or 0) + sum(r.cost_usd or 0 for r in outstanding),
            }
            for dimension, amount in proposals.items():
                limit = getattr(self.limits, f"max_{dimension}")
                if limit is not None and self.consumed[dimension] + amount > limit:
                    raise PrismError(
                        "compiled plan exceeds request resource budget",
                        "resource_limit",
                        413,
                    )
            reservation = Reservation(
                node_id, model.id, input_tokens, output_tokens, cost
            )
            self.reservations[node_id] = reservation
            return reservation

    async def start(self, reservation):
        async with self.lock:
            if reservation.started or reservation.finished:
                raise PrismError("invalid reservation state", "invalid_plan", 500)
            if not self.remaining_seconds():
                raise PrismError("request deadline exceeded", "deadline_exceeded", 504)
            reservation.started = True

    async def finish(self, reservation, usage=None, status="complete", elapsed_ms=None):
        async with self.lock:
            if reservation.finished:
                return
            reservation.finished = True
            if not reservation.started:
                return  # Only unstarted work can be refunded without evidence.
            valid_usage = isinstance(usage, dict) and all(
                type(usage.get(k)) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens")
            )
            inp = usage["prompt_tokens"] if valid_usage else reservation.input_tokens
            out = (
                usage["completion_tokens"] if valid_usage else reservation.output_tokens
            )
            self.consumed["calls"] += 1
            self.consumed["input_tokens"] += inp
            self.consumed["output_tokens"] += out
            # Keep the conservative price reservation; no unsupported billing guarantees.
            self.consumed["cost_usd"] += reservation.cost_usd or 0
            self.usage.append(
                {
                    "node_id": reservation.node_id,
                    "model_id": reservation.model_id,
                    "status": status,
                    "provider_usage": usage if valid_usage else None,
                    "usage_status": "reported" if valid_usage else "unknown",
                    "reserved_input_tokens": reservation.input_tokens,
                    "reserved_output_tokens": reservation.output_tokens,
                    "cost_upper_estimate_usd": reservation.cost_usd,
                    "elapsed_ms": elapsed_ms,
                    "cancellation_requested": status == "cancelled",
                    "remote_cancellation_confirmed": False
                    if status == "cancelled"
                    else None,
                }
            )
            if valid_usage and (
                inp > reservation.input_tokens or out > reservation.output_tokens
            ):
                raise PrismError(
                    "backend usage exceeded its configured accounting bound",
                    "backend_bound_violated",
                    502,
                )

    def snapshot(self):
        return {
            "consumed_or_conservatively_charged": self.consumed,
            "calls": self.usage,
            "outstanding_nodes": [
                r.node_id for r in self.reservations.values() if not r.finished
            ],
            "cost_status": "unknown"
            if any(u["cost_upper_estimate_usd"] is None for u in self.usage)
            else "upper_estimate",
        }

    def release_unstarted(self):
        # All owned tasks have been joined/cancelled before request finalization.
        for reservation in self.reservations.values():
            if not reservation.started:
                reservation.finished = True


@dataclass(frozen=True)
class PlanNode:
    id: str
    operator: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    strategy: str
    nodes: tuple[PlanNode, ...]
    max_depth: int = 3
    allowed_operators: frozenset[str] = field(
        default=frozenset({"generate", "extract", "verify", "synthesize"})
    )

    def validate(self, max_nodes):
        lookup = {n.id: n for n in self.nodes}
        if not self.nodes or len(lookup) != len(self.nodes) or len(lookup) > max_nodes:
            raise PrismError("invalid plan size or duplicate node IDs", "invalid_plan")
        depths = {}
        visiting = set()

        def visit(node_id):
            if node_id not in lookup or node_id in visiting:
                raise PrismError("unknown dependency or cycle", "invalid_plan")
            if node_id in depths:
                return depths[node_id]
            node = lookup[node_id]
            if node.operator not in self.allowed_operators:
                raise PrismError("unsupported plan operator", "invalid_plan")
            visiting.add(node_id)
            depth = 1 + max((visit(d) for d in node.dependencies), default=0)
            visiting.remove(node_id)
            if depth > self.max_depth:
                raise PrismError("plan exceeds graph depth", "invalid_plan")
            depths[node_id] = depth
            return depth

        for node in self.nodes:
            visit(node.id)
        return self


async def bounded_map(items, concurrency, function):
    """Allocate at most concurrency tasks, preserving input order and cancelling siblings."""
    iterator = iter(enumerate(items))
    results = [None] * len(items)

    async def worker():
        for index, item in iterator:
            results[index] = await function(item)

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(items))):
            group.create_task(worker())
    return results
