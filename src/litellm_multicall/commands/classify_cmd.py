import asyncio
import json
from contextlib import nullcontext

from ..classifier.artifacts import verify_controller
from ..classifier.calibration import fit
from ..classifier.executor import DecisionExecutor
from ..classifier.policy import DecisionPolicy
from ..classifier.types import DecisionRequest
from ..errors import PrismError
from ..storage import atomic_stream, read_json, write_json
from .output import envelope, render


def _services(ctx):
    profile = ctx.controller(ctx.args.controller)
    artifact = (
        verify_controller(ctx.artifact_lock, profile)
        if profile.backend != "rules"
        else None
    )
    calibration = (
        read_json(ctx.explicit(ctx.args.calibration)) if ctx.args.calibration else None
    )
    if calibration:
        from ..classifier.calibration import identity_for_profile, verify

        if artifact is None:
            raise PrismError("calibration requires a model-backed controller")
        verify(calibration, identity_for_profile(profile, artifact))
    return profile, DecisionExecutor(profile, artifact), DecisionPolicy(calibration)


def _request(text, profile):
    if len(text.encode()) > profile.max_request_bytes:
        raise PrismError("request byte limit exceeded", 6, "resource_budget")
    return DecisionRequest.model_validate_json(text)


def run(ctx):
    profile, executor, policy = _services(ctx)
    path = ctx.explicit(ctx.args.input)
    if path.stat().st_size > profile.max_request_bytes:
        raise PrismError("request byte limit exceeded", 6, "resource_budget")
    request = _request(path.read_text(), profile)

    async def execute():
        try:
            return policy.apply(
                await executor.decide(request, timeout=ctx.args.timeout)
            ).model_dump(mode="json")
        finally:
            await executor.close()

    data = asyncio.run(execute())
    if ctx.args.output:
        write_json(ctx.explicit(ctx.args.output), data)
    return data


def batch(ctx):
    if ctx.args.format == "json" and not ctx.args.output:
        raise PrismError(
            "batch JSON mode requires --output for per-row JSONL; use --format jsonl to stream results"
        )
    profile, executor, policy = _services(ctx)
    successes = errors = 0
    destination = ctx.explicit(ctx.args.output) if ctx.args.output else None
    if destination and destination.exists():
        raise PrismError("refusing to overwrite batch output")
    with atomic_stream(destination) if destination else nullcontext(None) as output:

        async def execute():
            nonlocal successes, errors
            try:
                with ctx.explicit(ctx.args.input).open(encoding="utf-8") as stream:
                    line_number = 0
                    while True:
                        text = stream.readline(profile.max_request_bytes + 1)
                        if not text:
                            break
                        line_number += 1
                        request = error = None
                        if len(text.encode()) > profile.max_request_bytes:
                            while text and not text.endswith("\n"):
                                text = stream.readline(profile.max_request_bytes + 1)
                            error = PrismError(
                                "request byte limit exceeded", 6, "resource_budget"
                            )
                        try:
                            if error:
                                raise error
                            request = _request(text, profile)
                            data = policy.apply(
                                await executor.decide(request, timeout=ctx.args.timeout)
                            ).model_dump(mode="json")
                            record = envelope("classify.batch.result", data)
                            successes += 1
                        except (ValueError, PrismError) as exc:
                            errors += 1
                            failed_id = f"invalid-row-{line_number}"
                            if request is not None:
                                failed_id = request.request_id
                            elif error is None:
                                try:
                                    raw = json.loads(text)
                                    if (
                                        isinstance(raw, dict)
                                        and isinstance(raw.get("request_id"), str)
                                        and raw["request_id"].strip()
                                    ):
                                        failed_id = raw["request_id"]
                                except (ValueError, TypeError):
                                    pass
                            record = envelope(
                                "classify.batch.result",
                                {
                                    "request_id": failed_id,
                                    "line": line_number,
                                },
                                error={
                                    "kind": exc.kind
                                    if isinstance(exc, PrismError)
                                    else "invalid_input",
                                    "message": "row failed validation/scoring",
                                    "exit_code": exc.code
                                    if isinstance(exc, PrismError)
                                    else 2,
                                },
                            )
                            ctx.args.exit_code = record["error"]["exit_code"]
                        if output:
                            output.write(json.dumps(record, allow_nan=False) + "\n")
                            output.flush()
                        if ctx.args.format in ("jsonl", "human"):
                            render(ctx.args, record)
                        if record["error"] and not ctx.args.continue_on_error:
                            break
            finally:
                await executor.close()

        asyncio.run(execute())
    return {
        "completed": successes,
        "errors": errors,
        "model_loads": executor.starts,
        "output": str(destination) if destination else None,
    }


def calibrate(ctx):
    artifact = fit(
        read_json(ctx.explicit(ctx.args.input)),
        ctx.args.risk_target,
        ctx.args.min_support,
    )
    write_json(ctx.explicit(ctx.args.output), artifact)
    return artifact
