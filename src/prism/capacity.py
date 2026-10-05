"""Small live behavior probes. Results are observations, never catalog assertions."""

import asyncio
import base64
import copy
import hashlib
import json
import re
import secrets
import struct
import time
import zlib
from datetime import UTC, datetime

import jsonschema

from .config import Limits
from .contracts import check_context, parse_json, validate_request
from .errors import PrismError
from .runtime import Ledger

REVISION = "capacity-v2"
FEATURES = ("json_object", "json_schema", "image", "reasoning")


def grade_reasoning(content, expected):
    # Score task answers separately from JSON/format adherence. Accept harmless
    # spaces, separators between letters, and surrounding final-answer prose.
    matches = re.findall(
        r"inventory\s*=\s*(\d+)\s*;\s*order\s*=\s*([A-D](?:[ \t,>\-]*[A-D]){3})",
        content,
        flags=re.IGNORECASE,
    )
    if not matches:
        return 0, 2
    inventory, order = matches[-1]
    order = re.sub(r"[^A-D]", "", order.upper())
    return int(int(inventory) == expected[0]) + int(order == expected[1]), 2


def image_challenge():
    """Random colored quadrants; the answer exists only in the image pixels."""
    palette = {
        "red": (230, 0, 0),
        "green": (0, 160, 0),
        "blue": (0, 0, 230),
        "yellow": (255, 230, 0),
        "purple": (140, 0, 180),
        "orange": (255, 130, 0),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
    }
    colors = secrets.SystemRandom().sample(list(palette), 4)
    size = 128
    pixels = bytearray()
    for y in range(size):
        pixels.append(0)
        for x in range(size):
            pixels.extend(palette[colors[(y >= size // 2) * 2 + (x >= size // 2)]])

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    prompt = "Name the four quadrant colors in this image in this order: top left, top right, bottom left, bottom right. Choose from red, green, blue, yellow, purple, orange, black, white. Reply with only four lowercase color names separated by commas."
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(png).decode()
                    },
                },
            ],
        }
    ], colors


def reasoning_challenge():
    rng = secrets.SystemRandom()
    start, incoming, size, sold, returned = (
        rng.randint(30, 80),
        rng.randint(2, 6),
        rng.randint(7, 15),
        rng.randint(10, 25),
        rng.randint(1, 7),
    )
    # First task requires tracking different operations; second requires obeying
    # a partial order rather than repeating the presented job order.
    jobs = rng.sample(["A", "B", "C", "D"], 4)
    order = rng.sample(jobs, 4)
    edges = [f"{a} must run before {b}" for a, b in zip(order, order[1:])]
    rng.shuffle(edges)
    constraints = "; ".join(edges)
    prompt = f"Solve both tasks. (1) A store starts with {start} units, receives {incoming} crates of {size} units each, sells {sold} units, then receives {returned} returned units. How many units remain? (2) Jobs are presented as {','.join(jobs)}. They must run one at a time. Constraints: {constraints}. Give the only valid sequence. Reply only as inventory=NUMBER;order=LETTERS with no spaces and the letters concatenated."
    return [{"role": "user", "content": prompt}], [
        start + incoming * size - sold + returned,
        "".join(order),
    ]


class CapacityEvaluator:
    def __init__(self, models, backend, settings):
        self.models, self.backend, self.settings = models, backend, settings
        self.cache = {}
        self.locks = {ref: asyncio.Lock() for ref in models}
        self.slots = asyncio.Semaphore(settings.max_parallel)

    def fingerprint(self, model):
        data = model.model_dump(mode="json")
        # A credential rotation invalidates results; neither key nor this hash is
        # returned in the API, traces, or probe records.
        data["resolved_credential"] = model.credential()
        return hashlib.sha256(
            json.dumps(
                [REVISION, data, self.settings.model_dump()], sort_keys=True
            ).encode()
        ).hexdigest()

    async def query(self, ids, refresh=False):
        unknown = set(ids) - self.models.keys()
        if unknown:
            raise PrismError(
                "unknown configured model combination", "model_not_found", 404
            )
        return list(await asyncio.gather(*(self.evaluate(ref, refresh) for ref in ids)))

    async def evaluate(self, ref, refresh=False):
        requested = time.monotonic()
        model = self.models[ref]
        try:
            fingerprint = self.fingerprint(model)
        except PrismError:
            fingerprint = None
        async with self.locks[ref]:
            previous = self.cache.get(ref)
            now = time.monotonic()
            if (
                previous
                and fingerprint is not None
                and previous[0] == fingerprint
                and (
                    previous[1] >= requested
                    or (not refresh and now - previous[1] < self.settings.ttl_seconds)
                )
            ):
                result = copy.deepcopy(previous[2])
                result["cached"] = True
                return result
            result = await self.run(model)
            self.cache[ref] = (fingerprint, time.monotonic(), result)
            return copy.deepcopy(result)

    async def probe(self, model, feature, messages, parameters, grade):
        started = time.monotonic()
        output = min(model.max_output_tokens, self.settings.output_tokens)
        result = {
            "supported": None,
            "status": "unavailable",
            "passed": 0,
            "total": 2 if feature == "reasoning" else 1,
        }
        try:
            # Ignore declared capabilities: admission still checks context/cost
            # bounds, while the actual request determines feature acceptance.
            tokens = check_context(messages, parameters, output, model)
            async with self.slots:
                ledger = Ledger(
                    Limits(
                        max_calls=1,
                        max_output_tokens=output,
                        deadline_seconds=self.settings.probe_timeout_seconds,
                    )
                )
                reservation = await ledger.reserve(
                    "capacity-" + feature, model, tokens, output
                )
                async with asyncio.timeout(ledger.remaining_seconds()):
                    data = await self.backend.complete(
                        model, messages, parameters, output, ledger, reservation
                    )
            choice = data["choices"][0]
            if choice["finish_reason"] != "stop":
                result.update(
                    status="inconclusive", error_code="probe_" + choice["finish_reason"]
                )
            else:
                passed, total = grade(choice["message"].get("content", ""))
                result.update(
                    supported=passed == total,
                    status="passed" if passed == total else "failed",
                    passed=passed,
                    total=total,
                )
            if feature == "reasoning":
                message = choice["message"]
                usage = data.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                result["reasoning_metadata_observed"] = bool(
                    message.get("reasoning_content")
                    or message.get("thinking_blocks")
                    or details.get("reasoning_tokens")
                )
        except TimeoutError:
            result["error_code"] = "deadline_exceeded"
        except PrismError as exc:
            result["error_code"] = exc.code
            if getattr(exc, "upstream_status", None) is not None:
                result["upstream_status"] = exc.upstream_status
            if exc.code in {
                "upstream_unsupported_feature",
                "unsupported_feature",
                "unsupported_coverage_contract",
            }:
                result.update(supported=False, status="rejected")
        except (ValueError, TypeError, KeyError, jsonschema.ValidationError):
            result.update(
                supported=False, status="failed", error_code="invalid_probe_answer"
            )
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        result["checked_at"] = datetime.now(UTC).isoformat()
        return result

    async def run(self, model):
        nonce = secrets.token_hex(6)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["tag", "count"],
            "properties": {
                "tag": {"type": "string", "enum": [nonce]},
                "count": {"type": "integer", "enum": [7]},
            },
        }

        def object_grade(content):
            value = parse_json(content)
            return int(
                isinstance(value, dict) and value == {"tag": nonce, "count": 7}
            ), 1

        def schema_grade(content):
            jsonschema.validate(parse_json(content), schema)
            return 1, 1

        def image_grade(content):
            answer = [s.strip().lower() for s in content.strip().split(",")]
            return int(answer == colors), 1

        def reasoning_grade(content):
            return grade_reasoning(content, expected)

        images, colors = image_challenge()
        reasoning, expected = reasoning_challenge()
        cases = [
            (
                "json_object",
                [
                    {
                        "role": "system",
                        "content": f'You output only valid JSON, without markdown or commentary. Reply as a JSON object with exactly tag="{nonce}" and count=7 (integer), no other fields.',
                    },
                    {"role": "user", "content": "Return the requested JSON object."},
                ],
                {"response_format": {"type": "json_object"}},
                object_grade,
            ),
            # Conflicting extra field/type requests expose ignored schema modes.
            (
                "json_schema",
                [
                    {
                        "role": "system",
                        "content": "You output only valid JSON, without markdown or commentary. Obey the supplied response schema.",
                    },
                    {
                        "role": "user",
                        "content": f'Reply as JSON with tag="{nonce}", count="7" (string), and an extra field debug=true. Follow the supplied response schema if it conflicts.',
                    },
                ],
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "prism_capacity",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                },
                schema_grade,
            ),
            ("image", images, {}, image_grade),
            ("reasoning", reasoning, {}, reasoning_grade),
        ]
        observations = {}
        # Sequential per combination respects small local engines and rate caps.
        for feature, messages, parameters, grade in cases:
            observations[feature] = await self.probe(
                model, feature, messages, parameters, grade
            )
        checked = time.time()
        return {
            "model": model.id,
            "evaluation": REVISION,
            "checked_at": datetime.fromtimestamp(checked, UTC).isoformat(),
            "expires_at": datetime.fromtimestamp(
                checked + self.settings.ttl_seconds, UTC
            ).isoformat(),
            "cached": False,
            "capabilities": observations,
        }


class ProfileBackend:
    """Run the same challenges through the actual public execution graph."""

    def __init__(self, engine):
        self.engine = engine

    async def complete(self, model, messages, parameters, output, ledger, reservation):
        await ledger.start(reservation)
        execution = None
        try:
            body = validate_request(
                {
                    "model": model.id,
                    "messages": messages,
                    **parameters,
                    "max_completion_tokens": output,
                }
            )
            execution = self.engine.prepare(body)
            result = await self.engine.execute(execution)
            await ledger.finish(reservation, result.get("usage"))
            return result
        except BaseException:
            if execution is not None:
                self.engine.finalize_trace(execution, "capacity_failed")
            await ledger.finish(reservation, status="failed")
            raise


class ProfileCapacityEvaluator(CapacityEvaluator):
    def __init__(self, engine, settings):
        self.engine = engine
        models = {
            alias: engine.models[profile.direct].model_copy(
                update={
                    "id": alias,
                    "max_output_tokens": min(
                        profile.public_max_output_tokens,
                        engine.models[profile.direct].max_output_tokens,
                    ),
                }
            )
            for alias, profile in engine.config.profiles.items()
        }
        super().__init__(models, ProfileBackend(engine), settings)

    def fingerprint(self, model):
        # Invalidate observations when any stage or its credential changes.
        return hashlib.sha256(
            json.dumps(
                [
                    REVISION,
                    self.engine.config.profiles[model.id].model_dump(mode="json"),
                    [
                        super(ProfileCapacityEvaluator, self).fingerprint(m)
                        for m in self.engine.models.values()
                    ],
                    self.settings.model_dump(),
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()

    async def run(self, model):
        result = await super().run(model)
        result["scope"] = "end_to_end_profile"
        return result
