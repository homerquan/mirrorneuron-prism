"""Seeded references for native-sized long-context qualification.

Gold evidence lives outside request packets and maps to original byte spans.
"""

import random

from .context import SourceArena

FAMILIES = (
    "distractor",
    "distributed",
    "exception",
    "chain",
    "revision",
    "unanswerable",
)


def build_case(family, seed, position, filler_records):
    if family not in FAMILIES or position not in {"start", "middle", "end"}:
        raise ValueError("Unknown long-context family or position")
    rng = random.Random(seed)
    key = f"svc-{rng.randrange(10000, 99999)}"
    role = rng.choice(["manager", "director", "supervisor"])
    days = rng.randrange(1, 5)
    facts = {
        "distractor": [f"For {key}, standard deployments require {role} approval."],
        "distributed": [
            f"For {key}, standard deployments require {role} approval.",
            f"For {key}, emergency reviews are due within {days} business days.",
        ],
        "exception": [
            f"For {key}, production deployments require {role} approval.",
            f"Exception for {key}: during an active outage, production rollback is permitted without approval.",
        ],
        "chain": [
            f"The escalation contact for {key} is team cobalt-{seed}.",
            f"The owner of team cobalt-{seed} is team amber-{seed}.",
            f"The approval role for team amber-{seed} is {role}.",
        ],
        "revision": [
            f"Policy revision 1 for {key}, effective 2025-01-01: approval role is auditor.",
            f"Policy revision 2 for {key}, effective 2026-01-01: approval role is {role}; it supersedes revision 1.",
        ],
        "unanswerable": [
            f"The archive for {key} specifies its display color as blue; its approval role is unspecified."
        ],
    }[family]
    question, expected = {
        "distractor": (
            f"For {key}, who approves standard deployments? Return JSON approval_role (snake_case).",
            {"approval_role": role},
        ),
        "distributed": (
            f"For {key}, return JSON approval_role (snake_case) and review_business_days (integer).",
            {"approval_role": role, "review_business_days": days},
        ),
        "exception": (
            f"For {key}, during an active outage is production rollback permitted without approval? Return JSON rollback_without_approval (boolean).",
            {"rollback_without_approval": True},
        ),
        "chain": (
            f"Follow the escalation contact and team ownership for {key} to find its approval role. Return JSON approval_role (snake_case).",
            {"approval_role": role},
        ),
        "revision": (
            f"As of 2026-02-01, what is the approval role for {key}? Return JSON approval_role (snake_case).",
            {"approval_role": role},
        ),
        "unanswerable": (
            f"Who approves deployments for {key}? Return JSON approval_role; use null when the evidence does not specify it.",
            {"approval_role": None},
        ),
    }[family]
    # Plausible vocabulary distractors with distinct entities, rather than identical padding.
    lines = [
        f"Archive {i:06d}: service other-{i:06d} approval metadata records sensor {rng.randrange(100, 999)}; deployment schedule remains unchanged."
        for i in range(filler_records)
    ]
    locations = {"start": 0, "middle": len(lines) // 2, "end": len(lines)}
    if family in {"distributed", "chain", "revision", "exception"}:
        rotation = {"start": 0, "middle": 1, "end": len(facts) - 1}[position]
        placements = [
            (len(lines) * ((index + rotation) % len(facts)) // (len(facts) - 1), fact)
            for index, fact in enumerate(facts)
        ]
        for location, fact in sorted(placements, reverse=True):
            lines.insert(location, fact)
    else:
        lines[locations[position] : locations[position]] = facts
    source = "\n".join(lines)
    request = {
        "messages": [
            {
                "role": "user",
                "content": question
                + '\n<prism-source id="archive">'
                + source
                + "</prism-source>",
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    arena = SourceArena(request["messages"])
    document = arena.documents[0]
    gold = []
    for fact in facts:
        start = document.byte_start + len(source[: source.index(fact)].encode())
        gold.append(
            {
                "source_id": document.source_id,
                "source_sha256": document.source_sha256,
                "byte_start": start,
                "byte_end": start + len(fact.encode()),
            }
        )
    case = {
        "id": f"{family}-seed{seed}-{position}",
        "description": f"{family} native-context qualification",
        "tags": [family, position],
        "request": request,
        "checks": [
            {"type": "json_equals", "path": [field], "value": value}
            for field, value in expected.items()
        ],
    }
    return case, gold


def evidence_recall(gold, spans):
    if not gold:
        return None
    return sum(
        any(
            s["source_id"] == g["source_id"]
            and s["source_sha256"] == g["source_sha256"]
            and s["byte_start"] <= g["byte_start"]
            and s["byte_end"] >= g["byte_end"]
            for s in spans
        )
        for g in gold
    ) / len(gold)
