"""Closed execution policies and deterministic focused-source selection."""

import re
from typing import Literal

PolicyName = Literal[
    "direct", "evidence_map", "batched_map", "verified_map", "retrieve_read"
]

POLICIES = {
    "direct": "One unchanged model call for a self-contained request that fits context",
    "evidence_map": "Extract each source partition independently, then synthesize complete evidence",
    "batched_map": "Group short source partitions into fewer extraction calls, then synthesize complete evidence",
    "verified_map": "Extract each partition, independently check its facts against quotes, then synthesize",
    "retrieve_read": "Lexically select relevant source partitions for a focused lookup, then answer from original spans",
}

STOP_WORDS = set(
    "a an and are as at be by cite document evidence for from give how in is it of on or original provided request return source sources supplied that the their then these this to using what which with answer json separately untrusted fields".split()
)


def rank_partitions(arena, partitions):
    text = " ".join(
        message["content"]
        if isinstance(message["content"], str)
        else " ".join(part["text"] for part in message["content"])
        for message in arena.instructions
        if message["role"] == "user"
    )
    terms = list(
        dict.fromkeys(
            token.casefold()
            for token in re.findall(r"[^\W_]+", text)
            if len(token) > 1 and token.casefold() not in STOP_WORDS
        )
    )
    terms = set(terms[:64] + terms[-64:])
    ranked = []
    for partition in partitions:
        tokens = set(re.findall(r"[^\W_]+", partition.text.casefold()))
        score = len(tokens & terms)
        if score:
            ranked.append((partition, score))
    return sorted(ranked, key=lambda item: -item[1])
