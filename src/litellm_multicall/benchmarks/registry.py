SUITES = {
    "semif-authored": {
        "availability": "implemented",
        "scope": "native synthetic decision quality",
        "file": "authored144.jsonl",
        "limitation": "model-reviewed synthetic annotations; not human-adjudicated or broad agent generalization",
    },
    "semif-shape": {
        "availability": "implemented",
        "scope": "scoring systems throughput; no gold labels",
        "file": "shape777.jsonl",
        "limitation": "timing does not establish semantic correctness",
    },
    "bfcl-decision": {
        "availability": "planned",
        "scope": "BFCL-derived decision probes",
    },
    "bfcl": {
        "availability": "planned",
        "scope": "official full function-call evaluation",
    },
    "tau": {"availability": "planned", "scope": "external agent task outcomes"},
    "swebench": {
        "availability": "planned",
        "scope": "external patch generation and official tests",
    },
}


def supported(name):
    from ..errors import PrismError

    suite = SUITES.get(name)
    if suite is None or suite["availability"] != "implemented":
        raise PrismError(
            f"benchmark adapter unavailable: {name}", 3, "unsupported_capability"
        )
    return suite
