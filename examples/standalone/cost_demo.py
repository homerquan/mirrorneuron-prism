"""Send a real text preparation/synthesis request and print server cost totals.

Start first: prism start --profile prism-mock-cost --show-cost --no-auth
Then: python examples/standalone/cost_demo.py --no-auth
Set OPENROUTER_API_KEY before starting Prism. Only token prices are hypothetical;
the request calls OpenRouter's free Nemotron Super and Ultra models.
"""

import argparse
import json
import os

import httpx


def demo_request(model="prism-mock-cost"):
    source = "\n".join(
        f"Team {n:02}: the weekly review found no service outage. "
        "The team completed routine documentation updates and confirmed the existing release process."
        for n in range(1, 33)
    )
    source += (
        "\nRelease policy: production releases require two reviewers and a passing integration suite."
        "\nSupport policy: critical incidents receive an initial response within 30 minutes."
        "\nRetention policy: operational logs are retained for 14 days."
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Summarize only the release, support, and retention policies in three short bullets. "
                'Ignore routine team updates.\n<prism-source id="review">\n'
                + source
                + "\n</prism-source>",
            }
        ],
        "max_completion_tokens": 1024,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="prism-mock-cost")
    parser.add_argument("--no-auth", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("PRISM_API_KEY")
    if not args.no_auth and not key:
        parser.error("set PRISM_API_KEY or pass --no-auth for an anonymous server")
    with httpx.Client(
        base_url=args.base_url.rstrip("/") + "/",
        headers={} if args.no_auth else {"Authorization": f"Bearer {key}"},
        timeout=300,
        trust_env=False,
    ) as client:
        response = client.post("chat/completions", json=demo_request(args.model))
        if response.is_error:
            raise SystemExit(
                f"Request failed (HTTP {response.status_code}): {response.text}"
            )
        print(response.json()["choices"][0]["message"]["content"])
        costs = client.get("prism/costs")
        costs.raise_for_status()
        print(json.dumps(costs.json(), indent=2))


if __name__ == "__main__":
    main()
