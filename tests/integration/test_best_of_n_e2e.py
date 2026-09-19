import json
import socket
import urllib.request
from urllib.parse import urlparse
import pytest

MODEL_A_URL = "http://10.0.4.32:8000/v1/chat/completions"
MODEL_B_URL = "http://localhost:12434/engines/v1/chat/completions"


def _reachable(url, timeout=2.0):
    try:
        parts = urlparse(url)
        host, port = parts.hostname, parts.port or 80
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _require_backends():
    # Manual live-endpoint smoke test only: skip (do not fail) when the
    # configured backends are unreachable so the default offline gate stays
    # green. This test exercises raw backends, not Prism expansion logic.
    missing = [u for u in (MODEL_A_URL, MODEL_B_URL) if not _reachable(u)]
    if missing:
        pytest.skip(f"live backends unreachable: {missing}")

def post_json(url, payload, timeout=15):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.load(resp)

def extract_content(body):
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "") or ""
    # fallback to reasoning if model returns reasoning-only
    if not content and "reasoning" in msg:
        content = msg.get("reasoning", "")
    return content

@pytest.mark.integration
def test_best_of_n_two_models():
    _require_backends()
    prompt = "Explain what a DAG is in one paragraph."
    # Candidate A
    payload_a = {
        "model": "muse-glimmer-30b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_completion_tokens": 256
    }
    status_a, body_a = post_json(MODEL_A_URL, payload_a)
    assert status_a == 200
    content_a = extract_content(body_a)
    assert len(content_a) > 50

    # Candidate B
    payload_b = {
        "model": "ai/gemma4:E2B",
        "messages": [{"role": "user", "content": prompt}]
    }
    status_b, body_b = post_json(MODEL_B_URL, payload_b)
    assert status_b == 200
    content_b = extract_content(body_b)
    assert len(content_b) > 50

    # Simple judge: pick longer answer as proxy for quality
    # In real deployment this would be an LLM judge
    winner = "A" if len(content_a) >= len(content_b) else "B"
    selected = content_a if winner == "A" else content_b

    assert len(selected) > 50
    # Ensure we made exactly 2 model calls
    # This test is a smoke check that both endpoints are reachable and return usable text
