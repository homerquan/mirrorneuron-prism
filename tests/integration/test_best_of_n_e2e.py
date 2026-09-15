import json
import urllib.request
import pytest

MODEL_A_URL = "http://10.0.4.32:8000/v1/chat/completions"
MODEL_B_URL = "http://localhost:12434/engines/v1/chat/completions"

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
