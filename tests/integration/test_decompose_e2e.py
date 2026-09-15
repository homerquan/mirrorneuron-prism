import json
import urllib.request
import pytest

MODEL_URL = "http://10.0.4.32:8000/v1/chat/completions"

def post_json(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(MODEL_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.load(resp)
    return body

def extract_content(body):
    msg = body.get("choices", [{}])[0].get("message", {})
    content = msg.get("content", "")
    if not content:
        content = msg.get("reasoning", "")
    return content

@pytest.mark.integration
def test_decompose_big_task_into_small_calls():
    # Big task
    big_task = "Explain DAGs, give a concrete example, and list three real-world use cases."
    
    # Decompose into sub-tasks
    subs = [
        "Give a concise definition of a Directed Acyclic Graph in 2 sentences.",
        "Give a concrete example of a DAG with nodes and edges.",
        "List three real-world use cases for DAGs."
    ]
    
    results = []
    for sub in subs:
        payload = {
            "model": "muse-glimmer-30b",
            "messages": [{"role": "user", "content": sub}],
            "temperature": 0.2,
            "max_completion_tokens": 256
        }
        body = post_json(payload)
        content = extract_content(body)
        results.append(content)
        assert len(content) > 10
    
    # Fuse
    fused = "\n\n".join([
        "Definition:\n" + results[0],
        "Example:\n" + results[1],
        "Use cases:\n" + results[2]
    ])
    
    assert "DAG" in fused or "Directed Acyclic Graph" in fused
    assert len(fused) > 100
