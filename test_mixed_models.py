"""
Mixed-model best-of-N demo.
Candidate A: muse-glimmer-30b @ http://10.0.4.32:8000/v1/chat/completions
Candidate B: ai/gemma4:E2B @ http://localhost:12434/engines/v1/chat/completions
"""
import asyncio
import json
import sys
from pathlib import Path

# Simple HTTP using urllib to avoid extra deps
import urllib.request

MODEL_A_URL = "http://10.0.4.32:8000/v1/chat/completions"
MODEL_B_URL = "http://localhost:12434/engines/v1/chat/completions"

def post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except Exception as e:
        return None, {"error": str(e)}

def call_muse_glimmer():
    payload = {
        "model": "muse-glimmer-30b",
        "messages": [{"role": "user", "content": "Hello! Please reply with one short sentence confirming that the model is working."}],
        "temperature": 0.2,
        "max_completion_tokens": 100
    }
    status, body = post_json(MODEL_A_URL, payload)
    return "muse-glimmer-30b", status, body

def call_gemma():
    payload = {
        "model": "ai/gemma4:E2B",
        "messages": [{"role": "user", "content": "Explain what a DAG is in one paragraph."}]
    }
    status, body = post_json(MODEL_B_URL, payload)
    return "ai/gemma4:E2B", status, body

def best_of_n_select(candidates):
    # Simple selector: first successful non-empty answer
    for name, status, body in candidates:
        if status and 200 <= status < 300:
            # Try to extract content
            choices = body.get("choices") or []
            msg = (choices[0].get("message") or {}).get("content") if choices else None
            if msg:
                return name, msg
    return None, "No valid candidate"

if __name__ == "__main__":
    print("Calling candidate models...")
    name_a, status_a, body_a = call_muse_glimmer()
    name_b, status_b, body_b = call_gemma()

    print(f"{name_a} -> HTTP {status_a}")
    print(f"{name_b} -> HTTP {status_b}")

    selected_name, selected_text = best_of_n_select([(name_a, status_a, body_a), (name_b, status_b, body_b)])
    print("\n=== Selected ===")
    print(f"Model: {selected_name}")
    print(f"Answer: {selected_text[:500]}")
