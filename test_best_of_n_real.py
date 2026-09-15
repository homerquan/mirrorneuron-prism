"""
Real best-of-N with budget tracker and LLM judge.
"""
import json
import urllib.request

MODEL_A_URL = "http://10.0.4.32:8000/v1/chat/completions"
MODEL_B_URL = "http://localhost:12434/engines/v1/chat/completions"

def post_json(url, payload, timeout=15):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except Exception as e:
        return None, {"error": str(e)}

class BudgetTracker:
    def __init__(self, max_model_calls: int = 5):
        self.max_model_calls = max_model_calls
        self.used = 0
    
    def reserve(self):
        if self.used >= self.max_model_calls:
            raise RuntimeError("Budget exhausted")
        self.used += 1
    
    def spend(self, n=1):
        self.used += n

budget = BudgetTracker(max_model_calls=5)

def call_model(name, url, payload):
    budget.reserve()
    print(f"[{budget.used}/{budget.max_model_calls}] Calling {name}...")
    status, body = post_json(url, payload)
    budget.spend()
    return name, status, body

def extract_content(body):
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    # Fallback to reasoning if content empty
    if not content:
        content = msg.get("reasoning", "")
    return content

# 1. Generate candidates
prompt = "Explain what a DAG is in one paragraph."

cand_a_payload = {
    "model": "muse-glimmer-30b",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.2,
    "max_completion_tokens": 256
}
cand_b_payload = {
    "model": "ai/gemma4:E2B",
    "messages": [{"role": "user", "content": prompt}]
}

name_a, status_a, body_a = call_model("muse-glimmer-30b", MODEL_A_URL, cand_a_payload)
name_b, status_b, body_b = call_model("ai/gemma4:E2B", MODEL_B_URL, cand_b_payload)

content_a = extract_content(body_a)
content_b = extract_content(body_b)

print("\n--- Candidate A ---")
print(content_a[:500])
print("\n--- Candidate B ---")
print(content_b[:500])

# 2. Judge
judge_prompt = f"""Question: {prompt}

Answer A: {content_a[:500]}
Answer B: {content_b[:500]}

Which is better? Reply with A or B only.
"""

judge_payload = {
    "model": "muse-glimmer-30b",
    "messages": [{"role": "user", "content": judge_prompt}],
    "temperature": 0,
    "max_completion_tokens": 256
}

name_j, status_j, body_j = call_model("judge", MODEL_A_URL, judge_payload)
print(f"Judge status: {status_j}")
print("Judge body:", json.dumps(body_j)[:1000])
judge_text = extract_content(body_j) or ""
print("\n--- Judge extracted ---")
print(repr(judge_text))

# Parse simple winner
winner = "A"
jt = judge_text.lower()
# Simple heuristic: look for isolated A or B at end
import re
match = re.search(r'\b([ab])\b', jt.strip()[-100:])
if match:
    winner = match.group(1).upper()
else:
    # fallback to length
    winner = "A" if len(content_a) > len(content_b) else "B"

print("\n=== Result ===")
print(f"Selected: {winner}")
if winner == "A":
    print(content_a)
else:
    print(content_b)

print(f"\nBudget used: {budget.used}/{budget.max_model_calls}")
