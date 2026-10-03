"""Use an optimization-enabled profile; see optimization/ for illustrative JSON."""

import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", api_key=os.environ["PRISM_API_KEY"]
)
response = client.chat.completions.create(
    model="prism-optimized",
    messages=[{"role": "user", "content": "Explain the tradeoffs of this design."}],
    max_completion_tokens=200,
    extra_body={
        "context_management": [
            {"prism_cost_priority": 0.8},
            {"prism_allowed_policies": ["direct", "draft_review"]},
            {"prism_max_calls": 3, "prism_max_cost_usd": 0.02},
        ]
    },
)
print(response.choices[0].message.content)
