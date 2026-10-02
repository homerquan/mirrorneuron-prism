"""Run after prism init, editing models.json, and prism serve."""

import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", api_key=os.environ["PRISM_API_KEY"]
)
response = client.chat.completions.create(
    model="prism",
    messages=[{"role": "user", "content": "Explain virtual context in two sentences."}],
    max_completion_tokens=200,
)
print(response.choices[0].message.content)
