"""Explicit source boundaries preserve the instructions on both sides of a corpus."""

import os
import sys
from pathlib import Path

from openai import OpenAI

corpus = Path(sys.argv[1]).read_text()
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", api_key=os.environ["PRISM_API_KEY"]
)
response = client.chat.completions.create(
    model="prism",
    messages=[
        {"role": "system", "content": "Answer only from supplied evidence."},
        {
            "role": "user",
            "content": "Summarize the policy exceptions and their qualifications.\n"
            + '<prism-source id="policies">\n'
            + corpus
            + "</prism-source>\n"
            + "Cite source spans. A missing fact should be stated as uncertain.",
        },
    ],
    max_completion_tokens=1000,
)
print(response.choices[0].message.content)
