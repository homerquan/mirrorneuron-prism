import httpx
import asyncio

async def call_model(url, payload):
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(url, json=payload)
            print(f"{url} -> {r.status_code}")
            print(r.json())
        except Exception as e:
            print(f"Error calling {url}: {e}")

async def main():
    payload1 = {
        "model": "muse-glimmer-30b",
        "messages": [{"role": "user", "content": "Hello! Please reply with one short sentence confirming that the model is working."}],
        "temperature": 0.2,
        "max_tokens": 100
    }
    payload2 = {
        "model": "ai/gemma4:E2B",
        "messages": [{"role": "user", "content": "Explain what a DAG is in one paragraph."}]
    }
    await call_model("http://10.0.4.32:8000/v1/chat/completions", payload1)
    # second endpoint different path
    await call_model("http://localhost:12434/engines/v1/chat/completions", payload2)

asyncio.run(main())
