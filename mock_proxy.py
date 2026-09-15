import json
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_A_URL = "http://10.0.4.32:8000/v1/chat/completions"
MODEL_B_URL = "http://localhost:12434/engines/v1/chat/completions"

def forward(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            models = {"data":[{"id":"smart-local"},{"id":"local-small"}]}
            self.wfile.write(json.dumps(models).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/v1/chat/completions"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
            req = json.loads(body)
            # Call both real models
            payload_a = dict(req)
            payload_a["model"] = "muse-glimmer-30b"
            payload_b = dict(req)
            payload_b["model"] = "ai/gemma4:E2B"
            try:
                resp_a = forward(MODEL_A_URL, payload_a)
                resp_b = forward(MODEL_B_URL, payload_b)
                # Simple best-of-2: pick A if it has content, else B
                content_a = resp_a.get("choices",[{}])[0].get("message",{}).get("content","")
                content_b = resp_b.get("choices",[{}])[0].get("message",{}).get("content","")
                chosen = resp_a if content_a else resp_b
                # Tag model
                chosen["model"] = "smart-local"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(chosen).encode())
                return
            except Exception as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                err = {"error":{"message":str(e)}}
                self.wfile.write(json.dumps(err).encode())
                return
        self.send_response(404)
        self.end_headers()

if __name__ == "__main__":
    print("Starting mock proxy on http://localhost:4001")
    HTTPServer(("0.0.0.0", 4001), Handler).serve_forever()
