#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request

class ProxyHandler(BaseHTTPRequestHandler):
    models = {}

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {"data":[{"id":name} for name in self.models]}
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        model_name = req.get("model")
        if not model_name:
            self.send_response(400)
            self.end_headers()
            return
        cfg = self.models.get(model_name)
        if not cfg:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error":{"message": f"Unknown model {model_name}"}}).encode())
            return
        target_url = cfg["base_url"].rstrip("/") + "/chat/completions"
        data = json.dumps(req).encode()
        req2 = urllib.request.Request(target_url, data=data, headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req2, timeout=60) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp_body)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error":{"message":str(e)}}).encode())

    def log_message(self, *args):
        sys.stderr.write(f"[{self.client_address[0]}] {args[1]}\n")

def load_models(path):
    data = json.loads(Path(path).read_text())
    models = {}
    # Support both {"models":[{"name":...}]} and flat list
    if "models" in data and isinstance(data["models"], list):
        for m in data["models"]:
            models[m["name"]] = {"base_url": m["base_url"], "api_key": m.get("api_key")}
    else:
        # assume dict
        for k,v in data.items():
            models[k] = v
    return models

def main():
    p = argparse.ArgumentParser(description="MirrorNeuron Prism simple proxy")
    p.add_argument("--file", required=True, help="JSON file defining models")
    p.add_argument("--port", type=int, default=4001, help="Binding port")
    p.add_argument("--host", default="0.0.0.0", help="Binding host")
    args = p.parse_args()
    models = load_models(args.file)
    ProxyHandler.models = models
    print(f"Loaded {len(models)} models from {args.file}")
    for name, cfg in models.items():
        print(f"  {name} -> {cfg['base_url']}")
    server = HTTPServer((args.host, args.port), ProxyHandler)
    print(f"Starting server on {args.host}:{args.port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
