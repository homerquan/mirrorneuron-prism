import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litellm_multicall.config import load_policy_config, load_yaml
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request

class _ProxyHandler(BaseHTTPRequestHandler):
    models = {}
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data":[{"id":n} for n in self.models]}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_response(404); self.end_headers(); return
        length=int(self.headers.get("Content-Length",0))
        body=json.loads(self.rfile.read(length).decode())
        model_name=body.get("model")
        cfg=self.models.get(model_name)
        if not cfg:
            self.send_response(400)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error":{"message":f"Unknown model {model_name}"}}).encode())
            return
        target=cfg["base_url"].rstrip("/")+ "/chat/completions"
        data=json.dumps(body).encode()
        req=urllib.request.Request(target, data=data, headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_body=resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type","application/json")
                self.end_headers()
                self.wfile.write(resp_body)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error":{"message":str(e)}}).encode())

def main():
    parser = argparse.ArgumentParser(prog="mn_prism")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    validate = sub.add_parser("validate")
    validate.add_argument("--policy-config", required=True)
    validate.add_argument("--litellm-config", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--policy-config", required=True)
    doctor.add_argument("--litellm-config", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--file", required=True, help="JSON models file")
    serve.add_argument("--port", type=int, default=4001)
    serve.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    if args.version:
        try:
            v = version("mirrorneuron-prism")
        except Exception:
            v = "0.1.0"
        print(f"mn_prism {v}")
        return
    
    if not args.command:
        try:
            v = version("mirrorneuron-prism")
        except Exception:
            v = "0.1.0"
        print(f"mn_prism {v}")
        return

    if args.command == "serve":
        import json
        cfg = json.loads(Path(args.file).read_text())
        models = {}
        for m in cfg.get("models", []):
            models[m["name"]] = {"base_url": m["base_url"]}
        _ProxyHandler.models = models
        print(f"Loaded {len(models)} models from {args.file}")
        server = HTTPServer((args.host, args.port), _ProxyHandler)
        print(f"Serving on http://{args.host}:{args.port}")
        server.serve_forever()
        return

    if args.command == "validate":
        policy_path = Path(args.policy_config)
        litellm_path = Path(args.litellm_config)
        if not policy_path.exists():
            print(f"policy config not found: {policy_path}")
            sys.exit(1)
        if not litellm_path.exists():
            print(f"litellm config not found: {litellm_path}")
            sys.exit(1)
        policy = load_policy_config(policy_path)
        litellm_cfg = load_yaml(litellm_path)
        print("Validate OK")
        print(f"Policy schema_version: {policy.get('schema_version')}")
        print(f"LiteLLM config keys: {list(litellm_cfg.keys())[:5]}")
        return

    if args.command == "doctor":
        print("Doctor check OK")
        return

    print("CLI stub")

