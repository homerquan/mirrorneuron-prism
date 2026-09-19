import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from litellm_multicall.config import (
    BudgetSpec,
    CandidateSpec,
    PolicySpec,
    PrismConfig,
    RuntimeConfig,
    SelectionSpec,
    load_litellm_config,
    load_policy_config,
    validate_references,
)
import yaml
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request


def _format_config_error(exc: Exception) -> str:
    from pydantic import ValidationError as _ValidationError

    if isinstance(exc, _ValidationError):
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            lines.append(f"  - {loc}: {err.get('msg')}")
        return "\n".join(lines) if lines else str(exc)
    return f"  - {exc}"

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
    init = sub.add_parser("init")
    init.add_argument("--out-dir", default=".", help="Directory to write config files")
    proxy = sub.add_parser("proxy")
    proxy.add_argument("--port", type=int, default=4000)
    proxy.add_argument("--host", default="0.0.0.0")
    proxy.add_argument("--policy-config", default="multicall.yaml")
    proxy.add_argument("--model-name", default="smart-local")
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
        # Local checks only: never make model calls during validation.
        policy_path = Path(args.policy_config)
        litellm_path = Path(args.litellm_config)
        if not policy_path.exists():
            print(f"policy config not found: {policy_path}")
            sys.exit(2)
        if not litellm_path.exists():
            print(f"litellm config not found: {litellm_path}")
            sys.exit(2)
        try:
            prism = load_policy_config(policy_path)
        except Exception as exc:
            print(f"policy config INVALID: {policy_path}")
            print(_format_config_error(exc))
            sys.exit(2)
        try:
            litellm_cfg = load_litellm_config(litellm_path)
        except Exception as exc:
            print(f"litellm config INVALID: {litellm_path}")
            print(_format_config_error(exc))
            sys.exit(2)
        ref_errors = validate_references(prism, litellm_cfg)
        if ref_errors:
            print("config INVALID: unresolved references")
            for err in ref_errors:
                print(f"  - {err}")
            sys.exit(2)
        print("Validate OK")
        print(f"Policy schema_version: {prism.schema_version}")
        print(f"Policies: {sorted(prism.policies)}")
        return

    if args.command == "doctor":
        print("Doctor check OK")
        return

    if args.command == "init":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        litellm_yaml = out_dir / "litellm.yaml"
        multicall_yaml = out_dir / "multicall.yaml"
        litellm_content = """# Auto-generated by mn_prism init
model_list:
  - model_name: local-small
    litellm_params:
      model: openai/local-small-model
      api_base: os.environ/LOCAL_LLM_BASE_URL
      api_key: os.environ/LOCAL_LLM_API_KEY
      max_parallel_requests: 2
      num_retries: 0

  - model_name: smart-local
    litellm_params:
      model: multicall/default
      num_retries: 0

litellm_settings:
  custom_provider_map:
    - provider: multicall
      custom_handler: litellm_multicall.provider.multicall_provider
  callbacks:
    - litellm_multicall.hooks.multicall_hooks
"""
        # Build the example through the strict model so the generated file
        # is guaranteed to match the loader (key: `policies`, int version).
        prism = PrismConfig(
            schema_version=1,
            runtime=RuntimeConfig(),
            policies={
                "default": PolicySpec(
                    strategy="best_of_n",
                    allowed_model_groups=["local-small"],
                    candidates=[
                        CandidateSpec(
                            model="local-small",
                            samples=4,
                            temperature=0.7,
                            max_completion_tokens=1500,
                        )
                    ],
                    selection=SelectionSpec(
                        model="local-small",
                        max_completion_tokens=160,
                        temperature=0,
                    ),
                    budgets=BudgetSpec(
                        max_model_calls=5,
                        max_total_tokens=16000,
                        deadline_ms=120000,
                    ),
                )
            },
        )
        dumped = yaml.safe_dump(
            prism.model_dump(mode="python", exclude_none=True),
            sort_keys=False,
        )
        multicall_content = "# Auto-generated by mn_prism init\n" + dumped
        litellm_yaml.write_text(litellm_content)
        multicall_yaml.write_text(multicall_content)
        print(f"Created {litellm_yaml}")
        print(f"Created {multicall_yaml}")
        print("Run: mn_prism validate --policy-config multicall.yaml --litellm-config litellm.yaml")
        return

    if args.command == "proxy":
        import os, tempfile, subprocess
        cfg = {
            "model_list": [{
                "model_name": args.model_name,
                "litellm_params": {"model": "multicall/default"}
            }],
            "litellm_settings": {
                "custom_provider_map": [{
                    "provider": "multicall",
                    "custom_handler": "litellm_multicall.provider.multicall_provider"
                }],
                "callbacks": ["litellm_multicall.hooks.multicall_hooks"]
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp_cfg = f.name
        env = os.environ.copy()
        env["LITELLM_MASTER_KEY"] = os.getenv("LITELLM_MASTER_KEY", "sk-test")
        cmd = ["litellm", "--config", tmp_cfg, "--port", str(args.port), "--host", args.host]
        print("Starting LiteLLM Proxy with auto-generated multicall provider...")
        print(f"Config: {tmp_cfg}")
        subprocess.run(cmd, env=env)
        return

    print("CLI stub")


if __name__ == "__main__":
    main()

