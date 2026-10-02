"""Isolated deprecated, buffered demo forwarder. No production/SSE claims."""

import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..errors import PrismError
from ..storage import read_json
from .proxy_cmd import loopback


def run(ctx):
    if not loopback(ctx.args.host):
        raise PrismError(
            "legacy unauthenticated demo requires loopback binding; use LiteLLM for serving"
        )
    data = read_json(ctx.explicit(ctx.args.file))
    models = {m["name"]: m for m in data["models"]} if "models" in data else data
    if not isinstance(models, dict) or not all(
        isinstance(m, dict) and m.get("base_url") for m in models.values()
    ):
        raise PrismError("invalid demo model descriptions")
    if ctx.args.offline:
        from ..config import local_endpoint

        if any(not local_endpoint(m.get("base_url", "")) for m in models.values()):
            raise PrismError(
                "--offline demo requires local/private inference endpoints"
            )

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_GET(self):
            self.reply(
                200, {"data": [{"id": name} for name in models]}
            ) if self.path == "/v1/models" else self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.reply(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 262144:
                    return self.reply(400, {"error": "invalid request size"})
                body = json.loads(self.rfile.read(size))
                if body.get("stream") or body.get("model") not in models:
                    return self.reply(
                        400, {"error": "unknown model or unsupported streaming"}
                    )
                cfg = models[body["model"]]
                request = urllib.request.Request(
                    cfg["base_url"].rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(
                    request, timeout=ctx.args.timeout or 60
                ) as response:
                    self.reply(response.status, json.load(response))
            except (ValueError, OSError):
                self.reply(502, {"error": "demo forwarding failed"})

    print(
        "Deprecated loopback-only demo; responses are buffered and streaming is unsupported.",
        file=sys.stderr,
    )
    with HTTPServer((ctx.args.host, ctx.args.port), Handler) as server:
        server.serve_forever()
