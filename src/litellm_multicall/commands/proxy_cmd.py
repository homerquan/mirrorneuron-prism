"""LiteLLM process ownership, without another gateway or Router."""

import copy
import ipaddress
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from ..errors import PrismError


def loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_config(llm):
    config = copy.deepcopy(llm)
    settings = config.setdefault("litellm_settings", {})
    providers = settings.setdefault("custom_provider_map", [])
    if not any(p.get("provider") == "multicall" for p in providers):
        providers.append(
            {
                "provider": "multicall",
                "custom_handler": "litellm_multicall.provider.multicall_provider",
            }
        )
    callbacks = settings.setdefault("callbacks", [])
    if "litellm_multicall.hooks.multicall_hooks" not in callbacks:
        callbacks.append("litellm_multicall.hooks.multicall_hooks")
    return config


def launch(config, policy_path, project_path, host, port, timeout=None):
    key = config.get("general_settings", {}).get("master_key") or os.environ.get(
        "LITELLM_MASTER_KEY"
    )
    if isinstance(key, str) and key.startswith("os.environ/"):
        key = os.environ.get(key.split("/", 1)[1])
    if not loopback(host) and (not key or key in ("sk-test", "sk-1234")):
        raise PrismError(
            "non-loopback binding requires a genuine configured master key"
        )
    env = os.environ.copy()
    env["PRISM_POLICY_CONFIG"] = str(policy_path)
    if project_path:
        env["PRISM_CONFIG"] = str(project_path)
    child = None
    previous = {}
    with tempfile.TemporaryDirectory(prefix="prism-proxy-") as temp:
        path = Path(temp) / "litellm.yaml"
        path.write_text(yaml.safe_dump(config))
        path.chmod(0o600)
        try:
            child = subprocess.Popen(
                ["litellm", "--config", str(path), "--host", host, "--port", str(port)],
                env=env,
                stdout=sys.stderr,
                stderr=sys.stderr,
            )

            def forward(sig, frame):
                if child.poll() is None:
                    child.send_signal(sig)

            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, forward)
            try:
                status = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise PrismError("proxy deadline exhausted", 6, "deadline") from exc
            return status if status >= 0 else 128 - status
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if child and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def run(ctx):
    _, llm, policy_path, _ = ctx.configs()
    if ctx.args.offline:
        from ..config import local_endpoint

        for entry in llm["model_list"]:
            parameters = entry["litellm_params"]
            if not parameters["model"].startswith(
                ("multicall/", "ollama/", "ollama_chat/")
            ) and not local_endpoint(parameters.get("api_base", "")):
                raise PrismError(
                    "--offline proxy requires explicitly local/private physical endpoints"
                )
    if ctx.args.model_name and ctx.args.model_name not in {
        m["model_name"] for m in llm["model_list"]
    }:
        raise PrismError(
            "--model-name must reference an alias in the complete LiteLLM configuration"
        )
    if any(
        m["litellm_params"]["model"].startswith("multicall/") for m in llm["model_list"]
    ):
        raise PrismError(
            "multicall inference is not implemented (P3); refusing to launch dummy completions",
            3,
            "unsupported_capability",
        )
    ctx.args.exit_code = launch(
        build_config(llm),
        policy_path,
        ctx.project_path,
        ctx.args.host,
        ctx.args.port,
        ctx.args.timeout,
    )
    return {"child_exit_code": ctx.args.exit_code}
