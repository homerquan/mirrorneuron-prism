import argparse
from importlib.metadata import version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from litellm_multicall.config import load_policy_config, load_yaml

def main():
    parser = argparse.ArgumentParser(prog="mirrorneuron-prism")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    validate = sub.add_parser("validate")
    validate.add_argument("--policy-config", required=True)
    validate.add_argument("--litellm-config", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--policy-config", required=True)
    doctor.add_argument("--litellm-config", required=True)
    args = parser.parse_args()
    
    if args.version:
        try:
            v = version("mirrorneuron-prism")
        except Exception:
            v = "0.1.0"
        print(f"mirrorneuron-prism {v}")
        return
    
    if not args.command:
        try:
            v = version("mirrorneuron-prism")
        except Exception:
            v = "0.1.0"
        print(f"mirrorneuron-prism {v}")
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

