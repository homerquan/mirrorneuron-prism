"""Compatibility entry point for the implemented benchmark CLI."""
import sys

from litellm_multicall.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
