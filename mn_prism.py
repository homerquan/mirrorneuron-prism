#!/usr/bin/env python3
"""Compatibility launcher; install Prism instead of changing sys.path."""

import sys


def main(argv=None):
    try:
        from litellm_multicall.cli import main as installed_main
    except ModuleNotFoundError as exc:
        if exc.name != "litellm_multicall":
            raise
        print("Install Prism first: python -m pip install -e '.[dev]'", file=sys.stderr)
        return 4
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--file" in arguments and "serve" not in arguments:
        arguments.insert(0, "serve")
    return installed_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
