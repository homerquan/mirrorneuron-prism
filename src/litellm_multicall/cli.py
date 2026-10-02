"""Stable compatibility facade. Heavy commands are imported only on dispatch."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version

from .commands.output import envelope, redact, render
from .errors import PrismError


def main(argv: Sequence[str] | None = None) -> int:
    from .commands.parser import build_parser

    parser = build_parser()
    args = argparse.Namespace(
        format="human", quiet=False, verbose=False, offline=False, timeout=None
    )
    try:
        args = parser.parse_args(argv, namespace=args)
        if args.timeout is not None and args.timeout <= 0:
            raise PrismError("--timeout must be positive")
        command = getattr(args, "command_id", "version")
        if getattr(args, "version", False) or not getattr(args, "handler", None):
            try:
                v = version("mirrorneuron-prism")
            except PackageNotFoundError:
                raise PrismError(
                    "install this checkout: python -m pip install -e '.[dev]'",
                    4,
                    "not_installed",
                )
            if args.format == "human":
                print(f"mn_prism {v}")
            else:
                render(args, envelope("version", {"version": v}))
            return 0
        import signal
        import threading

        from .commands.context import Context

        ctx = Context.resolve(args)
        module, function = args.handler.split(":")
        previous = None

        def terminate(sig, frame):
            raise KeyboardInterrupt

        if threading.current_thread() is threading.main_thread() and command.startswith(
            ("classify.", "benchmark.run", "models.verify")
        ):
            previous = signal.signal(signal.SIGTERM, terminate)
        try:
            result = getattr(
                importlib.import_module(f".commands.{module}", __package__), function
            )(ctx)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
        render(args, envelope(command, result))
        return getattr(args, "exit_code", 0)
    except PrismError as exc:
        error = exc
    except SystemExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        error = PrismError("interrupted", 130, "interrupted")
    except (ValueError, TypeError) as exc:
        # Pydantic's default string includes input values, which can be secret.
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            message = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                for e in exc.errors(include_input=False)
            )
        else:
            message = str(exc)
        error = PrismError(message)
    except FileNotFoundError as exc:
        error = PrismError(
            f"required file or executable unavailable: {exc.filename}", 3, "unavailable"
        )
    except OSError as exc:
        error = PrismError(
            f"file/process failure: {exc.strerror}", 5, "external_failure"
        )
    except Exception:
        error = PrismError("unexpected internal error", 1, "internal_error")
    render(
        args,
        envelope(
            getattr(args, "command_id", "cli"),
            error={
                "kind": error.kind,
                "message": redact(str(error)),
                "exit_code": error.code,
            },
        ),
    )
    if getattr(args, "verbose", False):
        print(f"{error.kind}: {redact(str(error))}", file=sys.stderr)
    return error.code
