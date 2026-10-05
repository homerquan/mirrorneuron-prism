"""Terminal presentation; redirected output remains ordinary JSON."""

import json
import sys
from contextlib import contextmanager, nullcontext

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def format_rate(value):
    return "unknown" if value is None else f"${value:g}"


class Output:
    def __init__(self, mode="auto", *, console=None, errors=None):
        self.console = console or Console(file=sys.stdout, highlight=False)
        self.errors = errors or Console(file=sys.stderr, highlight=False)
        self.human = mode == "table" or (mode == "auto" and self.console.is_terminal)
        self.mode = mode
        self.dashboard = None

    @contextmanager
    def cost_display(self, enabled, snapshot, *, host, port, profiles, no_auth):
        fullscreen = (
            enabled
            and self.human
            and self.console.is_terminal
            and self.errors.is_terminal
            and not self.errors.is_dumb_terminal
        )
        try:
            if fullscreen:
                from .cost_ui import CostDashboard

                with CostDashboard(
                    self.errors,
                    snapshot,
                    host=host,
                    port=port,
                    profiles=profiles,
                    no_auth=no_auth,
                ) as dashboard:
                    self.dashboard = dashboard
                    yield True
            else:
                self.serving(host, port, len(profiles), no_auth)
                if enabled:
                    self.costs(snapshot())
                yield False
        finally:
            self.dashboard = None
            if enabled:
                self.costs(snapshot())

    def status(self, message):
        return (
            self.errors.status(Text(message), spinner="dots")
            if self.human and self.errors.is_terminal
            else nullcontext()
        )

    def table(self, title, columns, rows):
        table = Table(
            title=Text(title), header_style="bold cyan", show_lines=False, expand=False
        )
        for column in columns:
            table.add_column(column, overflow="fold")
        for row in rows:
            table.add_row(*(Text(str(cell)) for cell in row))
        self.console.print(table)

    def emit(self, value):
        if not self.human:
            print(json.dumps(value, indent=2))
            return
        if "error" in value:
            error = value["error"]
            self.errors.print(
                Panel(
                    Text(
                        f"{error.get('code', 'error')}: {error.get('message', 'Command failed')}"
                    ),
                    title="Prism error",
                    border_style="red",
                )
            )
            return
        if "created" in value:
            self.table(
                "Configuration created", ["File"], [(p,) for p in value["created"]]
            )
            for command in value.get("next_steps", []):
                self.console.print(Text(command, style="cyan"))
        elif value.get("object") == "models":
            self.table(
                "Physical models",
                ["Model ID", "Configuration"],
                [
                    (
                        m["id"],
                        f"{m['name']}\n"
                        f"Context {m['context_window']:,} · output {m['max_output_tokens']:,}\n"
                        f"{', '.join(m['capabilities'])}\n"
                        + (
                            f"USD / 1M: input {format_rate(m.get('input_cost_per_million'))} · output {format_rate(m.get('output_cost_per_million'))}"
                            + (
                                " (hypothetical)"
                                if m.get("cost_rates_are_hypothetical")
                                else ""
                            )
                            + "\n"
                        )
                        + (
                            "Credential ready"
                            if m["credential_ready"]
                            else f"Set {m['credential_env']}"
                        ),
                    )
                    for m in value["data"]
                ],
            )
        elif value.get("object") == "profiles":
            self.table(
                "Virtual profiles",
                ["Alias", "Execution"],
                [
                    (
                        p["id"],
                        f"{p['strategy']}\n"
                        + (f"File: {p['file']}\n" if p.get("file") else "")
                        + f"Direct: {p['direct']}\n"
                        + (
                            f"Worker: {p['worker']}\nFinal: {p['synthesizer'] or p['direct']}\n"
                            if p["worker"]
                            else ""
                        )
                        + (f"Review: {p['verifier']}\n" if p.get("verifier") else "")
                        + f"JSON: {p['structured_output_model'] or 'default'}",
                    )
                    for p in value["data"]
                ],
            )
        elif value.get("object") == "capacity":
            self.table(
                "Measured capacity",
                ["Model / feature", "Result"],
                [
                    (
                        f"{m['model']}\n{feature}",
                        f"{observation['status']} · {observation['passed']}/{observation['total']}\n"
                        f"{observation['elapsed_ms']} ms\n"
                        f"{observation.get('error_code', m.get('scope', 'physical model'))}",
                    )
                    for m in value["data"]
                    for feature, observation in m["capabilities"].items()
                ],
            )
        elif "profiles" in value and value.get("valid"):
            self.console.print(Text("Configuration valid", style="bold green"))
            self.table(
                "Available aliases", ["Profile"], [(p,) for p in value["profiles"]]
            )
        elif "authentication_ready" in value:
            self.console.print(
                Text(
                    "Ready" if value["ready"] else "Not ready",
                    style="green" if value["ready"] else "yellow",
                )
            )
            self.table(
                "Backend health",
                ["Model", "Status", "Detail"],
                [
                    (
                        m["id"],
                        "ready"
                        if m.get("ready")
                        else "unprobed"
                        if not m["probed"]
                        else "failed",
                        m.get("error_code", ""),
                    )
                    for m in value["models"]
                ],
            )
            if not value["authentication_ready"]:
                self.errors.print(
                    Text(
                        "Set the configured Prism API key, or explicitly select --no-auth.",
                        style="yellow",
                    )
                )
        elif "policies" in value:
            self.table(
                "Execution policies",
                ["Policy", "Behavior"],
                [(p["id"], p["description"]) for p in value["policies"]],
            )
        else:
            self.console.print_json(data=value)

    def serving(self, host, port, profiles, no_auth):
        if self.human:
            self.console.print(
                Panel(
                    Text(
                        f"http://{host}:{port}\n{profiles} model aliases · client auth {'disabled (--no-auth)' if no_auth else 'enabled'}"
                    ),
                    title="Prism server",
                    border_style="cyan",
                )
            )

    def costs(self, value):
        if self.dashboard is not None:
            return  # The live renderer reads a consistent tracker snapshot.
        if not self.human or (self.mode == "auto" and not self.errors.is_terminal):
            self.errors.print(
                json.dumps(value), markup=False, highlight=False, soft_wrap=True
            )
            return
        total = value["total_cost_usd"]
        spend = (
            f"${total:.6f}"
            if total is not None
            else f"${value['known_cost_usd']:.6f} known; total incomplete"
        )
        saved = value["estimated_saved_usd"]
        percent = value["estimated_saved_percent"]
        savings = (
            "unavailable"
            if saved is None
            else f"${saved:.6f}"
            + (f" ({percent:.1f}%)" if percent is not None else " (zero-cost baseline)")
        )
        self.errors.print(
            Panel(
                Text(
                    f"Token cost since start: {spend} · {value['physical_calls']} calls\n"
                    f"Reported input tokens: {value['reported_input_tokens']:,} · output tokens: {value['reported_output_tokens']:,}\n"
                    f"Estimated savings vs direct: {savings} · {value['compared_requests']} compared requests\n"
                    f"Unpriced calls: {value['unpriced_calls']} · missing usage: {value['unreported_usage_calls']} · excluded requests: {value['excluded_requests']}"
                ),
                title="Prism simulated cost (hypothetical prices)"
                if value.get("pricing_mode") == "hypothetical"
                else "Prism cost",
                border_style="cyan",
            )
        )
