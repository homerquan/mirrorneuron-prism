"""Terminal presentation; redirected output remains ordinary JSON."""

import json
from contextlib import nullcontext

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Output:
    def __init__(self, mode="auto", *, console=None, errors=None):
        self.console = console or Console(highlight=False)
        self.errors = errors or Console(stderr=True, highlight=False)
        self.human = mode == "table" or (mode == "auto" and self.console.is_terminal)

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
                        f"{p['strategy']}\nDirect: {p['direct']}\n"
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
