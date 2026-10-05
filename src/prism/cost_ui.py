"""Live alternate-screen cost dashboard. No request contents are retained."""

import io
import logging
import re
import sys
import time
from collections import deque
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from threading import RLock

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def money(value):
    return "unavailable" if value is None else f"${value:.6f}"


class DashboardLogHandler(logging.Handler):
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def emit(self, record):
        # Uvicorn's status/access messages contain operational metadata. Other
        # libraries may log payloads; retain only their logger and severity.
        message = (
            record.getMessage()
            if record.name.startswith("uvicorn")
            else f"{record.name}: {record.levelname}"
        )
        self.dashboard.event(message)


class DashboardStream(io.TextIOBase):
    """Keep incidental library output from overwriting the alternate screen."""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def write(self, text):
        if text.strip():
            # Do not expose arbitrary library text, prompts, or credentials.
            self.dashboard.event("Library output received", deduplicate=True)
        return len(text)

    def flush(self):
        pass


class CostDashboard:
    def __init__(self, console, snapshot, *, host, port, profiles, no_auth):
        self.console = console
        self.snapshot = snapshot
        self.address = f"http://{host}:{port}"
        self.profiles = ", ".join(profiles)
        self.no_auth = no_auth
        self.state = "Starting · preparing Laya"
        self.events = deque(maxlen=8)
        self.lock = RLock()
        self.live = None

    def event(self, message, *, deduplicate=False):
        message = re.sub(r"[\x00-\x1f\x7f]", " ", message)[:200]
        with self.lock:
            if "Application startup complete" in message:
                self.state = "Ready"
            elif "Shutting down" in message:
                self.state = "Stopping"
            elif "Application startup failed" in message or "[Errno" in message:
                self.state = "Failed"
            if not deduplicate or not self.events or self.events[-1][1] != message:
                self.events.append((time.strftime("%H:%M:%S"), message))

    def render(self):
        value = self.snapshot()
        with self.lock:
            state, events = self.state, list(self.events)
        seconds = max(0, int(time.time() - value["started_at"]))
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        uptime = f"{hours:02}:{minutes:02}:{seconds:02}"
        hypothetical = value["pricing_mode"] == "hypothetical"
        compact = self.console.width < 70 or self.console.height < 24
        header = Panel(
            Text(
                f"{self.profiles} · {state} · up {uptime}"
                if compact
                else f"{self.profiles} · {self.address}\n{state} · up {uptime}",
                style="bold cyan",
            ),
            title="Prism · Cost dashboard",
            border_style="cyan",
        )
        total = value["total_cost_usd"]
        spend = Text(
            money(total if total is not None else value["known_cost_usd"]),
            style="bold cyan",
        )
        spend.append(
            "\n" + ("Known subtotal · incomplete" if total is None else "Since start")
        )
        spend.append(
            f"\nInput: {value['reported_input_tokens']:,}\nOutput: {value['reported_output_tokens']:,}"
        )
        spend.append(
            f"\n{value['physical_calls']:,} calls · {value['requests']:,} requests"
        )
        saved, percent = value["estimated_saved_usd"], value["estimated_saved_percent"]
        savings = Text(
            money(saved),
            style="bold green" if saved is not None and saved >= 0 else "bold yellow",
        )
        savings.append(
            "\n"
            + (
                f"{percent:.1f}% saved"
                if percent is not None
                else "Percent unavailable"
            )
        )
        savings.append(
            "\n"
            + (
                "Mixing cost more than direct"
                if saved is not None and saved < 0
                else "Estimated vs direct"
            )
        )
        baseline = Text(money(value["estimated_baseline_cost_usd"]), style="bold")
        baseline.append(f"\nActual: {money(value['compared_cost_usd'])}")
        baseline.append(
            f"\n{value['compared_requests']:,} compared · {value['excluded_requests']:,} excluded"
        )
        baseline.append("\nEligible requests only")
        metrics = Layout(name="metrics", size=7)
        metrics.split_row(
            Layout(
                Panel(
                    spend,
                    title="Simulated spend" if hypothetical else "Token spend",
                    border_style="cyan",
                )
            ),
            Layout(
                Panel(
                    savings,
                    title="Estimated savings",
                    border_style="green"
                    if saved is not None and saved >= 0
                    else "yellow",
                )
            ),
            Layout(Panel(baseline, title="Direct baseline", border_style="blue")),
        )
        if compact:
            summary = Text(
                f"{'Simulated spend' if hypothetical else 'Spend'}: {money(total if total is not None else value['known_cost_usd'])}"
                + (" (incomplete)" if total is None else "")
                + f"\nSaved: {money(saved)} · "
                + (f"{percent:.1f}%" if percent is not None else "percent unavailable")
                + f"\nBaseline: {money(value['estimated_baseline_cost_usd'])} · {value['compared_requests']} compared"
                + f"\nTokens: {value['reported_input_tokens']:,} in / {value['reported_output_tokens']:,} out"
                + f"\n{value['physical_calls']} calls · {value['requests']} requests"
            )
            metrics = Layout(
                Panel(summary, title="Since process start", border_style="cyan"), size=7
            )
        table = Table(expand=True, box=None, header_style="bold cyan")
        narrow = self.console.width < 90
        columns = (
            ("Model", "Calls", "Input / output", "Spend")
            if compact
            else ("Model", "Calls", "Input", "Output", "Spend", "USD / 1M in → out")
        )
        for title in columns:
            table.add_column(
                title, justify="left" if title == "Model" else "right", overflow="fold"
            )
        for model in value["models"]:
            rates = [
                "?" if model[k] is None else f"${model[k]:g}"
                for k in ("input_cost_per_million", "output_cost_per_million")
            ]
            cost = money(model["known_cost_usd"])
            if model["total_cost_usd"] is None:
                cost += " *"
            cells = (Text(model["model_id"]), str(model["physical_calls"]))
            if compact:
                cells += (
                    f"{model['reported_input_tokens']:,} / {model['reported_output_tokens']:,}",
                    cost,
                )
            else:
                cells += (
                    f"{model['reported_input_tokens']:,}",
                    f"{model['reported_output_tokens']:,}",
                    cost,
                    (" / " if narrow else " → ").join(rates),
                )
            table.add_row(*cells)
        detail = Text(
            f"Unpriced calls {value['unpriced_calls']} · missing usage {value['unreported_usage_calls']} · "
            f"client auth {'off' if self.no_auth else 'on'}\n"
        )
        if hypothetical:
            detail.append(
                "HYPOTHETICAL PRICES · real model calls · simulated dollars\n",
                style="bold yellow",
            )
        else:
            detail.append(
                "Configured token rates · estimated savings · excludes infrastructure\n"
            )
        detail.append(
            "Refreshes live · resize supported · Ctrl+C stops server", style="dim"
        )
        if compact:
            detail = Text(
                f"Unpriced {value['unpriced_calls']} · missing usage {value['unreported_usage_calls']}\n"
                + (
                    "HYPOTHETICAL PRICES · real calls\n"
                    if hypothetical
                    else "Configured rates · estimated savings\n"
                )
                + "Ctrl+C stops server · refreshes live"
            )
        layout = Layout()
        children = [
            Layout(header, size=3 if compact else 4),
            metrics,
            Layout(
                Panel(table, title="Physical model usage", border_style="cyan"),
                minimum_size=4 if compact else 6,
            ),
        ]
        if self.console.height >= 32:
            last = value["last_request"]
            activity = Text()
            if last:
                elapsed = last["elapsed_ms"]
                activity.append(
                    f"Last: {last['profile']} · {last['strategy']} · {last['stop_reason']}"
                    + (f" · {elapsed / 1000:.2f}s" if elapsed is not None else "")
                    + "\n",
                    style="bold",
                )
            for timestamp, message in events[-3:]:
                activity.append(f"{timestamp}  {message}\n")
            if not last and not events:
                activity.append("Waiting for the first request")
            children.append(
                Layout(
                    Panel(activity, title="Server activity", border_style="dim"), size=6
                )
            )
        children.append(
            Layout(
                Panel(detail, border_style="yellow" if hypothetical else "dim"), size=5
            )
        )
        layout.split_column(*children)
        return layout

    def __enter__(self):
        self.stack = ExitStack()
        self.handler = DashboardLogHandler(self)
        self.logging_state = []
        try:
            for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
                logger = logging.getLogger(name)
                self.logging_state.append(
                    (logger, logger.handlers[:], logger.level, logger.propagate)
                )
                logger.handlers = [self.handler]
                logger.propagate = False
            self.live = Live(
                console=self.console,
                screen=True,
                refresh_per_second=4,
                get_renderable=self.render,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.stack.enter_context(self.live)
            stream = DashboardStream(self)
            self.stack.enter_context(redirect_stdout(stream))
            self.stack.enter_context(redirect_stderr(stream))
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc):
        try:
            return self.stack.__exit__(*exc)
        finally:
            for logger, handlers, level, propagate in self.logging_state:
                logger.handlers, logger.level, logger.propagate = (
                    handlers,
                    level,
                    propagate,
                )
            self.handler.close()
