import io
import logging
import sys

import pytest
from rich.console import Console

from prism.cli_ui import Output
from prism.cost_ui import CostDashboard
from prism.costs import CostTracker

from .test_costs import priced_model


def dashboard(console, tracker):
    return CostDashboard(
        console,
        tracker.snapshot,
        host="127.0.0.1",
        port=8080,
        profiles=["cost-demo"],
        no_auth=True,
    )


@pytest.mark.parametrize("size", [(120, 40), (80, 24), (60, 24), (80, 20)])
def test_dashboard_dimensions_usage_and_negative_savings(size):
    stream = io.StringIO()
    console = Console(file=stream, width=size[0], height=size[1], color_system=None)
    tracker = CostTracker({"small": priced_model()})
    tracker.reported_input_tokens = 1200
    tracker.reported_output_tokens = 240
    tracker.compared_requests = 1
    tracker.compared_cost = 0.02
    tracker.baseline_cost = 0.01
    tracker.requests = 1
    tracker.last_request = {
        "profile": "demo",
        "strategy": "text_synthesis",
        "stop_reason": "complete",
        "elapsed_ms": 1250,
    }
    ui = dashboard(console, tracker)
    ui.event("Application startup complete.")
    console.print(ui.render())
    text = stream.getvalue()
    assert "$-0.010000" in text
    assert "-100.0%" in text
    assert "1,200" in text and "240" in text
    assert "Ready" in text
    assert "Ctrl+C" in text
    assert "small" in text
    assert len(text.splitlines()) == size[1]
    if size[1] >= 32:
        assert "text_synthesis" in text and "1.25s" in text


def test_incomplete_and_zero_baselines_do_not_claim_savings():
    stream = io.StringIO()
    console = Console(file=stream, width=100, height=32, color_system=None)
    tracker = CostTracker({})
    tracker.unpriced_calls = 1
    tracker.reported_cost = 0.5
    tracker.requests = tracker.compared_requests = 1
    console.print(dashboard(console, tracker).render())
    text = stream.getvalue()
    assert "Known subtotal" in text and "incomplete" in text
    assert "$0.500000" in text and "Percent unavailable" in text
    assert "0.0% saved" not in text


@pytest.mark.parametrize("raises", [False, True])
def test_alternate_screen_restores_streams_and_logging_even_on_error(
    raises, monkeypatch
):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dashboard-fixture-key")
    stream = io.StringIO()
    console = Console(
        file=stream, force_terminal=True, width=100, height=32, color_system=None
    )
    tracker = CostTracker(
        {
            "small": priced_model().model_copy(
                update={"cost_rates_are_hypothetical": True}
            )
        }
    )
    output = Output("table", console=console, errors=console)
    stdout, stderr = sys.stdout, sys.stderr
    logger = logging.getLogger("uvicorn.error")
    original = (logger.handlers[:], logger.level, logger.propagate)
    caught = False
    try:
        with output.cost_display(
            True,
            tracker.snapshot,
            host="127.0.0.1",
            port=8080,
            profiles={"demo": None},
            no_auth=True,
        ) as fullscreen:
            assert fullscreen and output.dashboard is not None
            logger.warning("Application startup complete.")
            print("secret library payload")
            logging.getLogger("other-library").warning("secret provider payload")
            output.costs(tracker.snapshot())
            output.dashboard.live.refresh()
            if raises:
                raise RuntimeError("test startup failure")
    except RuntimeError:
        caught = True
    assert caught == raises
    assert sys.stdout is stdout and sys.stderr is stderr
    assert (logger.handlers, logger.level, logger.propagate) == original
    assert output.dashboard is None
    text = stream.getvalue()
    assert "\x1b[?1049h" in text and "\x1b[?1049l" in text
    assert "\x1b[?25h" in text
    assert "HYPOTHETICAL PRICES" in text
    assert "Prism simulated cost" in text
    assert "secret" not in text


@pytest.mark.parametrize("mode,terminal", [("auto", False), ("json", True)])
def test_nonterminal_and_json_reports_have_no_terminal_control(mode, terminal):
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=terminal)
    output = Output(mode, console=console, errors=console)
    tracker = CostTracker({})
    with output.cost_display(
        True,
        tracker.snapshot,
        host="127.0.0.1",
        port=8080,
        profiles={"demo": None},
        no_auth=True,
    ) as fullscreen:
        assert not fullscreen
        output.costs(tracker.snapshot())
    assert "\x1b" not in stream.getvalue()
    assert stream.getvalue().count('"scope": "since_process_start"') == 3


def test_activity_is_bounded_and_control_characters_are_removed():
    console = Console(file=io.StringIO(), width=100, height=32)
    ui = dashboard(console, CostTracker({}))
    for i in range(20):
        ui.event(f"event {i}\x00\n")
    ui.event("Shutting down")
    assert len(ui.events) == 8 and ui.state == "Stopping"
    assert all(
        "\x00" not in message and "\n" not in message for _, message in ui.events
    )
    ui.event("Application startup failed")
    assert ui.state == "Failed"
    ui.event("duplicate", deduplicate=True)
    ui.event("duplicate", deduplicate=True)
    assert sum(message == "duplicate" for _, message in ui.events) == 1


def test_cli_connects_dashboard_to_real_app_tracker(monkeypatch):
    import uvicorn

    from prism import cli

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dashboard-fixture-key")
    stream = io.StringIO()
    console = Console(
        file=stream, force_terminal=True, width=120, height=40, color_system=None
    )
    output = Output("auto", console=console, errors=console)
    monkeypatch.setattr(cli, "Output", lambda mode: output)
    seen = []

    def run(app, **kwargs):
        assert kwargs["log_config"] is None
        assert kwargs["log_level"] == "info"
        assert output.dashboard is not None
        assert app.state.costs.snapshot()["pricing_mode"] == "hypothetical"
        logging.getLogger("uvicorn.error").warning("Application startup complete.")
        output.dashboard.live.refresh()
        seen.append(app)

    monkeypatch.setattr(uvicorn, "run", run)
    assert (
        cli.main(["start", "--profile", "prism-mock-cost", "--show-cost", "--no-auth"])
        == 0
    )
    assert seen and "Ready" in stream.getvalue()
    assert "hypothetical prices" in stream.getvalue()


def test_stderr_redirection_keeps_json_cost_reports():
    terminal = Console(file=io.StringIO(), force_terminal=True)
    stream = io.StringIO()
    output = Output("auto", console=terminal, errors=Console(file=stream))
    output.costs(CostTracker({}).snapshot())
    assert stream.getvalue().startswith('{"object": "costs"')
    assert "\x1b" not in stream.getvalue()
