import sys
from types import SimpleNamespace

import pytest

from prism.decision_worker import serve


@pytest.mark.parametrize("phase", ["load", "receive"])
def test_terminal_interrupt_closes_router_worker_without_traceback(phase, monkeypatch):
    sent = []
    closed = []

    def load(*args, **kwargs):
        if phase == "load":
            raise KeyboardInterrupt
        return object()

    def receive():
        raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(load=load))
    connection = SimpleNamespace(
        send=sent.append, recv=receive, close=lambda: closed.append(True)
    )
    serve(
        connection,
        {"model": "checkpoint", "revision": "pinned", "expected_sha256": None},
    )
    assert closed == [True]
    assert sent == ([{"ready": True}] if phase == "receive" else [])
