"""The worker's request loop, driven in-process.

`_serve` normally runs only inside the spawned worker, where mutmut's stats
pass cannot see which test reached it (#97: its mutants read "no tests").
It takes nothing but a pipe connection, so here it runs on a thread of the
test process and the test plays the parent's side of the protocol: send
`(name, args, kwargs, roots)`, receive `("ok", value)` or `("err", exc)`.
`Connection.send` pickles in both directions, so the unpicklable path is the
real one. The fixture swaps `os.abort` for a raise before the loop starts,
so `_abort` is testable and can never take the test process down.
"""

import multiprocessing as mp
import os
import threading

import pytest

from plantcv_mcp import paths, workers


class _AbortCalled(Exception):
    pass


def _no_abort():
    raise _AbortCalled("os.abort() reached")


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(os, "abort", _no_abort)
    parent, child = mp.Pipe()
    thread = threading.Thread(target=workers._serve, args=(child,), daemon=True)
    thread.start()

    def call(name, *args, roots=None, **kwargs):
        parent.send((name, args, kwargs, roots))
        assert parent.poll(10), f"no reply to {name!r} within 10 s"
        return parent.recv()

    try:
        yield call, parent, thread
    finally:
        try:
            parent.send(None)
        except (OSError, ValueError):
            pass
        parent.close()
        thread.join(timeout=5)
        child.close()
        paths.set_roots(None)


def test_each_request_carries_its_own_roots(served, tmp_path):
    call, _, _ = served
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert call("_configured_roots", roots=[str(a)]) == ("ok", [os.path.realpath(a)])
    assert call("_configured_roots", roots=[str(b)]) == ("ok", [os.path.realpath(b)])
    # A request with no roots clears them; the previous request's do not linger.
    assert call("_configured_roots") == ("ok", None)


def test_args_and_kwargs_both_reach_the_analysis(served):
    call, _, _ = served
    for reply, text in (
        (call("_raise_unpicklable", "by position"), "by position"),
        (call("_raise_unpicklable", message="by keyword"), "by keyword"),
    ):
        assert reply[0] == "err"
        assert str(reply[1]).endswith(f"original error: {text}")


def test_the_test_only_entries_are_registered(served):
    call, _, _ = served
    status, exc = call("_abort")
    assert (status, type(exc).__name__, str(exc)) == (
        "err",
        "_AbortCalled",
        "os.abort() reached",
    )
    assert call("_configured_roots") == ("ok", None)


def test_an_error_comes_back_as_itself_and_the_loop_survives(served):
    call, _, _ = served
    status, exc = call("no_such_analysis")
    assert status == "err"
    assert type(exc) is KeyError
    assert exc.args == ("no_such_analysis",)
    # Positive control: the same loop still answers the next request.
    assert call("_configured_roots") == ("ok", None)


def test_an_unpicklable_error_falls_back_to_its_text(served):
    call, _, _ = served
    status, exc = call("_raise_unpicklable", "the original message")
    assert status == "err"
    assert type(exc) is RuntimeError
    assert str(exc) == (
        "RuntimeError from '_raise_unpicklable' could not cross the process "
        "boundary (unpicklable); original error: the original message"
    )
    # Positive control: a picklable error in the same loop is not rewritten.
    status, exc = call("no_such_analysis")
    assert (status, type(exc)) == ("err", KeyError)
    assert call("_configured_roots") == ("ok", None)


def test_none_ends_the_loop(served):
    call, parent, thread = served
    assert call("_configured_roots") == ("ok", None)  # it was serving
    parent.send(None)
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_a_closed_pipe_ends_the_loop(served):
    call, parent, thread = served
    assert call("_configured_roots") == ("ok", None)  # it was serving
    parent.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
