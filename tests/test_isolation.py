"""Worker-process isolation.

The property is not that a worker runs — multiprocessing runs — but that a
native crash inside an analysis is a tool error the server survives, that the
numbers are the same in both modes, and that refusals keep their meaning across
the process boundary.
"""

import json

import numpy as np
import pytest

from plantcv_mcp import workers
from plantcv_mcp.morphology import MorphologyRefusedError
from plantcv_mcp.server import _measure_impl, _segment_impl, build_server
from plantcv_mcp.workers import WorkerCrashedError, dispatch, run_isolated


@pytest.fixture
def isolated():
    workers.set_isolation(True)
    try:
        yield
    finally:
        workers.set_isolation(None)
        workers.shutdown_worker()


def _green(tmp_path, name="green.png"):
    import cv2

    img = np.full((200, 200, 3), 128, np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


def test_a_crash_in_the_worker_is_an_error_the_server_survives(isolated, tmp_path):
    seg = _segment_impl(_green(tmp_path), "a", "otsu")
    # Positive control BEFORE the crash: the isolated path measures.
    assert _measure_impl(seg["session_id"])["traits"]["area"]["value"] == 10000.0

    with pytest.raises(WorkerCrashedError, match="signal"):
        run_isolated("_abort")

    # And AFTER: the next call gets a fresh worker and the same number.
    assert _measure_impl(seg["session_id"])["traits"]["area"]["value"] == 10000.0


@pytest.mark.anyio
async def test_a_crash_is_a_tool_error_over_the_real_mcp_layer(isolated, tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_server()
    path = _green(tmp_path)
    seg = json.loads(
        (
            await mcp.call_tool(
                "segment", {"image_path": path, "channel": "a", "method": "otsu"}
            )
        )
        .content[0]
        .text
    )
    # Make the worker die inside "measure" for exactly one call.
    real = workers.run_isolated

    def crash_once(name, *a, **k):
        workers.run_isolated = real
        return real("_abort")

    workers.run_isolated = crash_once
    with pytest.raises(ToolError, match="worker died"):
        await mcp.call_tool("measure", {"session_id": seg["session_id"]})
    ok = await mcp.call_tool("measure", {"session_id": seg["session_id"]})
    assert ok.structured_content["traits"]["area"]["value"] == 10000.0


def test_isolated_and_in_process_results_agree(tmp_path):
    import cv2

    img = np.full((400, 400, 3), 200, np.uint8)
    cv2.line(img, (200, 380), (200, 120), (40, 150, 40), 9)
    cv2.line(img, (200, 300), (260, 220), (40, 150, 40), 7)
    cv2.line(img, (200, 220), (140, 150), (40, 150, 40), 7)
    mask = np.where((img == (40, 150, 40)).all(axis=2), 255, 0).astype(np.uint8)

    args = {
        "measure": ((img, mask, ("size", "color"), 10.0, False), {}),
        "regions": (
            (img, mask),
            {
                "mode": "rect_grid",
                "nrows": 1,
                "ncols": 1,
                "coord": (100, 100),
                "height": 300,
                "width": 200,
                "spacing": (0, 0),
                "radius": None,
                "analyses": ("size",),
                "px_per_mm": None,
                "include_histograms": False,
            },
        ),
        "morphology": ((img, mask, 15, 25, None), {}),
    }
    workers.set_isolation(False)
    try:
        local = {k: dispatch(k, *a, **kw) for k, (a, kw) in args.items()}
    finally:
        workers.set_isolation(None)
    workers.set_isolation(True)
    try:
        remote = {k: dispatch(k, *a, **kw) for k, (a, kw) in args.items()}
    finally:
        workers.set_isolation(None)
        workers.shutdown_worker()

    assert local["measure"] == remote["measure"]
    assert local["regions"]["measurements"] == remote["regions"]["measurements"]
    assert local["regions"]["bboxes"] == remote["regions"]["bboxes"]
    assert local["morphology"].plant == remote["morphology"].plant
    assert local["morphology"].segments == remote["morphology"].segments
    assert np.array_equal(local["morphology"].overlay, remote["morphology"].overlay)


def test_refusals_keep_their_type_across_the_boundary(isolated):
    img = np.full((200, 200, 3), 200, np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    mask[40:160, 30:60] = 255
    mask[40:160, 140:170] = 255  # two comparable plants
    with pytest.raises(MorphologyRefusedError, match="measure_regions"):
        dispatch("morphology", img, mask, 15, 25, None)


def test_an_unpicklable_exception_still_reports_the_original_error(isolated):
    """An exception that cannot pickle must not be laundered into a generic
    crash: the parent gets the original error's text and the worker survives."""
    with pytest.raises(RuntimeError, match="boom"):
        run_isolated("_raise_unpicklable", "boom")
    # Positive control: the worker answers the next call.
    img = np.full((100, 100, 3), 200, np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[20:60, 20:60] = 255
    out = dispatch("measure", img, mask, ("size",), None, False)
    assert out["area"]["value"] == 1600.0


def test_recycle_boundary_and_a_crash_on_it(isolated, monkeypatch):
    """The worker recycles after WORKER_MAX_TASKS calls, and a crash landing
    exactly on the recycle boundary still leaves a working server."""
    monkeypatch.setattr(workers, "WORKER_MAX_TASKS", 2)
    img = np.full((100, 100, 3), 200, np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[20:60, 20:60] = 255
    args = ("measure", img, mask, ("size",), None, False)

    assert dispatch(*args)["area"]["value"] == 1600.0
    pid_first = workers._worker._proc.pid
    assert dispatch(*args)["area"]["value"] == 1600.0  # hits the boundary
    assert workers._worker._proc is None, "recycled after WORKER_MAX_TASKS"
    assert dispatch(*args)["area"]["value"] == 1600.0  # fresh worker
    assert workers._worker._proc.pid != pid_first

    # A crash on the boundary call: recycle bookkeeping must not double-stop
    # or wedge the restart.
    with pytest.raises(WorkerCrashedError, match="signal"):
        run_isolated("_abort")
    assert dispatch(*args)["area"]["value"] == 1600.0


def test_isolation_flag_comes_from_the_environment(monkeypatch):
    workers.set_isolation(None)
    monkeypatch.delenv("PLANTCV_MCP_ISOLATE", raising=False)
    assert workers.isolation_enabled() is True, "isolation is the default"
    monkeypatch.setenv("PLANTCV_MCP_ISOLATE", "0")
    assert workers.isolation_enabled() is False
    monkeypatch.setenv("PLANTCV_MCP_ISOLATE", "false")
    assert workers.isolation_enabled() is False
    monkeypatch.setenv("PLANTCV_MCP_ISOLATE", "1")
    assert workers.isolation_enabled() is True


def test_main_accepts_the_isolate_flag(monkeypatch):
    """`plantcv-mcp --isolate` turns isolation on before the server runs."""
    from plantcv_mcp import server

    seen = {}
    monkeypatch.setattr(
        server.MCPServer, "run", lambda self: seen.setdefault("ran", True)
    )
    monkeypatch.setenv("PLANTCV_MCP_ISOLATE", "0")  # env says off ...
    monkeypatch.setattr("sys.argv", ["plantcv-mcp", "--isolate"])
    workers.set_isolation(None)
    try:
        server.main()
        assert seen == {"ran": True}
        assert workers.isolation_enabled() is True  # ... the flag wins
        workers.set_isolation(None)
        monkeypatch.delenv("PLANTCV_MCP_ISOLATE")
        monkeypatch.setattr("sys.argv", ["plantcv-mcp", "--no-isolate"])
        server.main()
        assert workers.isolation_enabled() is False
    finally:
        workers.set_isolation(None)


def test_worker_serialises_concurrent_calls(isolated):
    """Two threads through one worker must both get their own numbers."""
    from concurrent.futures import ThreadPoolExecutor

    def scene(side):
        img = np.full((200, 200, 3), 200, np.uint8)
        mask = np.zeros((200, 200), np.uint8)
        lo = (200 - side) // 2
        mask[lo : lo + side, lo : lo + side] = 255
        return img, mask

    a, b = scene(40), scene(100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(5):
            fa = pool.submit(dispatch, "measure", *a, ("size",), None, False)
            fb = pool.submit(dispatch, "measure", *b, ("size",), None, False)
            assert fa.result()["area"]["value"] == 1600.0
            assert fb.result()["area"]["value"] == 10000.0


def test_tool_layer_sequence_runs_isolated():
    """The fresh-process driver from test_tool_layer_sequence, with isolation on."""
    import os
    import subprocess
    import sys

    from test_tool_layer_sequence import DRIVER

    proc = subprocess.run(
        [sys.executable, "-c", DRIVER],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "PLANTCV_MCP_ISOLATE": "1"},
    )
    assert "SEQUENCE_OK" in proc.stdout, proc.stderr[-3000:]


def test_cli_configured_roots_reach_the_worker(isolated, tmp_path):
    """`--root` configures roots via set_roots(), which spawn does NOT inherit
    (only the env var would survive). The worker must be told explicitly, or a
    future worker-side file read would be unrestricted under CLI roots."""
    import os

    from plantcv_mcp import paths

    paths.set_roots([str(tmp_path)])
    try:
        workers.shutdown_worker()  # a fresh worker must pick up the live roots
        assert run_isolated("_configured_roots") == [os.path.realpath(str(tmp_path))]
        # And a WARM worker must track a later change: roots travel with each
        # request, or a worker spawned under one policy silently enforces it
        # forever (found live: a batch refused paths the parent had re-allowed).
        paths.set_roots(None)
        assert run_isolated("_configured_roots") is None
    finally:
        paths.set_roots(None)
        workers.shutdown_worker()


def test_a_worker_that_survives_sigkill_is_reported_not_ignored():
    """A process in uninterruptible kernel sleep (state D) can survive SIGKILL.
    Silently starting a fresh worker on top would stack zombies holding memory
    and file handles; the survivor must be named loudly."""
    from plantcv_mcp.workers import _Worker

    class Immortal:
        pid = 4242

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

        def kill(self):
            pass

    w = _Worker()
    w._proc = Immortal()
    w._conn = None
    with pytest.raises(RuntimeError, match="4242"):
        w._stop()
    assert w._proc is None  # the dead handle is dropped either way
