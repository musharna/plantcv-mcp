"""Run analyses in a worker process, so a native crash cannot take the server.

Two facts motivate this. A SIGSEGV inside PlantCV/OpenCV kills the whole stdio
server — 0.5.0 closed one by validating inputs, which only covers the inputs
someone thought of. And PlantCV keeps process-global state (`pcv.outputs`,
`params.saved_color_scale`, `params.sample_label`) that 0.5.0 and 0.7.0 guard
with a lock and explicit resets — again only the globals someone found. A worker
process makes both structural: a crash becomes WorkerCrashedError and the server
answers the next call; the worker's globals are nobody else's.

Design:
* `spawn` context, never fork — the server runs anyio worker threads, and forking
  a threaded process is unsafe.
* One warm worker, started lazily, recycled after WORKER_MAX_TASKS calls so a leak
  in native code is bounded. Calls are serialised by a lock (as the in-process
  path is by PCV_OUTPUTS_LOCK).
* Exceptions raised inside the worker come back as the same exception object
  (pickled), so a MorphologyRefusedError still reads as one at the tool layer.
* Isolation is ON by default (+7.7% measured overhead on a 3000x3000 image);
  PLANTCV_MCP_ISOLATE=0 or `plantcv-mcp --no-isolate` turns it off, and
  `set_isolation(bool)` overrides for tests. `dispatch()` is the one call site
  that picks.

The worker imports the analysis registry by name, so the code executed is the
same module the in-process path uses; nothing is duplicated.
"""

import multiprocessing as mp
import os
import threading
from multiprocessing.connection import Connection
from typing import Any

WORKER_MAX_TASKS = 200
_ENV = "PLANTCV_MCP_ISOLATE"


class WorkerCrashedError(Exception):
    """The analysis worker died mid-call. The server itself is still running."""


_isolate: bool | None = None


def isolation_enabled() -> bool:
    """Default ON. Measured on the 3000x3000 fixture: +7.7% wall time for
    measure(), against a gate of 25% — cheap for what it buys. Set
    PLANTCV_MCP_ISOLATE=0 (or `plantcv-mcp --no-isolate`) to run in-process."""
    if _isolate is not None:
        return _isolate
    value = os.environ.get(_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def set_isolation(enabled: bool | None) -> None:
    """Override the environment (None restores it). Tests and the CLI use this."""
    global _isolate
    _isolate = enabled


def _serve(conn: Connection) -> None:  # runs in the worker process
    from .analysis import REGISTRY

    registry = dict(REGISTRY)
    registry["_abort"] = _abort_for_tests
    while True:
        try:
            request = conn.recv()
        except EOFError:
            return
        if request is None:
            return
        name, args, kwargs = request
        try:
            conn.send(("ok", registry[name](*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001 — forwarded, not swallowed
            conn.send(("err", exc))


def _abort_for_tests() -> None:
    """Die the way native code dies. Registered only inside the worker."""
    os.abort()


class _Worker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: mp.Process | None = None
        self._conn: Connection | None = None
        self._tasks = 0

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(
            target=_serve, args=(child,), daemon=True, name="plantcv-mcp-worker"
        )
        proc.start()
        child.close()
        self._proc, self._conn, self._tasks = proc, parent, 0

    def _stop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except (OSError, ValueError):
                pass
            self._conn.close()
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        self._proc, self._conn = None, None

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._stop()
                self._start()
            assert self._conn is not None and self._proc is not None
            try:
                self._conn.send((name, args, kwargs))
                status, payload = self._conn.recv()
            except (EOFError, OSError, BrokenPipeError) as exc:
                self._proc.join(timeout=5)
                code = self._proc.exitcode
                self._stop()
                signal_note = (
                    f"signal {-code}"
                    if code is not None and code < 0
                    else f"exit code {code}"
                )
                raise WorkerCrashedError(
                    f"The analysis worker died during {name!r} ({signal_note}). "
                    "That is a crash inside native PlantCV/OpenCV code, not a "
                    "measurement result; the server is still running and the next "
                    "call starts a fresh worker. Re-run segment() on this image and "
                    "look at the overlay before measuring again."
                ) from exc
            self._tasks += 1
            if self._tasks >= WORKER_MAX_TASKS:
                self._stop()
            if status == "err":
                raise payload
            return payload

    def shutdown(self) -> None:
        with self._lock:
            self._stop()


_worker = _Worker()


def run_isolated(name: str, *args: Any, **kwargs: Any) -> Any:
    return _worker.call(name, *args, **kwargs)


def shutdown_worker() -> None:
    _worker.shutdown()


def dispatch(name: str, *args: Any, **kwargs: Any) -> Any:
    """Run analysis `name` in the worker when isolation is on, else in-process."""
    if isolation_enabled():
        return run_isolated(name, *args, **kwargs)
    from .analysis import REGISTRY

    return REGISTRY[name](*args, **kwargs)
