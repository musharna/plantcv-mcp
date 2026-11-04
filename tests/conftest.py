"""Session hygiene for the worker processes the isolation layer spawns.

Two spawned workers were still alive when the suite finished (measured
2026-09-16 by running pytest in-process and listing the children of that
process). A plain `pytest` never notices: the interpreter exits and daemon
children die with it. mutmut notices: it runs this suite in-process for its
stats pass, so the leaked workers become children of mutmut itself, and when
one of them exits mid-run mutmut's reaper meets a pid it never registered and
aborts (`KeyError` in `read_one_child_exit_status`, 4,233 of 8,576 mutants in,
job green, issue titled as if complete).

So: shut the shared worker down when the session ends, then look for anything
still alive, and if there is, make the run red and name it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Under mutmut, run analyses in-process instead of in the isolation worker.

    mutmut decides which tests to run against a mutant from its stats pass,
    which records each call to a function in the process that makes it. A call
    the tool layer sends to the spawned worker is recorded in the worker and
    lost, so a test that reaches a function only through the server never runs
    against that function's mutants: split_by_seeds was credited with one test
    (a warning check) while the tests of its count and split went unseen, and
    `leaf_count=None` survived. In-process, every call is recorded.

    The env var rather than set_isolation(False): tests that reset with
    set_isolation(None) fall back to the env and stay in-process. mutmut sets
    MUTANT_UNDER_TEST before pytest starts, for the stats pass and each mutant.
    """
    if "MUTANT_UNDER_TEST" in os.environ:
        os.environ["PLANTCV_MCP_ISOLATE"] = "0"


def _live_children() -> list[tuple[int, str]]:
    me = os.getpid()
    out: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            ppid = next(
                line for line in status.splitlines() if line.startswith("PPid:")
            ).split()[1]
            if int(ppid) != me:
                continue
            cmd = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except (OSError, StopIteration):
            continue  # raced with exit; a process that is gone is not a leak
        # Full command line: a fixed-width cut once landed inside the word
        # "resource_tracker" (long worktree path + mutmut's -B) and the stdlib
        # tracker was reported as a leaked worker, failing mutmut's stats run.
        out.append((int(entry.name), cmd))
    return out


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    from plantcv_mcp import workers

    if (
        os.environ.get("PLANTCV_TEST_SKIP_WORKER_SHUTDOWN") != "1"
    ):  # control seam for the leak check
        # Kill first, then let shutdown_worker() clean the handles. The graceful
        # stop waits up to 5 s for a worker to answer; under a mutant that broke
        # the protocol it never does, and mutmut times a mutant's tests against
        # the clean run, so a 5 s wait on a sub-second test subset read as 364
        # "timeouts" in one night (2026-09-17). At session end the worker is
        # disposable: nothing will call it again.
        proc = getattr(workers._worker, "_proc", None)
        if proc is not None and proc.is_alive():
            proc.kill()
        workers.shutdown_worker()
    # The spawn context's resource_tracker is a stdlib helper that lives for the
    # whole process by design and exits with it; it is not a worker.
    leaked = [
        c
        for c in _live_children()
        if "multiprocessing" in c[1] and "resource_tracker" not in c[1]
    ]
    if leaked:
        lines = "\n".join(f"  pid {pid}: {cmd[:120]}" for pid, cmd in leaked)
        session.config.pluginmanager.get_plugin("terminalreporter").write_line(
            f"\nLEAKED WORKER PROCESSES after session teardown ({len(leaked)}):\n{lines}\n"
            "A worker left alive here is reaped by mutmut's parent and aborts the "
            "nightly mutation run. Shut it down in the test that started it.",
            red=True,
        )
        session.exitstatus = 1
