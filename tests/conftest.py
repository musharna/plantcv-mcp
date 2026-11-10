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
import resource
from pathlib import Path

import pytest

# A mutant's tests may use this much data memory (RLIMIT_DATA, the kernel's
# VmData: private writable mappings, the inherited heap included). mutmut's own
# limit is CPU time only, and with timeout_constant = 60 it gives each mutant at
# least 30 CPU minutes and 15 wall minutes. On 2026-11-09 the mutant
# `x += 1` -> `x = 1` in lens._largest_valid_rectangle looped forever appending
# to a list, grew ~28 MB/s to 12.8 GB in 541 s, and the 16 GB runner went down
# with it at 7,848 of 12,216 mutants: no results saved, #97 not updated. The
# 11-07 and 11-08 runs died in the same stretch. Under this cap the append
# raises MemoryError within seconds and the mutant is reported as a timeout.
# Sized from that night's [watch] notes: mutant processes peaked at 1.9 GB
# resident (p99 1.0 GB), and one runaway at the cap beside three ordinary
# mutants still fits the runner.
MUTANT_DATA_CAP = 6 * 1024**3
# mutmut's verdicts by exit code (mutmut/stats.py, status_by_exit_code).
_MUTMUT_TIMEOUT = 36
_MUTMUT_SUSPICIOUS = 35
_CAPPED = pytest.StashKey[bool]()
_HIT_CAP = pytest.StashKey[bool]()


def _vm_data() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmData:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no VmData line in /proc/self/status")


def _cap_mutant_memory(config: pytest.Config) -> None:
    """Cap a mutant run's data memory; PLANTCV_TEST_MUTANT_DATA_CAP=0 turns it off.

    A cap the run already half fills would turn ordinary tests into
    MemoryErrors, and mutmut would count every mutant as killed. Such a run
    stops before any test and reports itself as suspicious instead.
    """
    raw = os.environ.get("PLANTCV_TEST_MUTANT_DATA_CAP")
    cap = MUTANT_DATA_CAP if raw is None else int(raw)
    if cap <= 0:
        return
    # Clamp first: the guard must judge the cap that will be enforced, and an
    # inherited hard limit below the configured cap is what gets enforced.
    _soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    source = "configured"
    if hard != resource.RLIM_INFINITY and hard < cap:
        cap, source = hard, "inherited hard limit"
    used = _vm_data()
    if used > cap // 2:
        pytest.exit(
            f"mutant memory cap {cap} B ({source}) is too tight: this process "
            f"already uses {used} B of data memory before any test "
            "(tests/conftest.py)",
            returncode=_MUTMUT_SUSPICIOUS,
        )
    resource.setrlimit(resource.RLIMIT_DATA, (cap, hard))
    config.stash[_CAPPED] = True


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    if call.excinfo is not None and call.excinfo.errisinstance(MemoryError):
        item.config.stash[_HIT_CAP] = True


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

    OpenMP is pinned to one thread for the same runs. mutmut forks each mutant
    from the process that ran the stats pass, and that pass reaches
    scikit-learn's KMeans (pcv.roi.auto_grid -> GaussianMixture), which starts
    libgomp's thread pool. A forked child has none of those threads, so the
    first KMeans in a mutant run waited forever on them: every mutant whose
    tests reach auto_grid was cut off as a "timeout" after ~15 min (52 on
    2026-11-06, none of them a test result; 0 of 1,158 survivors reach it).
    libgomp reads the variable once, when scikit-learn first loads it; this
    hook runs before any test module is imported, so it is early enough.
    """
    if "MUTANT_UNDER_TEST" in os.environ:
        os.environ["PLANTCV_MCP_ISOLATE"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"


def pytest_sessionstart(session: pytest.Session) -> None:
    # Real mutants only: the stats pass ("stats") and mutmut's other runs
    # ("fail", "list_all_tests", "") go through the whole suite uncapped. Its own
    # hook, not pytest_configure: test_isolation calls that one directly, inside
    # the suite's own process, which must never be capped.
    if "__mutmut_" in os.environ.get("MUTANT_UNDER_TEST", ""):
        _cap_mutant_memory(session.config)


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
    # A mutant that ran into the cap is a runaway, what a time limit would have
    # caught given the memory: report it as one, not as a test that killed it.
    if session.config.stash.get(_CAPPED, False) and session.config.stash.get(
        _HIT_CAP, False
    ):
        session.exitstatus = _MUTMUT_TIMEOUT
