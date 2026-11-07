"""mutmut forks every mutant from a process that has already run the suite.

That process has run scikit-learn's KMeans (pcv.roi.auto_grid fits a
GaussianMixture, initialised by KMeans), so libgomp's OpenMP thread pool is
alive in it. A forked child has none of those threads; its first KMeans waits
on them forever. In the nightly that read as 52 "timeouts", each ~15 min of the
budget and none of them a test result (2026-11-06). tests/conftest.py pins
OpenMP to one thread whenever MUTANT_UNDER_TEST is set, so no pool is started.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

# Runs in a fresh interpreter, the way mutmut's main process starts: the suite's
# pytest_configure hook first, then scikit-learn, then a KMeans fit, then a fork
# whose child fits again. The parent kills a child that has not finished by the
# deadline, so a hang is reported, not left behind.
PROBE = """
import importlib.util, os, sys, time
spec = importlib.util.spec_from_file_location("_suite_conftest", sys.argv[1])
conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conftest)
conftest.pytest_configure(None)
assert "sklearn" not in sys.modules, "scikit-learn loaded before the hook ran"
import numpy as np
from sklearn.cluster import KMeans
X = np.random.default_rng(0).normal(size=(4000, 2))
KMeans(4, n_init=1, random_state=0).fit(X)
pid = os.fork()
if pid == 0:
    KMeans(4, n_init=1, random_state=0).fit(X)
    os._exit(0)
deadline = time.monotonic() + float(sys.argv[2])
while time.monotonic() < deadline:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        print("child exit", os.waitstatus_to_exitcode(status))
        sys.exit(0)
    time.sleep(0.2)
os.kill(pid, 9)
os.waitpid(pid, 0)
print("child hung")
"""


def _fork_after_kmeans(mutant_under_test: str | None) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("MUTANT_UNDER_TEST", "OMP_NUM_THREADS")
    }
    if mutant_under_test is not None:
        env["MUTANT_UNDER_TEST"] = mutant_under_test
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(CONFTEST), "30"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,  # the assertion below shows stderr; check=True would not
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="mutmut forks; no fork here")
@pytest.mark.skipif(
    (os.cpu_count() or 1) < 2,
    reason="a one-CPU host starts a one-thread pool, so the hazard cannot show",
)
def test_a_mutant_forked_after_kmeans_can_run_kmeans_again():
    # Control: without the hook's setting the child hangs. If this ever reads
    # "child exit 0", libgomp has become fork-safe here and the pin may go.
    assert _fork_after_kmeans(None) == "child hung"
    assert _fork_after_kmeans("stats") == "child exit 0"
    assert (
        _fork_after_kmeans("plantcv_mcp.imaging.x_render_region_overlay__mutmut_42")
        == "child exit 0"
    )


def test_the_hook_pins_openmp_only_under_mutmut(monkeypatch):
    spec = importlib.util.spec_from_file_location("_suite_conftest", CONFTEST)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)

    # The hook writes os.environ directly; registering the names first makes
    # monkeypatch put both back at teardown.
    monkeypatch.delenv("PLANTCV_MCP_ISOLATE", raising=False)
    # Outside mutmut a caller's thread count is left alone.
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    conftest.pytest_configure(None)
    # Into a local first: an assertion naming os.environ prints all of it.
    threads = os.environ.get("OMP_NUM_THREADS")
    assert threads == "8"

    # Under mutmut it is one thread, even if the caller asked for more.
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    conftest.pytest_configure(None)
    threads = os.environ.get("OMP_NUM_THREADS")
    assert threads == "1"
