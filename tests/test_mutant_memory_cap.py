"""The per-mutant memory cap in conftest.py, driven the way mutmut drives it.

Each case runs pytest in a subprocess with this directory's conftest loaded as
a plugin and MUTANT_UNDER_TEST set as mutmut sets it, then reads the exit code
mutmut would read. The allocating test stops by itself at 2 GiB, so a cap that
does not engage fails here as exit 0 instead of running away.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
MUTANT = "plantcv_mcp.lens.x__largest_valid_rectangle__mutmut_42"

ALLOCATES = """
def test_allocates():
    chunks = []
    for _ in range(32):  # 32 x 64 MiB = 2 GiB, then stop
        chunks.append(bytearray(64 * 2**20))
"""
IDLE = """
def test_idle():
    assert sum(range(10)) == 45
"""


def _run(
    tmp_path: Path, body: str, mutant: str, cap: int | None
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "test_case.py").write_text(body)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["MUTANT_UNDER_TEST"] = mutant
    env["PYTHONPATH"] = os.pathsep.join([str(TESTS), env.get("PYTHONPATH", "")])
    env.pop("PLANTCV_TEST_MUTANT_DATA_CAP", None)
    if cap is not None:
        env["PLANTCV_TEST_MUTANT_DATA_CAP"] = str(cap)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "conftest",
            "-p",
            "no:cacheprovider",
            "test_case.py",
        ],
        check=False,  # the exit code is the verdict under test
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA and /proc are Linux"
)
def test_a_mutant_that_runs_into_the_cap_is_reported_as_a_timeout(
    tmp_path: Path,
) -> None:
    out = _run(tmp_path, ALLOCATES, MUTANT, cap=1024 * 2**20)
    assert out.returncode == 36, out.stdout + out.stderr
    assert "MemoryError" in out.stdout
    # Positive control: the same mutant run, capped, passes a test that stays under it.
    idle = _run(tmp_path, IDLE, MUTANT, cap=1024 * 2**20)
    assert idle.returncode == 0, idle.stdout + idle.stderr


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA and /proc are Linux"
)
def test_the_default_cap_leaves_room_for_the_allocating_test(tmp_path: Path) -> None:
    out = _run(tmp_path, ALLOCATES, MUTANT, cap=None)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA and /proc are Linux"
)
def test_a_cap_the_run_already_half_fills_stops_as_suspicious_before_any_test(
    tmp_path: Path,
) -> None:
    out = _run(tmp_path, IDLE, MUTANT, cap=16 * 2**20)
    assert out.returncode == 35, out.stdout + out.stderr
    assert "mutant memory cap 16777216 B is too tight" in out.stdout + out.stderr
    assert "1 passed" not in out.stdout


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA and /proc are Linux"
)
@pytest.mark.parametrize("name", ["stats", "list_all_tests", "fail", ""])
def test_mutmuts_own_runs_are_not_capped(tmp_path: Path, name: str) -> None:
    out = _run(tmp_path, ALLOCATES, name, cap=16 * 2**20)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RLIMIT_DATA and /proc are Linux"
)
def test_zero_turns_the_cap_off(tmp_path: Path) -> None:
    out = _run(tmp_path, ALLOCATES, MUTANT, cap=0)
    assert out.returncode == 0, out.stdout + out.stderr
