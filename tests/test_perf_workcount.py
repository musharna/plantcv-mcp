"""The work-count ratchet's comparison: exact, named, and one-way.

The counting itself runs as its own CI step (`benchmarks/perf/workcount.py`,
every Python in the matrix); these tests pin the rules it applies to the
counts, and drive one real workload so the counter is seen to count.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "perf"))

import workcount

ROW = {
    "output": {"count": 7},
    "package": {"leaves.py:count_leaves": 1, "diagnostics.py:mask_report": 3},
    "cv2": {"cv2.findContours": 2},
    "plantcv": {"plantcv/plantcv/watershed.py:watershed_segmentation": 1},
    "spawns": {},
}


def _with(counter: str, key: str, n: int) -> dict:
    row = json.loads(json.dumps(ROW))
    row[counter][key] = n
    return row


def test_equal_counts_pass_and_one_extra_call_fails_naming_it() -> None:
    assert workcount.check({"count_leaves": ROW}, {"count_leaves": ROW}) == []
    lines = workcount.check(
        {"count_leaves": ROW},
        {"count_leaves": _with("package", "diagnostics.py:mask_report", 4)},
    )
    assert lines[0] == "count_leaves: package total 4 -> 5 (+1)"
    assert "+1  diagnostics.py:mask_report  (3 -> 4)" in lines[1]
    assert len(lines) == 2  # cv2, plantcv, spawns and output did not change


def test_a_drop_and_a_reshuffle_also_fail_until_recorded() -> None:
    drop = workcount.check({"w": ROW}, {"w": _with("cv2", "cv2.findContours", 1)})
    assert drop[0] == "w: cv2 total 2 -> 1 (-1)"
    moved = _with("package", "diagnostics.py:mask_report", 2)
    moved["package"]["leaves.py:helper"] = 1  # same total, different code path
    assert (
        workcount.check({"w": ROW}, {"w": moved})[0] == "w: package total 4 -> 4 (+0)"
    )


def test_a_different_output_fails_even_with_equal_counts() -> None:
    """The harness assertion: a workload that ran another input is caught by
    what it produced, not by the name it echoes."""
    other = dict(ROW, output={"count": 6})
    assert workcount.check({"w": ROW}, {"w": other}) == [
        "w: OUTPUT changed {'count': 7} -> {'count': 6}"
    ]


def test_missing_and_new_workloads_are_named() -> None:
    assert workcount.check({"a": ROW}, {}) == ["a: in the baseline but not run"]
    assert workcount.check({}, {"b": ROW}) == ["b: not in the baseline (run --update)"]


def test_the_real_counter_counts_one_workload_and_it_equals_the_baseline() -> None:
    """Real execution: a fresh interpreter runs list_methods through the tool
    layer under the profiler, and the result is the committed baseline row."""
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(workcount.__file__)),
            "--child",
            "list_methods",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert sum(got["package"].values()) > 0  # the profiler saw the package
    assert "server.py:list_methods_impl" in got["package"]
    baseline = json.loads(workcount.BASELINE.read_text())
    assert (
        workcount.check(
            {"list_methods": baseline["list_methods"]}, {"list_methods": got}
        )
        == []
    )
