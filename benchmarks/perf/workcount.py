"""Work-count ratchet: how much work each tool does, counted, never timed.

A wall-clock budget needs headroom to survive a slow host, and a regression
smaller than the headroom passes. A count of work is the same on every run, so
its ceiling is the baseline itself.

Each workload calls one tool through the real tool layer (``srv.call_tool``,
the path a client takes) on a fixed synthetic input, in a fresh interpreter,
with ``sys.setprofile`` (and ``threading.setprofile``: sync tools run on an
anyio worker thread) installed only around that one call. The code under test
is not edited. Counted:

- ``package``: calls of every function under ``src/plantcv_mcp``, keyed
  ``relpath:qualname``. List/dict/set comprehensions are skipped: Python 3.12
  inlined them (PEP 709), so counting their frames on 3.11 would make the
  counts differ by interpreter. A generator (a generator expression included)
  counts once, on its first entry, not on every resume: which resumes reach
  the profiler differs between patch releases (3.12.3 reports no event for
  closing a generator that ``any()`` left unfinished; 3.12.13 does).
- ``cv2``: OpenCV C functions called from Python (anywhere, PlantCV included).
- ``plantcv``: entries into PlantCV, i.e. calls of a PlantCV function from code
  that is not PlantCV, keyed ``plantcv/relpath:qualname``.
- ``spawns``: processes started (``fork_exec``, ``fork``, ``posix_spawn``).

Isolation is OFF (``set_isolation(False)``), so the analysis runs in this
process where it can be counted; with it on, all of it would happen in a
worker the profiler cannot see. ``batch``'s clock is a fake that advances one
second per read, so its time budget cannot branch on host speed.

    uv run python benchmarks/perf/workcount.py                    # check; exit 1 on any change
    uv run python benchmarks/perf/workcount.py --update           # record a drop (or a reshuffle)
    uv run python benchmarks/perf/workcount.py --update --allow-increase   # accept a rise
    uv run python benchmarks/perf/workcount.py --update --allow-output-change   # a workload's input changed
"""

from __future__ import annotations

import argparse
import asyncio
import dis
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINE = HERE / "baseline.json"
TESTS = REPO / "tests"
FIXTURES = TESTS / "fixtures"
INLINED_IN_312 = {"<listcomp>", "<dictcomp>", "<setcomp>"}
SPAWN_NAMES = {"fork_exec", "fork", "posix_spawn", "posix_spawnp"}
COUNTERS = ("package", "cv2", "plantcv", "spawns")
GENERATOR_FLAGS = (
    0x20 | 0x80 | 0x200
)  # CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR


# --- counting ---------------------------------------------------------------


class Count:
    """sys.setprofile counter for one tool call."""

    def __init__(self) -> None:
        import plantcv

        import plantcv_mcp

        self.pkg_root = str(Path(plantcv_mcp.__file__).resolve().parent) + os.sep
        self.pcv_root = str(Path(plantcv.__file__).resolve().parent) + os.sep
        self.kind: dict[str, tuple[str, str] | None] = {}
        # OpenCV's functions carry neither __module__ nor __self__, so they are
        # recognised by identity: every builtin reachable from a cv2 module.
        self.cv2_funcs: dict[int, str] = {}
        for modname, mod in list(sys.modules.items()):
            if mod is not None and (modname == "cv2" or modname.startswith("cv2.")):
                for attr, obj in list(vars(mod).items()):
                    if isinstance(obj, types.BuiltinFunctionType):
                        self.cv2_funcs[id(obj)] = f"{modname}.{attr}"
        self.first_yield: dict[types.CodeType, int] = {}
        self.package: Counter[str] = Counter()
        self.cv2: Counter[str] = Counter()
        self.plantcv: Counter[str] = Counter()
        self.spawns: Counter[str] = Counter()

    def _classify(self, filename: str) -> tuple[str, str] | None:
        k = self.kind.get(filename, False)
        if k is False:
            real = os.path.realpath(filename)
            if real.startswith(self.pkg_root):
                k = ("pkg", real[len(self.pkg_root) :])
            elif real.startswith(self.pcv_root):
                k = ("pcv", "plantcv/" + real[len(self.pcv_root) :])
            else:
                k = None
            self.kind[filename] = k
        return k

    def _resumed(self, frame: types.FrameType) -> bool:
        """A generator frame re-entered after a yield, not started."""
        code = frame.f_code
        if not code.co_flags & GENERATOR_FLAGS:
            return False
        first = self.first_yield.get(code)
        if first is None:
            first = min(
                (
                    i.offset
                    for i in dis.get_instructions(code)
                    if i.opname == "YIELD_VALUE"
                ),
                default=sys.maxsize,
            )
            self.first_yield[code] = first
        return frame.f_lasti >= first

    def __call__(self, frame: types.FrameType, event: str, arg: Any) -> None:
        if event == "call":
            code = frame.f_code
            k = self._classify(code.co_filename)
            if k is None or self._resumed(frame):
                return
            if k[0] == "pkg":
                if code.co_name not in INLINED_IN_312:
                    self.package[f"{k[1]}:{code.co_qualname}"] += 1
            else:
                caller = frame.f_back
                ck = (
                    self._classify(caller.f_code.co_filename)
                    if caller is not None
                    else None
                )
                if ck is None or ck[0] != "pcv":
                    self.plantcv[f"{k[1]}:{code.co_qualname}"] += 1
        elif event == "c_call":
            fn = self.cv2_funcs.get(id(arg))
            if fn is not None:
                self.cv2[fn] += 1
                return
            name = getattr(arg, "__name__", "")
            owner = getattr(arg, "__self__", None)
            # A method of an OpenCV object (CLAHE.apply, ...).
            if owner is not None and type(owner).__module__.startswith("cv2"):
                self.cv2[
                    f"{type(owner).__module__}.{getattr(arg, '__qualname__', name)}"
                ] += 1
            elif name in SPAWN_NAMES:
                mod = getattr(arg, "__module__", None) or getattr(
                    owner, "__name__", "?"
                )
                self.spawns[f"{mod}.{name}"] += 1

    def __enter__(self) -> Self:
        threading.setprofile(self)
        sys.setprofile(self)
        return self

    def __exit__(self, *exc: object) -> None:
        sys.setprofile(None)
        threading.setprofile(None)  # type: ignore[arg-type]

    def tables(self) -> dict[str, dict[str, int]]:
        return {c: dict(sorted(getattr(self, c).items())) for c in COUNTERS}


# --- workloads --------------------------------------------------------------
# Each takes (srv-call function, scratch dir), does its setup unprofiled, runs
# ONE tool call inside `counted(...)`, and returns a few numbers that only that
# input produces (the harness assertion: a workload that silently ran another
# input, or none, cannot reproduce them).


def _png(path: Path, rgb) -> str:
    import cv2

    cv2.imwrite(str(path), rgb[..., ::-1].copy())
    return str(path)


def _soil(h: int, w: int):
    import numpy as np

    img = np.empty((h, w, 3), np.uint8)
    img[:] = (96, 70, 52)
    return img


def _ellipse(shape, cx, cy, a, b, theta):
    import numpy as np

    yy, xx = np.mgrid[: shape[0], : shape[1]]
    c, s = np.cos(theta), np.sin(theta)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def _rosette(shape, n: int, cx=None, cy=None, a=55.0, b=16.0):
    import numpy as np

    h, w = shape
    cx = w / 2 if cx is None else cx
    cy = h / 2 if cy is None else cy
    yy, xx = np.mgrid[:h, :w]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= 10**2
    for k in range(n):
        t = 2 * np.pi * k / n + 0.3
        m |= _ellipse(
            shape, cx + np.cos(t) * (a + 6), cy + np.sin(t) * (a + 6), a, b, t
        )
    return m


def _plant_png(d: Path, name: str = "plant.png", n: int = 5) -> str:
    img = _soil(240, 260)
    img[_rosette((240, 260), n, a=45, b=15)] = (52, 148, 62)
    return _png(d / name, img)


def _segment(call, path: str) -> dict:
    return call("segment", {"image_path": path, "channel": "a", "method": "otsu"})


def w_list_methods(call, counted, d):
    with counted():
        r = call("list_methods", {})
    return {"n_channels": len(r["channels"]), "n_refine_ops": len(r["refine_ops"])}


def w_suggest_segmentation(call, counted, d):
    p = _plant_png(d)
    with counted():
        r = call("suggest_segmentation", {"image_path": p, "channel": "a"})
    return {"polarity_keys": sorted(r["polarity"])[:4]}


def w_segment(call, counted, d):
    p = _plant_png(d)
    with counted():
        r = _segment(call, p)
    return {"largest_area": r["largest_area"]}


def w_refine(call, counted, d):
    s = _segment(call, _plant_png(d))
    ops = [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}]
    with counted():
        r = call("refine", {"session_id": s["session_id"], "ops": ops})
    return {"after_largest_area": r["after"]["largest_area"]}


def w_measure_size_color(call, counted, d):
    s = _segment(call, _plant_png(d))
    with counted():
        r = call(
            "measure", {"session_id": s["session_id"], "analyses": ["size", "color"]}
        )
    t = r["traits"]
    return {
        "area": t["area"]["value"],
        "hue_mean": round(t["hue_circular_mean"]["value"], 3),
    }


def w_measure_regions(call, counted, d):
    import numpy as np

    img = _soil(240, 360)
    m = np.zeros((240, 360), bool)
    for r_ in range(2):
        for c_ in range(3):
            if (r_, c_) == (1, 2):
                continue  # one empty pot
            m |= _rosette((240, 360), 4, cx=60 + 120 * c_, cy=60 + 120 * r_, a=24, b=8)
    img[m] = (52, 148, 62)
    s = _segment(call, _png(d / "tray.png", img))
    with counted():
        r = call(
            "measure_regions", {"session_id": s["session_id"], "nrows": 2, "ncols": 3}
        )
    return {
        "measured": sum(bool(x.get("measured")) for x in r["regions"]),
        "areas": sorted(
            x["traits"]["area"]["value"] for x in r["regions"] if x.get("measured")
        ),
    }


def w_measure_regions_fragmented(call, counted, d):
    """A tray whose mask has many contours (holes), as a field tray's does.

    PlantCV's create_labels did a full-frame pass per ROI per contour, so its
    cv2 count grew as ROIs x contours (here 6 x 133 fillPoly calls). With
    `w_measure_regions`' 6 contours that product was invisible.
    """
    import numpy as np

    img = _soil(240, 360)
    yy, xx = np.mgrid[:240, :360]
    for r_ in range(2):
        for c_ in range(3):
            cx, cy = 60 + 120 * c_, 60 + 120 * r_
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 42**2] = (52, 148, 62)
            for hx in range(cx - 30, cx + 31, 12):
                for hy in range(cy - 30, cy + 31, 12):
                    if (hx - cx) ** 2 + (hy - cy) ** 2 <= 34**2:
                        img[(xx - hx) ** 2 + (yy - hy) ** 2 <= 9] = (96, 70, 52)
    s = _segment(call, _png(d / "tray.png", img))
    grid = {"mode": "rect_grid", "nrows": 2, "ncols": 3, "coord": [0, 0]}
    grid |= {"height": 120, "width": 120, "spacing": [120, 120]}
    with counted():
        r = call("measure_regions", {"session_id": s["session_id"], **grid})
    return {
        "measured": sum(bool(x.get("measured")) for x in r["regions"]),
        "areas": sorted(
            x["traits"]["area"]["value"] for x in r["regions"] if x.get("measured")
        ),
    }


def w_measure_morphology(call, counted, d):
    import numpy as np

    img = _soil(300, 240)
    m = np.zeros((300, 240), bool)
    m[60:270, 117:123] = True  # stem
    m |= _ellipse((300, 240), 150, 140, 34, 7, -0.5)
    m |= _ellipse((300, 240), 90, 190, 34, 7, 0.5 + np.pi)
    img[m] = (52, 148, 62)
    s = _segment(call, _png(d / "side.png", img))
    with counted():
        r = call("measure_morphology", {"session_id": s["session_id"]})
    return {"n_segments": len(r["segments"]), "tips": r["plant"]["tip_count"]}


def w_count_leaves(call, counted, d):
    s = _segment(call, _plant_png(d, n=6))
    with counted():
        r = call("count_leaves", {"session_id": s["session_id"]})
    return {
        "count": r["leaf_count"],
        "areas": sorted(x["area_px"] for x in r["instances"]),
    }


def w_calibrate_scale(call, counted, d):
    # The committed fixture: a 100 px disc as a 20 mm marker, exactly 5 px/mm.
    p = str(FIXTURES / "plant_with_marker.png")
    args = {
        "image_path": p,
        "x": 430,
        "y": 310,
        "w": 140,
        "h": 140,
        "marker_length_mm": 20.0,
    }
    with counted():
        r = call("calibrate_scale_from_marker", args)
    return {
        "px_per_mm": round(r["px_per_mm"], 6),
        "marker_length_px": r["marker_length_px"],
    }


def w_measure_images(call, counted, d):
    paths = [_plant_png(d, f"p{i}.png", n=n) for i, n in enumerate((4, 5, 6))]
    with counted():
        r = call(
            "measure_images", {"image_paths": paths, "channel": "a", "method": "otsu"}
        )
    return {"areas": [row["traits"]["area"]["value"] for row in r["results"]]}


def w_thermal(call, counted, d):
    import numpy as np
    from test_thermal import _scene  # the suite's own warm-disc scene

    frame, _ = _scene()
    p = d / "frame.npz"
    np.savez(p, frame)
    s = call("segment_thermal", {"path": str(p), "min_c": 25.0})
    with counted():
        r = call("measure_thermal", {"session_id": s["session_id"]})
    return {"mean_c": r["temperature"]["mean"], "pixels": r["pixel_count"]}


def w_hyperspectral(call, counted, d):
    from test_hyperspectral import _known_cube, _write_cube  # NDVI 0.6 disc

    raw = _write_cube(d, "cube", _known_cube())
    with counted():
        r = call(
            "segment_hyperspectral",
            {"envi_path": raw, "index": "ndvi", "threshold": 0.2},
        )
    return {"largest_area": r["largest_area"]}


def w_measure_spectral(call, counted, d):
    from test_hyperspectral import _known_cube, _write_cube

    raw = _write_cube(d, "cube", _known_cube())
    s = call(
        "segment_hyperspectral", {"envi_path": raw, "index": "ndvi", "threshold": 0.2}
    )
    with counted():
        r = call("measure_spectral", {"session_id": s["session_id"]})
    return {"ndvi_mean": round(r["indices"]["ndvi"]["mean"], 6)}


def w_correct_lens(call, counted, d):
    import cv2
    from test_lens import POSES, _distort, _view, _write_frames

    frames = d / "frames"
    _write_frames(frames, n=8)
    img = d / "scene.png"
    gray = _distort(_view(*POSES[0]))
    cv2.imwrite(str(img), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    args = {"image_path": str(img), "checkerboard_dir": str(frames), "row_corners": 6,
            "col_corners": 9}  # fmt: skip
    with counted():
        r = call("correct_lens_distortion", args)
    return {"corrected_size": r["corrected_size"], "frames_used": r["frames_used"]}


WORKLOADS: dict[str, Callable] = {
    "list_methods": w_list_methods,
    "suggest_segmentation": w_suggest_segmentation,
    "segment": w_segment,
    "refine": w_refine,
    "measure": w_measure_size_color,
    "measure_regions": w_measure_regions,
    "measure_regions_fragmented": w_measure_regions_fragmented,
    "measure_morphology": w_measure_morphology,
    "count_leaves": w_count_leaves,
    "calibrate_scale_from_marker": w_calibrate_scale,
    "measure_images": w_measure_images,
    "measure_thermal": w_thermal,
    "segment_hyperspectral": w_hyperspectral,
    "measure_spectral": w_measure_spectral,
    "correct_lens_distortion": w_correct_lens,
}


# --- one workload, in this process ------------------------------------------


def _fake_clock() -> types.SimpleNamespace:
    t = [0.0]

    def perf_counter() -> float:
        t[0] += 1.0
        return t[0]

    return types.SimpleNamespace(perf_counter=perf_counter)


def run_child(name: str) -> dict[str, Any]:
    sys.path.insert(0, str(TESTS))
    from plantcv_mcp import batch
    from plantcv_mcp.server import build_server
    from plantcv_mcp.workers import set_isolation

    set_isolation(False)
    batch.time = _fake_clock()  # every other `time` user is untouched
    srv = build_server()

    def call(tool: str, args: dict) -> Any:
        res = asyncio.run(srv.call_tool(tool, args))
        if getattr(res, "is_error", False) or getattr(res, "isError", False):
            raise RuntimeError(f"{tool} failed: {res.content}")
        structured = getattr(res, "structured_content", None) or getattr(
            res, "structuredContent", None
        )
        if structured is not None:
            return structured.get("result", structured)
        text = next(c.text for c in res.content if getattr(c, "text", None))
        return json.loads(text)

    counter = Count()
    with tempfile.TemporaryDirectory(prefix="workcount-") as d:
        output = WORKLOADS[name](call, lambda: counter, Path(d))
    return {"output": output, **counter.tables()}


# --- the gate ---------------------------------------------------------------


def measure_all(names: list[str]) -> dict[str, Any]:
    out = {}
    for name in names:
        proc = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child", name],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        if proc.returncode != 0:
            raise SystemExit(f"workload {name} crashed:\n{proc.stderr[-4000:]}")
        out[name] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def totals(row: dict[str, Any]) -> dict[str, int]:
    return {c: sum(row[c].values()) for c in COUNTERS}


def diff(
    name: str, want: dict[str, Any], got: dict[str, Any], top: int = 12
) -> list[str]:
    """Readable lines for every way `got` differs from `want`; [] if equal."""
    lines = []
    if want["output"] != got["output"]:
        lines.append(f"{name}: OUTPUT changed {want['output']} -> {got['output']}")
    for c in COUNTERS:
        if want[c] == got[c]:
            continue
        tw, tg = sum(want[c].values()), sum(got[c].values())
        lines.append(f"{name}: {c} total {tw} -> {tg} ({tg - tw:+d})")
        keys = set(want[c]) | set(got[c])
        deltas = sorted(
            ((got[c].get(k, 0) - want[c].get(k, 0), k) for k in keys),
            key=lambda x: (-abs(x[0]), x[1]),
        )
        for dlt, k in [x for x in deltas if x[0]][:top]:
            lines.append(
                f"    {dlt:+6d}  {k}  ({want[c].get(k, 0)} -> {got[c].get(k, 0)})"
            )
    return lines


def check(baseline: dict[str, Any], got: dict[str, Any]) -> list[str]:
    lines = []
    for name in sorted(set(baseline) | set(got)):
        if name not in got:
            lines.append(f"{name}: in the baseline but not run")
        elif name not in baseline:
            lines.append(f"{name}: not in the baseline (run --update)")
        else:
            lines += diff(name, baseline[name], got[name])
    return lines


def refusals(
    baseline: dict[str, Any],
    got: dict[str, Any],
    allow_increase: bool = False,
    allow_output_change: bool = False,
) -> list[str]:
    """Why ``--update`` must not write `got`: a rise in any total without
    --allow-increase, or a workload whose output changed (it ran a different
    input, so its counts are not comparable) without --allow-output-change.
    A drop or a reshuffle at the same total is recorded."""
    out = []
    for n in sorted(set(baseline) & set(got)):
        if not allow_increase:
            want, have = totals(baseline[n]), totals(got[n])
            out += [
                f"{n}: {c} {want[c]} -> {have[c]}"
                for c in COUNTERS
                if have[c] > want[c]
            ]
        if not allow_output_change and baseline[n]["output"] != got[n]["output"]:
            out.append(f"{n}: output {baseline[n]['output']} -> {got[n]['output']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--update", action="store_true", help="write the measured counts")
    ap.add_argument(
        "--allow-increase",
        action="store_true",
        help="with --update: accept a higher total",
    )
    ap.add_argument(
        "--allow-output-change",
        action="store_true",
        help="with --update: accept a workload whose output changed",
    )
    ap.add_argument("--only", default="", help="comma-separated workloads")
    a = ap.parse_args()
    if a.child:
        print(json.dumps(run_child(a.child), sort_keys=True))
        return
    names = a.only.split(",") if a.only else list(WORKLOADS)
    got = measure_all(names)
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    for name in names:
        print(
            f"{name:30s} " + "  ".join(f"{c}={v}" for c, v in totals(got[name]).items())
        )
    problems = check({k: v for k, v in baseline.items() if k in names}, got)
    if not a.update:
        if problems:
            print(
                "\nWORK COUNTS CHANGED (exact baseline, zero headroom):",
                *problems,
                sep="\n",
            )
            print("\nA drop: --update records it. A rise: --update --allow-increase.")
            sys.exit(1)
        print("\nwork counts equal the baseline")
        return
    refused = refusals(
        {k: v for k, v in baseline.items() if k in names},
        got,
        a.allow_increase,
        a.allow_output_change,
    )
    if refused:
        print(
            "\nREFUSED (a rise needs --allow-increase, an output change"
            " --allow-output-change):",
            *refused,
            sep="\n",
        )
        sys.exit(1)
    baseline.update(got)
    BASELINE.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {BASELINE.relative_to(REPO)}", *problems, sep="\n")


if __name__ == "__main__":
    main()
