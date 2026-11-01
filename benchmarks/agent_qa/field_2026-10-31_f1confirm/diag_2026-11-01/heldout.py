"""Evaluate threshold_inside_mode on held-out pot cells; see PREREG_threshold_inside_mode.md.

Usage, from the repo root (openpyxl reads the harvest sheet, as build.py does):

    uv run --with openpyxl==3.1.5 python \
      benchmarks/agent_qa/field_2026-10-31_f1confirm/diag_2026-11-01/heldout.py OUTDIR [--limit N]

Enumerates every eligible plant with build.py's own rules (field_lib.plants),
crops its pot cell as sweep.py does and runs it through the real tools. The 40
calibration cells run first as the positive control: their cell bounds and px
windows must equal the questions', their otsu areas must equal sweep.json, and
the warning must fire exactly where the calibration density is at least 0.4.
Any mismatch stops the run before a held-out cell is scored.
"""

import argparse
import asyncio
import datetime as dt
import glob
import json
import math
import signal
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmarks/agent_qa/field_2026-10-24"))
import field_lib as F

from plantcv_mcp.diagnostics import INSIDE_MODE, histogram_density_at
from plantcv_mcp.segmentation import to_gray
from plantcv_mcp.server import build_server

ARTS = Path.home() / "plantcv-mcp-artifacts"
TRAYS = {
    "031": ARTS / "field_2026-10-31_f1confirm/aberystwyth",
    "032": ARTS / "field_2026-10-24/aberystwyth",
}
QUESTIONS = [
    REPO / "benchmarks/agent_qa/field_2026-10-31_f1confirm/questions.json",
    REPO / "benchmarks/agent_qa/field_2026-10-24/questions.json",
]
CODE = "threshold_inside_mode"


def _walltime(*_: object) -> None:
    sys.stderr.write("walltime\n")
    sys.exit(2)


def raw_name(gt: Path) -> str:  # as build.py
    return gt.name.replace("_top_gt.png", "_top.png").replace("_top-gt.png", "_top.png")


def eligible() -> list[dict]:
    """Every eligible plant on both trays, with build.py's cell and px window."""
    out = []
    for tray, data in TRAYS.items():
        tv = f"PSI_Tray{tray}/tv"
        harvests = F.read_harvests(data / "AT20_graphed_GT.xlsx", int(tray))
        for gtp in sorted(Path(p) for p in glob.glob(str(data / tv / "gt" / "*.png"))):
            name = raw_name(gtp)
            img = cv2.imread(str(data / tv / name))
            g = F.grid(img)
            when = dt.date.fromisoformat(name.split("_")[2][:10])
            ps = F.plants(F.read_gt(gtp), name, g, F.occupied(harvests, when))
            xb, yb = g.bounds()
            for p in ps:
                if p.excluded:
                    continue
                er, di = F.mask_band(p.mask)
                out.append({
                    "tray": tray, "image": f"{tv}/{name}", "row": p.row + 1,
                    "col": p.col + 1, "date": str(when),
                    "cell_x": [round(xb[p.col], 1), round(xb[p.col + 1], 1)],
                    "cell_y": [round(yb[p.row], 1), round(yb[p.row + 1], 1)],
                    "area_px": int(p.mask.sum()), "px_lo": int(er.sum()),
                    "px_hi": int(di.sum()),
                })  # fmt: skip
    return out


def calibration() -> dict[tuple[str, int, int], dict]:
    cal = {}
    for qp in QUESTIONS:
        for q in json.loads(qp.read_text())["questions"]:
            if q["id"].startswith("f1"):
                w = q["where"]
                cal[(q["image"], w["row"], w["col"])] = q
    return cal


def main() -> None:
    signal.signal(signal.SIGALRM, _walltime)
    signal.alarm(3 * 3600)
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="held-out cells (pilot)")
    ap.add_argument(
        "--shard", type=int, default=0, help="this process's share, 0-based"
    )
    ap.add_argument("--of", type=int, default=1, help="number of shards")
    ap.add_argument("--score", action="store_true", help="score the shards in OUT")
    o = ap.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    if o.score:
        score(o.out, o.of)
        return
    server = build_server()

    def call(name: str, args: dict) -> dict:
        return json.loads(asyncio.run(server.call_tool(name, args)).content[0].text)

    def run(e: dict) -> dict:
        data = TRAYS[e["tray"]]
        img = cv2.imread(str(data / e["image"]))
        x0 = max(0, math.floor(e["cell_x"][0]))
        x1 = min(img.shape[1], math.ceil(e["cell_x"][1]))
        y0 = max(0, math.floor(e["cell_y"][0]))
        y1 = min(img.shape[0], math.ceil(e["cell_y"][1]))
        crop = img[y0:y1, x0:x1]
        path = o.out / f"cell_{o.shard}.png"
        cv2.imwrite(str(path), crop)
        res = {}
        for method in ("otsu", "triangle"):
            seg = call("segment", {"image_path": str(path), "channel": "a",
                                   "method": method, "object_type": "dark",
                                   "fill_size": 50})  # fmt: skip
            r = call("refine", {"session_id": seg["session_id"],
                                "ops": [{"op": "keep_largest", "n": 1}]})  # fmt: skip
            m = call("measure", {"session_id": r["session_id"], "traits": ["area"]})
            res[method] = {
                "area": m["traits"]["area"]["value"],
                "fires": CODE in [w["code"] for w in seg["warnings"]],
            }
        gray = to_gray(crop, "a")
        t = int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

        def label(a: float) -> str:
            return "OVER" if a > e["px_hi"] else "UNDER" if a < e["px_lo"] else "RIGHT"

        return {
            **e,
            "plant_frac": e["area_px"] / ((x1 - x0) * (y1 - y0)),
            "density": histogram_density_at(gray, t),
            "otsu": res["otsu"]["area"],
            "label": label(res["otsu"]["area"]),
            "fires": res["otsu"]["fires"],
            "triangle": res["triangle"]["area"],
            "triangle_label": label(res["triangle"]["area"]),
        }

    plants = eligible()
    cal = calibration()
    sweep = {r["id"]: r for r in json.loads((HERE / "sweep.json").read_text())}
    key = {(e["image"], e["row"], e["col"]): e for e in plants}

    # Positive control: the 40 calibration cells, before any held-out cell.
    bad = []
    for k, q in sorted(cal.items(), key=lambda kv: kv[1]["id"]):
        e = key.get(k)
        if e is None:
            bad.append((q["id"], "not enumerated"))
            continue
        w, t = q["where"], q["truth"]
        same = (
            e["cell_x"] == w["cell_x"] and e["cell_y"] == w["cell_y"]
            and (e["px_lo"], e["px_hi"]) == (t["px_lo"], t["px_hi"])
        )  # fmt: skip
        if not same:
            bad.append((q["id"], "cell or window differs"))
            continue
        r = run(e)
        if r["otsu"] != sweep[q["id"]]["otsu50"]:
            bad.append(
                (q["id"], f"otsu {r['otsu']} != sweep {sweep[q['id']]['otsu50']}")
            )
        if r["fires"] != (r["density"] >= INSIDE_MODE):
            bad.append((q["id"], f"fires {r['fires']} at density {r['density']:.2f}"))
        print(json.dumps({"control": q["id"], "density": round(r["density"], 3),
                          "fires": r["fires"], "label": r["label"]}), flush=True)  # fmt: skip
    if bad or len(cal) != 40:
        raise SystemExit(f"POSITIVE CONTROL FAILED ({len(cal)} cells): {bad}")
    print("positive control: 40/40", flush=True)

    held = [e for k, e in key.items() if k not in cal]
    print(f"held-out cells: {len(held)} (of {len(plants)} eligible)", flush=True)
    held = held[o.shard :: o.of]
    if o.limit:
        held = held[: o.limit]
    rows = []
    for e in held:
        rows.append(run(e))
        print(json.dumps(rows[-1]), flush=True)
    (o.out / f"heldout_{o.shard}of{o.of}.json").write_text(json.dumps(rows) + "\n")
    print(f"shard {o.shard}/{o.of}: {len(rows)} cells written", flush=True)


def score(out: Path, of: int) -> None:
    """Merge the shards and apply the pre-registered criteria. Every held-out
    cell must be present exactly once, or nothing is scored."""
    files = sorted(out.glob(f"heldout_*of{of}.json"))
    rows = [r for f in files for r in json.loads(f.read_text())]
    cal = calibration()
    want = {(e["image"], e["row"], e["col"]) for e in eligible()} - set(cal)
    got = [(r["image"], r["row"], r["col"]) for r in rows]
    if len(files) != of or len(got) != len(set(got)) or set(got) != want:
        raise SystemExit(
            f"INCOMPLETE: {len(files)}/{of} shards, {len(got)} rows "
            f"({len(set(got))} distinct) for {len(want)} held-out cells"
        )
    (out / "heldout.json").write_text(json.dumps(rows) + "\n")
    print(f"held-out cells scored: {len(rows)}")

    def n(pred) -> int:
        return sum(1 for r in rows if pred(r))

    over = n(lambda r: r["label"] == "OVER")
    right = n(lambda r: r["label"] == "RIGHT")
    over_f = n(lambda r: r["label"] == "OVER" and r["fires"])
    right_f = n(lambda r: r["label"] == "RIGHT" and r["fires"])
    p1 = over >= 10 and over_f >= 0.9 * over
    p2 = right > 0 and right_f <= 0.1 * right
    verdict = (
        "OPEN (fewer than 10 OVER)" if over < 10 else ("PASS" if p1 and p2 else "FAIL")
    )
    print(f"P1 fires on OVER: {over_f}/{over}  {'pass' if p1 else 'FAIL'}")
    print(f"P2 fires on RIGHT: {right_f}/{right}  {'pass' if p2 else 'FAIL'}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
