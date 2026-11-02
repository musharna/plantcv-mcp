"""Evaluate threshold_inside_mode for segment with rules; see
PREREG_rules_inside_mode.md.

Usage, from the repo root (openpyxl reads the harvest sheet, as build.py does):

    uv run --with openpyxl==3.1.5 python \
      benchmarks/agent_qa/field_2026-10-31_f1confirm/diag_2026-11-01/rules_check.py \
      OUTDIR [--shard I --of N | --of N --score]

The 40 calibration cells run first as the positive control: their labels must
equal rules_pilot.json under both combines, and the warning must fire exactly
where the pre-registered rule, applied to the pilot's densities, says.
"""

import argparse
import asyncio
import collections
import json
import math
import signal
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import heldout as H

from plantcv_mcp.diagnostics import INSIDE_MODE
from plantcv_mcp.server import build_server

RULES = [
    {"channel": "a", "method": "otsu", "object_type": "dark"},
    {"channel": "b", "method": "otsu", "object_type": "light"},
]
COMBINES = ("or", "and")
CODE = "threshold_inside_mode"


def _walltime(*_: object) -> None:
    sys.stderr.write("walltime\n")
    sys.exit(2)


def crop(e: dict):  # as heldout.py
    img = cv2.imread(str(H.TRAYS[e["tray"]] / e["image"]))
    x0 = max(0, math.floor(e["cell_x"][0]))
    x1 = min(img.shape[1], math.ceil(e["cell_x"][1]))
    y0 = max(0, math.floor(e["cell_y"][0]))
    y1 = min(img.shape[0], math.ceil(e["cell_y"][1]))
    return img[y0:y1, x0:x1]


def key(e: dict) -> str:
    return f"{e['image']}|{e['row']}|{e['col']}"


def expected_fire(dens_a: float, dens_b: float, combine: str) -> bool:
    fires = [dens_a >= INSIDE_MODE, dens_b >= INSIDE_MODE]
    return all(fires) if combine == "and" else any(fires)


def score(out: Path, of: int) -> None:
    held = {key(e): e for e in json.loads((HERE / "heldout.json").read_text())}
    rows = []
    for k in range(of):
        rows += json.loads((out / f"rules_{k}of{of}.json").read_text())
    got = collections.Counter(r["key"] for r in rows)
    if len(held) != 739 or set(got) != set(held) or max(got.values()) != 1:
        raise SystemExit(f"FAIL: cells {len(got)} of {len(held)}")
    res = {"cells": len(rows)}
    for comb in COMBINES:
        n = collections.Counter(r[comb]["label"] for r in rows)
        f = collections.Counter(r[comb]["label"] for r in rows if r[comb]["fires"])
        res[comb] = {lab: f"{f[lab]}/{n[lab]}" for lab in ("OVER", "RIGHT", "UNDER")}
        over, right = n["OVER"], n["RIGHT"]
        sens = f["OVER"] / over if over else None
        fa = f["RIGHT"] / right if right else None
        if comb == "or":
            res["R1"] = "PASS" if sens is not None and sens >= 0.9 else "FAIL"
            res["R2"] = "PASS" if fa is not None and fa <= 0.1 else "FAIL"
        else:
            res["R3"] = "OPEN" if over < 10 else "PASS" if sens >= 0.9 else "FAIL"
            res["R4"] = "PASS" if fa is not None and fa <= 0.1 else "FAIL"
    print(json.dumps(res, indent=1))


def main() -> None:
    signal.signal(signal.SIGALRM, _walltime)
    signal.alarm(3 * 3600)
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="held-out cells (pilot)")
    ap.add_argument("--score", action="store_true")
    o = ap.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    if o.score:
        score(o.out, o.of)
        return
    server = build_server()

    def call(name: str, args: dict) -> dict:
        return json.loads(asyncio.run(server.call_tool(name, args)).content[0].text)

    path = o.out / f"cell_{o.shard}.png"

    def run(e: dict) -> dict:
        cv2.imwrite(str(path), crop(e))
        out = {"key": key(e)}
        for comb in COMBINES:
            seg = call("segment", {"image_path": str(path), "rules": RULES,
                                   "combine": comb, "fill_size": 50})  # fmt: skip
            r = call("refine", {"session_id": seg["session_id"],
                                "ops": [{"op": "keep_largest", "n": 1}]})  # fmt: skip
            a = call("measure", {"session_id": r["session_id"], "traits": ["area"]})
            area = a["traits"]["area"]["value"]
            label = (
                "OVER" if area > e["px_hi"]
                else "UNDER" if area < e["px_lo"] else "RIGHT"
            )  # fmt: skip
            out[comb] = {
                "area": area,
                "label": label,
                "fires": CODE in [w["code"] for w in seg["warnings"]],
            }
        return out

    plants = H.eligible()
    cal = H.calibration()
    by_key = {(e["image"], e["row"], e["col"]): e for e in plants}
    pilot = {r["id"]: r for r in json.loads((HERE / "rules_pilot.json").read_text())}
    bad = []
    for k, q in sorted(cal.items(), key=lambda kv: kv[1]["id"]):
        p = pilot[q["id"]]
        r = run(by_key[k])
        for comb in COMBINES:
            if r[comb]["label"] != p[comb]:
                bad.append((q["id"], comb, "label", r[comb]["label"], p[comb]))
            want = expected_fire(p["dens_a"], p["dens_b"], comb)
            if r[comb]["fires"] != want:
                bad.append((q["id"], comb, "fires", r[comb]["fires"], want))
    if bad or len(cal) != 40:
        raise SystemExit(f"POSITIVE CONTROL FAILED ({len(cal)} cells): {bad}")
    print("positive control: 40/40", flush=True)

    held = json.loads((HERE / "heldout.json").read_text())[o.shard :: o.of]
    if o.limit:
        held = held[: o.limit]
    rows = [run(e) for e in held]
    (o.out / f"rules_{o.shard}of{o.of}.json").write_text(json.dumps(rows) + "\n")
    print(f"shard {o.shard}/{o.of}: {len(rows)} cells written", flush=True)


if __name__ == "__main__":
    main()
