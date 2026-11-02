"""Pilot for the rules form of threshold_inside_mode, on the 40 calibration cells.

Wrote rules_pilot.json (run before the warning existed for rules, so it records
each channel's density and the labels, not warnings). Usage, from the repo
root: uv run --with openpyxl==3.1.5 python <this file> OUTDIR
"""

import asyncio
import json
import math
import signal
import sys
from pathlib import Path

signal.alarm(3600)
REPO = Path.cwd()
D = REPO / "benchmarks/agent_qa/field_2026-10-31_f1confirm/diag_2026-11-01"
sys.path.insert(0, str(D))
import cv2
import heldout as H

from plantcv_mcp.diagnostics import histogram_density_at
from plantcv_mcp.segmentation import to_gray
from plantcv_mcp.server import build_server


def main():
    global s
    s = build_server()

    def call(n, a):
        try:
            return json.loads(asyncio.run(s.call_tool(n, a)).content[0].text)
        except Exception as ex:  # noqa: BLE001 - a crash is recorded per cell, not fatal
            return {"ERR": f"{n}: {str(ex)[:200]}"}

    plants = H.eligible()
    cal = H.calibration()
    key = {(e["image"], e["row"], e["col"]): e for e in plants}
    RULES = [
        {"channel": "a", "method": "otsu", "object_type": "dark"},
        {"channel": "b", "method": "otsu", "object_type": "light"},
    ]
    out = Path(sys.argv[1])
    rows = []
    for k in sorted(cal, key=lambda k: cal[k]["id"]):
        e = key[k]
        img = cv2.imread(str(H.TRAYS[e["tray"]] / e["image"]))
        x0, x1 = (
            max(0, math.floor(e["cell_x"][0])),
            min(img.shape[1], math.ceil(e["cell_x"][1])),
        )
        y0, y1 = (
            max(0, math.floor(e["cell_y"][0])),
            min(img.shape[0], math.ceil(e["cell_y"][1])),
        )
        c = img[y0:y1, x0:x1]
        p = out / "cell.png"
        cv2.imwrite(str(p), c)
        r = {"id": cal[k]["id"], "frac": e["area_px"] / c.shape[0] / c.shape[1]}
        for ch in "ab":
            g = to_gray(c, ch)
            t = int(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
            r[f"dens_{ch}"] = round(histogram_density_at(g, t), 3)
        for comb in ("or", "and"):
            seg = call(
                "segment",
                {
                    "image_path": str(p),
                    "rules": RULES,
                    "combine": comb,
                    "fill_size": 50,
                },
            )
            if "session_id" not in seg:
                r[comb] = "ERR " + str(seg)[:80]
                continue
            ref = call(
                "refine",
                {
                    "session_id": seg["session_id"],
                    "ops": [{"op": "keep_largest", "n": 1}],
                },
            )
            if "ERR" in ref:
                r[comb] = ref["ERR"]
                print("CRASH", r["id"], comb, ref["ERR"][:120], flush=True)
                continue
            a = call("measure", {"session_id": ref["session_id"], "traits": ["area"]})[
                "traits"
            ]["area"]["value"]
            r[comb] = (
                "OVER" if a > e["px_hi"] else "UNDER" if a < e["px_lo"] else "RIGHT"
            )
            r[comb + "_x"] = round(a / e["area_px"], 2)
        rows.append(r)
        print(json.dumps(r), flush=True)
    (out / "pilot.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
