"""Segment each F1 plant's pot cell through the real tools; see README.md.

Usage, from the repo root: python sweep.py OUTDIR  (writes OUTDIR/sweep.json)
"""

import asyncio
import json
import math
import signal
import sys
from pathlib import Path

import cv2

from plantcv_mcp.server import build_server

SETS = [
    (
        "benchmarks/agent_qa/field_2026-10-31_f1confirm/questions.json",
        "field_2026-10-31_f1confirm",
    ),
    ("benchmarks/agent_qa/field_2026-10-24/questions.json", "field_2026-10-24"),
]


def _walltime(*_: object) -> None:
    sys.stderr.write("walltime\n")
    sys.exit(2)


def main() -> None:
    signal.signal(signal.SIGALRM, _walltime)
    signal.alarm(1500)
    out = Path(sys.argv[1])
    server = build_server()
    rows = []

    def call(name: str, args: dict) -> dict:
        return json.loads(asyncio.run(server.call_tool(name, args)).content[0].text)

    def segment(path: Path, method: str, fill: int) -> dict:
        args = {"image_path": str(path), "channel": "a", "method": method}
        return call("segment", {**args, "object_type": "dark", "fill_size": fill})

    def area(sid: str) -> float:
        r = call("refine", {"session_id": sid, "ops": [{"op": "keep_largest", "n": 1}]})
        m = call("measure", {"session_id": r["session_id"], "traits": ["area"]})
        return m["traits"]["area"]["value"]

    for qpath, art in SETS:
        with open(qpath) as f:
            questions = [
                q for q in json.load(f)["questions"] if q["id"].startswith("f1")
            ]
        base = Path.home() / "plantcv-mcp-artifacts" / art / "aberystwyth"
        for q in questions:
            w, t = q["where"], q["truth"]
            img = cv2.imread(str(base / q["image"]))
            x0 = max(0, math.floor(w["cell_x"][0]))
            x1 = min(img.shape[1], math.ceil(w["cell_x"][1]))
            y0 = max(0, math.floor(w["cell_y"][0]))
            y1 = min(img.shape[0], math.ceil(w["cell_y"][1]))
            path = out / f"cell_{q['id']}.png"
            cv2.imwrite(str(path), img[y0:y1, x0:x1])
            s0 = segment(path, "otsu", 0)
            s50 = segment(path, "otsu", 50)
            s300 = segment(path, "otsu", 300)
            tri = segment(path, "triangle", 50)
            r = {
                "id": q["id"],
                "das": w["das"],
                "edge": w["edge"],
                "truth": t["area_px"],
                "lo": t["px_lo"],
                "hi": t["px_hi"],
                "plant_frac": t["area_px"] / ((x1 - x0) * (y1 - y0)),
                "pre_frac": s0["mask_fraction"],
                "pre_comp": s0.get("component_count"),
                "post50_frac": s50["mask_fraction"],
                "post50_comp": s50.get("component_count"),
                "otsu50": area(s50["session_id"]),
                "otsu300": area(s300["session_id"]),
                "tri50": area(tri["session_id"]),
                "warn50": [x["code"] for x in s50.get("warnings", [])],
            }
            r["removed_share"] = (
                1 - r["post50_frac"] / r["pre_frac"] if r["pre_frac"] else 0
            )
            rows.append(r)
            print(json.dumps(r), flush=True)
    (out / "sweep.json").write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
