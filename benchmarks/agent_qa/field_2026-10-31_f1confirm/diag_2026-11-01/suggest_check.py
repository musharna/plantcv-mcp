"""Check that suggest_segmentation reports threshold_inside_mode where segment
did; see PREREG_suggest_inside_mode.md.

Usage, from the repo root:

    uv run python benchmarks/agent_qa/field_2026-10-31_f1confirm/diag_2026-11-01/suggest_check.py OUTDIR --shard K --of N
    uv run python .../suggest_check.py OUTDIR --of N --score
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

from plantcv_mcp.diagnostics import histogram_density_at
from plantcv_mcp.segmentation import to_gray
from plantcv_mcp.server import build_server

HERE = Path(__file__).resolve().parent
ARTS = Path.home() / "plantcv-mcp-artifacts"
TRAYS = {
    "031": ARTS / "field_2026-10-31_f1confirm/aberystwyth",
    "032": ARTS / "field_2026-10-24/aberystwyth",
}
CODE = "threshold_inside_mode"


def _walltime(*_: object) -> None:
    sys.stderr.write("walltime\n")
    sys.exit(2)


def crop(e: dict):  # as heldout.py
    img = cv2.imread(str(TRAYS[e["tray"]] / e["image"]))
    x0 = max(0, math.floor(e["cell_x"][0]))
    x1 = min(img.shape[1], math.ceil(e["cell_x"][1]))
    y0 = max(0, math.floor(e["cell_y"][0]))
    y1 = min(img.shape[0], math.ceil(e["cell_y"][1]))
    return img[y0:y1, x0:x1]


def key(e: dict) -> str:
    return f"{e['image']}|{e['row']}|{e['col']}"


def score(out: Path, of: int) -> None:
    held = {key(e): e for e in json.loads((HERE / "heldout.json").read_text())}
    rows = []
    for k in range(of):
        rows += json.loads((out / f"suggest_{k}of{of}.json").read_text())
    got = collections.Counter(r["key"] for r in rows)
    if len(held) != 739 or set(got) != set(held) or max(got.values()) != 1:
        raise SystemExit(
            f"FAIL: cells {len(got)} of {len(held)}, duplicates "
            f"{sum(v > 1 for v in got.values())}"
        )
    bad = [r["key"] for r in rows if r["fires"] != held[r["key"]]["fires"]]
    table = collections.Counter((held[r["key"]]["label"], r["fires"]) for r in rows)
    silent = sum(
        held[r["key"]]["label"] == "OVER"
        and r["recommended"] == "dark"
        and not r["ambiguous"]
        and r["other_warnings"] == []
        for r in rows
    )
    over = sum(held[r["key"]]["label"] == "OVER" for r in rows)
    print(json.dumps({
        "cells": len(rows), "mismatches": len(bad), "mismatched": bad[:20],
        "label_fires": {f"{a}/{b}": n for (a, b), n in sorted(table.items())},
        "over_silent_before": f"{silent}/{over}",
        "S1": "PASS" if not bad else "FAIL",
    }, indent=1))  # fmt: skip


def main() -> None:
    signal.signal(signal.SIGALRM, _walltime)
    signal.alarm(2 * 3600)
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cells (pilot)")
    ap.add_argument("--score", action="store_true")
    o = ap.parse_args()
    o.out.mkdir(parents=True, exist_ok=True)
    if o.score:
        score(o.out, o.of)
        return
    server = build_server()
    held = json.loads((HERE / "heldout.json").read_text())[o.shard :: o.of]
    if o.limit:
        held = held[: o.limit]
    path = o.out / f"cell_{o.shard}.png"
    rows = []
    for e in held:
        c = crop(e)
        gray = to_gray(c, "a")
        t = int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        if histogram_density_at(gray, t) != e["density"]:
            raise SystemExit(f"crop differs from heldout.py's: {key(e)}")
        cv2.imwrite(str(path), c)
        res = asyncio.run(
            server.call_tool(
                "suggest_segmentation",
                {"image_path": str(path), "channel": "a", "method": "otsu"},
            )
        )
        pol = json.loads(res.content[0].text)["polarity"]
        codes = [w["code"] for w in pol["warnings"]]
        rows.append({
            "key": key(e), "fires": CODE in codes,
            "recommended": pol["recommended"], "ambiguous": pol["ambiguous"],
            "other_warnings": [c for c in codes if c != CODE],
        })  # fmt: skip
        if rows[-1]["fires"] != e["fires"]:
            print(f"MISMATCH {key(e)} suggest {rows[-1]['fires']} segment {e['fires']}")
    (o.out / f"suggest_{o.shard}of{o.of}.json").write_text(json.dumps(rows) + "\n")
    print(f"shard {o.shard}/{o.of}: {len(rows)} cells written", flush=True)


if __name__ == "__main__":
    main()
