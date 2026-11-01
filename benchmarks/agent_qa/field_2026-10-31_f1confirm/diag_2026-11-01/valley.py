"""Histogram height at the Otsu threshold, as a share of the peak; see README.md.

Usage: python valley.py OUTDIR  (OUTDIR holds sweep.py's cell crops and sweep.json)
"""

import json
import signal
import sys
from pathlib import Path

import cv2
import numpy as np

from plantcv_mcp.segmentation import to_gray

TRAY = Path.home() / "plantcv-mcp-artifacts/field_2026-10-31_f1confirm/aberystwyth"
# The raw-image crops the server arm segmented on its young-plant misses
# (x, y, w, h from the transcripts).
AGENT = {
    "f1c_02": ("2015-12-15--19-25-16", (2040, 220, 300, 200)),
    "f1c_03": ("2015-12-16--20-38-55", (1580, 150, 320, 300)),
    "f1c_04": ("2015-12-17--20-23-02", (250, 620, 280, 250)),
    "f1c_07": ("2015-12-21--20-32-51", (1560, 560, 380, 380)),
    "f1c_09": ("2015-12-23--22-56-06", (1100, 1000, 400, 390)),
}


def _walltime(*_: object) -> None:
    sys.stderr.write("walltime\n")
    sys.exit(2)


def stat(img: np.ndarray) -> tuple[float, int, int]:
    g = to_gray(img, "a")
    h = np.bincount(g.ravel(), minlength=256).astype(float)
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    t = int(cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    return hs[t] / hs.max(), t, int(hs.argmax())


def main() -> None:
    signal.signal(signal.SIGALRM, _walltime)
    signal.alarm(300)
    out_dir = Path(sys.argv[1])
    rows = {r["id"]: r for r in json.loads((out_dir / "sweep.json").read_text())}
    out = []
    for qid, r in rows.items():
        d, t, pk = stat(cv2.imread(str(out_dir / f"cell_{qid}.png")))
        right = r["lo"] <= r["otsu50"] <= r["hi"]
        out.append((d, qid, "cell", right, round(r["plant_frac"] * 100, 1), t, pk))
    for qid, (stem, (x, y, w, h)) in AGENT.items():
        img = cv2.imread(str(TRAY / f"PSI_Tray031/tv/PSI_Tray031_{stem}_top.png"))
        d, t, pk = stat(img[y : y + h, x : x + w])
        out.append((d, qid, "agent crop", False, None, t, pk))
    for d, qid, kind, ok, pf, t, pk in sorted(out):
        verdict = "right" if ok else "WRONG"
        print(
            f"{d:.2f}  {qid:7} {kind:10} otsu {verdict}  plant% {pf}  t={t} peak={pk}"
        )


if __name__ == "__main__":
    main()
