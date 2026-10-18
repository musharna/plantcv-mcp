"""Negative controls: answers that need no model, scored by the real scorer.

- ``constant``: one answer per kind of question, the median of the key's own
  answers for that kind. It is told the key, so it is a generous guess; a
  setup that does no better than it has not measured anything.
- ``threshold``: the shortcut the hardening was aimed at, one colour cut. A
  pixel is plant if its green exceeds both red and blue by more than 10; the
  largest connected part, holes filled, is the plant. From that mask:
  area = its pixels; height/width = its bounding box; area in mm² = its pixels
  ÷ (side of the darkest square ÷ the side in mm stated in the question)²;
  tray count = pots (the tray split evenly) holding more than 50 green pixels;
  tray area = the largest green part in the asked pot; mean hue = circular
  mean HSV hue of the mask; yellow share = pixels with hue below 70° among
  those with hue 35-170°, saturation ≥ 0.3 and value ≥ 0.25, largest part.
  Leaf counts have no colour-cut answer: ``UNKNOWN``.

Writes ``baselines.jsonl`` (one row per question and control). numpy, PIL and
scipy only; nothing from plantcv-mcp or PlantCV.

    uv run python benchmarks/agent_qa/baselines.py
"""

from __future__ import annotations

import colorsys
import json
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as nd
from score import score, truth_value

HERE = Path(__file__).resolve().parent


def kind(q: dict[str, Any]) -> str:
    return q["truth"].get("kind", q["category"])


def largest(mask: np.ndarray) -> np.ndarray:
    lab, n = nd.label(mask)
    if n == 0:
        return mask
    sizes = nd.sum(mask, lab, range(1, n + 1))
    return nd.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)


def green(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    return (g - r > 10) & (g - b > 10)


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = rgb.reshape(-1, 3) / 255.0
    h, s, v = np.array([colorsys.rgb_to_hsv(*p) for p in flat]).T
    shape = rgb.shape[:2]
    return h.reshape(shape) * 360.0, s.reshape(shape), v.reshape(shape)


def circular_mean_deg(h: np.ndarray) -> float:
    a = np.deg2rad(h)
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def threshold_answer(q: dict[str, Any]) -> str:
    rgb = np.asarray(Image.open(HERE / "images" / q["image"]).convert("RGB"))
    k = kind(q)
    if k == "leaf_count":
        return "UNKNOWN"
    if q["category"] == "tray":
        rows, cols = map(
            int, re.search(r"(\d+) rows x (\d+) columns", q["question"]).groups()
        )
        h, w = rgb.shape[:2]
        ch, cw = (h - 40) / rows, (w - 40) / cols
        g = green(rgb)

        def pot(r: int, c: int) -> np.ndarray:
            y0, x0 = int(20 + r * ch), int(20 + c * cw)
            return g[y0 : int(y0 + ch), x0 : int(x0 + cw)]

        if k == "count":
            return str(
                sum(pot(r, c).sum() > 50 for r in range(rows) for c in range(cols))
            )
        r, c = map(int, re.search(r"row (\d+), column (\d+)", q["question"]).groups())
        return str(int(largest(pot(r - 1, c - 1)).sum()))
    if k == "yellow_pct":
        hh, s, v = hsv(rgb)
        m = largest((hh > 35) & (hh < 170) & (s >= 0.3) & (v >= 0.25))
        return f"{100.0 * float((hh[m] < 70).mean()):.2f}"
    m = largest(green(rgb))
    if k == "hue":
        return f"{circular_mean_deg(hsv(rgb)[0][m]):.2f}"
    if q["category"] == "area":
        return str(int(m.sum()))
    if q["category"] == "extent":
        ys, xs = np.nonzero(m)
        span = ys if q["truth"]["axis"] == "height" else xs
        return str(int(span.max() - span.min() + 1))
    if q["category"] == "area_mm":
        side_mm = float(
            re.search(r"exactly (\d+(?:\.\d+)?) mm", q["question"]).group(1)
        )
        dark = largest(rgb.max(axis=2) < 40)
        ppm = np.sqrt(dark.sum()) / side_mm
        return f"{m.sum() / ppm**2:.1f}"
    raise ValueError(q["id"])


def main() -> None:
    qs = json.loads((HERE / "questions.json").read_text())["questions"]
    groups: dict[tuple[str, str], list[float]] = {}
    for q in qs:
        groups.setdefault((q["category"], kind(q)), []).append(truth_value(q))
    rows = []
    for q in qs:
        const = statistics.median(groups[(q["category"], kind(q))])
        if kind(q) in ("count", "leaf_count"):
            const = round(const)
        for arm, value in (
            ("baseline_constant", f"{const}"),
            ("baseline_threshold", threshold_answer(q)),
        ):
            rows.append(
                {
                    "arm": arm,
                    "qid": q["id"],
                    "category": q["category"],
                    "rep": 1,
                    "valid": True,
                    **score(q, f"ANSWER: {value}"),
                }
            )
    (HERE / "baselines.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for arm in ("baseline_constant", "baseline_threshold"):
        got = [r for r in rows if r["arm"] == arm]
        print(arm, sum(r["verdict"] == "right" for r in got), "of", len(got), "right")
        for cat in dict.fromkeys(q["category"] for q in qs):
            c = [r for r in got if r["category"] == cat]
            print(f"  {cat:10} {sum(r['verdict'] == 'right' for r in c)}/{len(c)}")


if __name__ == "__main__":
    main()
