"""Reference solution for the positive control: answers every field question
without plantcv-mcp, by a route that shares as little as it can with
build.py.

- The plant: the 25 px-dilated painted group (SciPy labelling, not OpenCV)
  with the most painted pixels inside the question's pot cell.
- Pixels: its painted pixel count.
- mm²: the DATASET AUTHORS' calibration (the shipped psi_camera_data.xml,
  one frame), not the builder's 14-board fit; the mask image is resampled
  with cv2.undistort (bilinear, summed), not integrated through an area
  Jacobian; the scale is the interior pots' rims measured on the
  undistorted first image itself, not on undistorted edge points. The first
  image is the tray's earliest annotated date, from the question's own tray.
- Leaves: distinct colours with at least 8 px, counted from packed RGB
  integers with bincount.

    python benchmarks/agent_qa/field_2026-10-24/reference.py [questions.json]   # one reply per question

Its replies must score right on every question (tests/test_agent_qa_field.py).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import field_lib as F
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent


def data_dir(doc: dict) -> Path:
    return Path(doc["images_dir"]).expanduser()


def gt_path(raw: Path) -> Path:
    for suffix in ("_top_gt.png", "_top-gt.png"):
        p = raw.parent / "gt" / raw.name.replace("_top.png", suffix)
        if p.exists():
            return p
    raise FileNotFoundError(f"no annotation for {raw}")


def plant_mask(rgb: np.ndarray, where: dict) -> np.ndarray:
    painted = (rgb[..., 0] >= 25) & ~(rgb == 255).all(axis=2)
    grown = ndimage.binary_dilation(painted, structure=np.ones((25, 25), bool))
    lab, _ = ndimage.label(grown, structure=np.ones((3, 3), bool))
    x0, x1 = (int(np.rint(v)) for v in where["cell_x"])
    y0, y1 = (int(np.rint(v)) for v in where["cell_y"])
    x0, y0 = max(x0, 0), max(y0, 0)
    inside = lab[y0:y1, x0:x1][painted[y0:y1, x0:x1]]
    best = np.bincount(inside[inside > 0]).argmax()
    return painted & (lab == best)


def first_image(tray_dir: Path) -> Path:
    """The tray's earliest annotated raw image: every pot still unoccluded."""
    gts = sorted((tray_dir / "gt").glob("*.png"))
    if not gts:
        raise FileNotFoundError(f"no annotations in {tray_dir / 'gt'}")
    name = (
        gts[0]
        .name.replace("_top_gt.png", "_top.png")
        .replace("_top-gt.png", "_top.png")
    )
    return tray_dir / name


@lru_cache(maxsize=2)
def shipped_scale(
    data: str, tray_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lens = F.shipped(Path(data) / "camera_correction" / "psi_camera_data.xml")
    path = first_image(Path(data) / tray_dir)
    first = cv2.imread(str(path))
    if first is None:
        raise FileNotFoundError(f"cannot read the rim image {path}")
    und = cv2.undistort(first, lens.K, lens.dist, None, lens.K)
    c = F.cream(und)
    # interior pots: rows 2-3, columns 2-4; cell bounds from the corrected
    # image's own gap peaks
    colp, rowp = c[200:1750].mean(axis=0), c[:, 300:2300].mean(axis=1)

    def peaks(prof: np.ndarray, near: tuple[int, ...]) -> list[int]:
        return [a - 60 + int(np.argmax(prof[a - 60 : a + 61])) for a in near]

    gx, gy = peaks(colp, F.GAP_X), peaks(rowp, F.GAP_Y)
    sizes = []
    for r in (0, 1):  # between gy[r] and gy[r+1]
        for k in (0, 1, 2):  # between gx[k] and gx[k+1]
            x0, x1, y0, y1 = gx[k], gx[k + 1], gy[r], gy[r + 1]
            for y in range(int(y0 + 0.3 * (y1 - y0)), int(y0 + 0.7 * (y1 - y0)), 4):
                d = np.nonzero(~c[y, x0:x1])[0]
                sizes.append(d[-1] - d[0] + 1)
            for x in range(int(x0 + 0.3 * (x1 - x0)), int(x0 + 0.7 * (x1 - x0)), 4):
                d = np.nonzero(~c[y0:y1, x])[0]
                sizes.append(d[-1] - d[0] + 1)
    return lens.K, lens.dist, np.array(sizes), float(np.mean(sizes) / F.POT_MM)


def answer(q: dict, data: Path) -> str:
    raw = data / q["image"]
    rgb = cv2.imread(str(gt_path(raw)), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1]
    mask = plant_mask(rgb, q["where"])
    if q["category"] == "tray_leaf_count":
        packed = (rgb[mask].astype(np.int64) * [65536, 256, 1]).sum(axis=1)
        _, counts = np.unique(packed, return_counts=True)
        return f"ANSWER: {int((counts >= 8).sum())}"
    K, dist, _, ppm = shipped_scale(str(data), q["image"].rsplit("/", 1)[0])
    und = cv2.undistort(mask.astype(np.float32), K, dist, None, K)
    mm2 = float(und.sum()) / ppm**2
    return f"ANSWER: {int(mask.sum())} px, {mm2:.2f} mm²"


def main(argv: list[str]) -> int:
    doc = json.loads(Path(argv[0] if argv else HERE / "questions.json").read_text())
    for q in doc["questions"]:
        print(q["id"], answer(q, data_dir(doc)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
