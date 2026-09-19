"""Count error of count_leaves() against the Aberystwyth Leaf Evaluation Dataset.

Not run in CI: the tray images are 8 MB each inside a 61 GB archive
(https://doi.org/10.5281/zenodo.168158, CC BY 4.0). Fetch a tray image and its
`gt/` annotation, then:

    python scripts/eval_leaf_count.py TRAY.png TRAY_gt.png [more pairs ...]

Ground truth: in a `_gt.png`, each leaf of a plant is painted in its own colour;
white and near-black (4,4,4) are annotation furniture. A plant is a connected
group of painted pixels after a 25 px dilation; its leaf count is the number of
distinct colours covering at least MIN_LEAF_PX pixels.

Two masks are scored, because they answer different questions:
  gt-mask    the annotation's own foreground: counting error alone
  segment()  the plant's box written to disk and sent through segment()
             (LAB a, otsu, dark, default fill_size), refine(keep_largest 1),
             then count_leaves(): what a user of the server gets, segmentation
             error included. keep_largest is needed because a box around a
             grown rosette contains slivers of its neighbours, each of which
             the watershed chops into many instances (measured: +15 leaves mean
             error without it); it also drops any leaf of THIS plant that
             segmented apart from the rosette.
  +fill      the same with fill_holes first: pinholes in a thresholded leaf
             each add distance-transform peaks
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from plantcv_mcp.leaves import count_leaves
from plantcv_mcp.server import _count_leaves_impl, _refine_impl, _segment_impl
from plantcv_mcp.workers import set_isolation, shutdown_worker

MIN_LEAF_PX = 8
DISTANCES = (3, 4, 5, 6, 8, 10, 15)
# Wide enough that the largest rosette stays under half its box (count_leaves
# refuses a mask over 50% of the frame as probably inverted).
PAD = 60


def plants(gt: np.ndarray):
    furniture = (gt == 255).all(axis=2) | (gt == 4).all(axis=2)
    fg = gt.any(axis=2) & ~furniture
    grouped = cv2.dilate(fg.astype(np.uint8), np.ones((25, 25), np.uint8))
    n, lab = cv2.connectedComponents(grouped)
    for i in range(1, n):
        sel = (lab == i) & fg
        if sel.sum() < 150:
            continue
        codes = gt[sel]
        _, counts = np.unique(codes, axis=0, return_counts=True)
        truth = int((counts >= MIN_LEAF_PX).sum())
        ys, xs = np.nonzero(sel)
        box = (
            max(ys.min() - PAD, 0),
            ys.max() + PAD,
            max(xs.min() - PAD, 0),
            xs.max() + PAD,
        )
        yield sel, truth, box


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    set_isolation(False)
    for image_path, gt_path in zip(argv[::2], argv[1::2], strict=True):
        img = cv2.imread(image_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[..., :3]
        found = list(plants(gt))
        truths = [t for _, t, _ in found]
        print(f"{image_path}: {len(found)} plants, truth {truths}")
        tmp = Path(tempfile.mkdtemp(prefix="leafeval-"))
        for kind in ("gt-mask", "segment()", "+fill"):
            for d in DISTANCES:
                errors = []
                for k, (sel, truth, (y0, y1, x0, x1)) in enumerate(found):
                    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
                    if kind == "gt-mask":
                        mask = np.where(sel[y0:y1, x0:x1], 255, 0).astype(np.uint8)
                        n = count_leaves(crop, mask, d).leaf_count
                    else:
                        path = tmp / f"plant{k}.png"
                        if not path.exists():
                            cv2.imwrite(str(path), crop)
                        seg = _segment_impl(str(path), "a", "otsu")
                        sid = seg["session_id"]
                        ops: list[dict] = [{"op": "keep_largest", "n": 1}]
                        if kind == "+fill":
                            ops.insert(0, {"op": "fill_holes"})
                        sid = _refine_impl(sid, ops)["session_id"]
                        n = _count_leaves_impl(sid, d)["leaf_count"]
                    errors.append(n - truth)
                e = np.array(errors)
                print(
                    f"  {kind:9s} min_distance={d:2d}  mean error {e.mean():+6.2f}  "
                    f"MAE {np.abs(e).mean():5.2f}  exact {int((e == 0).sum())}/{len(e)}"
                )
    shutdown_worker()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
