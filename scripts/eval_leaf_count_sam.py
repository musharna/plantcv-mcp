"""Count error of segment_leaves_sam() against the Aberystwyth dataset.

The counterpart of eval_leaf_count.py, on the same plants, boxes and ground
truth (imported from it), through the same path a user of the server takes:
the plant's box written to disk, segment(a, otsu), refine(fill_holes,
keep_largest 1), then segment_leaves_sam(). Not run in CI (data size, and the
`sam` extra).

    CUDA_VISIBLE_DEVICES="" python scripts/eval_leaf_count_sam.py \
        CHECKPOINT.pth TRAY.png TRAY_gt.png [more pairs ...] [--margin F]

Runs on CPU. `--margin` overrides CROP_MARGIN_FRACTION and exists for the one
choice that was made with this script, on Tray 031; the figures in docs/EVAL.md
use the default. Seconds per plant cover the tool call only (image re-read,
checkpoint already loaded after the first plant, model, post-filter, overlay).
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_leaf_count import plants

from plantcv_mcp import sam_leaves
from plantcv_mcp.server import (
    _refine_impl,
    _segment_impl,
    _segment_leaves_sam_impl,
)
from plantcv_mcp.workers import set_isolation, shutdown_worker


def main(argv: list[str]) -> int:
    if "--margin" in argv:
        at = argv.index("--margin")
        sam_leaves.CROP_MARGIN_FRACTION = float(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
    if len(argv) < 3 or len(argv) % 2 == 0:
        print(__doc__)
        return 2
    checkpoint = argv[0]
    set_isolation(False)
    print(f"crop margin {sam_leaves.CROP_MARGIN_FRACTION}")
    with tempfile.TemporaryDirectory(prefix="leafeval-sam-") as holder:
        for image_path, gt_path in zip(argv[1::2], argv[2::2], strict=True):
            img = cv2.imread(image_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)[..., :3]
            errors, seconds, coverage = [], [], []
            for k, (_sel, truth, (y0, y1, x0, x1)) in enumerate(plants(gt)):
                path = Path(holder) / f"{Path(image_path).stem}_{k}.png"
                cv2.imwrite(str(path), np.ascontiguousarray(img[y0:y1, x0:x1]))
                sid = _segment_impl(str(path), "a", "otsu")["session_id"]
                sid = _refine_impl(
                    sid, [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}]
                )["session_id"]
                started = time.perf_counter()
                out = _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint)
                seconds.append(time.perf_counter() - started)
                errors.append(out["leaf_count"] - truth)
                coverage.append(out["mask_coverage"])
            e = np.array(errors)
            print(
                f"{image_path}: {len(e)} plants  mean error {e.mean():+.2f}  "
                f"MAE {np.abs(e).mean():.2f}  exact {int((e == 0).sum())}/{len(e)}  "
                f"s/plant mean {np.mean(seconds[1:]):.1f} max {np.max(seconds[1:]):.1f} "
                f"(first, with model load: {seconds[0]:.1f})  "
                f"mask_coverage min {min(coverage):.2f} median {np.median(coverage):.2f}"
            )
    shutdown_worker()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
