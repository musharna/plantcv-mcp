"""F2 counterfactual: the released server's leaf counters on each F2 question's plant.

No model calls. Run with plantcv-mcp[sam]==1.17.2 on CPU:
    python f2_counterfactual.py ../questions.json sam_vit_b_01ec64.pth rows.json
Rules fixed before any count was seen (2026-10-27 ~01:26 EDT; the one earlier
run stopped on f2_01's refusal before counting):
 (A) counter ceiling = docs/EVAL.md's path: the plant's box from the annotation
     (eval_leaf_count.plants), segment(a, otsu), refine(fill_holes, keep_largest 1),
     then count_leaves (default) and segment_leaves_sam. Box from the annotation,
     never the count.
 (B) realistic = the pot cell the question's row/col names, segment(a, otsu,
     polarity_report's recommendation), the same refine, the same counters.
A refusal is a miss. Scored as the PREREG scores F2: exact; |DiC| <= 1.
Cross-check: F2's truth must equal EVAL's truth for the matched plant.
"""

import json
import signal
import sys
import tempfile
import time
from pathlib import Path

# eval_leaf_count.plants: the plant boxes and truth docs/EVAL.md uses
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import cv2
import numpy as np
from eval_leaf_count import plants

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.leaves import LeafCountRefusedError
from plantcv_mcp.refine import RefinementErasedMaskError
from plantcv_mcp.server import (
    _count_leaves_impl,
    _refine_impl,
    _segment_impl,
    _segment_leaves_sam_impl,
    load_image,
    polarity_report,
)
from plantcv_mcp.workers import set_isolation, shutdown_worker

# The server's named refusals of a mask; anything else is a bug and stops the run.
REFUSALS = (DegenerateMaskError, LeafCountRefusedError, RefinementErasedMaskError)

signal.signal(
    signal.SIGALRM,
    lambda *_: (sys.stderr.write("aborting: walltime guard\n"), sys.exit(2)),
)
signal.alarm(3300)

qfile, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
q = json.loads(Path(qfile).read_text())
root = Path(q["images_dir"]).expanduser()
f2 = [x for x in q["questions"] if x["family"] == "F2"]
assert len(f2) == 10, len(f2)
set_isolation(False)


def run(path, ot):
    r = {}
    try:
        sid = _segment_impl(str(path), "a", "otsu", ot)["session_id"]
        sid = _refine_impl(sid, [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}])[
            "session_id"
        ]
    except REFUSALS as e:
        return {
            "watershed": None,
            "sam": None,
            "refused": f"segment/refine {type(e).__name__}: {str(e)[:90]}",
        }
    for k, f in (
        ("watershed", lambda: _count_leaves_impl(sid)),
        ("sam", lambda: _segment_leaves_sam_impl(sid, checkpoint_path=ckpt)),
    ):
        try:
            t = time.perf_counter()
            o = f()
            r[k] = o["leaf_count"]
            r[k + "_s"] = round(time.perf_counter() - t, 1)
            r[k + "_warn"] = [w["code"] for w in o.get("warnings", [])]
        except REFUSALS as e:
            r[k] = None
            r[k + "_refused"] = f"{type(e).__name__}: {str(e)[:90]}"
    return r


rows = []
with tempfile.TemporaryDirectory(prefix="f2cf-") as hold:
    for x in f2:
        ip = root / x["image"]
        img = cv2.imread(str(ip))
        assert img is not None, ip
        gtp = [
            g
            for g in (
                ip.parent / "gt" / (ip.stem + "_gt.png"),
                ip.parent / "gt" / (ip.stem + "-gt.png"),
            )
            if g.is_file()
        ]
        assert len(gtp) == 1, (x["id"], gtp)
        gt = cv2.imread(str(gtp[0]), cv2.IMREAD_UNCHANGED)[..., :3]
        (cx0, cx1), (cy0, cy1) = [
            [round(v) for v in x["where"][k]] for k in ("cell_x", "cell_y")
        ]
        hits = [
            (t, b)
            for _, t, b in plants(gt)
            if cx0 <= (b[2] + b[3]) / 2 < cx1 and cy0 <= (b[0] + b[1]) / 2 < cy1
        ]
        assert len(hits) == 1, (x["id"], len(hits))
        etruth, (y0, y1, x0, x1) = hits[0]
        assert etruth == x["truth"]["count"], (x["id"], etruth, x["truth"]["count"])
        pa = Path(hold) / f"{x['id']}_A.png"
        cv2.imwrite(str(pa), np.ascontiguousarray(img[y0:y1, x0:x1]))
        pb = Path(hold) / f"{x['id']}_B.png"
        cv2.imwrite(str(pb), np.ascontiguousarray(img[cy0:cy1, cx0:cx1]))
        rec = polarity_report(load_image(str(pb)), "a", "otsu")["recommended"]
        row = {
            "id": x["id"],
            "truth": x["truth"]["count"],
            "date_median": x["truth"]["date_median"],
            "A": run(pa, "dark"),
            "B_polarity": rec,
            "B": run(pb, rec),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
shutdown_worker()
Path(out).write_text(json.dumps(rows, indent=1) + "\n")
