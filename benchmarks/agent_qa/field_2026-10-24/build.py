"""Build the field families F1 (per-plant area, px and mm2) and F2 (leaf count)
from the tray 032 hand annotations of the Aberystwyth Leaf Evaluation Dataset.

Bell J, Dee HM (2016) Aberystwyth Leaf Evaluation Dataset. Zenodo.
https://doi.org/10.5281/zenodo.168158 (CC BY 4.0).

Run fetch.py first, then (openpyxl only reads the harvest sheet):

    uv run --with openpyxl==3.1.5 python benchmarks/agent_qa/field_2026-10-24/build.py

Writes questions.json beside this file (the set run.py reads with
--questions) and build_report.json to the artifacts directory: the lens
calibration, the pot scale, the retention of every annotated plant with the
reason for each exclusion, and the shortcut measurements. Deterministic: the
calibration is OpenCV's, and the sampling is seeded.

Never imports or calls plantcv-mcp.
"""

from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import field_lib as F
import numpy as np

HERE = Path(__file__).resolve().parent
ART = Path.home() / "plantcv-mcp-artifacts" / "field_2026-10-24"
DATA = ART / "aberystwyth"
TV = "PSI_Tray032/tv"
BOARDS = "camera_correction/images"
SEED = {"F1": 20261024, "F2": 20261025}
N = 10
SOWN = dt.date(2015, 11, 23)
FIRST = "PSI_Tray032_2015-12-14--12-54-51_top.png"  # every pot unoccluded
SET = "field_2026-10-24"

TRAY_TEXT = (
    "This is a top-down photo of a tray of Arabidopsis rosettes in square pots, "
    "4 rows by 5 columns; some pots may be empty. "
)
WHERE = (
    "the plant in the pot at row {r}, column {c} (row 1 is the top row of pots in "
    "the image, column 1 the leftmost)"
)
F1_TEXT = (
    TRAY_TEXT + "What is the projected leaf area of " + WHERE + ", counting that "
    "plant only: (a) in pixels of this image as it is, and (b) in square "
    "millimetres, with the lens distortion removed and the scale taken from the "
    "pot rims, which are 60 mm across. The camera lens has barrel distortion; the "
    "folder checkerboards in the same directory as the image holds photos of a "
    "flat checkerboard with 12 x 9 inner corners, taken with the same camera."
)
F2_TEXT = (
    TRAY_TEXT + "How many leaves does " + WHERE + " have? Count every leaf, including "
    "small young leaves at the centre and leaves mostly hidden under others."
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def raw_name(gt: Path) -> str:
    return gt.name.replace("_top_gt.png", "_top.png").replace("_top-gt.png", "_top.png")


def main() -> int:
    lenses = F.calibrate(DATA / BOARDS, DATA / "camera_correction/psi_camera_data.xml")
    jac = {k: F.area_jacobian(v) for k, v in lenses.items()}
    edges = F.pot_edges(
        cv2.imread(str(DATA / TV / FIRST)), F.grid(cv2.imread(str(DATA / TV / FIRST)))
    )
    raw_sizes = F.pot_sizes(edges, None)
    sizes = {k: F.pot_sizes(edges, v) for k, v in lenses.items()}
    ppm = {k: F.px_per_mm(s) for k, s in sizes.items()}
    prim = F.PRIMARY
    iv = np.array([sizes[prim][r, c] for r, c in F.INTERIOR]).ravel()
    # Scale uncertainty: the standard error of the interior pots' mean size,
    # doubled for an area and taken at 95%.
    scale_se = float(iv.std(ddof=1) / np.sqrt(len(iv)) / iv.mean())
    scale_term = 2 * 1.96 * scale_se
    ppm_raw = {"interior": F.px_per_mm(raw_sizes), "all": F.px_per_mm(raw_sizes, "all")}
    harvests = F.read_harvests(DATA / "AT20_graphed_GT.xlsx")

    retention, eligible = [], []
    leaves_by_date: dict[str, list[int]] = {}
    for gtp in sorted(Path(p) for p in glob.glob(str(DATA / TV / "gt" / "*.png"))):
        name = raw_name(gtp)
        img = cv2.imread(str(DATA / TV / name))
        g = F.grid(img)
        when = dt.date.fromisoformat(name.split("_")[2][:10])
        occ = F.occupied(harvests, when)
        ps = F.plants(F.read_gt(gtp), name, g, occ)
        why = Counter(p.excluded or "kept" for p in ps)
        retention.append({"image": name, "date": str(when), "das": (when - SOWN).days,
                          "plants": len(occ), **why, "grid_x": g.gx, "grid_y": g.gy})  # fmt: skip
        xb, yb = g.bounds()
        for p in ps:
            if p.excluded:
                continue
            leaves_by_date.setdefault(str(when), []).append(p.leaf_count)
            er, di = F.mask_band(p.mask)
            mm = {k: float(jac[k][p.mask].sum() / ppm[k] ** 2) for k in lenses}
            cal_term = max(abs(v / mm[prim] - 1) for k, v in mm.items() if k != prim)
            e = cal_term + scale_term
            area = int(p.mask.sum())
            eligible.append({
                "image": name, "date": str(when), "das": (when - SOWN).days,
                "row": p.row + 1, "col": p.col + 1,
                "edge": p.row in (0, F.ROWS - 1) or p.col in (0, F.COLS - 1),
                "cell_x": [round(xb[p.col], 1), round(xb[p.col + 1], 1)],
                "cell_y": [round(yb[p.row], 1), round(yb[p.row + 1], 1)],
                "area_px": area, "px_lo": int(er.sum()), "px_hi": int(di.sum()),
                "area_mm2": mm[prim],
                "mm2_lo": float(jac[prim][er].sum() / ppm[prim] ** 2) * (1 - e),
                "mm2_hi": float(jac[prim][di].sum() / ppm[prim] ** 2) * (1 + e),
                "mm2_rel_terms": {"calibration": cal_term, "scale": scale_term},
                "mm2_by_lens": {k: v for k, v in mm.items() if not k.startswith("loo_")},
                "uncorrected_mm2": {k: area / v ** 2 for k, v in ppm_raw.items()},
                "leaf_count": p.leaf_count,
                # How the count moves with the leaf-size rule (truth sensitivity).
                "leaf_count_at_px": {
                    str(t): sum(1 for n in p.colours.values() if n >= t)
                    for t in (1, 8, 30, 100)
                },
            })  # fmt: skip

    # Shortcut controls, on every eligible plant (not only the sampled ones).
    def inside(x: float, e: dict) -> bool:
        return e["mm2_lo"] <= x <= e["mm2_hi"]

    shortcut = {}
    for k, k_ppm in ppm_raw.items():
        fail = [e for e in eligible if not inside(e["uncorrected_mm2"][k], e)]
        rel = [e["uncorrected_mm2"][k] / e["area_mm2"] - 1 for e in eligible]
        edge = [r for r, e in zip(rel, eligible) if e["edge"]]
        mid = [r for r, e in zip(rel, eligible) if not e["edge"]]
        shortcut[f"uncorrected_px_per_mm_{k}"] = {
            "px_per_mm": k_ppm,
            "fails": len(fail), "of": len(eligible),
            "fails_edge": sum(e["edge"] for e in fail),
            "fails_interior": sum(not e["edge"] for e in fail),
            "edge_plants": sum(e["edge"] for e in eligible),
            "rel_error_edge_median": float(np.median(edge)),
            "rel_error_edge_range": [float(min(edge)), float(max(edge))],
            "rel_error_interior_median": float(np.median(mid)),
            "rel_error_interior_range": [float(min(mid)), float(max(mid))],
        }  # fmt: skip
    medians = {d: int(np.median(v)) for d, v in leaves_by_date.items()}
    guess = [e for e in eligible if medians[e["date"]] == e["leaf_count"]]
    guess1 = [e for e in eligible if abs(medians[e["date"]] - e["leaf_count"]) <= 1]

    # Sampling, seeded, fixed before any arm runs: 10 distinct images per
    # family (of those with an eligible plant), one eligible plant from each.
    by_image: dict[str, list[dict]] = {}
    for e in eligible:
        by_image.setdefault(e["image"], []).append(e)
    images = sorted(by_image)
    questions, sampled = [], {}
    boards = sorted(Path(p) for p in glob.glob(str(DATA / BOARDS / "*.png")))
    board_sha = {b.name: sha(b) for b in boards}
    for fam in ("F1", "F2"):
        rng = np.random.default_rng(SEED[fam])
        pick = sorted(rng.choice(len(images), size=N, replace=False).tolist())
        sampled[fam] = []
        for n, i in enumerate(pick, 1):
            cands = by_image[images[i]]
            e = cands[int(rng.integers(len(cands)))]
            sampled[fam].append((e["image"], e["row"], e["col"]))
            q = {
                "id": f"{fam.lower()}_{n:02d}",
                "family": fam,
                "image": f"{TV}/{e['image']}",
                "sha256": sha(DATA / TV / e["image"]),
                "answer_format": (
                    "two numbers, the area in pixels and then in mm², e.g. 12345 px, 678.9 mm²"
                    if fam == "F1" else "an integer, e.g. 12"
                ),
                "question": (F1_TEXT if fam == "F1" else F2_TEXT).format(r=e["row"], c=e["col"]),
                "where": {k: e[k] for k in ("date", "das", "row", "col", "edge", "cell_x", "cell_y")},
            }  # fmt: skip
            if fam == "F1":
                q["category"] = "tray_plant_area"
                q["extra_dirs"] = {"checkerboards": BOARDS}
                q["extra_sha256"] = {
                    f"checkerboards/{k}": v for k, v in board_sha.items()
                }
                q["truth"] = {k: e[k] for k in (
                    "area_px", "px_lo", "px_hi", "area_mm2", "mm2_lo", "mm2_hi",
                    "mm2_rel_terms", "mm2_by_lens", "uncorrected_mm2")}  # fmt: skip
                q["truth"]["px_per_mm"] = ppm[prim]
                q["how_built"] = (
                    "hand-annotation pixels of the plant's 25 px-dilated group; px window = the "
                    "mask eroded / dilated by 1 px (3x3); mm2 = sum over the mask of the "
                    "undistortion area Jacobian (OpenCV k1k2p1p2k3 fit to the shipped "
                    "checkerboards) / px_per_mm^2, px_per_mm from the 6 interior pot rims "
                    "(60 mm) in corrected coordinates; window widened by the lens-model spread "
                    "and 2 x 1.96 x the scale's standard error"
                )
            else:
                q["category"] = "tray_leaf_count"
                q["truth"] = {
                    "count": e["leaf_count"],
                    "date_median": medians[e["date"]],
                    "count_at_min_px": e["leaf_count_at_px"],
                }
                q["how_built"] = (
                    "distinct annotation colours covering >= 8 px of the plant's 25 px-dilated "
                    "group (the docs/EVAL.md rule); one colour per leaf (dataset PDF 2.3.1)"
                )
            questions.append(q)

    sampled_f1 = [q for q in questions if q["family"] == "F1"]
    report = {
        "set": SET,
        "source": {"doi": "10.5281/zenodo.168158", "licence": "CC BY 4.0",
                   "citation": "Bell J, Dee HM (2016) Aberystwyth Leaf Evaluation Dataset. Zenodo.",
                   "manifest_sha256": sha(DATA / "MANIFEST.json"), "tray": "032"},
        "lenses": {k: v.to_json() for k, v in lenses.items()},
        "distortion": {
            k: {"max_area_jacobian": float(jac[k].max()), "min_area_jacobian": float(jac[k].min()),
                "max_displacement_px": float(np.max(np.linalg.norm(
                    F.undistort_pts(_corners(), lenses[k]) - _corners(), axis=1)))}
            for k in lenses if not k.startswith("loo_")
        },
        "pot_sizes_raw_px": raw_sizes.tolist(),
        "pot_sizes_corrected_px": sizes[prim].tolist(),
        "px_per_mm": {k: v for k, v in ppm.items() if not k.startswith("loo_")},
        "px_per_mm_all_pots": F.px_per_mm(sizes[prim], "all"),
        "px_per_mm_raw": ppm_raw,
        "scale_rel_se": scale_se,
        "retention": retention,
        "eligible": len(eligible),
        "annotated": sum(r["plants"] for r in retention),
        "shortcut_uncorrected_all_eligible": shortcut,
        "shortcut_uncorrected_sampled": {
            k: sum(not (q["truth"]["mm2_lo"] <= q["truth"]["uncorrected_mm2"][k] <= q["truth"]["mm2_hi"])
                   for q in sampled_f1) for k in ppm_raw
        },
        "shortcut_date_median_leaf_count": {"exact": len(guess), "within_one": len(guess1),
                                            "of": len(eligible), "medians": medians},
        "sampled": sampled,
        "seeds": SEED,
    }  # fmt: skip
    ART.mkdir(parents=True, exist_ok=True)
    F.save_json(ART / "build_report.json", report)
    out = {
        "set": SET,
        "built": "2026-10-24",
        "seed": SEED,
        "images_dir": "~/plantcv-mcp-artifacts/field_2026-10-24/aberystwyth",
        "alt_web": False,
        "source": report["source"],
        "build_report_sha256": sha(ART / "build_report.json"),
        # Measured on every eligible plant before any arm ran (build_report.json
        # has the detail); the tests hold the key to these numbers.
        "controls": {
            "annotated_plants": report["annotated"],
            "eligible_plants": report["eligible"],
            "retention": [
                {k: r.get(k, 0) for k in ("date", "plants", "kept", "touching", "not_found", "stray_group")}
                for r in retention
            ],
            "shortcut_uncorrected_all_eligible": shortcut,
            "shortcut_uncorrected_sampled": report["shortcut_uncorrected_sampled"],
            "shortcut_date_median_leaf_count": {
                k: v for k, v in report["shortcut_date_median_leaf_count"].items() if k != "medians"
            },
            "max_area_jacobian": report["distortion"][prim]["max_area_jacobian"],
            "max_displacement_px": report["distortion"][prim]["max_displacement_px"],
        },
        "questions": questions,
    }  # fmt: skip
    F.save_json(HERE / "questions.json", out)
    print(json.dumps({k: report[k] for k in ("eligible", "annotated", "shortcut_uncorrected_all_eligible",
                      "shortcut_uncorrected_sampled", "px_per_mm", "scale_rel_se")}, indent=1))  # fmt: skip
    return 0


def _corners() -> np.ndarray:
    w, h = F.SIZE
    return np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)


if __name__ == "__main__":
    sys.exit(main())
