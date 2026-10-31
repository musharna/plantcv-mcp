"""Truth for the field families, from the Aberystwyth hand annotations of tray 032.

Bell J, Dee HM (2016) Aberystwyth Leaf Evaluation Dataset. Zenodo.
https://doi.org/10.5281/zenodo.168158 (CC BY 4.0).

Nothing here imports or calls plantcv-mcp: the key comes from the annotators'
painted leaves, the dataset's own checkerboard photographs and harvest sheet,
and OpenCV/NumPy.

Pieces, each used by build.py and checked by the tests:

- ``calibrate``: OpenCV camera calibration from the shipped checkerboard
  photographs (13 x 10 squares of 26 mm, so 12 x 9 inner corners; the square
  size is in the shipped psi_camera_data.xml). Several lens models are fitted,
  because the truth's own uncertainty includes the choice of model.
- ``area_jacobian``: for one lens model, the area that each raw pixel covers
  after the distortion is removed (P = K, so the image centre keeps its
  scale). A raw mask's corrected area is the sum of this map over the mask:
  no resampling of the mask, so no resampling error.
- ``grid`` and ``pot_edges``: the tray's 4 x 5 pot grid from the cream holder
  between the pots, and the outer edges of each pot rim along scan lines.
- ``plants``: the annotated plants of one image, each matched to its pot, and
  kept only if it is one plant alone (see ``Plant.excluded``).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROWS, COLS = 4, 5
POT_MM = 60.0  # "PSI 6cm square pots" (dataset PDF, section 2.1)
BOARD = (12, 9)  # inner corners
SQUARE_MM = 26.0  # psi_camera_data.xml square_Size
SIZE = (2560, 1920)
# The eval_leaf_count rule (docs/EVAL.md): a plant is the painted pixels that
# connect after a 25 px dilation (petioles are partly buried, so a plant's
# annotation is not always connected); a leaf is an annotation colour covering
# at least MIN_LEAF_PX pixels of it.
GROUP_DILATE = 25
MIN_LEAF_PX = 8
# Where to look for the three internal horizontal and four internal vertical
# gaps between pots (raw image px, measured on 2015-12-14); every image's
# gaps are searched within +-GAP_SEARCH of these and must be found there.
GAP_X = (619, 1069, 1520, 1968)
GAP_Y = (519, 968, 1418)
GAP_SEARCH = 40
INTERIOR = [(r, c) for r in (1, 2) for c in (1, 2, 3)]  # 0-based rows 2-3, cols 2-4


# --------------------------------------------------------------- calibration
def board_corners(boards_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    """Inner corners of every board OpenCV's standard detector finds."""
    names, found = [], []
    files = sorted(
        glob.glob(str(boards_dir / "*.png")), key=lambda p: int(Path(p).stem)
    )
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None or g.shape[::-1] != SIZE:
            raise RuntimeError(f"{f}: unreadable or not {SIZE}")
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, c = cv2.findChessboardCorners(g, BOARD, flags=flags)
        if ok:
            found.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit))
            names.append(Path(f).name)
    return names, found


def _objp() -> np.ndarray:
    o = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    o[:, :2] = np.mgrid[0 : BOARD[0], 0 : BOARD[1]].T.reshape(-1, 2) * SQUARE_MM
    return o


MODELS = {
    # name: calibrateCamera flags
    "k1k2p1p2k3": 0,  # OpenCV's default model: the primary
    "k1k2k3": cv2.CALIB_ZERO_TANGENT_DIST,
    "k1k2p1p2": cv2.CALIB_FIX_K3,  # the model plantcv-mcp fits
}
PRIMARY = "k1k2p1p2k3"


@dataclass
class Lens:
    name: str
    K: np.ndarray
    dist: np.ndarray
    rms: float | None
    boards: list[str]

    def to_json(self) -> dict:
        return {"name": self.name, "K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
                "rms": self.rms, "boards": self.boards}  # fmt: skip


def fit(names: list[str], corners: list[np.ndarray], flags: int, name: str) -> Lens:
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [_objp()] * len(corners), corners, SIZE, None, None, flags=flags
    )
    return Lens(name, K, dist, float(rms), list(names))


def shipped(xml: Path) -> Lens:
    fs = cv2.FileStorage(str(xml), cv2.FILE_STORAGE_READ)
    K = fs.getNode("Camera_Matrix").mat()
    d = fs.getNode("Distortion_Coefficients").mat()
    fs.release()
    if K is None or d is None:
        raise RuntimeError(f"{xml}: no Camera_Matrix / Distortion_Coefficients")
    return Lens("shipped_xml", K, d, None, ["(one frame, 2016-01-21)"])


def calibrate(boards_dir: Path, xml: Path) -> dict[str, Lens]:
    """Every lens model the truth considers: the three OpenCV models on all
    detected boards, the dataset's shipped one-frame file, and the primary
    model with each board left out in turn."""
    names, corners = board_corners(boards_dir)
    if len(corners) < 10:
        raise RuntimeError(f"only {len(corners)} boards detected in {boards_dir}")
    lenses = {m: fit(names, corners, f, m) for m, f in MODELS.items()}
    lenses["shipped_xml"] = shipped(xml)
    for i, n in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        lenses[f"loo_{n}"] = fit(
            [names[j] for j in keep], [corners[j] for j in keep], MODELS[PRIMARY],
            f"loo_{n}",
        )  # fmt: skip
    return lenses


def undistort_pts(pts: np.ndarray, lens: Lens) -> np.ndarray:
    """Raw pixel coordinates (N, 2) -> corrected coordinates, same K."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, lens.K, lens.dist, P=lens.K).reshape(-1, 2)


def area_jacobian(lens: Lens, step: int = 1) -> np.ndarray:
    """|det d(corrected)/d(raw)| at every raw pixel centre (central
    differences of half a pixel), shape (H, W)."""
    w, h = SIZE
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    base = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    e = 0.5
    dx = (undistort_pts(base + [e, 0], lens) - undistort_pts(base - [e, 0], lens)) / (
        2 * e
    )
    dy = (undistort_pts(base + [0, e], lens) - undistort_pts(base - [0, e], lens)) / (
        2 * e
    )
    j = np.abs(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0])
    return j.reshape(ys.shape)


# ---------------------------------------------------------------- pot grid
def cream(img: np.ndarray) -> np.ndarray:
    """The cream holder plate between the pots (not soil, rim, leaf or frame)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (v > 100) & (s >= 20) & (s <= 90) & (h <= 35)


@dataclass
class Grid:
    gx: list[int]  # x of the 4 internal vertical gaps
    gy: list[int]  # y of the 3 internal horizontal gaps
    strength: list[float]

    def bounds(self) -> tuple[list[float], list[float]]:
        """Cell boundaries in x (6) and y (5): the internal gaps, and one
        mean pitch beyond the outermost."""
        px = float(np.mean(np.diff(self.gx)))
        py = float(np.mean(np.diff(self.gy)))
        xb = [self.gx[0] - px, *self.gx, self.gx[-1] + px]
        yb = [self.gy[0] - py, *self.gy, self.gy[-1] + py]
        return xb, yb

    def centre(self, r: int, c: int) -> tuple[float, float]:
        xb, yb = self.bounds()
        return (xb[c] + xb[c + 1]) / 2, (yb[r] + yb[r + 1]) / 2


def grid(img: np.ndarray) -> Grid:
    """The internal gaps: peaks of the cream fraction near the expected
    positions. A peak under 0.10 is no gap: refused, not guessed."""
    c = cream(img)
    colp = c[200:1750].mean(axis=0)
    rowp = c[:, 300:2300].mean(axis=1)
    gx, gy, st = [], [], []
    for prof, want, out in ((colp, GAP_X, gx), (rowp, GAP_Y, gy)):
        for a in want:
            w = prof[a - GAP_SEARCH : a + GAP_SEARCH + 1]
            k = int(np.argmax(w))
            if w[k] < 0.10:
                raise RuntimeError(f"no gap near {a}: peak cream fraction {w[k]:.2f}")
            out.append(a - GAP_SEARCH + k)
            st.append(round(float(w[k]), 3))
    return Grid(gx, gy, st)


def pot_edges(img: np.ndarray, g: Grid) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Per pot, the raw-image outer rim edges along scan lines through the
    middle 40% of its cell: horizontal lines give (left, right) point pairs,
    vertical lines (top, bottom). The rim's outer edge is the first and last
    non-cream pixel of the line within the cell (the cell runs gap centre to
    gap centre, so each side keeps half a gap of cream)."""
    c = cream(img)
    xb, yb = g.bounds()
    out = {}
    for r in range(ROWS):
        for col in range(COLS):
            x0, x1 = int(np.rint(xb[col])), int(np.rint(xb[col + 1]))
            y0, y1 = int(np.rint(yb[r])), int(np.rint(yb[r + 1]))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, SIZE[0] - 1), min(y1, SIZE[1] - 1)
            hp, vp = [], []
            for y in range(int(y0 + 0.3 * (y1 - y0)), int(y0 + 0.7 * (y1 - y0)), 4):
                dark = np.nonzero(~c[y, x0:x1])[0]
                if len(dark):
                    hp.append(((x0 + dark[0] - 0.5, y), (x0 + dark[-1] + 0.5, y)))
            for x in range(int(x0 + 0.3 * (x1 - x0)), int(x0 + 0.7 * (x1 - x0)), 4):
                dark = np.nonzero(~c[y0:y1, x])[0]
                if len(dark):
                    vp.append(((x, y0 + dark[0] - 0.5), (x, y0 + dark[-1] + 0.5)))
            out[(r, col)] = {"h": np.array(hp, float), "v": np.array(vp, float)}
    return out


def pot_sizes(edges: dict, lens: Lens | None) -> np.ndarray:
    """(ROWS, COLS, 2) median width and height of each pot, in raw px if
    lens is None, else in that model's corrected px."""
    out = np.zeros((ROWS, COLS, 2))
    for (r, c), e in edges.items():
        for k, key in enumerate(("h", "v")):
            pairs = e[key]
            a, b = pairs[:, 0], pairs[:, 1]
            if lens is not None:
                a, b = undistort_pts(a, lens), undistort_pts(b, lens)
            out[r, c, k] = float(np.median(np.linalg.norm(b - a, axis=1)))
    return out


def px_per_mm(sizes: np.ndarray, which: str = "interior") -> float:
    """Scale from the pot rims: the mean of the widths and heights of the
    interior pots (rows 2-3, columns 2-4) over 60 mm, or of all 20 pots."""
    if which == "interior":
        vals = np.array([sizes[r, c] for r, c in INTERIOR]).ravel()
    else:
        vals = sizes.reshape(-1)
    return float(vals.mean() / POT_MM)


# ------------------------------------------------------------------ plants
@dataclass
class Plant:
    image: str  # raw image file name
    row: int  # 0-based
    col: int
    mask: np.ndarray = field(repr=False)  # bool, raw image frame
    colours: dict[tuple[int, int, int], int] = field(repr=False)
    excluded: str | None = None

    @property
    def leaf_count(self) -> int:
        return sum(1 for n in self.colours.values() if n >= MIN_LEAF_PX)


def read_gt(path: Path) -> np.ndarray:
    gt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if gt is None or gt.shape[:2] != (SIZE[1], SIZE[0]):
        raise RuntimeError(f"{path}: unreadable or wrong size")
    return np.ascontiguousarray(gt[..., :3][..., ::-1])  # RGB


def foreground(rgb: np.ndarray) -> np.ndarray:
    """Painted pixels. Every leaf colour has red >= 25 (dataset PDF 2.3.1:
    leaves start at R 25-30 and all have a higher red than anything else);
    pure white and the near-black (4,4,4) are annotation furniture
    (eval_leaf_count.py), as is any pixel with red under 25."""
    furniture = (rgb == 255).all(axis=2) | (rgb[..., 0] < 25)
    return rgb.any(axis=2) & ~furniture


def occupied(
    harvests: dict[tuple[int, int], dt.date], when: dt.date
) -> set[tuple[int, int]]:
    """Pots still holding a plant on the image's date: a plant harvested on
    day d is gone from that day's images (dataset Table 3 counts agree)."""
    return {(r, c) for r in range(ROWS) for c in range(COLS) if harvests[(r, c)] > when}


def read_harvests(xlsx: Path, tray: int = 32) -> dict[tuple[int, int], dt.date]:
    """Harvest date per pot from the dataset's 'AT20 graphed GT.xlsx', sheet
    'sorted data'. Positions a1-d5: row letter a-d top to bottom, column 1-5
    left to right (checked against the empty pots in the images). The sheet
    writes the December 2015 harvests as 2016-12-xx; they are read as 2015."""
    import openpyxl

    ws = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)["sorted data"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] != tray:
            continue
        pos, when = str(row[1]), row[2]
        d = when.date() if isinstance(when, dt.datetime) else when
        if d.year == 2016 and d.month == 12:
            d = d.replace(year=2015)
        out[("abcd".index(pos[0]), int(pos[1]) - 1)] = d
    if len(out) != ROWS * COLS:
        raise RuntimeError(f"tray {tray}: {len(out)} harvest rows, want 20")
    return out


def plants(
    gt_rgb: np.ndarray, image: str, g: Grid, occ: set[tuple[int, int]]
) -> list[Plant]:
    """One Plant per occupied pot. A pot's plant is the 25 px-dilated group
    nearest the pot's centre (distance from the centre to the group's nearest
    painted pixel). Excluded, with the reason kept:

    - ``touching``: two or more occupied pots have the same nearest group, so
      the annotation cannot say which pixels are whose (neighbours overlap);
    - ``not_found``: the nearest group is over a quarter pitch from the centre;
    - ``stray_group``: a group that is no pot's nearest lies within
      STRAY_PX of this plant, so part of this plant may be in it.
    """
    fg = foreground(gt_rgb)
    grouped = cv2.dilate(
        fg.astype(np.uint8), np.ones((GROUP_DILATE, GROUP_DILATE), np.uint8)
    )
    n, lab = cv2.connectedComponents(grouped)
    lab = np.where(fg, lab, 0)  # groups hold painted pixels only
    pix = {i: np.argwhere(lab == i)[:, ::-1].astype(np.float64) for i in range(1, n)}
    pix = {i: p for i, p in pix.items() if len(p)}
    pitch = float(np.mean(np.diff(g.gx)))
    nearest: dict[tuple[int, int], tuple[int, float]] = {}
    for rc in occ:
        cx, cy = g.centre(*rc)
        d = {
            i: float(np.min(np.hypot(p[:, 0] - cx, p[:, 1] - cy)))
            for i, p in pix.items()
        }
        i = min(d, key=d.get)
        nearest[rc] = (i, d[i])
    owners: dict[int, list[tuple[int, int]]] = {}
    for rc, (i, _) in nearest.items():
        owners.setdefault(i, []).append(rc)
    strays = [i for i in pix if i not in owners]
    out = []
    for rc in sorted(occ):
        i, d = nearest[rc]
        sel = lab == i
        u, k = np.unique(gt_rgb[sel], axis=0, return_counts=True)
        cols = {tuple(int(v) for v in c): int(m) for c, m in zip(u, k)}
        why = None
        if d > pitch / 4:
            why, sel, cols = "not_found", np.zeros_like(fg), {}
        elif len(owners[i]) > 1:
            why = "touching"
        else:
            mine = pix[i]
            for s in strays:
                gap = _min_dist(mine, pix[s])
                if gap < STRAY_PX:
                    why = "stray_group"
                    break
        out.append(Plant(image, *rc, sel, cols, why))
    return out


STRAY_PX = 100


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two point sets."""
    from scipy.spatial import cKDTree

    return float(cKDTree(a).query(b, k=1)[0].min())


def mask_band(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mask eroded and dilated by one pixel (3 x 3 square): the +-1 px
    boundary band of a hand-drawn outline."""
    k = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    return cv2.erode(m, k).astype(bool), cv2.dilate(m, k).astype(bool)


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
