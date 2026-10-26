"""PlantCV's `create_labels(mask, rois, roi_type="partial")`, computed locally.

What PlantCV 4.11.3 does (`plantcv/create_labels.py:45-55` and
`plantcv/_helpers.py:_roi_filter`, lines 513-533 for "partial"):

1. ``contours, hierarchy = findContours(mask, RETR_TREE, CHAIN_APPROX_NONE)``,
   once, on the original mask.
2. For each ROI ``i``, in order:

   a. ``base = drawContours(zeros, contours, -1, 255, FILLED, LINE_8,
      hierarchy)``: the mask redrawn.
   b. ``roi = fillPoly(zeros, [vstack(rois.contours[i][0])], 255)``: only the
      ROI's first contour counts.
   c. For EVERY contour ``c``, holes included: if ``fillPoly(zeros,
      [contour c], 255) AND roi`` is all zero, ``drawContours(base, contours,
      c, 0, FILLED, LINE_8, hierarchy)``, which erases ``c`` together with
      every contour nested in it.
   d. ``kept = findContours(base)``, then ``drawContours(labeled, kept, -1,
      i + 1, FILLED)`` with no hierarchy.

   On OpenCV 4.11 a FILLED drawContours is ONE ``fillPoly`` over all the
   contours it selects (`modules/imgproc/src/drawing.cpp:2536`): an even-odd
   fill plus every edge drawn as a line (`drawing.cpp:1288`). So in (d) a hole
   is NOT labelled, an island inside a hole is, and erasing a hole in (c) also
   erases the ring of plant pixels its contour runs through. Each step
   allocates and fills a full frame, and (c) does so once per contour: cost
   ROIs x contours x frame, over 900 s per call on a 2560x1920 tray with 6.6k
   contours.

The same result, per contour tree (an outer contour and every contour nested in
it). A tree lies inside its root's bounding box and two trees are 8-separated,
so their polygons are disjoint or nested only within one tree; an even-odd fill
over several trees is then the union of their separate fills, and every step
above acts on each tree independently:

- a tree with no contour overlapping ROI ``i`` is erased whole by (c): the
  erase draws exactly the contours (a) drew for it. It is skipped.
- a tree that is touched has (a), (c) and (d) replayed inside its root's
  bounding box, with PlantCV's own drawContours calls on its contours (same
  flags, the hierarchy renumbered to the tree), translated to the box.
- the overlap test (c) runs in the contour's own bounding box against the ROI
  rasterised in its own box, only for contours whose box meets the ROI's.
  Neither polygon is clipped, and fillPoly is exact under an integer shift.
- a tree with nothing erased gives the same labels for every ROI that touches
  it, so it is computed once.

Later ROIs overwrite earlier ones pixel by pixel, as in PlantCV, so an object
straddling two cells goes whole to the last one. Cost: O(frame) once plus, per
ROI, the area and contour count of the trees it touches. Pixel identity with
the PlantCV call is tested on synthetic masks and real tray masks
(tests/test_labels.py).
"""

from typing import Any

import cv2
import numpy as np
from plantcv import plantcv as pcv

_FILLED = -1
_LINE_8 = 8
_LEAF = np.full((1, 1, 4), -1, dtype=np.int32)


def _replayable(hierarchy: np.ndarray) -> bool:
    """Whether the per-tree replay is exact for this hierarchy.

    OpenCV walks children with ``while (next > 0)`` (`drawing.cpp:2523`), so a
    contour at index 0 is never reached as a child, and the replay renumbers
    each tree so its root is index 0. Both are safe only when every parent
    precedes its children, which findContours has always produced here; if it
    ever does not, the caller falls back to PlantCV itself.
    """
    parents = hierarchy[:, 3]
    return bool(np.all(parents < np.arange(len(parents))))


class _Forest:
    """The mask's contours grouped into trees, with per-tree geometry."""

    def __init__(self, mask: np.ndarray, contours: Any, hierarchy: np.ndarray) -> None:
        self.height, self.width = int(mask.shape[0]), int(mask.shape[1])
        self.contours = contours
        self.hier = hierarchy
        n = len(contours)
        pts = np.concatenate([c.reshape(-1, 2) for c in contours])
        starts = np.cumsum([0] + [len(c) for c in contours[:-1]])
        self.x0 = np.minimum.reduceat(pts[:, 0], starts)
        self.y0 = np.minimum.reduceat(pts[:, 1], starts)
        self.x1 = np.maximum.reduceat(pts[:, 0], starts)
        self.y1 = np.maximum.reduceat(pts[:, 1], starts)
        self.children: list[list[int]] = [[] for _ in range(n)]
        self.root = np.arange(n)
        for c in range(n):  # parents precede children (_replayable)
            p = int(hierarchy[c, 3])
            if p >= 0:
                self.children[p].append(c)
                self.root[c] = self.root[p]
        self._subtree: dict[int, list[int]] = {}
        self._sub_hier: dict[int, np.ndarray] = {}
        self._tree: dict[int, dict[str, Any]] = {}
        self._fill: dict[int, np.ndarray] = {}
        self._whole: dict[int, np.ndarray] = {}

    def subtree(self, c: int) -> list[int]:
        """`c` and every contour nested in it, in index order (`c` first)."""
        if c not in self._subtree:
            out, stack = [], [c]
            while stack:
                k = stack.pop()
                out.append(k)
                stack.extend(self.children[k])
            self._subtree[c] = sorted(out)
        return self._subtree[c]

    def sub_hierarchy(self, c: int) -> np.ndarray:
        """The hierarchy of `subtree(c)`, renumbered to positions in it, with
        `c` made a root: no parent and no siblings outside the list."""
        if c not in self._sub_hier:
            members = self.subtree(c)
            if len(members) == 1:
                self._sub_hier[c] = _LEAF
            else:
                m = np.asarray(members)
                rows = self.hier[m]
                at = np.minimum(np.searchsorted(m, rows), len(m) - 1)
                out = np.where((rows >= 0) & (m[at] == rows), at, -1).astype(np.int32)
                out[0, [0, 1, 3]] = -1
                self._sub_hier[c] = out[None]
        return self._sub_hier[c]

    def tree(self, r: int) -> dict[str, Any]:
        """Window and window-local contours for the tree rooted at `r`.

        The window is the root's bounding box plus a 1-px margin, clipped to
        the frame, so findContours inside it sees the same zero surround, or
        the same frame edge, as it does on the whole frame.
        """
        if r not in self._tree:
            wx0, wy0 = max(int(self.x0[r]) - 1, 0), max(int(self.y0[r]) - 1, 0)
            wx1 = min(int(self.x1[r]) + 2, self.width)
            wy1 = min(int(self.y1[r]) + 2, self.height)
            shift = np.array([wx0, wy0], dtype=np.int32)
            members = self.subtree(r)
            self._tree[r] = {
                "box": (wx0, wy0, wx1, wy1),
                "members": members,
                "local": {c: self.contours[c] - shift for c in members},
            }
        return self._tree[r]

    def contour_fill(self, c: int) -> np.ndarray:
        """`fillPoly` of contour `c` alone (step c's test), in its own box."""
        if c not in self._fill:
            x0, y0 = int(self.x0[c]), int(self.y0[c])
            buf = np.zeros(
                (int(self.y1[c]) - y0 + 1, int(self.x1[c]) - x0 + 1), np.uint8
            )
            pts = np.vstack(self.contours[c]) - np.array([x0, y0], dtype=np.int32)
            cv2.fillPoly(buf, [pts], 255)
            self._fill[c] = buf > 0
        return self._fill[c]

    def labels_for(self, r: int, erased: list[int]) -> np.ndarray:
        """Steps (a), (c), (d) for tree `r`, inside its window: the pixels
        create_labels gives this ROI's label."""
        if not erased and r in self._whole:
            return self._whole[r]
        t = self.tree(r)
        wx0, wy0, wx1, wy1 = t["box"]
        local = t["local"]
        win = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
        cv2.drawContours(
            win,
            [local[c] for c in t["members"]],
            -1,
            255,
            _FILLED,
            lineType=_LINE_8,
            hierarchy=self.sub_hierarchy(r),
        )
        for c in erased:
            cv2.drawContours(
                win,
                [local[k] for k in self.subtree(c)],
                0,
                0,
                _FILLED,
                lineType=_LINE_8,
                hierarchy=self.sub_hierarchy(c),
            )
        kept = cv2.findContours(win, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
        out = np.zeros_like(win)
        cv2.drawContours(out, kept, -1, 1, _FILLED)
        sel = out > 0
        if not erased:
            self._whole[r] = sel
        return sel


def _roi_raster(
    points: np.ndarray, height: int, width: int
) -> tuple[np.ndarray, int, int]:
    """Step (b): the ROI's fillPoly, returned as (raster, x0, y0).

    Rasterised in its own bounding box when the polygon lies inside the frame;
    otherwise on the full frame exactly as PlantCV does, since clipping at the
    frame edge can change which pixels fillPoly draws.
    """
    xs, ys = points[..., 0], points[..., 1]
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if x0 >= 0 and y0 >= 0 and x1 < width and y1 < height:
        buf = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        cv2.fillPoly(buf, [points - np.array([x0, y0], dtype=points.dtype)], 255)
        return buf > 0, x0, y0
    buf = np.zeros((height, width), np.uint8)
    cv2.fillPoly(buf, [points], 255)
    return buf > 0, 0, 0


def create_labels_partial(mask: np.ndarray, rois: Any) -> tuple[np.ndarray, int]:
    """Pixel-identical to `pcv.create_labels(mask, rois, roi_type="partial")`
    on PlantCV 4.11.3 / OpenCV 4.11, without its per-contour full-frame passes.

    Returns (labeled int32 mask, number of ROIs), as PlantCV does: the count is
    `len(rois.contours)`, empty ROIs included.
    """
    contours, hierarchy = cv2.findContours(
        np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )[-2:]
    h, w = int(mask.shape[0]), int(mask.shape[1])
    labeled = np.zeros((h, w), dtype=np.int32)
    n_rois = len(rois.contours)
    if not len(contours):
        return labeled, n_rois
    if not _replayable(hierarchy[0]):
        return pcv.create_labels(mask=mask, rois=rois, roi_type="partial")
    forest = _Forest(mask, contours, hierarchy[0])
    for i in range(n_rois):
        roi, rx0, ry0 = _roi_raster(np.vstack(rois.contours[i][0]), h, w)
        rx1, ry1 = rx0 + roi.shape[1] - 1, ry0 + roi.shape[0] - 1
        meets = np.flatnonzero(
            (forest.x1 >= rx0)
            & (forest.x0 <= rx1)
            & (forest.y1 >= ry0)
            & (forest.y0 <= ry1)
        )
        overlapping: set[int] = set()
        for c in meets.tolist():
            cx0, cy0 = int(forest.x0[c]), int(forest.y0[c])
            fill = forest.contour_fill(c)
            ix0, iy0 = max(cx0, rx0), max(cy0, ry0)
            ix1 = min(cx0 + fill.shape[1], rx1 + 1)
            iy1 = min(cy0 + fill.shape[0], ry1 + 1)
            a = fill[iy0 - cy0 : iy1 - cy0, ix0 - cx0 : ix1 - cx0]
            b = roi[iy0 - ry0 : iy1 - ry0, ix0 - rx0 : ix1 - rx0]
            if np.logical_and(a, b).any():
                overlapping.add(c)
        for r in sorted({int(forest.root[c]) for c in overlapping}):
            t = forest.tree(r)
            erased = [c for c in t["members"] if c not in overlapping]
            sel = forest.labels_for(r, erased)
            wx0, wy0, wx1, wy1 = t["box"]
            labeled[wy0:wy1, wx0:wx1][sel] = i + 1
    return labeled, n_rois
