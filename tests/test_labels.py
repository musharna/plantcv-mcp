"""The region labeller must equal PlantCV's create_labels(roi_type="partial").

`create_labels_partial` replaces PlantCV's call because PlantCV's costs ROIs x
contours x frame: a 4x5 grid on a 2560x1920 field tray took over 900 s. The
replacement is only acceptable if its labels are PlantCV's, pixel for pixel,
so every test here compares against the real `pcv.create_labels` on the same
input, and most also assert the behaviour that comparison protects (which
cell a straddling object goes to, that a hole is not labelled), so a broken
reference could not make them pass either.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from plantcv import plantcv as pcv
from plantcv.plantcv.classes import Objects

from plantcv_mcp import labels
from plantcv_mcp.labels import create_labels_partial
from plantcv_mcp.regions import build_regions, measure_regions

FIXTURES = Path(__file__).parent / "fixtures" / "aberystwyth"


@pytest.fixture(autouse=True)
def _quiet_plantcv(monkeypatch):
    monkeypatch.setattr(pcv.params, "debug", None)


def _img(mask: np.ndarray) -> np.ndarray:
    return np.zeros((*mask.shape, 3), np.uint8)


def _rects(mask, coord, h, w, spacing, nrows, ncols):
    return pcv.roi.multi_rect(
        img=_img(mask),
        coord=coord,
        h=h,
        w=w,
        spacing=spacing,
        nrows=nrows,
        ncols=ncols,
    )


def _same_as_plantcv(mask: np.ndarray, rois) -> np.ndarray:
    """Assert identity with PlantCV and return the labels."""
    want, want_n = pcv.create_labels(mask=mask.copy(), rois=rois, roi_type="partial")
    got, got_n = create_labels_partial(mask.copy(), rois)
    assert got_n == want_n
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    differ = np.argwhere(got != want)
    assert differ.size == 0, (
        f"{len(differ)} px differ from PlantCV, first at (row, col) "
        f"{differ[0].tolist()}: ours {got[tuple(differ[0])]}, PlantCV "
        f"{want[tuple(differ[0])]}"
    )
    return got


def test_straddling_object_goes_whole_to_the_last_cell():
    mask = np.zeros((100, 200), np.uint8)
    cv2.ellipse(mask, (100, 50), (30, 15), 0, 0, 360, 255, -1)  # crosses x=100
    cv2.circle(mask, (40, 50), 10, 255, -1)  # only in cell 0
    got = _same_as_plantcv(mask, _rects(mask, (0, 0), 100, 100, (100, 0), 1, 2))

    straddler = np.zeros_like(mask)
    cv2.ellipse(straddler, (100, 50), (30, 15), 0, 0, 360, 255, -1)
    assert set(np.unique(got[straddler > 0])) == {2}, "whole object, last cell"
    # Positive control: cell 0 does label what is only in cell 0.
    assert set(np.unique(got[35:66, 30:51][mask[35:66, 30:51] > 0])) == {1}


def test_hole_is_not_labelled_and_island_in_it_is():
    mask = np.zeros((120, 120), np.uint8)
    cv2.circle(mask, (60, 60), 40, 255, -1)
    cv2.circle(mask, (60, 60), 25, 0, -1)
    cv2.circle(mask, (60, 60), 8, 255, -1)  # island inside the hole
    got = _same_as_plantcv(mask, _rects(mask, (0, 0), 120, 120, (0, 0), 1, 1))

    assert got[60, 60] == 1, "island labelled"
    assert got[60, 75] == 0, "hole left open"
    assert got[60, 90] == 1, "ring labelled"


def test_hole_outside_the_roi_that_keeps_its_object():
    """PlantCV erases each contour that misses the ROI, holes included, and
    erasing a hole erases the ring of plant pixels its contour runs through.
    So an object kept by one cell loses the ring around a hole that lies in
    another cell."""
    mask = np.zeros((100, 220), np.uint8)
    cv2.rectangle(mask, (20, 20), (200, 80), 255, -1)
    cv2.circle(mask, (170, 50), 15, 0, -1)  # hole wholly in cell 1
    cv2.circle(mask, (50, 50), 12, 0, -1)  # hole wholly in cell 0
    _same_as_plantcv(mask, _rects(mask, (0, 0), 100, 110, (110, 0), 1, 2))

    only_cell0 = _same_as_plantcv(mask, _rects(mask, (0, 0), 100, 110, (0, 0), 1, 1))
    lost = (mask > 0) & (only_cell0 == 0)
    xs = np.nonzero(lost)[1]
    assert xs.size and xs.min() >= 150, "pixels lost only around the far hole"
    # Positive control: the near hole's ring survives in the cell that holds it.
    assert only_cell0[50, 50 - 13] == 1


def test_objects_outside_every_cell_and_empty_cells():
    mask = np.zeros((120, 300), np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)  # cell 0
    cv2.circle(mask, (280, 110), 8, 255, -1)  # outside every cell
    rois = _rects(mask, (10, 10), 80, 80, (90, 0), 1, 3)  # cells 1, 2 empty
    got = _same_as_plantcv(mask, rois)
    _, n = create_labels_partial(mask, rois)

    assert n == 3, "the count is the number of ROIs, empty ones included"
    assert set(np.unique(got)) == {0, 1}
    assert got[110, 280] == 0, "an object no ROI touches is not labelled"
    assert got[50, 50] == 1


def test_touching_cells_and_specks_on_their_border():
    mask = np.zeros((90, 180), np.uint8)
    mask[44:47, 59:62] = 255  # 3x3 speck across the x=60 border
    mask[0, 0] = mask[89, 179] = 255  # single pixels at the frame corners
    for x in range(5, 175, 7):
        mask[(x * 3) % 90, x] = 255
    _same_as_plantcv(mask, _rects(mask, (0, 0), 45, 60, (60, 45), 2, 3))


def test_circle_rois():
    mask = np.zeros((150, 300), np.uint8)
    cv2.ellipse(mask, (150, 75), (40, 20), 30, 0, 360, 255, -1)  # between circles
    cv2.circle(mask, (75, 75), 30, 255, 4)
    cv2.line(mask, (10, 140), (290, 10), 255, 1)  # one-pixel diagonal
    rois = pcv.roi.multi(
        img=_img(mask), coord=(75, 75), radius=74, spacing=(150, 0), nrows=1, ncols=2
    )
    got = _same_as_plantcv(mask, rois)
    assert set(np.unique(got)) == {0, 1, 2}


def test_roi_partly_outside_the_frame():
    """A polygon fillPoly must clip is rasterised on the full frame."""
    mask = np.zeros((80, 80), np.uint8)
    cv2.circle(mask, (10, 10), 8, 255, -1)
    cv2.circle(mask, (60, 60), 8, 255, -1)
    rois = Objects()
    pts = np.array([[[-30, -30]], [[-30, 25]], [[25, 25]], [[25, -30]]], np.int32)
    rois.append([pts], np.array([[[-1, -1, -1, -1]]], np.int32))
    got = _same_as_plantcv(mask, rois)
    assert got[10, 10] == 1 and got[60, 60] == 0


def test_empty_mask():
    mask = np.zeros((40, 40), np.uint8)
    got = _same_as_plantcv(mask, _rects(mask, (0, 0), 20, 20, (20, 20), 2, 2))
    assert not got.any()


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for _ in range(int(rng.integers(1, 12))):
        cx, cy = int(rng.integers(-5, w + 5)), int(rng.integers(-5, h + 5))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            r = int(rng.integers(5, 25))
            cv2.circle(m, (cx, cy), r, 255, -1)
            cv2.circle(m, (cx, cy), int(r * rng.uniform(0.3, 0.9)), 0, -1)
            if rng.random() < 0.6:
                cv2.circle(m, (cx, cy), max(1, r // 6), 255, -1)
        elif kind == 1:
            end = (cx + int(rng.integers(-50, 50)), cy + int(rng.integers(-50, 50)))
            cv2.line(m, (cx, cy), end, 255, int(rng.integers(1, 3)))
        elif kind == 2:
            ys, xs = rng.integers(0, h, 30), rng.integers(0, w, 30)
            m[ys, xs] = 255
        else:  # a patch of noise: many holes sharing one-pixel walls
            x, y = max(cx, 0), max(cy, 0)
            ph, pw = int(rng.integers(5, 30)), int(rng.integers(5, 30))
            patch = (rng.random((ph, pw)) < rng.uniform(0.3, 0.8)) * 255
            sub = m[y : y + ph, x : x + pw]
            sub[...] = np.maximum(sub, patch[: sub.shape[0], : sub.shape[1]])
    return m


@pytest.mark.parametrize("seed", range(30))
def test_random_masks_and_grids(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(30, 100)), int(rng.integers(30, 120))
    mask = _random_mask(rng, h, w)
    nrows, ncols = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    if seed % 2:
        cw, ch = w // ncols, h // nrows  # touching or overlapping rectangles
        sx = cw + int(rng.integers(-3, 4))
        sy = ch + int(rng.integers(-3, 4))
        rois = _rects(
            mask,
            (0, 0),
            ch,
            cw,
            (max(sx, 0), max(sy, 0)),
            nrows,
            ncols,
        )
    else:
        # largest radius whose grid still fits the frame at spacing <= 2r
        r = max(2, min(h // (2 * nrows), w // (2 * ncols)) - 1)
        rois = pcv.roi.multi(
            img=_img(mask),
            coord=(r, r),
            radius=r,
            spacing=(int(rng.integers(r, 2 * r + 1)), int(rng.integers(r, 2 * r + 1))),
            nrows=nrows,
            ncols=ncols,
        )
    _same_as_plantcv(mask, rois)


def _crop_2015_12_18_cells(mask):
    # the benchmark agent's two cells: 420 px wide, 23 px apart
    return _rects(mask, (0, 0), 420, 210, (233, 0), 1, 2)


def _crop_2015_12_18_touching(mask):
    return _rects(mask, (0, 0), 210, 150, (150, 210), 2, 3)


def _crop_2016_01_17_circles(mask):
    return pcv.roi.multi(
        img=_img(mask), coord=(100, 100), radius=99, spacing=(200, 0), nrows=1, ncols=2
    )


def _crop_2016_01_17_touching(mask):
    return _rects(mask, (0, 0), 100, 200, (200, 100), 2, 2)


@pytest.mark.parametrize(
    ("name", "grid"),
    [
        ("tray032_2015-12-18_mask_crop.png", _crop_2015_12_18_cells),
        ("tray032_2015-12-18_mask_crop.png", _crop_2015_12_18_touching),
        ("tray032_2016-01-17_mask_crop.png", _crop_2016_01_17_circles),
        ("tray032_2016-01-17_mask_crop.png", _crop_2016_01_17_touching),
    ],
)
def test_real_tray_masks(name, grid):
    mask = cv2.imread(str(FIXTURES / name), cv2.IMREAD_GRAYSCALE)
    holes = int(
        (
            cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1][0][
                :, 3
            ]
            >= 0
        ).sum()
    )
    assert holes > 50, "the fixture is the many-holed real mask"
    rois = grid(mask)
    got = _same_as_plantcv(mask, rois)
    assert len(set(np.unique(got)) - {0}) >= 2, "more than one cell labels plant"


def test_unreplayable_hierarchy_falls_back_to_plantcv(monkeypatch):
    child_first = np.array([[-1, -1, -1, 1], [-1, -1, 0, -1]], np.int32)
    real = cv2.findContours(
        cv2.imread(
            str(FIXTURES / "tray032_2016-01-17_mask_crop.png"), cv2.IMREAD_GRAYSCALE
        ),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )[1][0]
    assert not labels._replayable(child_first)
    assert labels._replayable(real)

    calls = []
    real_create_labels = pcv.create_labels

    def spy(**kwargs):
        calls.append(kwargs["roi_type"])
        return real_create_labels(**kwargs)

    mask = np.zeros((60, 60), np.uint8)
    cv2.circle(mask, (30, 30), 12, 255, -1)
    rois = _rects(mask, (0, 0), 60, 60, (0, 0), 1, 1)
    monkeypatch.setattr(pcv, "create_labels", spy)
    create_labels_partial(mask, rois)
    assert calls == [], "a replayable hierarchy never reaches PlantCV"
    monkeypatch.setattr(labels, "_replayable", lambda h: False)
    got, n = create_labels_partial(mask, rois)
    assert calls == ["partial"]
    assert n == 1 and got[30, 30] == 1


def test_measure_regions_no_longer_calls_create_labels(monkeypatch):
    """The per-region path must not reach PlantCV's labeller (ROIs x contours x
    frame); the work-count ratchet catches the same regression by count."""

    def refuse(**kwargs):
        raise AssertionError("pcv.create_labels reached from measure_regions")

    mask = np.zeros((100, 200), np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    cv2.circle(mask, (150, 50), 30, 255, -1)
    img = _img(mask)
    img[mask > 0] = (40, 200, 40)
    regions = build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=1,
        ncols=2,
        coord=(0, 0),
        height=100,
        width=100,
        spacing=(100, 0),
    )
    monkeypatch.setattr(pcv, "create_labels", refuse)
    out = measure_regions(img, mask, regions)
    areas = [r["traits"]["area"]["value"] for r in out]
    assert areas[0] < areas[1], "both cells measured, in order"
