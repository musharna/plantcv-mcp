"""Scale calibration, colour correction, and unattended batch.

Each of these was deferred once with a stated reason, and each reason was removed
by finding a way to test it against ground truth rather than by lowering the bar:

* scale — a synthetic disc of KNOWN diameter, so px_per_mm has a right answer.
* colour — a synthetic Macbeth chart plus a KNOWN distortion, so "corrected" means
  measurably closer to the undistorted original, not merely "different".
* batch — the assertion that matters is not that it measures, but that it REFUSES
  to measure the images whose guards fire.
"""

import cv2
import numpy as np
import pytest

from plantcv_mcp.batch import MAX_BATCH, BatchTooLargeError, measure_batch
from plantcv_mcp.color import ColorCardNotFoundError, correct_color
from plantcv_mcp.scale import MarkerNotFoundError, calibrate_scale

FIXTURE = "tests/fixtures/multi_specimen.png"

# 24-patch Macbeth ColorChecker, sRGB, 4 rows x 6 columns.
MACBETH = [
    (115, 82, 68), (194, 150, 130), (98, 122, 157), (87, 108, 67),
    (133, 128, 177), (103, 189, 170), (214, 126, 44), (80, 91, 166),
    (193, 90, 99), (94, 60, 108), (157, 188, 64), (224, 163, 46),
    (56, 61, 150), (70, 148, 73), (175, 54, 60), (231, 199, 31),
    (187, 86, 149), (8, 133, 161), (243, 243, 242), (200, 200, 200),
    (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]  # fmt: skip


def _disc_image(diameter_px=80, canvas=300):
    """A dark disc of exactly `diameter_px` on a light field — known ground truth."""
    img = np.full((canvas, canvas, 3), 240, np.uint8)
    cv2.circle(img, (canvas // 2, canvas // 2), diameter_px // 2, (30, 30, 30), -1)
    return img


def _color_card(chip=60, gap=8, margin=40):
    img = np.full((560, 760, 3), 30, np.uint8)
    for i, rgb in enumerate(MACBETH):
        r, c = divmod(i, 6)
        y = margin + r * (chip + gap)
        x = margin + c * (chip + gap)
        img[y : y + chip, x : x + chip] = rgb[::-1]  # BGR
    return img


def _mean_abs_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


# --------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------


def test_scale_recovers_a_known_marker_size_from_a_tight_crop():
    """This is the case that defeated pcv.report_size_marker_area: with a tight ROI
    it reported 348 px for an 80 px disc, a silent 4.35x scale error. Cropping
    before thresholding removes the mechanism, so a tight region is now correct."""
    img = _disc_image(diameter_px=80)
    est = calibrate_scale(img, x=100, y=100, w=100, h=100, marker_length_mm=20.0)

    assert est.marker_length_px == pytest.approx(80, abs=2)
    assert est.px_per_mm == pytest.approx(4.0, rel=0.03)
    assert not est.warnings

    # The whole frame must agree with the tight crop, or the crop is doing something
    # other than isolating the marker.
    whole = calibrate_scale(img, x=0, y=0, w=300, h=300, marker_length_mm=20.0)
    assert whole.marker_length_px == est.marker_length_px


def test_wrong_polarity_is_flagged_rather_than_silently_rescaling():
    """The inverted case selects the background, which fills the crop. Coverage
    cannot detect it — measured, both polarities give crop_fraction 0.50 — so the
    guard is edge contact."""
    img = _disc_image(diameter_px=80)
    good = calibrate_scale(img, 100, 100, 100, 100, 20.0, object_type="dark")
    bad = calibrate_scale(img, 100, 100, 100, 100, 20.0, object_type="light")

    assert bad.crop_fraction == pytest.approx(good.crop_fraction, abs=0.02), (
        "this test is only meaningful while coverage CANNOT discriminate"
    )
    assert "marker_touches_crop_edge" in [w.code for w in bad.warnings]
    # Positive control in the same test: the correct polarity stays clean.
    assert "marker_touches_crop_edge" not in [w.code for w in good.warnings]


def test_scale_rejects_nonsense_input():
    img = _disc_image()
    for mm in (0, -1):
        with pytest.raises(ValueError):
            calibrate_scale(img, 100, 100, 100, 100, mm)
    with pytest.raises(ValueError):
        calibrate_scale(img, 5000, 5000, 50, 50, 20.0)  # crop outside the image
    with pytest.raises(MarkerNotFoundError):
        calibrate_scale(np.full((100, 100, 3), 240, np.uint8), 0, 0, 100, 100, 20.0)
    # Positive control: a valid call still succeeds.
    assert calibrate_scale(img, 100, 100, 100, 100, 20.0).px_per_mm > 0


# --------------------------------------------------------------------------
# colour correction
# --------------------------------------------------------------------------


def test_correction_moves_a_distorted_image_back_toward_the_original():
    base = _color_card()
    # A known warm cast: boost red, cut blue.
    distorted = np.clip(
        base.astype(np.float32) * np.array([0.75, 1.0, 1.30]), 0, 255
    ).astype(np.uint8)

    corrected = correct_color(distorted)

    before = _mean_abs_diff(distorted, base)
    after = _mean_abs_diff(corrected, base)
    assert after < before, f"correction made it worse: {before:.2f} -> {after:.2f}"
    # Not merely "different from distorted" — meaningfully closer to the truth.
    assert after < before * 0.75


def test_missing_color_card_raises_instead_of_returning_the_image_unchanged():
    """Silently skipping correction would hand back colour traits that look
    corrected and are not."""
    plain = cv2.imread(FIXTURE)
    with pytest.raises(ColorCardNotFoundError):
        correct_color(plain)
    # Positive control in the same test: an image WITH a card still corrects.
    assert correct_color(_color_card()) is not None


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------


def _write(tmp_path, name, img):
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


def test_batch_measures_good_images_and_refuses_the_rest(tmp_path):
    """The load-bearing property: a batch never returns a number the server could
    not validate."""
    blank = _write(tmp_path, "blank.png", np.full((200, 200, 3), 255, np.uint8))
    good = FIXTURE

    out = measure_batch([good, blank], channel="a", method="otsu")

    assert out["summary"]["submitted"] == 2
    assert out["summary"]["measured"] == 1
    assert out["summary"]["needs_review"] == 1
    assert blank in out["summary"]["review_paths"]

    by_path = {r["image_path"]: r for r in out["results"]}
    assert by_path[good]["measured"] is True
    assert by_path[good]["traits"]["area"]["value"] > 0

    assert by_path[blank]["measured"] is False
    assert by_path[blank]["traits"] is None, "a refused image must carry NO traits"
    assert "empty_mask" in by_path[blank]["refused_because"]


def test_batch_refuses_an_inverted_mask_rather_than_measuring_the_background(tmp_path):
    """Channel 's' with the default polarity yields 96% coverage on the fixture.
    Unattended, that must not become a number."""
    out = measure_batch([FIXTURE], channel="s", method="otsu", object_type="dark")
    entry = out["results"][0]

    assert entry["measured"] is False
    assert entry["traits"] is None
    assert "implausible_coverage" in entry["refused_because"]

    # Positive control: the correct polarity on the same image DOES measure.
    ok = measure_batch([FIXTURE], channel="s", method="otsu", object_type="light")
    assert ok["results"][0]["measured"] is True


def test_batch_keeps_advisory_warnings_without_withholding_traits():
    """multi_specimen means 'this number describes a group', not 'this number is
    invalid'. It must be surfaced but must not suppress the measurement."""
    out = measure_batch([FIXTURE], channel="a", method="otsu")
    entry = out["results"][0]
    assert entry["measured"] is True
    assert "multi_specimen" in [w["code"] for w in entry["warnings"]]
    assert entry["traits"] is not None


def test_batch_survives_one_bad_path_without_losing_the_others(tmp_path):
    out = measure_batch(
        [FIXTURE, str(tmp_path / "does_not_exist.png")], channel="a", method="otsu"
    )
    assert out["summary"]["measured"] == 1
    assert out["summary"]["needs_review"] == 1
    bad = next(r for r in out["results"] if not r["measured"])
    assert bad["refused_because"], "the failure reason must be reported, not swallowed"


def test_batch_size_and_emptiness_are_bounded():
    with pytest.raises(ValueError):
        measure_batch([], channel="a", method="otsu")
    with pytest.raises(BatchTooLargeError):
        measure_batch([FIXTURE] * (MAX_BATCH + 1), channel="a", method="otsu")
    # Positive control: a batch at the boundary is accepted (1 image).
    assert (
        measure_batch([FIXTURE], channel="a", method="otsu")["summary"]["submitted"]
        == 1
    )


def test_batch_passes_scale_and_analyses_through():
    out = measure_batch(
        [FIXTURE],
        channel="a",
        method="otsu",
        analyses=("size", "color"),
        px_per_mm=10.0,
    )
    traits = out["results"][0]["traits"]
    assert traits["area"]["unit"] == "mm2"
    assert "hue_circular_mean" in traits
    assert out["recipe"]["px_per_mm"] == 10.0


def test_a_rotated_square_marker_is_measured_by_its_side_not_its_bbox():
    """A 45deg-rotated square's axis-aligned bbox is sqrt(2)x the side (~41%
    overstated scale). The marker's length must come from rotation-invariant
    geometry, not the bbox."""
    import cv2

    canvas = np.full((300, 300, 3), 240, np.uint8)
    corners = cv2.boxPoints(((150.0, 150.0), (100.0, 100.0), 45.0)).astype(np.int32)
    cv2.fillPoly(canvas, [corners], (30, 30, 30))
    est = calibrate_scale(canvas, 40, 40, 220, 220, marker_length_mm=20.0)
    assert est.marker_length_px == pytest.approx(100, abs=3)
    assert est.px_per_mm == pytest.approx(5.0, rel=0.04)


def test_an_elongated_rotated_marker_still_trips_marker_not_round():
    """A 120x20 bar rotated 45deg has a nearly SQUARE axis-aligned bbox, so a
    bbox-ratio roundness check cannot see it. The rotated rect can."""
    import cv2

    canvas = np.full((300, 300, 3), 240, np.uint8)
    corners = cv2.boxPoints(((150.0, 150.0), (120.0, 20.0), 45.0)).astype(np.int32)
    cv2.fillPoly(canvas, [corners], (30, 30, 30))
    est = calibrate_scale(canvas, 30, 30, 240, 240, marker_length_mm=20.0)
    assert "marker_not_round" in [w.code for w in est.warnings]
    assert est.marker_length_px == pytest.approx(120, abs=4)


def test_a_partially_out_of_frame_crop_is_refused_not_silently_clamped():
    """A crop hanging over the edge used to be clamped quietly; a marker cut by
    the clamped edge then produced a plausible but wrong scale. Refuse instead."""
    img = _disc_image()  # 300x300
    for box in [(250, 100, 100, 100), (-20, 100, 100, 100), (100, 260, 100, 100)]:
        with pytest.raises(ValueError, match="300x300"):
            calibrate_scale(img, *box, marker_length_mm=20.0)
    # Positive control: the same-size crop fully inside still calibrates.
    assert calibrate_scale(img, 100, 100, 100, 100, 20.0).px_per_mm > 0


# --- batch dogfood 2026-08-29: six findings from the first real-photo batch ---


def _speckled(tmp_path, name="speck.png", n=60):
    """One plant blob plus n dark 18x18 specks (each > fill_size=200): the
    sorghum-in-chamber case that batch measured as one 650k-px 'plant'."""
    img = np.full((400, 400, 3), 200, np.uint8)
    img[180:220, 180:220] = (30, 30, 30)
    rng = np.random.default_rng(5)
    placed = 0
    while placed < n:
        y, x = rng.integers(0, 382, 2)
        if 150 <= y <= 240 and 150 <= x <= 240:
            continue
        if (img[y : y + 18, x : x + 18, 0] == 30).any():
            continue
        img[y : y + 18, x : x + 18] = (30, 30, 30)
        placed += 1
    return _write(tmp_path, name, img)


def test_batch_reports_timing_and_stops_at_the_time_budget(tmp_path):
    disc = _write(tmp_path, "d1.png", _disc_image())
    disc2 = _write(tmp_path, "d2.png", _disc_image(60))
    disc3 = _write(tmp_path, "d3.png", _disc_image(40))
    out = measure_batch(
        [disc, disc2, disc3], channel="l", method="otsu", max_seconds=0.0
    )
    rows = out["results"]
    assert rows[0]["measured"] is True and rows[0]["seconds"] > 0
    assert [r["measured"] for r in rows[1:]] == [False, False]
    assert all("time budget" in r["refused_because"] for r in rows[1:])
    assert all(r["seconds"] is None for r in rows[1:])
    assert out["summary"]["not_run"] == 2
    assert out["summary"]["not_run_paths"] == [disc2, disc3]
    assert out["summary"]["needs_review"] == 0  # not run is not "needs review"
    assert out["elapsed_s"] >= rows[0]["seconds"]
    assert out["recipe"]["max_seconds"] == 0.0
    # Positive control: with the default budget all three run and time is reported.
    full = measure_batch([disc, disc2, disc3], channel="l", method="otsu")
    assert full["summary"]["not_run"] == 0 and full["summary"]["measured"] == 3
    assert sum(r["seconds"] for r in full["results"]) <= full["elapsed_s"] + 1e-6


def test_batch_refuses_a_bad_recipe_before_running_anything():
    """channel='zz' used to run every image and return N identical
    UnknownChannelError rows; a recipe error is one error, raised up front."""
    from plantcv_mcp.measurement import UnknownAnalysisError
    from plantcv_mcp.segmentation import (
        UnknownChannelError,
        UnknownMethodError,
        UnknownObjectTypeError,
    )

    with pytest.raises(UnknownChannelError):
        measure_batch([FIXTURE], channel="zz", method="otsu")
    with pytest.raises(UnknownMethodError):
        measure_batch([FIXTURE], channel="a", method="nope")
    with pytest.raises(UnknownObjectTypeError):
        measure_batch([FIXTURE], channel="a", method="otsu", object_type="up")
    with pytest.raises(UnknownAnalysisError):
        measure_batch([FIXTURE], channel="a", method="otsu", analyses=("shape",))
    with pytest.raises(ValueError, match="max_seconds"):
        measure_batch([FIXTURE], channel="a", method="otsu", max_seconds=-1)


def test_batch_dedupes_paths_and_says_so():
    out = measure_batch([FIXTURE, FIXTURE], channel="a", method="otsu")
    assert len(out["results"]) == 1
    assert out["summary"]["submitted"] == 2
    assert out["summary"]["unique"] == 1
    assert out["summary"]["duplicates_dropped"] == [FIXTURE]


def test_batch_summary_counts_rows_with_advisories():
    out = measure_batch([FIXTURE], channel="a", method="otsu")
    assert out["summary"]["measured"] == 1
    assert out["summary"]["with_advisories"] == 1
    assert out["summary"]["advisory_counts"]["multi_specimen"] == 1


def test_batch_withholds_traits_from_a_noisy_segmentation(tmp_path):
    speck = _speckled(tmp_path)
    clean = _write(tmp_path, "clean.png", _disc_image())
    out = measure_batch([speck, clean], channel="l", method="otsu")
    by_path = {r["image_path"]: r for r in out["results"]}
    row = by_path[speck]
    assert row["measured"] is False and row["traits"] is None
    assert "noisy_segmentation" in row["refused_because"]
    assert row["component_count"] >= 50
    msg = next(
        w["message"] for w in row["warnings"] if w["code"] == "noisy_segmentation"
    )
    assert "components" in msg and "refine" in msg
    # Positive control in the same batch: the clean disc is measured.
    assert by_path[clean]["measured"] is True


def test_batch_measures_each_plant_when_given_a_grid():
    """The four-view panel came back as ONE area with a multi_specimen
    advisory; with the grid it comes back as four rows, like measure_regions."""
    out = measure_batch([FIXTURE], channel="a", method="otsu", nrows=2, ncols=2)
    row = out["results"][0]
    assert row["measured"] is True
    assert row["traits"] is None
    assert out["recipe"]["regions"] == {"mode": "auto_grid", "nrows": 2, "ncols": 2}
    regions = row["regions"]
    assert len(regions) == 4
    assert [r["measured"] for r in regions] == [True] * 4
    assert all(r["traits"]["area"]["value"] > 0 for r in regions)
    assert row["regions_measured"] == 4
    # multi_specimen is no longer an advisory on a row that was measured per plant
    assert "multi_specimen" not in [w["code"] for w in row["warnings"]]
    # a bad grid is a recipe error, raised before anything runs
    with pytest.raises(ValueError, match="nrows"):
        measure_batch([FIXTURE], channel="a", method="otsu", nrows=0, ncols=2)


# --- panel audit of 1.5.1 (2026-08-29) ---


def _late_germination_tray(tmp_path, name="well96.png"):
    """10 germinated wells (r=25) and 86 late ones (r=7) on a 10x10 plate: 100
    components, 10 major, 90 minor, largest 6% of the mask. Whole-mask
    is_noisy says texture; with a grid every well is a real plant."""
    tray = np.full((1100, 1100, 3), 200, np.uint8)
    rng = np.random.default_rng(2)
    big = {tuple(p) for p in rng.integers(0, 10, (10, 2))}
    for r in range(10):
        for c in range(10):
            cv2.circle(
                tray,
                (55 + c * 110, 55 + r * 110),
                25 if (r, c) in big else 7,
                (30, 30, 30),
                -1,
            )
    return _write(tmp_path, name, tray)


def test_batch_with_a_grid_measures_a_late_germination_tray(tmp_path):
    p = _late_germination_tray(tmp_path)
    out = measure_batch(
        [p], channel="l", method="otsu", fill_size=50, nrows=10, ncols=10
    )
    row = out["results"][0]
    assert row["measured"] is True
    assert row["regions_measured"] >= 90
    assert "noisy_segmentation" in [
        w["code"] for w in row["warnings"]
    ]  # kept as advice
    # Positive control: without a grid the same mask is still withheld.
    blocked = measure_batch([p], channel="l", method="otsu", fill_size=50)
    assert blocked["results"][0]["measured"] is False
    assert "noisy_segmentation" in blocked["results"][0]["refused_because"]


def test_batch_grid_withholds_rows_whose_object_exceeds_the_cell(tmp_path):
    """Interactive measure_regions keeps the numbers next to the numbered
    overlay; the batch has no overlay, so a row the guard has already called a
    merge must not come back measured."""
    img = np.full((100, 300, 3), 200, np.uint8)
    cv2.ellipse(
        img, (100, 50), (75, 20), 0, 0, 360, (30, 30, 30), -1
    )  # spans cells 0/1
    cv2.circle(img, (250, 50), 25, (30, 30, 30), -1)  # inside cell 2
    p = _write(tmp_path, "spill.png", img)
    out = measure_batch(
        [p],
        channel="l",
        method="otsu",
        mode="rect_grid",
        nrows=1,
        ncols=3,
        coord=(0, 0),
        height=100,
        width=100,
        spacing=(100, 0),
    )
    rows = out["results"][0]["regions"]
    exceeding = [
        r
        for r in rows
        if any(w["code"] == "object_exceeds_region" for w in r["warnings"])
    ]
    assert exceeding, "fixture must produce a spilling cell"
    assert all(r["measured"] is False and r["traits"] is None for r in exceeding)
    assert all("object_exceeds_region" in r["reason"] for r in exceeding)
    assert rows[2]["measured"] is True and rows[2]["traits"]["area"]["value"] > 0


def test_batch_refuses_bad_grid_geometry_before_running_anything():
    from plantcv_mcp.regions import MAX_REGIONS

    with pytest.raises(ValueError, match="mode"):
        measure_batch(
            [FIXTURE], channel="a", method="otsu", nrows=2, ncols=2, mode="hex_grid"
        )
    with pytest.raises(ValueError, match="radius"):
        measure_batch([FIXTURE], channel="a", method="otsu", nrows=2, ncols=2, radius=0)
    with pytest.raises(ValueError, match="ncols"):
        measure_batch([FIXTURE], channel="a", method="otsu", nrows=2)
    with pytest.raises(ValueError, match="nrows"):
        measure_batch([FIXTURE], channel="a", method="otsu", ncols=2)
    with pytest.raises(ValueError, match=str(MAX_REGIONS)):
        measure_batch([FIXTURE], channel="a", method="otsu", nrows=MAX_REGIONS, ncols=2)


def test_batch_keeps_measuring_a_tray_shaped_mask(tmp_path):
    """The not-noisy side of is_noisy: a 30-plant tray with a 3x size spread
    (28-36 components on the real arabidopsis trays) must stay measurable, or
    a tightened constant would refuse every tray while every test stayed green."""
    tray = np.full((700, 700, 3), 200, np.uint8)
    rng = np.random.default_rng(4)
    for r in range(6):
        for c in range(5):
            cv2.circle(
                tray,
                (70 + c * 130, 60 + r * 110),
                int(rng.integers(12, 36)),
                (30, 30, 30),
                -1,
            )
    p = _write(tmp_path, "tray30.png", tray)
    out = measure_batch([p], channel="l", method="otsu", fill_size=50)
    row = out["results"][0]
    assert row["component_count"] == 30
    assert row["measured"] is True
    assert "noisy_segmentation" not in [w["code"] for w in row["warnings"]]


def test_a_marker_that_fills_its_crop_is_flagged():
    """marker_fills_crop had no test; a crop that is all 'marker' is a crop of
    background under the wrong polarity."""
    canvas = np.full((200, 200, 3), 240, np.uint8)
    canvas[52:148, 52:148] = 30  # 96x96 dark block inside a 100x100 crop: 92%
    est = calibrate_scale(canvas, 50, 50, 100, 100, marker_length_mm=20.0)
    codes = [w.code for w in est.warnings]
    assert "marker_fills_crop" in codes
    # Positive control: a real marker inside its crop is not flagged.
    clean = calibrate_scale(_disc_image(), 100, 100, 100, 100, marker_length_mm=20.0)
    assert "marker_fills_crop" not in [w.code for w in clean.warnings]


# --- panel audit of 1.5.4 (2026-08-30) -------------------------------------


def _noisy_scene(tmp_path, plant_px=90, name="noisy.png"):
    """The calibrated noisy mask from test_diagnostics as a photo: one
    plant_px-square plant and 60 18x18 specks. At 90 px the plant is a third
    of the mask, so is_noisy fires on the whole frame."""
    rng = np.random.default_rng(3)
    img = np.full((600, 600, 3), 200, np.uint8)
    occupied = np.zeros((600, 600), bool)
    img[100 : 100 + plant_px, 100 : 100 + plant_px] = 30
    occupied[100 : 100 + plant_px, 100 : 100 + plant_px] = True
    placed = 0
    while placed < 60:
        y, x = rng.integers(0, 600 - 18, 2)
        if 80 <= y <= 100 + plant_px + 2 and 80 <= x <= 100 + plant_px + 2:
            continue
        if occupied[y - 2 : y + 20, x - 2 : x + 20].any():
            continue
        img[y : y + 18, x : x + 18] = 30
        occupied[y : y + 18, x : x + 18] = True
        placed += 1
    return _write(tmp_path, name, img)


def test_batch_grid_does_not_explain_a_noisy_mask(tmp_path):
    """1.5.2 demoted noisy_segmentation for ANY grid, on the premise that the
    per-cell floor guards each cell. It guards near-empty cells only: the
    calibrated noisy scene under a 1x2 grid came back as two measured plants
    of 1,620 and 2,592 px — clusters of specks. A grid explains components
    only when there are about as many objects as cells."""
    p = _noisy_scene(tmp_path)
    blocked = measure_batch([p], channel="l", method="otsu", fill_size=1)
    assert blocked["results"][0]["measured"] is False  # fixture is noisy
    for nrows, ncols in ((1, 2), (2, 2)):
        out = measure_batch(
            [p], channel="l", method="otsu", fill_size=1, nrows=nrows, ncols=ncols
        )
        row = out["results"][0]
        assert row["measured"] is False, (nrows, ncols)
        assert "noisy_segmentation" in row["refused_because"]
        assert "grid" in row["refused_because"]
        assert out["summary"]["needs_review"] == 1
    # Positive control: the late-germination tray (100 objects, 100 cells)
    # is still measured with its grid.
    tray = _late_germination_tray(tmp_path)
    out = measure_batch(
        [tray], channel="l", method="otsu", fill_size=50, nrows=10, ncols=10
    )
    assert out["results"][0]["measured"] is True
    assert out["results"][0]["regions_measured"] >= 90


def _dense_tray(tmp_path, name="dense.png"):
    """Two discs each filling its 1x2 cell: 72% of the frame is plant."""
    img = np.full((200, 400, 3), 200, np.uint8)
    cv2.circle(img, (100, 100), 96, (30, 30, 30), -1)
    cv2.circle(img, (300, 100), 96, (30, 30, 30), -1)
    return _write(tmp_path, name, img)


def test_batch_grid_measures_a_dense_tray_and_still_refuses_it_inverted(tmp_path):
    """Two discs filling their cells are 72% of the frame; the whole-frame
    implausible_coverage block ran before the grid and refused a valid tray.
    With two or more cells the grid itself catches an inverted mask: the
    background is one object spanning every cell."""
    p = _dense_tray(tmp_path)
    cells = {
        "mode": "rect_grid",
        "nrows": 1,
        "ncols": 2,
        "coord": (0, 0),
        "height": 200,
        "width": 200,
        "spacing": (200, 0),
    }
    out = measure_batch([p], channel="l", method="otsu", **cells)
    row = out["results"][0]
    assert row["measured"] is True
    assert row["mask_fraction"] > 0.5
    assert row["regions_measured"] == 2
    assert "implausible_coverage" in [w["code"] for w in row["warnings"]]
    # Positive control: the same tray thresholded the wrong way.
    inv = measure_batch([p], channel="l", method="otsu", object_type="light", **cells)
    assert inv["results"][0]["measured"] is False
    assert inv["summary"]["needs_review"] == 1


def test_batch_refuses_an_image_whose_every_region_was_withheld(tmp_path):
    """One ellipse spanning both cells: both rows object_exceeds_region, yet
    the image was measured=True and the summary said measured 1, needs_review
    0. Nobody looks at a batch; an image with no measured row is a review."""
    img = np.full((100, 200, 3), 200, np.uint8)
    cv2.ellipse(img, (100, 50), (90, 20), 0, 0, 360, (30, 30, 30), -1)
    p = _write(tmp_path, "spill.png", img)
    out = measure_batch(
        [p],
        channel="l",
        method="otsu",
        mode="rect_grid",
        nrows=1,
        ncols=2,
        coord=(0, 0),
        height=100,
        width=100,
        spacing=(100, 0),
    )
    row = out["results"][0]
    assert row["measured"] is False
    assert "0 of 2" in row["refused_because"]
    assert "object_exceeds_region" in row["refused_because"]
    assert row["regions"] is not None  # the per-cell reasons travel with it
    assert out["summary"]["measured"] == 0
    assert out["summary"]["needs_review"] == 1
    assert out["summary"]["review_paths"] == [p]


@pytest.mark.parametrize(
    "kw",
    [
        {"mode": "rect_grid"},
        {"mode": "bogus"},
        {"radius": -5},
        {"coord": (0, 0), "height": 100, "width": 100, "spacing": (100, 0)},
    ],
)
def test_batch_grid_arguments_without_a_grid_are_refused(tmp_path, kw):
    """Every one of these ran a whole-frame measurement with no error: the
    validator only looked at grid arguments once nrows/ncols were given."""
    img = np.full((300, 300, 3), 200, np.uint8)
    cv2.circle(img, (150, 150), 40, (30, 30, 30), -1)
    p = _write(tmp_path, "one.png", img)
    with pytest.raises(ValueError, match="nrows"):
        measure_batch([p], channel="l", method="otsu", **kw)
    # Positive control: without them the image measures.
    assert measure_batch([p], channel="l", method="otsu")["results"][0]["measured"]


def test_batch_dedupes_the_same_file_under_different_spellings(tmp_path):
    img = np.full((300, 300, 3), 200, np.uint8)
    cv2.circle(img, (150, 150), 40, (30, 30, 30), -1)
    p = _write(tmp_path, "one.png", img)
    link = str(tmp_path / "link.png")
    (tmp_path / "link.png").symlink_to(p)
    dotted = f"{tmp_path}/./one.png"  # pathlib would fold the dot away
    out = measure_batch([p, dotted, link, p], channel="l", method="otsu")
    assert len(out["results"]) == 1
    assert out["summary"]["unique"] == 1
    assert out["summary"]["duplicates_dropped"] == [dotted, link, p]
