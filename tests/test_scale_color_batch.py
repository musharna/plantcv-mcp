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
