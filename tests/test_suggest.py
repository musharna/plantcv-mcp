from pathlib import Path

import cv2
import numpy as np
import pytest

from plantcv_mcp.segmentation import UnknownChannelError
from plantcv_mcp.suggest import (
    GREEN_FLOOR,
    GREENNESS_MARGIN,
    colorspace_sheet,
    polarity_report,
    threshold_sheet,
)


def _img():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    return img


def test_colorspace_sheet_returns_exact_shape():
    sheet = colorspace_sheet(_img())
    assert sheet.ndim == 3
    assert sheet.shape == (275, 600, 3)


def test_threshold_sheet_returns_exact_shape():
    img = _img()
    sheet = threshold_sheet(img, channel="a")
    assert sheet.ndim == 3
    assert sheet.shape == (200, 200, 3)
    # Ensure it's not just a passthrough of the input image
    assert not np.array_equal(sheet, img), "Output must not be identical to input"


def test_threshold_sheet_unknown_channel_raises_and_valid_channel_works():
    img = _img()
    # Negative control: invalid channel must raise
    with pytest.raises(UnknownChannelError) as exc:
        threshold_sheet(img, channel="zzz")
    assert "zzz" in str(exc.value)

    # Positive control: valid channel must still work
    sheet = threshold_sheet(img, channel="a")
    assert sheet.shape == (200, 200, 3)


# --- suggest says when the recommended polarity is noise (real-photo finding #4) ---


def _speckled_scene(size: int = 400) -> np.ndarray:
    """One 40x40 dark 'plant' plus 60 dark 18x18 specks (each > fill_size=200).

    On a real sorghum photo the `a`/otsu polarity report recommended 'dark'
    with 118 components at 12.9% coverage -- chamber-wall noise -- and said
    `ambiguous: false`, because ambiguity only compares the two polarities'
    coverage. A recommendation made of specks needs its own signal.
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    img[180:220, 180:220] = 30
    rng = np.random.default_rng(3)
    placed = 0
    while placed < 60:
        y, x = rng.integers(0, size - 18, 2)
        if 150 <= y <= 240 and 150 <= x <= 240:
            continue
        if (img[y : y + 18, x : x + 18] == 30).any():
            continue
        img[y : y + 18, x : x + 18] = 30
        placed += 1
    return img


def test_polarity_report_flags_a_recommendation_made_of_specks():

    report = polarity_report(_speckled_scene(), "l")
    assert report["recommended"] == "dark"
    assert report["ambiguous"] is False
    dark = report["dark"]
    assert dark["component_count"] >= 50
    assert dark["largest_fraction"] < 0.2  # the plant is a minority of the mask
    codes = [w["code"] for w in report["warnings"]]
    assert codes == ["noisy_segmentation"]
    msg = report["warnings"][0]["message"]
    assert "61 components" in msg or f"{dark['component_count']} components" in msg
    assert "ambiguous" in msg  # says ambiguous is not a quality verdict
    assert "channel" in msg  # remedy: a different channel, or refine()

    # Positive control: a clean single-object scene reports no quality warning
    # and carries the same new field.
    clean = np.full((200, 200, 3), 200, dtype=np.uint8)
    clean[60:140, 60:140] = 30
    clean_report = polarity_report(clean, "l")
    assert clean_report["recommended"] == "dark"
    assert clean_report["warnings"] == []
    assert clean_report["dark"]["largest_fraction"] == 1.0


def test_polarity_report_warns_when_the_recommended_polarity_selects_nothing():
    """A blank frame with two dust pixels: 'dark' covers 0.0% and wins the
    less-coverage rule with no components at all, reported as unambiguous and
    clean. An empty recommendation is not a recommendation."""
    im = np.full((200, 200, 3), 128, np.uint8)
    im[10, 10] = 0
    im[100, 150] = 255
    rep = polarity_report(im, "l", method="otsu")
    assert rep["dark"]["component_count"] == 0
    codes = [w["code"] for w in rep["warnings"]]
    assert "empty_mask" in codes
    msg = rep["warnings"][0]["message"]
    assert "nothing" in msg or "no objects" in msg
    assert "overlay" in msg
    # Positive control: a real object scene carries no such warning.
    clean = np.full((200, 200, 3), 200, np.uint8)
    clean[60:140, 60:140] = 30
    assert polarity_report(clean, "l")["warnings"] == []


# --- a plant that fills its pot (field benchmark F2, 2026-10-27) ---

POTCELLS = Path(__file__).parent / "fixtures" / "aberystwyth"


@pytest.mark.parametrize(
    ("channel", "plant_side"), [("a", "dark"), ("b", "light"), ("s", "light")]
)
def test_polarity_report_picks_the_rosette_that_fills_its_pot(channel, plant_side):
    """A real grown rosette in its own pot cell. The plant is the larger side,
    so "the plant covers less of the frame" recommended the soil, and the SAM
    count on that mask was 2 for a 29-leaf plant. The green side is the plant."""
    img = cv2.imread(str(POTCELLS / "tray032_2016-01-17_potcell_r1c5.png"))
    rep = polarity_report(img, channel)
    # The premise: the plant side really is the bigger one here.
    assert rep[plant_side]["mask_fraction"] > 0.5
    assert rep["recommended"] == plant_side
    assert rep["ambiguous"] is False
    assert rep["basis"].startswith("greenness")
    assert rep[plant_side]["excess_green"] >= GREEN_FLOOR


def test_polarity_report_keeps_a_purple_plant_on_grey_with_coverage():
    """Grey scores 0 and purple scores negative, so "the greener side" is the
    background. The green floor keeps the scene with coverage, which picks the
    square, and the basis says so. Positive control: the same scene in green
    is decided by greenness, to the square. Which side the square is on `l`
    depends on its colour, so the test finds it by its 9% coverage."""
    for colour, basis in (((150, 40, 150), "coverage"), ((40, 150, 40), "greenness")):
        scene = np.full((200, 200, 3), 128, np.uint8)
        scene[70:130, 70:130] = colour  # BGR
        rep = polarity_report(scene, "l")
        (square,) = [k for k in ("dark", "light") if rep[k]["mask_fraction"] < 0.2]
        grey = "light" if square == "dark" else "dark"
        assert rep["recommended"] == square, colour
        assert rep["basis"].startswith(basis), (colour, rep["basis"])
        if basis == "coverage":
            # The premise: a gap that, with no floor, would choose the grey.
            gap = rep[grey]["excess_green"] - rep[square]["excess_green"]
            assert gap >= GREENNESS_MARGIN
            assert rep[grey]["excess_green"] < GREEN_FLOOR
