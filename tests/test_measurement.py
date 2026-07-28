import numpy as np
import pytest

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.measurement import measure_traits


def _img_and_mask(fill=True):
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    if fill:
        mask[50:150, 50:150] = 255
    return img, mask


def test_empty_mask_refuses_and_valid_mask_returns_traits():
    """The single most important test in the suite. PlantCV returns 17
    zero-valued traits with in_bounds=True on an empty mask; we must refuse.
    The positive control is in the SAME test."""
    img, empty = _img_and_mask(fill=False)
    with pytest.raises(DegenerateMaskError):
        measure_traits(img, empty)

    img, valid = _img_and_mask(fill=True)
    traits = measure_traits(img, valid)  # must NOT raise
    assert traits["area"]["value"] > 0


def test_traits_carry_units():
    img, mask = _img_and_mask()
    traits = measure_traits(img, mask)
    assert traits["area"]["unit"] == "pixels"
    assert "solidity" in traits


def test_area_matches_the_known_mask_size():
    img, mask = _img_and_mask()  # 100x100 filled square
    traits = measure_traits(img, mask)
    assert traits["area"]["value"] == pytest.approx(10000, rel=0.02)


def test_successive_measurements_do_not_contaminate():
    """Regression test for global-state contamination. pcv.outputs.observations
    accumulates process-wide; without pcv.outputs.clear(), old observation groups
    could persist and contaminate next() iteration. This test checks exact areas
    to detect if observations from prior measurements leak in."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)

    # First: 100×100 square (area=10000)
    mask_100 = np.zeros((200, 200), dtype=np.uint8)
    mask_100[50:150, 50:150] = 255
    traits_100_first = measure_traits(img, mask_100)
    # Use exact equality: if contamination occurs, the area will mismatch
    assert traits_100_first["area"]["value"] == 10000, (
        f"First 100x100 measurement returned {traits_100_first['area']['value']}, expected 10000"
    )

    # Second: 30×30 square (area=900)
    mask_30 = np.zeros((200, 200), dtype=np.uint8)
    mask_30[85:115, 85:115] = 255
    traits_30_second = measure_traits(img, mask_30)
    assert traits_30_second["area"]["value"] == 900, (
        f"First 30x30 measurement returned {traits_30_second['area']['value']}, expected 900"
    )

    # Reverse order: verify contamination doesn't flow backward either
    traits_30_first = measure_traits(img, mask_30)
    assert traits_30_first["area"]["value"] == 900, (
        f"Second 30x30 measurement returned {traits_30_first['area']['value']}, expected 900"
    )

    traits_100_second = measure_traits(img, mask_100)
    assert traits_100_second["area"]["value"] == 10000, (
        f"Second 100x100 measurement returned {traits_100_second['area']['value']}, expected 10000"
    )
