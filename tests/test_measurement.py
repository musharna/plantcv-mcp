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
