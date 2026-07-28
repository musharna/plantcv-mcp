import numpy as np
import pytest
from plantcv import plantcv as pcv

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


def test_successive_measurements_return_correct_results():
    """Verify that successive measure_traits() calls return correct independent
    results, regardless of calling order. Each call should return traits matching
    its own mask, with exact area equality (10000 for 100x100, 900 for 30x30)."""
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


def test_clear_guards_against_foreign_sample_label_contamination():
    """Regression test for contamination from foreign sample_label keys in
    pcv.outputs.observations. If a prior call used a different sample_label
    (e.g., 'priorstuff_1'), that key persists in the global dict. When
    measure_traits() calls next(iter(...)), it returns whichever key was
    inserted FIRST, which could be the stale foreign group. This test seeds
    a foreign key, verifies measure_traits returns traits for the NEW mask
    (not the foreign group), and demonstrates that pcv.outputs.clear()
    prevents the bug."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)

    # Seed the observations dict with a foreign group under a different label
    # Simulate what would happen if another code path had written observations
    # with sample_label='priorstuff'
    original_label = pcv.params.sample_label
    try:
        pcv.params.sample_label = "priorstuff"
        roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
        foreign_mask = np.zeros((200, 200), dtype=np.uint8)
        foreign_mask[30:80, 30:80] = 255  # 50×50 = 2500
        labeled, n = pcv.create_labels(mask=foreign_mask, rois=roi, roi_type="partial")
        pcv.analyze.size(img=img, labeled_mask=labeled, n_labels=n)

        # Observations dict now has 'priorstuff_1' with area 2500
        assert "priorstuff_1" in pcv.outputs.observations, (
            "Setup failed: foreign key not seeded"
        )
        foreign_area = pcv.outputs.observations["priorstuff_1"]["area"]["value"]

        # Restore default label and call measure_traits
        pcv.params.sample_label = original_label

        # Now measure a DIFFERENT mask with known area
        test_mask = np.zeros((200, 200), dtype=np.uint8)
        test_mask[85:115, 85:115] = 255  # 30×30 = 900

        traits = measure_traits(img, test_mask)

        # The returned traits MUST be for the 30×30 mask (900), not the foreign
        # group (2500). Without pcv.outputs.clear(), next(iter(...)) would pick
        # 'priorstuff_1' (inserted first) and return 2500.
        assert traits["area"]["value"] == 900, (
            f"measure_traits returned {traits['area']['value']}, expected 900 "
            f"(foreign group had {foreign_area}). This means next(iter(...)) "
            f"returned the stale foreign key instead of the new measurement."
        )
    finally:
        pcv.params.sample_label = original_label
        pcv.outputs.observations.clear()
