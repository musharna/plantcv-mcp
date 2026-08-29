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


def test_keyed_lookup_immune_to_foreign_sample_labels():
    """Regression test: keyed lookup is immune to foreign sample_label keys.
    pcv.outputs.observations is global and keyed by sample_label. If prior
    code used a different sample_label (e.g., 'priorstuff_1'), that key
    persists. With next(iter(...)), this would corrupt results. With keyed
    lookup (current implementation), the correct group is selected regardless
    of foreign keys. This test seeds a foreign key, verifies measure_traits
    returns traits for the measured mask (not the foreign group), proving
    that the keyed-lookup implementation is immune to contamination."""
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
        # group (2500). The keyed lookup selects 'default_1' explicitly, so the
        # presence of 'priorstuff_1' does not affect the result.
        assert traits["area"]["value"] == 900, (
            f"measure_traits returned {traits['area']['value']}, expected 900 "
            f"(foreign group had {foreign_area}). Keyed lookup should be immune "
            f"to foreign keys, but got the wrong area instead."
        )
    finally:
        pcv.params.sample_label = original_label
        pcv.outputs.observations.clear()


def test_keyed_lookup_raises_on_missing_key(monkeypatch):
    """Regression test: KeyError raised when expected observation key is missing.
    When pcv.analyze.size() fails to create the expected observation group,
    measure_traits must raise KeyError (not silently fall back to next(iter(...))).
    This test verifies both the positive control (normal case works) and the
    negative control (missing key raises with a clear error message)."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    # Positive control: normal case MUST NOT raise
    traits = measure_traits(img, mask)
    assert traits["area"]["value"] == 10000, "Positive control: normal case failed"

    # Negative control: make analyze.size() a no-op so no observation group is created
    def no_op(*args, **kwargs):
        pass

    monkeypatch.setattr(pcv.analyze, "size", no_op)

    # Now measure_traits will call clear(), run the analysis (no-op), and try to
    # look up 'default_1' in an empty observations dict. It MUST raise KeyError.
    with pytest.raises(KeyError) as exc_info:
        measure_traits(img, mask)

    # Verify the error message names the expected key and lists actual keys
    error_message = str(exc_info.value)
    assert "default_1" in error_message, "Error message should name expected key"
    assert "Available keys" in error_message, "Error message should list actual keys"


def test_isolated_section_starts_with_an_empty_palette_and_hands_the_hosts_back():
    """PlantCV's morphology functions share params.saved_color_scale, sized by
    whichever call filled it and indexed by every later one: a host that left
    a one-colour palette behind gives IndexError inside our morphology run,
    and our run must not leave OUR palette in the host's process either."""
    from plantcv_mcp.measurement import isolated_pcv_outputs

    host_palette = [[1, 2, 3]]
    pcv.params.saved_color_scale = host_palette
    try:
        with isolated_pcv_outputs():
            assert pcv.params.saved_color_scale is None
            pcv.params.saved_color_scale = [[9, 9, 9]] * 5
        assert pcv.params.saved_color_scale is host_palette
        # And on the error path.
        with pytest.raises(RuntimeError), isolated_pcv_outputs():
            pcv.params.saved_color_scale = [[7, 7, 7]]
            raise RuntimeError("mid-analysis")
        assert pcv.params.saved_color_scale is host_palette
    finally:
        pcv.params.saved_color_scale = None
