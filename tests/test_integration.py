"""Real execution: drives the full tool path against a real file on disk."""

from pathlib import Path

import pytest

from plantcv_mcp.diagnostics import analyze_mask
from plantcv_mcp.imaging import load_image
from plantcv_mcp.segmentation import segment_mask
from plantcv_mcp.server import _measure_impl, _segment_impl

FIXTURE = Path(__file__).parent / "fixtures" / "multi_specimen.png"


def test_real_render_fires_multi_specimen_warning_end_to_end():
    """Integration test against a real four-view render. Verifies the full
    segmentation → measurement pipeline and that the multi_specimen guard
    correctly identifies multiple plants in a single image."""
    result = _segment_impl(str(FIXTURE), channel="a", method="otsu")

    # Verify warning fired and segmentation is not degenerate
    codes = {w["code"] for w in result["warnings"]}
    assert "multi_specimen" in codes, (
        "The multi-specimen guard did not fire on a known four-view render. "
        f"diagnostics: {result['component_count']} components, "
        f"{result['major_object_count']} major objects."
    )

    # Pin exact deterministic values: this is the only real-image test in the
    # suite and must catch any segmentation regression on real data.
    assert result["component_count"] == 9
    assert result["major_object_count"] == 4
    assert result["mask_fraction"] == pytest.approx(0.030925, rel=1e-3)

    # Verify top-four component areas match known values using structured data
    # instead of parsing the prose message. This asserts on the actual segmentation
    # result, not on message formatting.
    img = load_image(str(FIXTURE))
    mask = segment_mask(img, channel="a", method="otsu")
    diag = analyze_mask(mask)
    assert diag.areas[:4] == [8628, 7981, 7106, 6748], (
        f"Top-four areas do not match known values: {diag.areas[:4]}"
    )

    # Verify end-to-end measurement succeeds
    traits = _measure_impl(result["session_id"])["traits"]
    assert traits["area"]["value"] > 0
