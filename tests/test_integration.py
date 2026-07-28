"""Real execution: drives the full tool path against a real file on disk."""

from pathlib import Path


from plantcv_mcp.server import _measure_impl, _segment_impl

FIXTURE = Path(__file__).parent / "fixtures" / "multi_specimen.png"


def test_real_render_fires_multi_specimen_warning_end_to_end():
    result = _segment_impl(str(FIXTURE), channel="a", method="otsu")
    codes = {w["code"] for w in result["warnings"]}
    assert "multi_specimen" in codes, (
        "The multi-specimen guard did not fire on a known four-view render. "
        f"diagnostics: {result['component_count']} components, "
        f"{result['major_object_count']} major objects."
    )
    assert result["major_object_count"] >= 2
    traits = _measure_impl(result["session_id"])["traits"]
    assert traits["area"]["value"] > 0


def test_real_render_segmentation_is_not_degenerate():
    """Positive control for the integration path: a real image must produce a
    usable mask, so a always-degenerate bug cannot hide behind the warning test."""
    result = _segment_impl(str(FIXTURE), channel="a", method="otsu")
    assert result["mask_fraction"] > 0.001
    assert result["component_count"] > 0
