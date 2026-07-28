from pathlib import Path

from plantcv_mcp.server import _measure_impl, _segment_impl

FIXTURE = Path(__file__).parent / "fixtures" / "multi_specimen.png"


def test_same_image_and_params_give_identical_traits():
    a = _measure_impl(_segment_impl(str(FIXTURE), "a", "otsu")["session_id"])["traits"]
    b = _measure_impl(_segment_impl(str(FIXTURE), "a", "otsu")["session_id"])["traits"]
    assert a == b, "Identical inputs produced different traits — not deterministic."
