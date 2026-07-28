import numpy as np
import pytest

from plantcv_mcp.server import _measure_impl, _segment_impl, list_methods_impl


def _write_green_png(tmp_path):
    import cv2

    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    p = tmp_path / "green.png"
    cv2.imwrite(str(p), img)
    return str(p)


def test_list_methods_reports_pinned_plantcv_version():
    info = list_methods_impl()
    assert info["plantcv_version"] == "4.11.3"
    assert "a" in info["channels"]
    assert "otsu" in info["methods"]


def test_segment_returns_no_traits_and_measure_needs_its_session(tmp_path):
    """The load-bearing API constraint: traits are unreachable without first
    receiving a segmentation overlay."""
    path = _write_green_png(tmp_path)
    seg = _segment_impl(path, channel="a", method="otsu")
    assert "traits" not in seg
    assert seg["overlay_png_bytes"] > 0
    assert "session_id" in seg

    traits = _measure_impl(seg["session_id"])
    assert traits["traits"]["area"]["value"] > 0


def test_measure_rejects_an_unknown_session_id():
    with pytest.raises(Exception) as exc:
        _measure_impl("not-a-real-session")
    assert "not-a-real-session" in str(exc.value)
