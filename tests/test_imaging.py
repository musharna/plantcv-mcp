import numpy as np
import pytest

from plantcv_mcp.imaging import downscale, encode_png, load_image, render_overlay


def test_load_image_raises_on_missing_file_with_path_in_message():
    with pytest.raises(RuntimeError) as exc:
        load_image("/nonexistent/definitely_not_here.png")
    assert "definitely_not_here.png" in str(exc.value)


def test_downscale_shrinks_large_and_reports_scale():
    big = np.zeros((2048, 1024, 3), dtype=np.uint8)
    out, scale = downscale(big, max_edge=1024)
    assert max(out.shape[:2]) == 1024
    assert scale == pytest.approx(0.5)


def test_downscale_leaves_midsize_untouched_and_reports_one():
    mid = np.zeros((300, 300, 3), dtype=np.uint8)
    out, scale = downscale(mid, max_edge=1024)
    assert out.shape == mid.shape
    assert scale == 1.0


def test_render_overlay_tints_only_masked_pixels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, :] = 255
    out = render_overlay(img, mask)
    assert out[0, 0].sum() > 0  # masked -> tinted
    assert out[9, 0].sum() == 0  # unmasked -> untouched


def test_encode_png_returns_png_magic_bytes():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert encode_png(img)[:8] == b"\x89PNG\r\n\x1a\n"


def test_tiny_frames_are_upscaled_so_the_overlay_can_be_looked_at():
    """A 31x43 ENVI cube's overlay at 1:1 is an unreadable thumbnail, and
    'look at the overlay' is the product's core discipline. Small frames are
    upscaled by an integer factor (crisp pixels), and the scale is reported."""
    from plantcv_mcp.imaging import downscale

    out, scale = downscale(np.zeros((31, 43, 3), np.uint8))
    assert scale > 1.0 and scale == int(scale), "integer upscale, reported"
    assert max(out.shape[:2]) >= 256
    # Mid-size and large frames keep their existing behavior.
    same, s1 = downscale(np.zeros((400, 400, 3), np.uint8))
    assert s1 == 1.0 and same.shape == (400, 400, 3)
    big, s2 = downscale(np.zeros((3000, 1500, 3), np.uint8))
    assert s2 < 1.0 and max(big.shape[:2]) == 1024
