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


def _png_bytes(arr):
    import cv2

    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def test_load_image_refuses_grayscale_by_name(tmp_path):
    from plantcv_mcp.imaging import NotColorImageError

    p = tmp_path / "gray.png"
    p.write_bytes(_png_bytes(np.full((20, 30), 128, dtype=np.uint8)))
    with pytest.raises(NotColorImageError) as exc:
        load_image(str(p))
    msg = str(exc.value)
    assert "1 channel" in msg and "gray.png" in msg
    assert "segment_thermal" in msg  # the likely intent, named
    # positive control: a colour image of the same shape loads
    q = tmp_path / "rgb.png"
    q.write_bytes(_png_bytes(np.full((20, 30, 3), 128, dtype=np.uint8)))
    assert load_image(str(q)).shape == (20, 30, 3)


def test_load_image_names_undecodable_file(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# not an image\n")
    with pytest.raises(RuntimeError, match="not a decodable image"):
        load_image(str(p))


def test_overlay_outlines_the_mask_in_cyan_so_red_subjects_stay_legible():
    """Found on a real photo of red beans: a red tint on a red subject is
    invisible, so 'look at the overlay' could not show what was selected. The
    boundary of the mask is drawn in cyan, INSIDE the mask, so unmasked pixels
    are still untouched (the existing tint test) and the outline is always a
    different colour from the fill."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 220)  # a red subject, BGR
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    out = render_overlay(img, mask)
    assert tuple(out[10, 20]) == (255, 255, 0)  # boundary pixel: cyan
    assert tuple(out[20, 20]) != (255, 255, 0)  # interior: tinted, not outlined
    assert tuple(out[5, 5]) == (0, 0, 220)  # unmasked: untouched
