"""Image I/O and rendering. The only module that touches the filesystem."""

import hashlib

import cv2
import numpy as np
from plantcv import plantcv as pcv

OVERLAY_BGR = np.array([0, 0, 255], dtype=np.float64)  # red in BGR
OVERLAY_ALPHA = 0.55


def load_image(path: str) -> np.ndarray:
    """Load an image as BGR.

    pcv.readimage already raises RuntimeError("Failed to open <path>") for both
    missing and non-image files, so we let it propagate with the path intact
    rather than wrapping it into something vaguer.
    """
    img, _, _ = pcv.readimage(path)
    return img


def file_digest(path: str) -> str:
    """SHA-256 of the file's bytes.

    The stale-image guard used to compare only the image's SHAPE, so swapping the
    file for a DIFFERENT image of identical dimensions passed the check and
    measured the old mask against new content. Shape is a weak proxy for
    identity; the bytes are the identity.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def downscale(img: np.ndarray, max_edge: int = 1024) -> tuple[np.ndarray, float]:
    """Shrink so the longest edge is <= max_edge. Returns (image, scale).

    Scale is always returned so downsampling is never silent.
    """
    longest = max(img.shape[:2])
    if longest <= max_edge:
        return img, 1.0
    scale = max_edge / longest
    resized = cv2.resize(
        img,
        (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def render_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint masked pixels so a viewer can see what was measured."""
    out = img.copy()
    sel = mask > 0
    out[sel] = ((1 - OVERLAY_ALPHA) * out[sel] + OVERLAY_ALPHA * OVERLAY_BGR).astype(
        np.uint8
    )
    return out


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image")
    return buf.tobytes()
