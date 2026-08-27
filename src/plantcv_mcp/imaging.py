"""Image I/O and rendering. The only module that touches the filesystem."""

import hashlib

import cv2
import numpy as np

OVERLAY_BGR = np.array([0, 0, 255], dtype=np.float64)  # red in BGR
OVERLAY_ALPHA = 0.55


def read_image_bytes(path: str) -> bytes:
    """Read the file ONCE. Everything derived from it — pixels and identity —
    comes from these bytes, never from a second look at the path."""
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        # Same message shape pcv.readimage used, so callers and tests that
        # match on "Failed to open <path>" keep working.
        raise RuntimeError(f"Failed to open {path}: {exc.strerror or exc}") from exc


def decode_image(data: bytes, path: str) -> np.ndarray:
    """Decode bytes the way pcv.readimage(mode="native") decodes a file.

    Native mode is cv2.IMREAD_UNCHANGED, falling back to a 3-channel read when
    the result carries an alpha channel. Reproduced here rather than delegated
    because pcv.readimage only accepts a PATH, and re-reading the path is the
    time-of-check/time-of-use hole this module exists to close.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.shape[2] == 4:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to open {path}")
    return img


def digest_bytes(data: bytes) -> str:
    """SHA-256 of the bytes.

    The stale-image guard used to compare only the image's SHAPE, so swapping the
    file for a DIFFERENT image of identical dimensions passed the check and
    measured the old mask against new content. Shape is a weak proxy for
    identity; the bytes are the identity.
    """
    return hashlib.sha256(data).hexdigest()


def load_image_with_digest(path: str) -> tuple[np.ndarray, str]:
    """Load an image and the SHA-256 of the exact bytes it was decoded from.

    Hashing the PATH after decoding left a window: a same-shape replacement
    landing between the decode and the hash bound the old pixels' mask to the
    new file's identity, and the later integrity check then passed. One read,
    one hash of that read, one decode of that read — there is no window.
    """
    data = read_image_bytes(path)
    return decode_image(data, path), digest_bytes(data)


def load_image(path: str) -> np.ndarray:
    """Load an image as BGR (native mode; alpha dropped)."""
    return load_image_with_digest(path)[0]


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


MEASURED_BGR = (0, 220, 0)  # green: this region produced traits
EMPTY_BGR = (0, 200, 255)  # amber: this region was refused, no plant found


def render_region_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    measured: list[bool],
) -> np.ndarray:
    """Tint the mask, then outline and number every region.

    The whole point of returning per-region numbers is that a reader can tell
    WHICH plant each row describes. A tinted mask alone cannot do that on a tray
    of twenty seedlings, so the region index is drawn onto the region itself.

    Refused regions are drawn too, in a different colour. Omitting them would
    make an empty cell indistinguishable from a cell the grid never covered —
    and those call for opposite fixes (re-segment vs. correct the geometry).
    """
    out = render_overlay(img, mask)
    for i, (x, y, w, h) in enumerate(bboxes):
        colour = MEASURED_BGR if i < len(measured) and measured[i] else EMPTY_BGR
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            out,
            str(i),
            (x + 4, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
    return out


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image")
    return buf.tobytes()
