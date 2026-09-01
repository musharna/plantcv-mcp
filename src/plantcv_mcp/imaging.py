"""Image I/O and rendering. The only module that touches the filesystem."""

import errno
import hashlib
import os

import cv2
import numpy as np

from .paths import check_readable


class NotColorImageError(Exception):
    """The file decoded, but not to the 3-channel colour image the RGB tools need."""


OVERLAY_BGR = np.array([0, 0, 255], dtype=np.float64)  # red in BGR
OVERLAY_ALPHA = 0.55
OUTLINE_BGR = (255, 255, 0)  # cyan: the fill's complement, visible on red subjects


def read_image_bytes(path: str) -> bytes:
    """Read the file ONCE. Everything derived from it — pixels and identity —
    comes from these bytes, never from a second look at the path.

    The read-root check runs HERE, at the one place bytes leave the disk, so a
    path no tool layer validated — an ENVI sibling derived from a .hdr, a future
    reader — is still contained. The open() uses the realpath the check
    returned: the path that was checked is the path that is read.
    """
    real = check_readable(path)
    try:
        with open(real, "rb") as fh:
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
        raise RuntimeError(
            f"Failed to open {path}: not a decodable image (expected a PNG/JPEG/TIFF "
            "photograph). ENVI cubes go to segment_hyperspectral(), radiometric "
            "thermal files to segment_thermal()."
        )
    if img.ndim != 3 or img.shape[2] != 3:
        # Guard HERE, at the one place pixels enter, so segment(),
        # suggest_segmentation(), calibrate_scale_from_marker() and the batch
        # refuse the same way instead of each dying in a different cvtColor.
        channels = 1 if img.ndim == 2 else img.shape[2]
        raise NotColorImageError(
            f"{path} decodes to {img.shape[1]}x{img.shape[0]} with {channels} channel"
            f"{'' if channels == 1 else 's'}, not the 3-channel colour photograph the "
            "RGB tools measure. If this is a thermal frame, use segment_thermal(); if "
            "it is a mask or a single band, it is not an image to segment."
        )
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


def write_image(path: str, img: np.ndarray, *, exclusive: bool = False) -> None:
    """Write an image without ever following a symlink at `path`.

    cv2.imwrite follows symlinks: measured on the one writing tool, a
    pre-existing `<image>_undistorted.png` symlink sent the corrected image
    into an unrelated victim file (a target outside the read roots included).
    So the bytes are encoded here and written through os.open(O_NOFOLLOW);
    `exclusive` adds O_EXCL, making refuse-if-exists atomic instead of an
    exists()-then-write race.
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise OSError(f"Could not encode image for {path!r}")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o644)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(
                f"{path!r} is a symlink; images are written only to real "
                "files, so the link's target was left untouched. Remove the "
                "link or pass a different output_path."
            ) from exc
        raise
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf.tobytes())


def downscale(
    img: np.ndarray, max_edge: int = 1024, min_edge: int = 256
) -> tuple[np.ndarray, float]:
    """Rescale so the longest edge is <= max_edge AND >= min_edge.

    Returns (image, scale); scale is always returned so rescaling is never
    silent. Small frames are UPSCALED by an integer factor (nearest neighbour,
    so mask pixels stay crisp): a 31x43 hyperspectral overlay at 1:1 is an
    unreadable thumbnail, and looking at the overlay is the point.
    """
    longest = max(img.shape[:2])
    if longest > max_edge:
        scale = max_edge / longest
        resized = cv2.resize(
            img,
            (int(img.shape[1] * scale), int(img.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale
    if longest < min_edge:
        k = -(-min_edge // longest)  # ceil: smallest integer factor that reaches
        resized = cv2.resize(
            img,
            (img.shape[1] * k, img.shape[0] * k),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized, float(k)
    return img, 1.0


def render_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint masked pixels red and outline the mask in cyan.

    The outline exists because a red tint on a red subject is invisible (found
    on a photo of red beans: nothing in the overlay showed what was selected).
    It is drawn on the mask's own boundary pixels -- inside the mask -- so no
    unmasked pixel is ever touched and the fill and the outline can never be
    the same colour.
    """
    out = img.copy()
    sel = mask > 0
    out[sel] = ((1 - OVERLAY_ALPHA) * out[sel] + OVERLAY_ALPHA * OVERLAY_BGR).astype(
        np.uint8
    )
    # Thickness scales with the frame: overlays are downscaled to ~1024 px for
    # the client, and a 1 px outline on a 4000 px photo disappears at that
    # scale (measured on the beans photo that motivated the outline).
    t = max(1, round(max(img.shape[:2]) / 1024))
    k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
    eroded = cv2.erode(sel.astype(np.uint8), k).astype(bool)
    out[sel & ~eroded] = OUTLINE_BGR
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
