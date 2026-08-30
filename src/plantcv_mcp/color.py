"""Colour-card correction.

Colour traits are only comparable across images if the images are colour-corrected
first: the same leaf under two lighting setups yields two different hues. PlantCV
can detect a Macbeth-style ColorChecker in the frame and correct to a standard
reference, which is what this wraps.

The important behaviour here is the failure mode. If correction is requested and no
card can be found, this RAISES. Silently returning the uncorrected image would hand
back colour traits that look corrected and are not — the same class of confident
wrongness as an inverted mask.
"""

import numpy as np
from plantcv.plantcv import transform as tr

from .diagnostics import Advisory


class ColorCardNotFoundError(Exception):
    """Raised when colour correction is requested but no colour card is detected."""


def _card_region(img: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the detected colour card as an (x0, y0, x1, y1) frame region.

    The detector labels a sample circle INSIDE each chip, so the raw extent
    stops short of the chips' edges and the card sheet; padding by the median
    chip extent covers the whole card. The same detector just succeeded inside
    auto_correct_color, so a failure here is real and propagates.
    """
    labeled = tr.detect_color_card(rgb_img=img)
    ys, xs = np.nonzero(labeled)
    extents = []
    for value in np.unique(labeled):
        if value == 0:
            continue
        yy, xx = np.nonzero(labeled == value)
        extents.append(int(max(xx.max() - xx.min(), yy.max() - yy.min())) + 1)
    pad = int(np.median(extents))
    frame_h, frame_w = labeled.shape[:2]
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(frame_w, int(xs.max()) + 1 + pad),
        min(frame_h, int(ys.max()) + 1 + pad),
    )


def exclude_card(
    mask: np.ndarray, card: tuple[int, int, int, int]
) -> tuple[np.ndarray, int]:
    """Zero the card's region of a mask; return (mask, pixels removed)."""
    x0, y0, x1, y1 = card
    removed = int((mask[y0:y1, x0:x1] > 0).sum())
    if removed:
        mask = mask.copy()
        mask[y0:y1, x0:x1] = 0
    return mask, removed


def color_card_excluded_advisory(
    removed: int, card: tuple[int, int, int, int]
) -> "Advisory | None":
    """The note that travels with a mask the card was cut out of."""
    if not removed:
        return None
    x0, y0, x1, y1 = card
    return Advisory(
        code="color_card_excluded",
        message=(
            f"{removed} px of the mask lay inside the detected colour card "
            f"(x {x0}-{x1}, y {y0}-{y1}) and were removed before measurement "
            "— the card is the instrument, not a specimen. If the mask is now "
            "empty, the threshold was selecting only the card. To use a card "
            "chip as a size marker, call calibrate_scale_from_marker()."
        ),
    )


def correct_color(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Colour-correct an image against a ColorChecker in the frame.

    Corrects to PlantCV's standard reference matrix, so the result is comparable
    across images and sessions rather than merely internally consistent. Returns
    the corrected image AND the card's frame region, so callers can keep the
    card — a known non-specimen the server itself just located — out of the
    measured mask (found measured as the largest "plant" in a real photo,
    2026-08-30).

    Raises ColorCardNotFoundError when no card is detected — never falls back to
    the uncorrected image.
    """
    try:
        corrected = tr.auto_correct_color(rgb_img=img)
    except Exception as exc:  # PlantCV raises assorted types when detection fails
        raise ColorCardNotFoundError(
            "Colour correction was requested but no colour card could be detected "
            f"in this image ({type(exc).__name__}: {exc}). Correction needs a "
            "Macbeth-style ColorChecker visible in the frame. Re-run with "
            "color_correct=false to measure the image as shot — but note that "
            "colour traits will then not be comparable with other images."
        ) from exc

    if corrected is None:
        raise ColorCardNotFoundError(
            "Colour correction returned nothing, which means no colour card was "
            "detected. Re-run with color_correct=false to measure as shot."
        )
    return corrected, _card_region(img)
