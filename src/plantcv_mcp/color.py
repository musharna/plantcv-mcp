"""Colour-card correction, and the card as a region to keep out of the mask.

Colour traits are only comparable across images if the images are colour-corrected
first: the same leaf under two lighting setups yields two different hues. PlantCV
can detect a Macbeth-style ColorChecker in the frame and correct to a standard
reference, which is what this wraps.

The important behaviour here is the failure mode. If correction is requested and no
card can be found, this RAISES. Silently returning the uncorrected image would hand
back colour traits that look corrected and are not — the same class of confident
wrongness as an inverted mask.

The card is also the instrument, not a specimen. Its region is derived here so the
callers can exclude it from the mask: on a real photo its chips were the largest
"plant" in the scene (2026-08-30).
"""

import cv2
import numpy as np
from plantcv.plantcv import transform as tr

from .diagnostics import Advisory

# The exclusion polygon extends this many chip pitches beyond the outermost
# chip CENTRES. Chips reach half a pitch past their centres and the card's
# border about another half, so one pitch covers the sheet; a leaf within a
# pitch of the card is clipped and the advisory says how many pixels went.
CARD_PAD_PITCHES = 1.0

# Per-chip residual (0-1 RGB units) a chip may read from its reference AFTER
# the fitted correction. PlantCV verifies that every detected chip holds one
# grid centre, not that every centre has a chip, so a card with a chip erased
# is accepted and the fit is distorted for every pixel (mean shift 19 levels
# on the fixture). Measured: complete cards — the fixture, a CameraTrax card
# beside beans, a maize photo's card, an X-Rite in a booth — all read <= 0.3;
# the fixture with one interior chip erased reads 0.6.
CARD_CHIP_RESIDUAL_MAX = 0.45

CardRegion = list[list[int]]  # four [x, y] corners of the card polygon


class ColorCardNotFoundError(Exception):
    """Raised when colour correction is requested but no colour card is detected."""


def _detect(img: np.ndarray) -> np.ndarray:
    """PlantCV's card detector, with its assorted failure types made one."""
    try:
        labeled = tr.detect_color_card(rgb_img=img)
    except Exception as exc:  # PlantCV raises assorted types when detection fails
        raise ColorCardNotFoundError(
            "No colour card could be detected in this image "
            f"({type(exc).__name__}: {exc}). Correction needs a Macbeth-style "
            "ColorChecker visible in the frame. Re-run with color_correct=false to "
            "measure the image as shot — but note that colour traits will then not "
            "be comparable with other images."
        ) from exc
    if labeled is None or not np.any(labeled):
        raise ColorCardNotFoundError(
            "Card detection returned nothing, which means no colour card was "
            "detected. Re-run with color_correct=false to measure as shot."
        )
    return labeled


def _card_polygon(labeled: np.ndarray) -> CardRegion:
    """The card as a rotated rectangle around the detected chip lattice.

    The detector labels a fixed-radius sample circle inside each chip, so the
    circles say nothing about chip size; the LATTICE does. The median
    nearest-neighbour distance between chip centres is the chip pitch, and the
    minimum-area rectangle of the centres, grown by CARD_PAD_PITCHES pitches
    on every side, covers the chips and the sheet at any rotation. An
    axis-aligned box did not: it left the outer chips of a large card in the
    mask and took 18% bench beside a card rotated 30 degrees.
    """
    centres = []
    for value in np.unique(labeled):
        if value == 0:
            continue
        ys, xs = np.nonzero(labeled == value)
        centres.append((float(xs.mean()), float(ys.mean())))
    if len(centres) < 2:
        raise ColorCardNotFoundError(
            f"Card detection labelled {len(centres)} chip(s); a card region needs "
            "the chip lattice. Re-run with color_correct=false to measure as shot."
        )
    pts = np.array(centres, np.float32)
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    pitch = float(np.median(dist.min(axis=1)))
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    pad = 2.0 * CARD_PAD_PITCHES * pitch
    box = cv2.boxPoints(((cx, cy), (w + pad, h + pad), angle))
    return [[int(np.rint(x)), int(np.rint(y))] for x, y in box]


def detect_card_region(img: np.ndarray) -> CardRegion:
    """Locate the card without correcting the colours (exclude_color_card)."""
    return _card_polygon(_detect(img))


def exclude_card(mask: np.ndarray, card: CardRegion) -> tuple[np.ndarray, int]:
    """Zero the card's polygon in a mask; return (mask, pixels removed)."""
    region = np.zeros(mask.shape[:2], np.uint8)
    cv2.fillPoly(region, [np.array(card, np.int32)], 1)
    inside = (mask > 0) & (region == 1)
    removed = int(inside.sum())
    if removed:
        mask = mask.copy()
        mask[inside] = 0
    return mask, removed


def color_card_excluded_advisory(
    removed: int, card: CardRegion, note: str = ""
) -> "Advisory | None":
    """The note that travels with a mask the card was cut out of."""
    if not removed:
        return None
    pts = np.array(card)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return Advisory(
        code="color_card_excluded",
        message=(
            f"{removed} px of the mask lay inside the detected colour card "
            f"(polygon within x {x0}-{x1}, y {y0}-{y1}, one chip pitch beyond "
            "the outer chips) and were removed before measurement — the card "
            f"is the instrument, not a specimen.{note} If the mask is now "
            "empty, the threshold was selecting only the card. To use a card "
            "chip as a size marker, call calibrate_scale_from_marker()."
        ),
    )


def correct_color(img: np.ndarray) -> tuple[np.ndarray, CardRegion]:
    """Colour-correct an image against a ColorChecker in the frame.

    Runs PlantCV's own auto_correct_color pipeline step by step — detect,
    sample the chips, fit to the standard reference — so the card is detected
    ONCE and the fit can be checked. Returns the corrected image AND the card's
    region, so callers can keep the card out of the measured mask.

    Raises ColorCardNotFoundError when no card is detected, or when the
    detected card is incomplete (a chip missing or covered distorts the fit
    for every pixel) — never falls back to the uncorrected image.
    """
    labeled = _detect(img)
    try:
        _, source = tr.get_color_matrix(rgb_img=img, mask=labeled)
        target = tr.std_color_matrix(pos=3)
        corrected = tr.affine_color_correction(
            rgb_img=img, source_matrix=source, target_matrix=target
        )
        _, achieved = tr.get_color_matrix(rgb_img=corrected, mask=labeled)
    except Exception as exc:
        raise ColorCardNotFoundError(
            "A colour card was detected but the correction could not be fitted "
            f"({type(exc).__name__}: {exc}). Re-run with color_correct=false to "
            "measure as shot."
        ) from exc
    if corrected is None:
        raise ColorCardNotFoundError(
            "Colour correction returned nothing. Re-run with color_correct=false "
            "to measure as shot."
        )
    residual = np.linalg.norm(achieved[:, 1:4] - target[:, 1:4], axis=1)
    worst = int(residual.argmax())
    if residual[worst] > CARD_CHIP_RESIDUAL_MAX:
        raise ColorCardNotFoundError(
            "A colour card was detected but it is incomplete or partly covered: "
            f"chip {worst + 1} of 24 reads {residual[worst]:.2f} from its reference "
            f"after correction (limit {CARD_CHIP_RESIDUAL_MAX}; complete cards read "
            "under 0.3). A missing or covered chip distorts the fitted correction "
            "for EVERY pixel. Uncover the whole card, or re-run with "
            "color_correct=false to measure as shot."
        )
    return corrected, _card_polygon(labeled)
