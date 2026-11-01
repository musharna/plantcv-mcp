"""Contact sheets that make the channel/method choice informed rather than blind."""

import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import (
    Advisory,
    analyze_mask,
    excess_green,
    is_noisy,
    largest_fraction,
)
from .segmentation import OBJECT_TYPES, segment_mask, to_gray


def colorspace_sheet(img: np.ndarray) -> np.ndarray:
    """Grid of L,A,B,H,S,V,C,M,Y,K plus the original — which channel separates plant from background."""
    return pcv.visualize.colorspaces(rgb_img=img, original_img=True)


def threshold_sheet(img: np.ndarray, channel: str) -> np.ndarray:
    """Otsu and triangle side by side on one channel, each labelled with its
    threshold — which method works here.

    Built here, not with pcv.visualize.auto_threshold_methods: that returned a
    list whose first item is the Otsu panel alone, and its triangle panel is
    Otsu as well (PlantCV 4.11.3 thresholds with THRESH_OTSU after computing
    the triangle value). The sheet compared nothing. Each panel is the
    threshold segment() applies (threshold_mask's otsu and triangle are these
    same cv2 calls; tests hold the two equal), in "light" polarity as PlantCV
    drew it, from one channel conversion.
    """
    import cv2

    gray = to_gray(img, channel)
    w = gray.shape[1]
    panels = []
    for name, flag in (("Otsu", cv2.THRESH_OTSU), ("Triangle", cv2.THRESH_TRIANGLE)):
        value, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + flag)
        panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.putText(
            panel,
            f"{name} {value:.0f}",
            (max(w // 8, 1), max(gray.shape[0] // 8, 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            pcv.params.text_size,
            (255, 0, 255),
            pcv.params.text_thickness,
        )
        panels.append(panel)
    return np.hstack(panels)


def noisy_segmentation_warning(polarity: str, stats: dict) -> Advisory | None:
    """The shared noise rule (diagnostics.is_noisy), worded for the polarity
    report: a sorghum-in-chamber photo gave 118 components at 12.9% under
    a/otsu and the polarity comparison alone called it unambiguous."""
    if not is_noisy(
        stats["component_count"], stats["largest_fraction"], stats["major_object_count"]
    ):
        return None
    return Advisory(
        code="noisy_segmentation",
        message=(
            f"The recommended polarity ('{polarity}') is {stats['component_count']} "
            f"components and the largest is only {stats['largest_fraction']:.0%} of "
            "the masked pixels: this channel/method is picking up background "
            "texture, not a plant. `ambiguous` only compares the two polarities' "
            "coverage and is not a quality verdict. Try a different channel "
            "(look at the colorspace sheet), or segment() and refine() with "
            "median_blur/opening plus keep_largest and check the overlay."
        ),
    )


# Greenness decides the polarity when the greener side is itself green
# (mean excess green at least GREEN_FLOOR) and leads the other by at least
# GREENNESS_MARGIN. Calibrated 2026-10-28 with otsu on channels a, b, s, v, l.
# On Aberystwyth tray 032 (450 px pot cells around every annotated plant on
# three dates), in every case where one side matched the annotation it was
# the greener side, by 0.154 to 0.256, at 0.144 or more; less coverage missed
# the 3 grown rosettes that fill their pot. On the 46 synthetic benchmark
# images, in the 25 cases where one side's area was within 25% of the true
# plant area, the plant side scored 0.194 or more whenever it was green; the
# greener side was wrong only twice, both by under 0.035. The floor is what
# keeps a purple or red plant on a grey background (grey scores 0) with the
# coverage rule instead of handing the grey to it as "greener".
GREENNESS_MARGIN = 0.05
GREEN_FLOOR = 0.1


def polarity_report(
    img: np.ndarray,
    channel: str,
    method: str = "otsu",
    fill_size: int = 200,
    ambiguity_margin: float = 0.1,
) -> dict:
    """What each object_type would actually yield on THIS image.

    object_type is the single easiest way to get a confidently wrong answer: pick
    the wrong polarity and the mask is the background, while every trait remains
    plausible and correctly united. The server refuses to guess it, so this
    measures both and hands back the numbers.

    'recommended' is the GREENER polarity when that side is green (mean
    excess green at least GREEN_FLOOR) and leads the other by at least
    GREENNESS_MARGIN. Otherwise it is the polarity that
    covers LESS of the frame, which holds for a small plant on a big
    background but picked the soil on every rosette that filled its pot
    (field benchmark F2, 2026-10-27). A plant that is not green (purple,
    senescent, a fluorescence or grey image) falls to the coverage rule, and
    `basis` says which rule decided. When neither decides (no greenness gap
    and coverages within `ambiguity_margin`), `ambiguous` is set and the
    caller is told to look at the overlay.
    """
    per_polarity = {}
    for object_type in OBJECT_TYPES:
        mask = segment_mask(
            img, channel, method, object_type=object_type, fill_size=fill_size
        )
        diag = analyze_mask(mask)
        per_polarity[object_type] = {
            "mask_fraction": diag.mask_fraction,
            "component_count": diag.component_count,
            "major_object_count": diag.major_object_count,
            "largest_fraction": largest_fraction(diag),
            "excess_green": excess_green(img, mask),
        }

    fractions = {k: v["mask_fraction"] for k, v in per_polarity.items()}
    greens = {k: v["excess_green"] for k, v in per_polarity.items()}
    by_green = (
        None not in greens.values()
        and max(greens.values()) >= GREEN_FLOOR
        and abs(greens["dark"] - greens["light"]) >= GREENNESS_MARGIN
    )
    if by_green:
        recommended = max(greens, key=lambda k: greens[k])
        ambiguous = False
    else:
        recommended = min(fractions, key=lambda k: fractions[k])
        ambiguous = abs(fractions["dark"] - fractions["light"]) < ambiguity_margin

    report = dict(per_polarity)
    report["recommended"] = recommended
    report["ambiguous"] = ambiguous
    advisories = []
    if per_polarity[recommended]["component_count"] == 0:
        # min(coverage) happily picks a polarity that selected NOTHING (a
        # blank frame, or a threshold past every pixel) and calls it
        # unambiguous. An empty recommendation is not a recommendation.
        advisories.append(
            Advisory(
                code="empty_mask",
                message=(
                    f"The recommended polarity ('{recommended}') selects nothing "
                    "on this image: no objects at all. There may be no plant "
                    "in the frame, or this channel/method does not separate "
                    "it. Try another channel from the colourspace sheet and "
                    "check the overlay from segment()."
                ),
            )
        )
    noisy = noisy_segmentation_warning(recommended, per_polarity[recommended])
    if noisy:
        advisories.append(noisy)
    report["warnings"] = [{"code": w.code, "message": w.message} for w in advisories]
    if by_green:
        report["basis"] = (
            f"greenness: the '{recommended}' mask is the greener one (mean excess "
            f"green {greens[recommended]:+.3f} against "
            f"{greens['light' if recommended == 'dark' else 'dark']:+.3f}), so it "
            "is taken to be the plant whatever share of the frame it covers. "
            "Check the overlay if the subject is not green."
        )
    elif not ambiguous:
        report["basis"] = (
            "coverage: neither mask is clearly the green one (or one is empty), "
            "so the polarity covering less of the frame is assumed to be the "
            "plant. Check the overlay before trusting this on a subject that "
            "fills the frame."
        )
    else:
        report["basis"] = (
            "undecided: neither mask is clearly the green one, and they cover a "
            "similar share of the frame, so neither rule can decide. Inspect the "
            "overlay from segment() under each object_type."
        )
    return report
