"""Unattended measurement across many images.

**The design tension, stated rather than hidden.** This server's central claim is
that you cannot obtain a number without first being handed the picture it came
from. A batch of two hundred images cannot honour that literally: nobody is going
to look at two hundred overlays, and returning two hundred base64 images would be
useless anyway.

So the overlay is replaced by the only honest substitute — automated validation
plus explicit refusal. Every image runs the SAME guards as the interactive path
(`segmentation_warnings`). Any image tripping a BLOCKING guard gets **no traits at
all**, only the reason and an instruction to inspect it with `segment()`. Advisory
warnings that do not invalidate the measurement, like `multi_specimen`, are
attached to the traits rather than suppressing them.

The result is that a batch never hands back a number the system could not validate.
That is weaker than a human looking at a mask, and it is stated here so nobody
mistakes it for the same thing.
"""

from dataclasses import dataclass, field

from plantcv import plantcv as pcv

from . import plantcv_version
from .color import correct_color
from .diagnostics import BLOCKING_CODES, Advisory, analyze_mask, segmentation_warnings
from .imaging import load_image
from .measurement import measure_traits
from .segmentation import threshold_mask

# A cap, so a model passing a directory glob of ten thousand files fails fast and
# loudly instead of pinning a CPU for an hour.
MAX_BATCH = 200


class BatchTooLargeError(Exception):
    """Raised when more images are submitted than MAX_BATCH."""


@dataclass
class BatchEntry:
    image_path: str
    measured: bool
    mask_fraction: float | None = None
    component_count: int | None = None
    warnings: list[Advisory] = field(default_factory=list)
    traits: dict | None = None
    refused_because: str | None = None


def measure_batch(
    image_paths: list[str],
    channel: str,
    method: str,
    object_type: str = "dark",
    fill_size: int = 200,
    ksize: int = 11,
    offset: int = 2,
    analyses: tuple[str, ...] = ("size",),
    px_per_mm: float | None = None,
    include_histograms: bool = False,
    color_correct: bool = False,
) -> dict:
    """Segment and measure many images with one fixed recipe.

    The recipe is EVERYTHING segment() takes — channel, method, object_type,
    fill_size, ksize, offset, color_correct — so a recipe settled interactively
    reproduces here exactly. A batch that silently ran default ksize/offset for a
    'mean' threshold, or skipped the colour correction, measured a different mask
    than the one the user had looked at: 1900 px vs 4536 px on the same file.

    Returns a per-image entry and a summary. Images whose guards fire are reported
    with `measured: false` and a reason, never with traits. An image that cannot
    be colour-corrected when asked is refused the same way — never measured raw.
    """
    if not image_paths:
        raise ValueError("image_paths is empty; nothing to measure.")
    if len(image_paths) > MAX_BATCH:
        raise BatchTooLargeError(
            f"{len(image_paths)} images submitted, limit is {MAX_BATCH}. Split the "
            "batch, or narrow the selection."
        )

    entries: list[BatchEntry] = []
    for path in image_paths:
        try:
            img = load_image(path)
            if color_correct:
                # Raises ColorCardNotFoundError without a card; caught below and
                # reported per-image, so the run continues and the image is
                # refused rather than measured uncorrected.
                img = correct_color(img)
            pre_fill = threshold_mask(
                img,
                channel,
                method,
                object_type=object_type,
                ksize=ksize,
                offset=offset,
            )
            mask = pcv.fill(bin_img=pre_fill, size=fill_size)
            diag = analyze_mask(mask)
            warnings = segmentation_warnings(
                mask, diag, analyze_mask(pre_fill), fill_size
            )
        except Exception as exc:  # noqa: BLE001 — see below
            # Deliberately blind: one unreadable or unsegmentable file must not
            # abort a 200-image run. The failure is reported per-image with its
            # exception type, so nothing is swallowed silently.
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    refused_because=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        blocking = [w for w in warnings if w.code in BLOCKING_CODES]
        if blocking:
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    mask_fraction=diag.mask_fraction,
                    component_count=diag.component_count,
                    warnings=warnings,
                    refused_because=(
                        f"{', '.join(w.code for w in blocking)} — traits withheld "
                        "because the mask probably does not describe the plant. "
                        "Inspect this image with segment() and look at the overlay."
                    ),
                )
            )
            continue

        try:
            traits = measure_traits(
                img,
                mask,
                analyses=analyses,
                px_per_mm=px_per_mm,
                include_histograms=include_histograms,
            )
        except Exception as exc:  # noqa: BLE001 — same rationale as above:
            # a single image failing to measure must not lose the other 199.
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    mask_fraction=diag.mask_fraction,
                    component_count=diag.component_count,
                    warnings=warnings,
                    refused_because=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        entries.append(
            BatchEntry(
                image_path=path,
                measured=True,
                mask_fraction=diag.mask_fraction,
                component_count=diag.component_count,
                warnings=warnings,
                traits=traits,
            )
        )

    measured = [e for e in entries if e.measured]
    refused = [e for e in entries if not e.measured]
    return {
        "recipe": {
            "channel": channel,
            "method": method,
            "object_type": object_type,
            "fill_size": fill_size,
            "ksize": ksize,
            "offset": offset,
            "color_correct": color_correct,
            "analyses": list(analyses),
            "px_per_mm": px_per_mm,
        },
        # Which PlantCV produced these numbers, travelling WITH them — the same
        # provenance measure() carries, for the same reason.
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "summary": {
            "submitted": len(image_paths),
            "measured": len(measured),
            "needs_review": len(refused),
            "review_paths": [e.image_path for e in refused],
        },
        "results": [
            {
                "image_path": e.image_path,
                "measured": e.measured,
                "mask_fraction": e.mask_fraction,
                "component_count": e.component_count,
                "warnings": [
                    {"code": w.code, "message": w.message} for w in e.warnings
                ],
                "traits": e.traits,
                "refused_because": e.refused_because,
            }
            for e in entries
        ],
    }
