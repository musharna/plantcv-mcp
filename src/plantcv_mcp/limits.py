"""The one place a client-supplied size, count or iteration is range-checked.

Every integer that sizes an allocation or a loop reaches native code (OpenCV,
PlantCV, NumPy) with no ceiling of its own. The audit of 2026-09-22 found four
of them unguarded at once: refine `iterations` (a 3x3 dilation 200000 times asked
OpenCV for 160 GB), checkerboard corner counts (10.9 TiB), the adaptive
threshold `ksize` (40 s of CPU for one 300 px image at 10^6; an OpenCV overflow
at 2^31) and negative `fill_size`, which PlantCV silently treats as "remove
nothing". Issue #117 found the refine `ksize` the same way. Each had a lower
bound and no upper one, checked — when at all — in a different module.

So the ranges are stated here, once, as named constants with the reason for
each, and every entry point calls `require_int`. A ceiling is either what the
value cannot meaningfully exceed (a kernel wider than the mask has already
reached every pixel; an offset past 255 on an 8-bit channel changes nothing) or,
where no such fact exists, a generous practical limit stated as one.
"""

from typing import Any


class ParameterRangeError(ValueError):
    """A size/count/iteration argument outside its documented range."""


# Adaptive threshold neighbourhood ('mean', 'gaussian'). PlantCV's own floor is
# 3 (it rounds even sizes up to odd). The ceiling is practical: a 1001 px block
# is a sixth of a 6000 px frame, far wider than any local-contrast window, and
# a Gaussian that wide already costs seconds per megapixel.
THRESHOLD_KSIZE_MIN = 3
THRESHOLD_KSIZE_MAX = 1001

# The offset is subtracted from a local mean of an 8-bit channel; beyond ±255
# every pixel lands on the same side of the threshold, so a larger value
# means nothing a smaller one does not.
THRESHOLD_OFFSET_MAX = 255

# Inner corners per side of a calibration checkerboard. OpenCV needs a few
# pixels per square to find a board at all, and printed boards are counted in
# tens of squares; 200 per side (40000 corners) is far past any real target
# while bounding the object-point grid to kilobytes.
CHECKERBOARD_CORNERS_MIN = 2
CHECKERBOARD_CORNERS_MAX = 200


def require_int(
    name: str,
    value: Any,
    *,
    lo: int,
    hi: int | None = None,
    why: str = "",
) -> int:
    """Return `value` if it is an integer in [lo, hi]; raise otherwise.

    bool is refused although it is an int subclass: `True` as a kernel size is
    a bug, not a 1. The message names the parameter, the range and, when given,
    why the range is what it is.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterRangeError(f"{name} must be an integer, got {value!r}.")
    if value < lo or (hi is not None and value > hi):
        bound = f">= {lo}" if hi is None else f"between {lo} and {hi}"
        raise ParameterRangeError(
            f"{name} must be {bound}, got {value}." + (f" {why}" if why else "")
        )
    return value


def require_fill_size(fill_size: Any) -> int:
    """fill_size removes components smaller than itself. PlantCV accepts a
    negative size and silently removes nothing, recording a recipe that did not
    run as written. No ceiling: a size past the mask's area erases it, and the
    fill_erased_mask advisory already says so."""
    return require_int(
        "fill_size",
        fill_size,
        lo=0,
        why="0 disables the speck filter; a negative size is not a smaller one.",
    )


def require_threshold_params(method: str, ksize: Any, offset: Any) -> None:
    """The adaptive methods' kernel and offset, checked before any image is
    read. Only 'mean' and 'gaussian' use them; the global methods ignore both,
    so a recipe carrying the defaults for an 'otsu' run is not refused."""
    if method not in ("mean", "gaussian"):
        return
    require_int(
        "ksize",
        ksize,
        lo=THRESHOLD_KSIZE_MIN,
        hi=THRESHOLD_KSIZE_MAX,
        why="It is the side of the local neighbourhood, in pixels.",
    )
    require_int(
        "offset",
        offset,
        lo=-THRESHOLD_OFFSET_MAX,
        hi=THRESHOLD_OFFSET_MAX,
        why="The channel is 8-bit; a larger offset moves every pixel to one side.",
    )


def require_kernel_extent(
    what: str, ksize: int, iterations: int, shape: tuple[int, ...]
) -> None:
    """A structuring element applied `iterations` times reaches
    (ksize - 1) * iterations + 1 pixels; OpenCV builds exactly that kernel.
    Once it spans the mask's longest edge every pixel already sees every other,
    so a larger one changes nothing and only costs memory."""
    extent = (ksize - 1) * iterations + 1
    longest = max(int(shape[0]), int(shape[1]))
    if extent > longest:
        raise ParameterRangeError(
            f"{what}: ksize={ksize} applied {iterations} time(s) reaches "
            f"{extent} px, wider than the mask's longest edge ({longest} px). "
            "A kernel past the mask's own extent changes nothing further; use a "
            "smaller ksize or fewer iterations."
        )
