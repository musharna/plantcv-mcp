"""EVAL: are the traits right, measured against shapes whose answer is known?

Every other test here checks that the code does what the code intends, or that
a guard fires. This one checks the NUMBERS against geometry — the area of a
rectangle and of a disc were settled long before this server existed.

Two shape families, deliberately:

**Rectangles give EXACT ground truth.** w x h pixels have area w*h, width w and
height h with no discretisation ambiguity at all, so any disagreement is a
measurement error rather than a rounding convention. Measured: exact on every
case, no tolerance required.

**Discs give an ANALYTIC area** (pi*r^2) that a pixel grid can only approximate,
which is what makes them the useful second case: they exercise the same code on
a shape whose true value is irrational.

The pairing is what makes the disc convention interpretable. A disc drawn with
`x^2 + y^2 <= r^2` spans pixels -r..+r INCLUSIVE, so its pixel width is 2r+1,
not 2r. On its own that looks like an off-by-one in the measurement; alongside
the rectangles, which are exact, it is clearly the disc's geometry. The eval
asserts 2r+1 exactly rather than hiding it under a tolerance — a tolerance wide
enough to absorb it would also absorb a real off-by-two.
"""

import math

import numpy as np
import pytest

from plantcv_mcp.measurement import measure_traits

SIZE = 400


def _canvas() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((SIZE, SIZE, 3), 30, np.uint8),
        np.zeros((SIZE, SIZE), np.uint8),
    )


def _disc(radius: int) -> tuple[np.ndarray, np.ndarray]:
    img, mask = _canvas()
    yy, xx = np.ogrid[:SIZE, :SIZE]
    sel = (xx - SIZE // 2) ** 2 + (yy - SIZE // 2) ** 2 <= radius**2
    mask[sel] = 255
    img[sel] = (40, 200, 40)
    return img, mask


def _rect(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    img, mask = _canvas()
    mask[100 : 100 + height, 80 : 80 + width] = 255
    img[100 : 100 + height, 80 : 80 + width] = (40, 200, 40)
    return img, mask


@pytest.mark.parametrize(("width", "height"), [(60, 30), (100, 100), (25, 140)])
def test_rectangle_traits_are_exact(width, height):
    """No tolerance. A rectangle's pixel area IS width times height."""
    traits = measure_traits(*_rect(width, height), analyses=("size",))
    assert traits["area"]["value"] == width * height
    assert traits["width"]["value"] == width
    assert traits["height"]["value"] == height


@pytest.mark.parametrize("radius", [20, 40, 60, 90])
def test_disc_area_matches_pi_r_squared(radius):
    """Measured error over these radii: 0.01%-0.18%."""
    traits = measure_traits(*_disc(radius), analyses=("size",))
    true_area = math.pi * radius**2
    error = abs(traits["area"]["value"] - true_area) / true_area
    assert error < 0.005, (
        f"disc of radius {radius}: measured area {traits['area']['value']} "
        f"against pi*r^2 = {true_area:.1f} ({error:.2%} error)"
    )


@pytest.mark.parametrize("radius", [20, 40, 60, 90])
def test_disc_width_is_two_r_plus_one_which_is_the_disc_not_a_bug(radius):
    """Pinned EXACTLY, not absorbed by a tolerance.

    `x^2 + y^2 <= r^2` includes both -r and +r, so the disc spans 2r+1 pixels.
    The rectangle cases above are exact, which is what proves this is the
    shape's geometry rather than an off-by-one in the measurement. A tolerance
    loose enough to hide it would equally hide a real off-by-two.
    """
    traits = measure_traits(*_disc(radius), analyses=("size",))
    assert traits["width"]["value"] == 2 * radius + 1
    assert traits["height"]["value"] == 2 * radius + 1


def test_area_scales_as_the_square_of_size():
    """THE resolution control.

    Every assertion above compares one shape to one number, which a measurer
    that returned plausible constants could survive at a single size. Doubling
    the radius must QUADRUPLE the area — a relationship no constant satisfies,
    and one that also fails if area were secretly a linear measure.
    """
    small = measure_traits(*_disc(30), analyses=("size",))["area"]["value"]
    large = measure_traits(*_disc(60), analyses=("size",))["area"]["value"]
    ratio = large / small
    assert abs(ratio - 4.0) < 0.05, (
        f"doubling the radius scaled area by {ratio:.3f}, not 4. Area is not "
        "behaving as a two-dimensional quantity."
    )


def test_real_world_units_convert_by_the_square_for_areas():
    """mm2 must divide by px_per_mm SQUARED, not by px_per_mm.

    This is the specific error the explicit unit table exists to prevent:
    PlantCV labels both `area` and `width` as "pixels", so a unit-derived rule
    would scale area linearly and leave every value wrong by exactly one factor
    of px_per_mm — plausibly, and in the right ballpark.
    """
    px_per_mm = 10.0
    pixels = measure_traits(*_rect(100, 100), analyses=("size",))
    mm = measure_traits(*_rect(100, 100), analyses=("size",), px_per_mm=px_per_mm)

    assert mm["area"]["unit"] == "mm2"
    assert mm["area"]["value"] == pytest.approx(pixels["area"]["value"] / px_per_mm**2)
    assert mm["width"]["unit"] == "mm"
    assert mm["width"]["value"] == pytest.approx(pixels["width"]["value"] / px_per_mm)

    # The two conversions must DIFFER. If area were scaled linearly it would
    # come out 10x too large here, which is exactly the plausible-looking bug.
    assert mm["area"]["value"] != pytest.approx(pixels["area"]["value"] / px_per_mm)
