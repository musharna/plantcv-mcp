# Eval — checking the traits against geometry, not against ourselves

Every other test here checks that the code does what the code intends, or that a
guard fires. A regression test can tell you a number stopped changing; it cannot
tell you the number is wrong. This file records the check against shapes whose
answers were settled long before this server existed.

## Two shape families, on purpose

**Rectangles give exact ground truth.** A w×h block of pixels has area `w*h`,
width `w` and height `h` with no discretisation ambiguity, so any disagreement
is a measurement error rather than a rounding convention. Measured: **exact on
every case, no tolerance required.**

| rectangle | measured area | true  | width | height |
| --------- | ------------- | ----- | ----- | ------ |
| 60 × 30   | 1800          | 1800  | 60    | 30     |
| 100 × 100 | 10000         | 10000 | 100   | 100    |
| 25 × 140  | 3500          | 3500  | 25    | 140    |

**Discs give an analytic area** that a pixel grid can only approximate, which is
what makes them useful: the same code on a shape whose true value is irrational.

| radius | measured area | π·r²    | error |
| ------ | ------------- | ------- | ----- |
| 20     | 1257.0        | 1256.6  | 0.03% |
| 40     | 5025.0        | 5026.5  | 0.03% |
| 60     | 11289.0       | 11309.7 | 0.18% |
| 90     | 25445.0       | 25446.9 | 0.01% |

## The 2r+1 width, and why the pairing matters

A disc drawn with `x² + y² <= r²` includes both −r and +r, so it spans **2r+1**
pixels, not 2r. On its own that looks like an off-by-one in the measurement.

Alongside the rectangles — which are exact — it is unmistakably the disc's
geometry. So the eval asserts `2r+1` **exactly** rather than hiding it under a
tolerance: a tolerance wide enough to absorb one pixel would equally absorb a
real off-by-two.

## The resolution control

Every assertion above compares one shape to one number, which a measurer
returning plausible constants could survive at a single size. Doubling the
radius must **quadruple** the area — a relationship no constant satisfies, and
one that also fails if area were secretly a linear measure.

## Units

`mm²` must divide by `px_per_mm` **squared**. This is the specific error the
explicit unit table exists to prevent: PlantCV labels both `area` and `width` as
`"pixels"`, so a unit-derived rule would scale area linearly and leave every
value wrong by exactly one factor of `px_per_mm` — plausibly, and in the right
ballpark.

## Seen to fail

| mutant                                                    | result |
| --------------------------------------------------------- | ------ |
| areas converted linearly instead of by the square         | RED    |
| `area` removed from `AREA_TRAITS` (the unit-derived rule) | RED    |

## Not covered

Colour traits have no comparable ground truth here — a synthetic ColorChecker
gives a known _distortion_ to correct, which is checked in the ordinary suite,
not a known hue distribution to recover. Shape descriptors beyond area, width
and height (solidity, ellipse axes) are not evaluated against analytic values.
