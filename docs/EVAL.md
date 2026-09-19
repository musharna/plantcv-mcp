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

## Leaf instances

`count_leaves()` is checked two ways, and the second is not flattering.

**Known geometry** (`tests/test_leaves.py`): 1, 2, 3, 5 and 8 seeded, separated
ellipses are exactly that many instances, each centred on a drawn ellipse, and the
instances partition the mask pixel for pixel. Two overlapping ellipses forming ONE
connected component are two instances split within 5% of evenly; one ellipse alone
is one. A single long ellipse (70×24 px) is **two** at the default distance and one
at 30 — the over-segmentation is pinned, not avoided.

**Real rosettes**: the Aberystwyth Leaf Evaluation Dataset (Bell and Dee, 2016,
<https://doi.org/10.5281/zenodo.168158>, CC BY 4.0) — top-down Arabidopsis trays with
every leaf hand-painted, including leaves mostly hidden under others. Truth per
plant = distinct annotation colours. `scripts/eval_leaf_count.py` reproduces the
table from a tray image and its `gt/` file.

Tray 031 was used to look at the method: it is where the pinhole and
neighbour-sliver effects were found, and the only data any design choice was made
on. **Tray 032 is what is reported.** It was not run only once — its figures were
seen during development, before and after the border-ring fix (they moved by at
most 0.1) — but nothing was chosen from them. The default
`min_distance` is PlantCV's (10) and was not tuned on either. Mean error in leaves
(count − truth), mean absolute error, exact matches:

| Tray 032 (held out)             | mask                                   | `min_distance` 6    | 8                   | 10 (default)        | 15                  |
| ------------------------------- | -------------------------------------- | ------------------- | ------------------- | ------------------- | ------------------- |
| 2015-12-21, 20 plants, 8–16 lvs | annotation's own                       | −0.2 / 1.2 / 7      | −1.7 / 1.8 / 3      | −2.3 / 2.3 / 1      | −3.4 / 3.4 / 0      |
|                                 | `segment` + `keep_largest`             | −0.2 / 1.2 / 5      | −1.7 / 1.8 / 3      | −2.2 / 2.3 / 0      | −3.3 / 3.3 / 0      |
|                                 | … + `fill_holes`                       | −2.0 / 2.1 / 5      | −2.7 / 2.7 / 2      | −3.2 / 3.2 / 0      | −4.4 / 4.4 / 0      |
| 2016-01-05, 13 plants, 18–25    | annotation's own                       | +6.9 / 6.9 / 1      | +1.7 / 3.9 / 2      | −1.6 / 3.9 / 1      | −5.4 / 6.5 / 0      |
|                                 | `segment` + `keep_largest`             | +28.2 / 28.2 / 0    | +16.5 / 16.5 / 0    | +9.6 / 9.9 / 1      | +0.8 / 4.9 / 0      |
|                                 | … + `fill_holes`                       | +1.8 / 5.5 / 0      | −3.1 / 7.1 / 0      | −5.4 / 7.9 / 0      | −8.4 / 9.5 / 0      |

Read plainly: at best about one leaf in ten wrong on young plants with a distance
fitted to the camera, two to eight leaves wrong on grown ones, and the best
`min_distance` differs between the two dates of the same tray. Tray 031 gave the
same picture (default, `fill_holes` + `keep_largest`: −3.8 / 3.8 on 20 young
plants, −6.1 / 6.2 on 14 grown ones). Two scale-relative rules for `min_distance`
(a fraction of the largest inscribed radius; of √area) and h-maxima markers were
tried on Tray 031 and were no better than a fixed value, so none was adopted.

Two fixture crops (plant 0 of each held-out image, chosen before any count was
read) pin the numbers in the suite: 11 annotated leaves → 7 (default) and 8 (at 6);
25 → 13 and 19; and the 25-leaf plant reads 30 with its pinholes left in.

### Seen to fail (leaf instances)

| mutant                                                             | result                                                   |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| zero ring no wider than 2 px                                       | RED — `LeafCountRefusedError`, no peak at the frame edge |
| crop offset dropped from centroid/bbox                             | RED — centroid `[56, 48]`, not `[700, 450]`              |
| empty-mask guard removed                                           | RED — numpy `ValueError`, not `DegenerateMaskError`      |
| inverted-mask refusal removed                                      | RED — did not raise                                      |
| area scaled by `px_per_mm`, not its square                         | RED — 714.0 vs 178.5                                     |
| hole counter returns 0                                             | RED — no `mask_has_holes`                                |
| sensitivity threshold never reached                                | RED — no `min_distance_sensitive`                        |
| `multi_object_mask` never raised                                   | RED                                                      |
| `min_distance >= 1` check removed                                  | RED — did not raise                                      |
| watershed replaced by connected components                         | RED — touching ellipses count 1, not 2                   |
| instance outlines not drawn                                        | RED — instance colour absent from the overlay            |
| `pcv.outputs` not isolated                                         | RED — host table gained a `leaves` sample                |
| mask-level warnings not carried by the tool                        | RED — no `frame_clipping`                                |
| tool function not registered under its name                        | RED — tool-set assertion                                 |
| `min_distance` ≤ mask-extent check removed (run under a 4 GB cap)  | RED — `ArrayMemoryError`, 149 GiB asked                  |
| comparison pass not capped at the extent                           | RED — keyed `194`, not `97`                              |

Each was run with `python -B -p no:cacheprovider`, green before the mutation, and the
source's md5 checked after the restore.
