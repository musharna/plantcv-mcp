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

## Leaf instances: Segment Anything

`segment_leaves_sam()` (optional `sam` extra) is scored on the same plants, boxes and
hand counts as the watershed above, through the path a user takes: the plant's box
written to disk, `segment(a, otsu)`, `refine(fill_holes, keep_largest 1)`, then the
tool. `scripts/eval_leaf_count_sam.py` reproduces it; it imports the plant boxes and
the truth from `scripts/eval_leaf_count.py`. It is not run in CI (data size).

**Model.** Segment Anything, ViT-B (Kirillov et al., 2023,
<https://doi.org/10.1109/ICCV51070.2023.00371>), used as released: not fine-tuned and
not trained on leaves. Code and checkpoints are Apache-2.0 (the upstream repository's
`LICENSE` and README). `segment-anything==1.0`, torch 2.14.0 CPU build, 6 threads, no
GPU. The checkpoint is `sam_vit_b_01ec64.pth` from
`https://dl.fbaipublicfiles.com/segment_anything/`. Upstream publishes no SHA-256 for
it (none in its README or repository, checked 2026-09-19); the six hex digits in the
file name are the start of its MD5, which the file fetched here matches. The SHA-256
pinned in `sam_leaves.py`
(`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`, 375,042,383
bytes) is the hash of the file fetched from that URL on 2026-09-19. It is loaded with
`torch.load(weights_only=True)`, which this checkpoint allows: it is a plain
`state_dict` of tensors.

**Method.** The image is cropped to the mask's bounding box plus a margin, a 32×32
grid of point prompts is laid over the crop and only the points on the mask are
kept, and SAM's candidate masks are filtered: predicted IoU ≥ 0.80 and stability
≥ 0.85, at least 80% on the plant mask, between 0.2% and 40% of the plant's area,
then smallest first, dropping a mask once more than half of it is already covered.

**What was chosen on which tray.** Every constant was chosen on Tray 031. The grid
and the filter were fixed from SAM's raw masks on Tray 031 before Tray 032 was run.
The crop margin was then chosen on Tray 031 through the finished tool, from three
values, by grown-rosette MAE:

| Tray 031 (development), crop margin | 20 young: mean / MAE / exact | 14 grown: mean / MAE / exact |
| ----------------------------------- | ---------------------------- | ---------------------------- |
| 0.10 (adopted)                      | −0.65 / 1.35 / 2             | −1.00 / 2.00 / 2             |
| 0.25                                | −0.55 / 1.25 / 2             | −1.36 / 2.21 / 3             |
| 0.50                                | −0.50 / 1.20 / 5             | −1.36 / 2.21 / 3             |

The three differ by less than a quarter of a leaf on 14 plants, and the young plants
rank them the other way. The margin is not a sensitive choice and the table should not
be read as showing that 0.10 is better.

**Tray 032 was then run once through the finished tool**, with nothing changed
afterwards. SAM's raw masks on Tray 032 had been scored once before, with the same
filter and no crop-margin step (young −0.75 / 0.95 / 10, grown −0.38 / 2.08 / 1), to
decide whether the tool was worth building; no constant was chosen from that run. Both
methods at their defaults, same sessions:

| tray, plants                       | method               | mean error | MAE  | exact | s/plant (CPU) |
| ---------------------------------- | -------------------- | ---------- | ---- | ----- | ------------- |
| 032 (held out), 20 young, 8–16 lvs | `count_leaves`       | −3.20      | 3.20 | 0/20  | under 0.1     |
|                                    | `segment_leaves_sam` | −0.60      | 1.10 | 6/20  | 20.8 (max 32) |
| 032 (held out), 13 grown, 18–25    | `count_leaves`       | −5.38      | 7.85 | 0/13  | under 0.1     |
|                                    | `segment_leaves_sam` | +0.23      | 1.77 | 2/13  | 19.3 (max 21) |
| 031 (development), 20 young        | `count_leaves`       | −3.80      | 3.80 | 0/20  |               |
|                                    | `segment_leaves_sam` | −0.65      | 1.35 | 2/20  | see below     |
| 031 (development), 14 grown        | `count_leaves`       | −6.07      | 6.21 | 0/14  |               |
|                                    | `segment_leaves_sam` | −1.00      | 2.00 | 2/14  | see below     |

Seconds per plant cover the tool call only, with the model already loaded; the first
call adds the load (about 2 s more here). The Tray 031 runs shared the machine with
other jobs (load average over 40) and took 28–35 s per plant at margin 0.10, against
13–21 s at 0.50 on a quieter machine, so they are not a timing measurement. Peak
resident memory of the evaluation process was 3.8 GB.

Read plainly: about one leaf wrong on young rosettes and two on grown ones, against
three and eight for the watershed, and still exact on fewer than a third of plants.
The annotation counts leaves mostly hidden under others, which no top-view method
sees. `mask_coverage` (the share of the plant mask that received an instance) ran
0.83–0.92 on Tray 032 and never fell under 0.64 on either tray, so the
`low_instance_coverage` advisory (under 0.50) did not fire on this dataset; it is
tested with a stubbed model. One dataset, one species, one camera: the figures say
nothing about other plants.

**Dataset.** Aberystwyth Leaf Evaluation Dataset (Bell and Dee, 2016,
<https://doi.org/10.5281/zenodo.168158>), CC BY 4.0, as above.

### Seen to fail (Segment Anything)

Tests that need no model (`tests/test_sam_leaves.py`, run in every CI job):

| mutant                                                 | result                                                  |
| ------------------------------------------------------ | ------------------------------------------------------- |
| require_sam never raises                               | RED — no `SamNotInstalledError` in the tool error       |
| extra checked after the checkpoint is resolved         | RED — the cache directory exists: a download came first |
| open_verified skips the hash comparison                | RED — did not raise `CheckpointVerificationError`       |
| open_verified skips the read-root check                | SURVIVED — see below                                    |
| download ignores the permission flag                   | RED — did not raise `CheckpointMissingError`            |
| download hash not checked                              | RED — did not raise `CheckpointVerificationError`       |
| download size cap removed                              | RED — no "more than the expected" in the error          |
| OSError not wrapped as CheckpointDownloadError         | RED — bare `OSError: network is unreachable`            |
| cache not held to read roots before download           | RED — did not raise `PathOutsideRootsError`             |
| unavailable cuda silently becomes cpu                  | RED — did not raise `SamDeviceError`                    |
| overlap rule removed                                   | RED — 3 instances, not 2                                |
| max-area rule removed                                  | SURVIVED — see below                                    |
| on-plant rule removed                                  | RED — 3 instances, not 2                                |
| kept mask relabels taken pixels                        | RED — labels `{1, 2}` where `{1}` was kept              |
| session kind not checked                               | RED — `WrongSessionKindError` on the RGB control        |
| tool annotated read-only closed-world                  | RED — `read_only_hint` is True                          |
| crop offset dropped from instances                     | RED — bbox x 33, not 76                                 |
| area scaled by px_per_mm not its square                | RED — 1157.5 vs 2315 / 4                                |
| low coverage never flagged                             | RED — no `low_instance_coverage`                        |
| zero instances returned as a count                     | RED — did not raise `LeafCountRefusedError`             |
| inverted-mask refusal removed                          | RED — did not raise `LeafCountRefusedError`             |
| empty-mask guard removed                               | RED — numpy `ValueError`, not `DegenerateMaskError`     |
| model loaded before the mask checks                    | RED — `load_model` called on a refused mask             |
| multi_object_mask never raised                         | RED — no `multi_object_mask`                            |
| mask-level warnings not carried by the tool            | RED — no `frame_clipping`                               |
| tool not registered under its name                     | RED — tool-set assertion                                |
| torch present in the base environment (guard inverted) | RED — "torch is installed in the base test environment" |

Tests that load the model (`tests/test_sam_leaves_model.py`, the `sam` CI job), on CPU:

| mutant                                                    | result                                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| load_model opens the file unverified                      | RED — did not raise `CheckpointVerificationError` on the flipped byte |
| weights_only=False                                        | RED — `torch.load` saw `[False]`, not `[True]`                        |
| stale-image check bypassed                                | RED — did not raise `ImageChangedSinceSegmentationError`              |
| instances not clipped to the plant mask                   | RED — labels off the mask                                             |
| labels written at the frame origin, not the crop          | RED — labels off the mask                                             |
| unavailable cuda silently becomes cpu (with the model)    | RED — did not raise `SamDeviceError`                                  |
| prompt grid ignores the mask                              | RED — 0 candidates, `LeafCountRefusedError`                           |
| max-area rule removed (re-run after the added assertions) | RED — 1 instance from a rosette-sized mask, not 0                     |
| both read-root checks on the checkpoint removed           | RED — did not raise `PathOutsideRootsError`                           |

Each was run with `python -B -p no:cacheprovider`, green before the mutation, and the
source's md5 checked after the restore. Two mutants survived their first run and the
tests were changed: removing the 40% area ceiling was hidden by the overlap rule until
the ceiling was asserted with a single candidate, and removing `check_readable` from
`open_verified` is still caught by `check_open_fd` on the open handle, so the two were
removed together.
