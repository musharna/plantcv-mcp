# plantcv-mcp guide

Everything the [README](https://github.com/musharna/plantcv-mcp#readme) does not say: each tool's parameters and guards, the measured facts behind them, and the full warning-code reference. Sections are in workflow order.

- [Segmenting](#segmenting)
- [What it measures](#what-it-measures)
  - [Real-world units](#real-world-units)
- [Getting the polarity right](#getting-the-polarity-right)
- [Refining a mask](#refining-a-mask)
- [Colour correction](#colour-correction)
- [Measuring a tray](#measuring-a-tray)
- [Morphology: leaves, stem, branch points](#morphology-leaves-stem-branch-points)
- [Measuring many images](#measuring-many-images)
- [Hyperspectral and thermal](#hyperspectral-and-thermal)
- [Warnings and refusals](#warnings-and-refusals)
- [Crash containment: the analysis worker](#crash-containment-the-analysis-worker)
- [Restricting what the server may read](#restricting-what-the-server-may-read)
- [Security and trust boundary](#security-and-trust-boundary)
- [Limitations](#limitations)

## Segmenting

| parameter       | default  | what it does                                                                              |
| --------------- | -------- | ----------------------------------------------------------------------------------------- |
| `image_path`    | required | image to read from the host filesystem                                                    |
| `channel`       | required | one of `l a b h s v` — never guessed for you                                              |
| `method`        | required | one of `otsu triangle mean gaussian`                                                      |
| `object_type`   | `"dark"` | which side of the threshold is the plant. **See [polarity](#getting-the-polarity-right)** |
| `fill_size`     | `200`    | drops components smaller than this; can erase a small specimen                            |
| `ksize`         | `11`     | neighbourhood size, `mean` and `gaussian` only                                            |
| `offset`        | `2`      | constant subtracted from the local mean, `mean`/`gaussian` only                           |
| `color_correct` | `false`  | correct to a ColorChecker in the frame first. See [colour correction](#colour-correction) |

The `segment()` response for the README's example image — verbatim, apart from a
shortened `session_id` and an elided warning message:

```json
{
  "session_id": "9d2384c8-…",
  "channel": "a",
  "method": "otsu",
  "object_type": "dark",
  "fill_size": 200,
  "mask_fraction": 0.031,
  "component_count": 9,
  "major_object_count": 4,
  "largest_area": 8628,
  "overlay_scale": 1.0,
  "overlay_png_bytes": 748233,
  "warnings": [
    {
      "code": "multi_specimen",
      "message": "4 comparably-sized objects detected (areas: [8628, 7981, 7106, 6748]). …"
    }
  ]
}
```

The overlay arrives alongside this as a second content block, as an image. Grayscale
files (a dataset's mask PNGs, say) are refused by name at the load boundary rather than
failing three different ways downstream; a thermal frame goes to `segment_thermal()`.

## What it measures

One `measure()` call returns seventeen traits, each with a unit.

| group         | traits                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| size          | `area`, `convex_hull_area`, `perimeter`, `total_edge_length`, `width`, `height`, `longest_path` (pixels) |
| shape         | `solidity`, `convex_hull_vertices`, `ellipse_eccentricity` (unitless)                                    |
| ellipse fit   | `ellipse_major_axis`, `ellipse_minor_axis` (pixels), `ellipse_angle` (degrees)                           |
| position      | `center_of_mass`, `ellipse_center` (x, y)                                                                |
| PlantCV flags | `in_bounds`, `object_in_frame`                                                                           |

The last two are PlantCV's own flags. They are passed through as **information, never as
validity signals** — on an all-zero mask PlantCV reports both as `True` while returning
seventeen zeros. They are bounds checks, not success checks.

Passing `analyses=["size", "color"]` adds hue, saturation and value statistics —
`hue_circular_mean`, `hue_circular_std`, `hue_median` (degrees), `saturation_mean`,
`saturation_median`, `value_mean`, `value_median` (percent). The three frequency histograms
that accompany them total 692 numbers, so they are withheld unless you ask for them with
`include_histograms=true`.

Every number-bearing result carries `engine` (PlantCV's name and version) and, when the
mask came from `refine()`, `lineage` — the ops that produced it.

### Real-world units

**Traits are in pixels by default, and pixel sizes are not comparable between images shot at
different distances or zoom levels.** Pass `px_per_mm` to `measure()` and spatial traits come
back in `mm` and `mm2`:

```
measure(session_id, px_per_mm=12.5)
  area    207.533 mm2     (32427 pixels)
  width    54.880 mm      (686 pixels)
```

Lengths divide by `px_per_mm`, areas by its square. That distinction is a hard-coded table
rather than something inferred from PlantCV's unit strings, because PlantCV labels **both**
`area` and `width` as `"pixels"` — scaling everything with that label linearly would leave
every area wrong by exactly a factor of `px_per_mm`, plausibly and silently. Positions
(`center_of_mass`, `ellipse_center`) stay in pixels, since a millimetre coordinate means
nothing without a defined origin.

If you have a marker of known real size in the frame — a coin, a printed disc — put a box
around it and let the server measure it:

```
calibrate_scale_from_marker(image_path, x=100, y=100, w=100, h=100, marker_length_mm=20)
  -> px_per_mm 4.05, marker_length_px 81
```

`marker_length_mm` is the marker's longest real dimension. **Check `marker_length_px` against
what you expect**, because a wrong scale silently rescales every trait you measure afterwards.
The region is cropped before thresholding, so nothing outside your box can be selected;
PlantCV's own `report_size_marker_area` takes an ROI instead, and measured against a disc of
known 80 px diameter it returns 348 with a tight ROI — a silent 4.35× error. If the detected
object reaches the crop edge you get a `marker_touches_crop_edge` warning, which usually means
the polarity is wrong and the background was measured — or that the candidate object simply
continues beyond the crop (a pot contiguous with its soil and plant is ONE dark object), in
which case there is no isolatable marker in the scene and traits should stay in pixels: on a
real photo, four plausible crops of the same pot returned px_per_mm from 10.8 to 18.1.

## Correcting lens distortion

A wide-angle or fisheye lens makes `px_per_mm` a fiction: the scale varies across the frame
and by direction. Measured on the PlantCV fisheye tutorial photo, distortion inflated the
centre-frame plant's area **2.13×** (width 1.69×, height 1.22×), shifted shape traits
(eccentricity 0.56 → 0.76), and scaled the same pot's rim, face height and base by
1.73×/1.31×/1.59× — three different pixel scales on one object, so no marker calibration can
compensate: a scale taken from the rim mis-sizes area by −29%, one from the face height by
+24%. If straight edges bow in your images, correct first and do everything else on the
corrected file:

```
correct_lens_distortion(image_path, checkerboard_dir, row_corners=13, col_corners=19)
  -> corrected_image_path ".../img_undistorted.png", frames_used 7,
     frames_skipped [bad_checkerboard.png],
     frames_outliers [{checkerboard1.png, rms_px 37.2, reason "fits at 37.2 px against the other views' median of 3.8 px, and moves the focal length by 13% on its own, 33 times the uncertainty of the fit without it"},
                      {checkerboard2.png, reason "moves the focal length by 2% on its own, 5 times the uncertainty of the fit without it — a view consistent with some camera but not this one: a bent board, a rolling-shutter frame"}],
     rms 3.1, focal_uncertainty 0.0044, crop_fraction 0.19, residual_void_px 0,
     board_coverage 0.71,
     warnings [lens_corrected, outlier_frames_dropped, frames_skipped]
```

(That is the PlantCV tutorial set as this release reads it: one frame does not detect,
two are left out for the reasons quoted, and the remaining seven fit at 3.1 px — the 13 px
that earlier releases quoted for this set was the first of those frames. The second is the
one 1.11.1 added: it moves the focal length 2%, which the 3% floor before it waved through.
Both are real. Measured independently of the fit, by how straight the board's own rows come
out after correction, `checkerboard2` is the most warped board in the set — 10.7 px of
residual bow against 2.4–5.5 px for every frame kept — and dropping it changes the
correction the tool applies by nothing you can see: mean residual bow over the seven is
4.858 px, against 4.860 px when it is kept.)

`checkerboard_dir` holds several photos of a checkerboard taken with the **same camera at
the same resolution**, the board **tilted differently in each — by ten degrees and more, in
different directions**; the corner counts are the board's INNER corners. Frames where no
board is detected, or that differ from the majority size, are skipped and named; fewer than
three detections is refused (PlantCV's own `checkerboard_calib` crashes on that directory,
and would happily "calibrate" from one frame). So is a set showing fewer than three distinct
board **orientations**: a camera model is determined by the board's angles, not its
positions, and copies of one photo, the board slid around the frame, moved nearer or farther,
or turned in its own plane all count as ONE (orientations more than 5° apart are counted, in
an order that depends on the set and not on filenames — and NOT as a chain, so a board
swept smoothly through 40° in 4° steps, a phone video of a nodding board, counts every 5°
of it). Byte-identical files count once (`frames_duplicates`, warning
`duplicate_frames_ignored`): copies add no geometry and only make every uncertainty the fit
reports look smaller. Measured with the orientation refusal disabled on a synthetic camera
of known intrinsics, ten copies of one pose "calibrated" to fx=252 against a true 400 at
rms 0.19 and eight positions of a board facing the camera to fx=228 — confidently wrong
models no error metric could flag.

The fit itself is started from several focal lengths and the lowest residual kept: OpenCV's
own initial estimate ignores distortion, and on a strongly distorted lens seen through
near-frontal boards it can be far enough off that the optimiser settles in a wrong basin —
a camera with a residual a few times the right one that nothing downstream can tell is
wrong (measured: 41 of 91 twelve-view subsets of an honest synthetic set calibrated to
fx 480–8270; every one recovers the camera from a start at the frame width). Views that
do not belong are then dropped one at a time, the camera refitted after each, and named
with the reason (`frames_outliers`, warning `outlier_frames_dropped`): a view whose
reprojection error stands above three times the OTHER views' median and 0.25% of the
diagonal is a bad detection (the PlantCV tutorial set's 37-px frame, which moved fx by
13%), and a view that fits but, left out, moves the focal length by more than 3% and more
than four times the uncertainty of the fit made without it is a bent board or a
rolling-shutter frame, consistent with some camera but not this one (influence alone is
not guilt: the steepest view of a weakly tilted set moves the answer too, and leaves a
loose fit behind that says so) (measured: one view sheared by 5%
among eight moved fx from 400 to 459 with the correction 111 px wrong and its own
residual unremarkable). Judged against the others, not a whole-set median, so two bad
views of four cannot hide each other; judged only where there are enough views to have
"others" (four for the residual test, five for the influence test).

Three orientations is a floor, not a guarantee, and no statistic of the fit's own
consistency is one either: what can be said honestly is the fit's uncertainty on the focal
length relative to its value (`focal_uncertainty`). Above 4% the calibration is refused as
undetermined (measured: two mis-detected views of four 4.9%, ±3° about one axis 12%);
above 2.5% the `focal_length_uncertain` warning says how loose it is (±7° about two axes
3.6% with the correction 54 px at worst — the documented soft spot; a board tilted ±7°
about one axis 2.4% with 16 px; the PlantCV tutorial set 0.4%; a sound three-view set
0.6%). Tilt the board ten degrees and beyond, in different directions, and none of this
fires.
Mirrored frames are a limit, not a check: a set that is
_all_ mirrored calibrates the reflected camera consistently, but a set with _some_ frames
mirrored returns a camera that is neither (measured with cx=285, the mixed set fitted cx=319
at a low rms) and nothing can detect it — do not mix. Both the calibration frames and the
image are read as stored, ignoring any EXIF orientation tag, so a camera's own frames never
mismatch its scenes. An image whose size differs from the calibration frames' is refused
(`CalibrationResolutionMismatchError`): intrinsics are in pixels at the calibration
resolution, and measured, a 1280×960 image through a 640×480 calibration came back as a
49×127 crop.

The model is fitted where boards were and **extrapolated everywhere else**. `board_coverage`
reports the fraction of the frame the detected corner grids covered (the board's outer ring
of squares lies outside its inner corners, so a board covers a little more than the number
says); under 40% the `low_calibration_coverage` warning says so (measured: 24% coverage left
the frame corners 31 px wrong; 48% held them within 10 px at the 95th percentile and 25 px at
the worst pixel; the tutorial set covers 71%). The sixth-order radial term k3 is not fitted —
left free, it is fitted from the boards' footprint alone and folds the correction over
outside it (measured: 659 px at the corners from 24% coverage, and on the real fisheye set a
k3 of −44 that turned a fifth of the frame into voids). The price is a limit that nothing in
the fit can detect: a lens whose distortion genuinely needs a sixth-order term is corrected
to fourth order only, and the error grows toward the corners (measured with a true k3 of 0.1:
rms 0.34 px, no warning, and the applied correction 77 px wrong at the corners). Fit quality
is reported as a fraction of the frame diagonal so it means the same at any resolution;
`high_reprojection_error` fires above 0.15%. That number is the fit at the detected corners,
not a bound on the residual distortion elsewhere — measured on the real set, two models that
fit its boards at rms 13.0 and 13.1 px disagree by 34% on a centre plant's corrected area.
A calibration that earns the warning corrects visibly and not exactly; sharper, more varied,
wider-spread board views tighten it.

The corrected image is cropped to the **largest rectangle of real pixels**, computed from
an actual validity mask rather than OpenCV's approximate ROI (which, measured on the test
camera model, retained 566 fabricated black pixels — and on the real fisheye photo
over-cropped by half again as much as needed: 0.58 of the frame against the true 0.39 under
the 1.8 model, 0.19 with k3 fixed). When no usable rectangle exists the frame is returned
uncropped with the count of fabricated pixels — void and void-blended border alike — in
`residual_void_px` and a `distortion_voids_remain` warning. The calibration is cached on
the directory's content, so batches re-use it. Segment the corrected file, and take
`px_per_mm` from `calibrate_scale_from_marker` **on the corrected file**: a scale
calibrated on the original belongs to a geometry that no longer exists.

## Getting the polarity right

`object_type` decides which side of the threshold is the plant, and it is the easiest way to
get a confidently wrong answer — that is the right-hand image at the top of the README.

Two things guard against it. `suggest_segmentation` reports what **both** polarities yield on
your image before you commit, alongside a contact sheet of every colourspace:

![colourspace contact sheet](https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/suggest-colorspaces.png)

And `segment` emits an `implausible_coverage` warning when the mask covers more than half the
frame. Neither refuses the measurement, because a macro shot of a single leaf legitimately
fills the frame — they make the choice visible rather than making it for you.

`ambiguous` in the polarity report only compares the two polarities' coverage; it is not a
quality verdict. A recommendation made of specks — on a sorghum photo, `a`/otsu recommended
'dark' with 118 components of chamber-wall texture — is flagged `noisy_segmentation`, with each
polarity's `largest_fraction` (largest component / masked pixels) reported so you can see why.
The same rule fires as an advisory on `segment()` and blocks in the batch.

`fill_size` deletes any component smaller than itself, so a small specimen can vanish
entirely. When that happens `segment` reports `fill_erased_mask` and names the size to drop
below, rather than letting it look like a bad channel choice.

## Refining a mask

When the overlay is nearly right — a hole in a leaf, specks on the background, a
pot rim segmented alongside the plant — `refine()` fixes the mask instead of sending
you back to hunt for a threshold:

```json
{
  "session_id": "…",
  "ops": [{ "op": "fill_holes" }, { "op": "keep_largest", "n": 1 }]
}
```

Ops run in the order given: `fill_holes`, `fill(size)`, `erode(ksize, iterations)`,
`dilate(ksize, iterations)`, `opening(ksize)`, `closing(ksize)`, `median_blur(ksize)`,
and `keep_largest(n)`; `list_methods()` documents each parameter's constraints.
Every op is validated before any runs — PlantCV silently does nothing for
`fill(size=-1)` or `erode(iterations=0)`, and a no-op recorded as a refinement is a
lie — and a refinement that leaves no measurable plant is **refused** rather than
turned into a session that measures zeros.

`refine()` mints a **new** session and returns its overlay; the original stays
measurable, so a refinement you dislike is simply discarded. Trait tables from a
refined session carry `lineage`, the ops that produced their mask, so two tables
made differently can be told apart.

The refinement is traced op by op. Any component at least 10% of the largest one
present that an op removes entirely is reported as `refine_dropped_object`, naming
the op and the earlier op that split it off — on a real sorghum photo `opening`
detached a leaf and `keep_largest` threw it away, a 17% change that sat under the
`refine_large_change` alarm and showed only in the overlay.

## Colour correction

`segment(..., color_correct=true)` detects a Macbeth-style ColorChecker in the frame and
corrects to a standard reference, which is what makes colour traits comparable between images
shot under different lighting. `measure()` re-applies the same correction, so traits are always
measured on the pixels the mask was drawn on.

If no card is found this **raises** rather than quietly measuring the uncorrected image —
returning colour traits that look corrected and are not would be the same kind of confident
wrongness as an inverted mask. A partially occluded card (a leaf over half the chips) counts
as not found: the correction matrix cannot be fit on missing chips.

**The card is excluded from the mask.** The card the correction just located is the
instrument, not a specimen, so its region is removed from the mask before any diagnostics or
traits, and the result carries a `color_card_excluded` advisory saying how many pixels went.
The region is a rotated rectangle around the detected chip lattice, one chip pitch beyond the
outer chips, so it follows a tilted card and scales with the chips (1.6.0 used an axis-aligned
box padded by PlantCV's fixed 20-px sample circle: on the real beans photo, whose chips are
~200 px, 32,093 px of chip material stayed in the mask). A leaf within a pitch of the card is
clipped, and the advisory's pixel count is how you notice. The exclusion is a property of the
session: `refine()` re-applies it if an op grows the plant into the card, and `measure()`
repeats the advisory.
On a real photo of beans beside a card, the card's warm chips otherwise merged into the
largest object in the scene — which suppressed `multi_specimen` (the beans all read as
minor next to it) and dominated the group traits, with nothing warned. If the mask is empty
after exclusion, the threshold was selecting only the card and the image is refused as
`empty_mask`. To measure a card chip as a size marker, that is `calibrate_scale_from_marker`,
not `segment`.

**An incomplete card is refused.** PlantCV checks that every chip it finds holds one grid
centre, not that every centre has a chip, so a card with a chip missing or covered corrects
"successfully" with the fit distorted for every pixel (a mean shift of 19 levels with one chip
erased). After the fit each chip is read back against its reference; a chip more than 0.45
(0–1 RGB units) off refuses the image, naming the chip. Complete real cards read under 0.3.

**`exclude_color_card=true`** keeps the card out of the mask WITHOUT correcting colours, for
size-only work where comparable colours are not needed; it raises if there is no card. With
neither flag the server never looks for a card, so a card left in the frame WILL be measured
as plant material — crop it out of the photo, or keep it out of the cells of a `rect_grid`.
A second, differently sized card that PlantCV's chip-size filter drops is not excluded either.

## Measuring a tray

`measure()` treats the whole image as one region, so on a tray it merges every plant into one
object — the `multi_specimen` warning in the README's example. `measure_regions(session_id,
nrows, ncols)` measures each plant separately and returns the overlay with every region
outlined and numbered, so a row can be matched to a plant:

![region overlay: four numbered cells](https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/regions-overlay.png)

`mode="auto_grid"` infers the grid from the mask; `mode="rect_grid"` takes explicit
`coord`/`height`/`width`/`spacing` when the mask is too sparse to infer a layout or the cells
must line up with pots. An empty region returns `measured=false` and a reason, never zeros.
It works on every session kind: RGB traits, thermal temperatures, or hyperspectral index
statistics (`indices`) per plant.

Two guards come from a real X-Rite tray photo. PlantCV assigns an object that overlaps two
cells **whole** to one of them, so a misaligned grid returned objects 620–785 px wide inside
369-px cells — two plants per row — and called the other cell empty. A region whose object's
bounding box is ≥1.25× the cell (leaf-tip overhang on a clean tray reaches 1.02×; merged
neighbours 1.68–2.13×) carries `object_exceeds_region`; the cell whose material went to a
neighbour says `object_claimed_by_neighbour` naming it; and under `auto_grid` both together
raise `grid_misaligned`, pointing at `rect_grid`. A cell whose labelled object is several
comparably-sized components carries `multi_specimen` — several plants share the cell, or one
plant was split by the threshold; the overlay tells which. There is deliberately no per-cell
coverage check: a plant filling a tight cell is the happy path, not an inverted mask. What a
cell CAN report wrongly is a fragment: an inverted tray under a 1×2 `rect_grid` handed one cell
the whole background (≥1.25×, caught) and the other a 544-px outline of it, 195×195 — inside
the ratio, above the floor, measured as a plant. A cell whose own object is under 20% of the
mask material inside it is refused as `object_claimed_by_neighbour` (a clean real tray owns
≥ 99.9% of every cell; the misaligned X-Rite tray's intruded-upon cells own 35–39% and their
own object is their plant, so they stay measured).

`auto_grid` fits one cluster per row and per column, so it needs at least that many objects,
spread over the rows and columns asked; a mask that cannot support the layout (one object, or
four objects in one row with two rows asked) is refused by name — before 1.5.5 it surfaced
sklearn's "Found array with 1 sample(s)" or an OpenCV drawing error. A single plant is
`measure()`; a known layout is `rect_grid`.

## Morphology: leaves, stem, branch points

`measure_morphology(session_id)` skeletonises the mask and returns PlantCV's
skeleton traits for **one plant**: per segment, path length, euclidean length,
curvature, angle, tangent angle and insertion angle; per plant, stem height, length
and angle, tip and branch-point counts, cycles, and segment widths. It returns the
**numbered-segment overlay** with the table — a segment `id` is the number drawn on
the picture — because a per-segment number is unreadable without it:

![morphology overlay: numbered skeleton segments](https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/morphology-overlay.png)

Three things here are guards, not pass-throughs, all measured on PlantCV 4.11.3
against a synthetic plant of known geometry:

- A perfectly vertical stem makes PlantCV report `stem_angle = -14373°`. That is not
  an angle, so it is returned as `null` with `stem_angle_undefined`. The same stem
  makes the line PlantCV measures insertion angles against undrawable (its
  extrapolation overflows the frame — a crash inside OpenCV on a real 233-px
  seedling); every `insertion_angle` is then `null` with `insertion_angle_undefined`,
  and the rest of the table is unaffected.
- A mask that is the background (`implausible_coverage` on the session) is refused
  before anything is skeletonised: on a real photo it cost 80 s to skeletonise the
  frame and was then refused for the wrong reason. A stem PlantCV cannot join into
  one piece is refused naming the refine chains that measured the real photos
  (`median_blur 11`; `opening 9` + `median_blur 21`) — `closing` did not repair a
  stem the chain had cut, so it is not offered. When PlantCV loses track of its
  own insertion segments (an internal list desync), every `insertion_angle` is
  `null` with `insertion_angle_undefined` as well.
- `tangent_size` (default 25 px, chosen from a bias sweep) is the window PlantCV fits
  tangents on, from **each** end of a segment. A window longer than half a leaf
  collapses that leaf's insertion angle to `0.0`; `tangent_window_exceeds_segment`
  says so instead of letting the zero read as a measurement.
- `prune_size` decides how many segments a skeleton has. When the count changes by
  more than 30% at twice the value, `prune_size_sensitive` tells you the table
  describes the parameter, not the plant. A skeleton PlantCV cannot analyse at all
  is refused with those counts — and the refusal stops recommending a larger
  `prune_size` once doubling it no longer changes the count; `refine()` is the
  remedy that works there.

Multi-plant masks are refused by name — `refine(keep_largest)` isolates one plant (that is
how the picture above was made from the four-view render). Lengths scale with `px_per_mm`;
angles are always degrees.

## Measuring many images

`measure_images(image_paths, channel, method, ...)` applies one fixed recipe across up to 200
images.

This is the one place the two-step discipline cannot hold literally: nobody reviews two hundred
overlays. So the overlay is replaced by the only honest substitute — **every image runs the same
guards as `segment()`, and any image that trips a blocking guard comes back with no traits at
all**, just a reason and an instruction to inspect it individually. Advisory warnings such as
`multi_specimen` are attached to the traits rather than suppressing them. With a grid,
`noisy_segmentation` stops blocking **when the grid explains the components** — at most four
per cell (a 96-well plate with ten germinated and 86 late wells is 100 objects in 100 cells;
the calibrated noisy scene, 61 specks under a 1×2 grid, came back as two measured "plants"
of 1,620 and 2,592 px before this rule). With two or more cells `implausible_coverage` stops
blocking too (two discs filling their cells are 72% of the frame): an inverted mask is caught
per cell instead — the background is one object spanning every cell, or, when dark dividers cut
it into one island per cell, each island fills its cell and is withheld as `probable_background`
(≥ 85% of the cell in a mask covering most of the frame; the fullest real cells are 0.19–0.51)
— and an image with **no measured row** is refused as `no_region_measured` with the per-cell
reasons, so it reaches the review list. The count rule has a per-cell half too: the calibrated
noisy scene satisfied `61 ≤ 4 × 16` under a 4×4 grid and came back as 13 measured "plants";
now a cell holding several comparable specks in a mask that is texture overall is a
`noise_cluster`, and one such cell refuses the image as `noisy_segmentation` — the grid explained
the count, not the mask. Rows whose object the guard has already called a merge
(`object_exceeds_region`) are withheld — there is no overlay here to check them against. Grid
arguments (`mode`, `coord`, `radius`, …) without `nrows`/`ncols` are refused before any image
runs rather than silently ignored, and duplicates are judged by the file, not the spelling
(`./a.png`, a symlink and the absolute path are one image).

| parameter                                                                           | default            | what it does                                                                                                           |
| ----------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `image_paths`                                                                       | required           | up to 200 paths; duplicates are measured once and listed                                                               |
| `channel`, `method`, `object_type`, `fill_size`, `ksize`, `offset`, `color_correct` | as `segment`       | the recipe — settle it on one image with `segment()` first, then pass the SAME values                                  |
| `analyses`, `px_per_mm`                                                             | `["size"]`, `null` | as `measure`                                                                                                           |
| `max_seconds`                                                                       | `300`              | wall-clock budget; images not started in time return as `not_run` with their paths listed                              |
| `nrows`, `ncols`, `mode`, `coord`, `height`, `width`, `spacing`, `radius`           | none               | give a grid and each image is measured per plant as `measure_regions()` does; rows carry `regions` instead of `traits` |

A recipe error (unknown channel, method, analysis) is raised once, before any image runs.
Every row reports its own `seconds`; the batch reports `elapsed_s`. The summary splits rows
into `measured`, `with_advisories` (with a per-code count), `needs_review` and `not_run`:

```json
{
  "summary": {
    "submitted": 3,
    "unique": 3,
    "measured": 1,
    "with_advisories": 1,
    "advisory_counts": { "frame_clipping": 1 },
    "needs_review": 1,
    "review_paths": ["blank.png"],
    "not_run": 1,
    "not_run_paths": ["late.png"]
  },
  "elapsed_s": 21.4,
  "results": [
    {
      "image_path": "blank.png",
      "measured": false,
      "traits": null,
      "seconds": 0.8,
      "refused_because": "empty_mask — traits withheld because the mask probably does not describe the plant. …"
    },
    {
      "image_path": "late.png",
      "measured": false,
      "traits": null,
      "seconds": null,
      "refused_because": "not run: the 20 s time budget was used up after 2 image(s) (21 s). …"
    }
  ]
}
```

A batch never returns a number the server could not validate — which is weaker than a human
looking at a mask, and is stated plainly rather than implied.

## Hyperspectral and thermal

Sessions are typed (`rgb`, `hsi`, `thermal`); each kind has its own segmenter and measurer,
every measurer refuses the wrong kind naming the right tool, and `measure_regions` accepts all
three. Refusals and advisories are worded per modality — "a different channel or method" is
RGB advice; a cube is told about its threshold and a thermal frame about its band.

| tool                    | key parameters                                                                                                                            | notes                                                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `segment_hyperspectral` | `envi_path` (`.raw` or `.hdr`), `index="ndvi"`, `threshold=0.2`, `object_type="light"`, `white_reference` + `dark_reference`, `fill_size` | one of PlantCV's 31 indices; overlay on the pseudo-RGB             |
| `measure_spectral`      | `indices=[…]`, `include_spectrum`, `include_histograms`                                                                                   | mean/median/std/min/max per index; the per-band spectrum is opt-in |
| `segment_thermal`       | `path` (FLIR `.jpg` via `flyr`, `.csv`, `.npz` of °C), `min_c`, `max_c`, `fill_size`                                                      | segments **by temperature**; overlay on a grey rendering           |
| `measure_thermal`       | `include_histograms`                                                                                                                      | max/min/mean/median °C via PlantCV's `analyze.thermal`             |

Measured facts that drive the guards:

- Cubes are usually **integer counts**, and an index computed on integer data
  wraps around silently — NDVI on a uint16 test cube read 65.3 on a [-1, 1]
  index. So the cube is either **calibrated** to reflectance from
  `white_reference` + `dark_reference` (refused if `white − dark` is not positive
  everywhere: PlantCV would clip the degenerate case to 1.0 silently), or cast to
  float with the `uncalibrated_cube` warning — the indices are then relative, not
  reflectance. The references are pinned by digest at segmentation and re-checked
  at measurement.
- An index the cube's wavelength range cannot support is **refused by name**,
  not returned as null.
- A threshold past either end of the index's range selects everything or nothing;
  on a real leaf cube (NDVI 0.19–0.89) the default `threshold=0.2` selected every
  pixel. `threshold_outside_range` says which side, with the range.
- A thermal frame is a different sensor from the RGB camera, so a mask is never
  borrowed from an RGB session. `measure_thermal` is validated against PlantCV's own
  test frame to six decimal places and against a real FLIR JPEG.
- Calling `segment_thermal` without a band refuses **with** the frame's range and
  p5–p95 percentiles (the plant is usually the cool tail); a band entirely outside
  the frame is refused naming the range, and one enclosing the whole frame gets
  `threshold_outside_range`.

## Warnings and refusals

Every result carries a `warnings` array of `{code, message}`; each message names the next
action. **Blocking** codes withhold traits in the unattended batch; everything else is
attached to the numbers it qualifies. A **refusal** is an error with a named class and the
same kind of guidance (`WrongSessionKindError: … Use measure_thermal() for it`), raised
instead of a result.

| code                                                                | raised by                                              | meaning                                                                                                               | batch    |
| ------------------------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | -------- |
| `implausible_coverage`                                              | any segmenter, `measure`                               | mask covers > 50% of the frame — probably the background                                                              | blocking |
| `empty_mask`                                                        | any segmenter                                          | nothing selected                                                                                                      | blocking |
| `fill_erased_mask`                                                  | any segmenter                                          | thresholding found objects; `fill_size` deleted them all — names the size to use                                      | blocking |
| `noisy_segmentation`                                                | any segmenter, `suggest`                               | ≥50 non-major components and no dominant object — background texture                                                  | blocking |
| `multi_specimen`                                                    | any segmenter, `measure`, `measure_regions` (per cell) | several comparably-sized objects; the number describes the group → `measure_regions`                                  | advisory |
| `frame_clipping`                                                    | any segmenter, `measure`                               | a major object touches the frame edge; size traits are lower bounds (background slivers at the edge do not count)     | advisory |
| `color_card_excluded`                                               | `segment`, `refine`, `measure`, `measure_images`       | the detected colour card's region was removed from the mask — the card is the instrument                              | advisory |
| `probable_background` (per cell)                                    | `measure_images` with a grid                           | in a mask covering most of the frame, this cell's object fills ≥ 85% of it: background between dividers, not a plant  | withheld |
| `noise_cluster` (per cell)                                          | `measure_images` with a grid                           | several comparable specks in one cell of a mask that is texture overall; the image is refused as `noisy_segmentation` | refused  |
| `threshold_outside_range`                                           | `segment_hyperspectral`, `segment_thermal`             | the threshold or band lies past the data's range; selects everything or nothing                                       | advisory |
| `uncalibrated_cube`                                                 | `segment_hyperspectral`                                | integer counts cast to float; indices are relative                                                                    | advisory |
| `nan_pixels`                                                        | thermal and spectral measurers                         | non-finite values excluded from statistics; counts given                                                              | advisory |
| `refine_large_change`                                               | `refine`                                               | mask changed > 25% — a different outline, not a cleanup                                                               | —        |
| `refine_dropped_object`                                             | `refine`                                               | an op removed a component ≥ 10% of the largest — a leaf, not a speck; names the op that split it                      | —        |
| `object_exceeds_region`                                             | `measure_regions`                                      | a cell's object is ≥ 1.25× the cell — two plants, or a misaligned grid                                                | —        |
| `object_claimed_by_neighbour`                                       | `measure_regions`                                      | this cell's material was assigned whole to a neighbour, or its own object is < 20% of it                              | —        |
| `grid_misaligned`                                                   | `measure_regions` (`auto_grid`)                        | empty cells plus exceeded cells → use `rect_grid`                                                                     | —        |
| `region_count_mismatch`                                             | `measure_regions` (`auto_grid`)                        | PlantCV built fewer regions than asked                                                                                | —        |
| `implausible_longest_path`                                          | `measure`, `measure_regions`                           | PlantCV's `longest_path` is shorter than the bounding box allows                                                      | —        |
| `stem_angle_undefined`                                              | `measure_morphology`                                   | a vertical stem; PlantCV's angle is not an angle → `null`                                                             | —        |
| `insertion_angle_undefined`                                         | `measure_morphology`                                   | a vertical stem; the stem line cannot be drawn, every `insertion_angle` → `null`                                      | —        |
| `tangent_window_exceeds_segment`                                    | `measure_morphology`                                   | `tangent_size` longer than half a segment; its angles collapse to 0                                                   | —        |
| `prune_size_sensitive`                                              | `measure_morphology`                                   | segment count changes > 30% at 2× `prune_size`                                                                        | —        |
| `skeleton_has_cycles`, `no_leaf_segments`, `no_stem_segment`        | `measure_morphology`                                   | skeleton topology PlantCV's leaf/stem split cannot use                                                                | —        |
| `marker_touches_crop_edge`, `marker_not_round`, `marker_fills_crop` | `calibrate_scale_from_marker`                          | the detected marker is probably not the marker                                                                        | —        |
| `minor_components_inflate_extent`                                   | any segmenter, `refine`, `measure`                     | material far from the main object stretches width/height/ellipse/hull traits; names the extent driver and the remedy  | advisory |
| `lens_corrected`                                                    | `correct_lens_distortion`                              | which file to measure from here on; re-calibrate any scale on it                                                      | —        |
| `thin_calibration`                                                  | `correct_lens_distortion`                              | fewer than 5 checkerboard frames went into the calibration                                                            | —        |
| `high_reprojection_error`                                           | `correct_lens_distortion`                              | the calibration fits its own boards worse than 0.15% of the frame diagonal; the correction is only as good as the fit | —        |
| `low_calibration_coverage`                                          | `correct_lens_distortion`                              | the boards covered under 40% of the frame; the correction is extrapolated toward the edges and corners                | —        |
| `distortion_voids_remain`                                           | `correct_lens_distortion`                              | no usable all-valid crop exists; fabricated pixels remain (void and void-blended border, count in `residual_void_px`) | —        |
| `outlier_frames_dropped`                                            | `correct_lens_distortion`                              | views that fit far worse than the others, or move the answer far more, were dropped by name (`frames_outliers`)       | —        |
| `duplicate_frames_ignored`                                          | `correct_lens_distortion`                              | byte-identical checkerboard files counted once (`frames_duplicates`)                                                  | —        |
| `focal_length_uncertain`                                            | `correct_lens_distortion`                              | the views determine the focal length only loosely (`focal_uncertainty` above 2.5%); tilt the board more               | —        |

Refusals you will meet: a degenerate mask at `measure` (`DegenerateMaskError`), a refinement
that erases the plant (`RefinementErasedMaskError`) or an invalid op list (`RefineSpecError`,
nothing applied), a skeleton PlantCV cannot analyse (`MorphologyRefusedError`, with counts),
the wrong session kind (`WrongSessionKindError`), a file that changed since segmentation
(`ImageChangedSinceSegmentationError`, SHA-256), a grayscale or undecodable file
(`NotColorImageError`), no colour card when one was asked for (`ColorCardNotFoundError`), a
checkerboard directory with fewer than three detectable boards, duplicate poses, a
meaningless fit, or boards that leave the focal length undetermined (`LensCalibrationError`,
with the per-frame accounting), an image whose
resolution differs from its lens calibration (`CalibrationResolutionMismatchError`), and
unknown channel / method / index / analysis names, each listing the valid ones.

## Crash containment: the analysis worker

Every PlantCV analysis (`measure`, `measure_regions`, `measure_morphology`,
`measure_images`, `refine`) runs in a **worker subprocess** by default. Two things
follow. A native crash inside PlantCV/OpenCV — the class 0.5.0 closed one instance
of by validating `rect_grid` geometry — becomes a tool error (_"the analysis worker
died during 'measure' (signal 11) … the server is still running"_) and the next call
starts a fresh worker; the server itself never executes native analysis code. And
PlantCV's process-global state (`pcv.outputs`, its cached colour palette, its
sample label) lives in the worker, not in the server.

Measured cost on a 3000×3000 image: **+7.7 %** wall time per `measure()` (the arrays
cross a pipe), against a gate of 25 % that would have made it opt-in. The first
analysis after start-up pays the worker's import (~1 s). Turn it off with
`plantcv-mcp --no-isolate` or `PLANTCV_MCP_ISOLATE=0`.

## Restricting what the server may read

By default the server reads any image the host user can read (see
[Security](#security-and-trust-boundary)). To confine it, name one or more directories:

```bash
plantcv-mcp --root /data/phenotyping --root /data/trials
# or
PLANTCV_MCP_ROOTS=/data/phenotyping:/data/trials plantcv-mcp
```

Every path argument — `segment`, `suggest_segmentation`,
`calibrate_scale_from_marker`, every entry of `measure_images`, ENVI cubes and their
`.hdr` siblings, thermal files — is resolved with symlinks and `..` followed
**first**, then checked against the roots; a symlink inside a root pointing outside
is refused, and a batch with one stray path is refused whole before anything is
read. `list_methods()` reports the policy as `read_roots`, and the roots themselves are
resolved once (a root directory renamed and replaced by a symlink after the server read
its configuration does not move the policy). The check on the name is the early, legible
refusal; with roots configured the binding one is made on the file that was actually
opened — the descriptor is asked where it lives (Linux `/proc`, macOS `F_GETPATH`), so a
directory renamed and replaced by an outside symlink between the check and the open is
refused rather than followed — and only regular files are read (a FIFO, socket or device
at a checkerboard member's name is a named skip, whether it was there from the start or
swapped in after the check). Written files — the corrected image — go through a descriptor
of their directory, opened once and, wherever the platform can say where it lives, checked
to be the directory intended, so the same swap cannot redirect a write; the image is
written to a random hidden `.<name>.<token>.partial` sibling first and swapped into place,
so a crash can leave such a residue and never a half-written output. A platform that
cannot report a descriptor's path refuses to READ with roots configured and refuses to
verify the write's directory only if roots are configured; without roots it writes
unverified, as 1.9.0 did.

## Security and trust boundary

**Unless you configure read roots, this server reads image files anywhere on the
host filesystem, and returns them to the model as images.**

Every path-taking tool passes the path to a reader; with no `--root` there is no
allow-list and no sandbox, and any path the model asks for — that the
operating-system user running the server can read — will be decoded and returned as
a base64 image in the model's context.

Practical consequences:

- Treat it like any other local filesystem MCP server. Run it as a user whose
  read access you are comfortable exposing to the model driving it, and set
  `--root` to the directories that hold the imagery.
- A prompt-injected or adversarial model can use it to view arbitrary **image**
  files the roots allow. Non-image files fail to decode and raise, but the error
  message discloses whether the path exists.
- Do not run it as root, and do not expose it to untrusted input on a machine
  holding sensitive imagery.

The read-root allow-list is the whole of the sandboxing; there is no network access,
no execution of anything but PlantCV, and the only write is
`correct_lens_distortion`'s corrected image — next to its input, or at an explicit
`output_path` that must not already exist and must not be a symlink.

## Limitations

Morphology traits are single-plant (`measure_morphology()` refuses a tray); per-region
morphology is not implemented. Hyperspectral support covers ENVI cubes; other cube
formats (nd2, ArcGIS) and photosynthesis (PSII) data are not exposed.

The batch is serial and in-process: real photographs take seconds each, which is what
`max_seconds` exists for.

Sessions are in-memory and capped (8 by default, LRU-evicted). They do not
survive a server restart.

The server is developed and tested on Linux and relies on POSIX file semantics
(`O_NOFOLLOW`, directory descriptors, hard links, `/proc`): the containment guards
around reading and writing files are inert on Windows, read roots cannot be enforced
there, and the corrected image is written through plain exclusive creates.

Images on these pages are rendered from `tests/fixtures/multi_specimen.png`, an original render
by the author, and regenerate from committed code.
