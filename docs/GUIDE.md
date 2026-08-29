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
the polarity is wrong and the background was measured.

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
wrongness as an inverted mask.

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
raise `grid_misaligned`, pointing at `rect_grid`.

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
  an angle, so it is returned as `null` with `stem_angle_undefined`.
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
`multi_specimen` are attached to the traits rather than suppressing them.

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

| code                                                                | raised by                                  | meaning                                                                                          | batch    |
| ------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------ | -------- |
| `implausible_coverage`                                              | any segmenter, `measure`                   | mask covers > 50% of the frame — probably the background                                         | blocking |
| `empty_mask`                                                        | any segmenter                              | nothing selected                                                                                 | blocking |
| `fill_erased_mask`                                                  | any segmenter                              | thresholding found objects; `fill_size` deleted them all — names the size to use                 | blocking |
| `noisy_segmentation`                                                | any segmenter, `suggest`                   | ≥50 non-major components and no dominant object — background texture                             | blocking |
| `multi_specimen`                                                    | any segmenter, `measure`                   | several comparably-sized objects; the number describes the group → `measure_regions`             | advisory |
| `frame_clipping`                                                    | any segmenter, `measure`                   | plant touches the frame edge; size traits are lower bounds                                       | advisory |
| `threshold_outside_range`                                           | `segment_hyperspectral`, `segment_thermal` | the threshold or band lies past the data's range; selects everything or nothing                  | advisory |
| `uncalibrated_cube`                                                 | `segment_hyperspectral`                    | integer counts cast to float; indices are relative                                               | advisory |
| `nan_pixels`                                                        | thermal and spectral measurers             | non-finite values excluded from statistics; counts given                                         | advisory |
| `refine_large_change`                                               | `refine`                                   | mask changed > 25% — a different outline, not a cleanup                                          | —        |
| `refine_dropped_object`                                             | `refine`                                   | an op removed a component ≥ 10% of the largest — a leaf, not a speck; names the op that split it | —        |
| `object_exceeds_region`                                             | `measure_regions`                          | a cell's object is ≥ 1.25× the cell — two plants, or a misaligned grid                           | —        |
| `object_claimed_by_neighbour`                                       | `measure_regions`                          | this cell's material was assigned whole to a neighbour                                           | —        |
| `grid_misaligned`                                                   | `measure_regions` (`auto_grid`)            | empty cells plus exceeded cells → use `rect_grid`                                                | —        |
| `region_count_mismatch`                                             | `measure_regions` (`auto_grid`)            | PlantCV built fewer regions than asked                                                           | —        |
| `implausible_longest_path`                                          | `measure`, `measure_regions`               | PlantCV's `longest_path` is shorter than the bounding box allows                                 | —        |
| `stem_angle_undefined`                                              | `measure_morphology`                       | a vertical stem; PlantCV's angle is not an angle → `null`                                        | —        |
| `tangent_window_exceeds_segment`                                    | `measure_morphology`                       | `tangent_size` longer than half a segment; its angles collapse to 0                              | —        |
| `prune_size_sensitive`                                              | `measure_morphology`                       | segment count changes > 30% at 2× `prune_size`                                                   | —        |
| `skeleton_has_cycles`, `no_leaf_segments`, `no_stem_segment`        | `measure_morphology`                       | skeleton topology PlantCV's leaf/stem split cannot use                                           | —        |
| `marker_touches_crop_edge`, `marker_not_round`, `marker_fills_crop` | `calibrate_scale_from_marker`              | the detected marker is probably not the marker                                                   | —        |

Refusals you will meet: a degenerate mask at `measure` (`DegenerateMaskError`), a refinement
that erases the plant (`RefinementErasedMaskError`) or an invalid op list (`RefineSpecError`,
nothing applied), a skeleton PlantCV cannot analyse (`MorphologyRefusedError`, with counts),
the wrong session kind (`WrongSessionKindError`), a file that changed since segmentation
(`ImageChangedSinceSegmentationError`, SHA-256), a grayscale or undecodable file
(`NotColorImageError`), no colour card when one was asked for (`ColorCardNotFoundError`), and
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
read. `list_methods()` reports the policy as `read_roots`.

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
no writing, and no execution of anything but PlantCV.

## Limitations

Morphology traits are single-plant (`measure_morphology()` refuses a tray); per-region
morphology is not implemented. Hyperspectral support covers ENVI cubes; other cube
formats (nd2, ArcGIS) and photosynthesis (PSII) data are not exposed.

The batch is serial and in-process: real photographs take seconds each, which is what
`max_seconds` exists for.

Sessions are in-memory and capped (8 by default, LRU-evicted). They do not
survive a server restart.

Images on these pages are rendered from `tests/fixtures/multi_specimen.png`, an original render
by the author, and regenerate from committed code.
