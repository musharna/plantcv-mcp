# Changelog

All notable changes to `plantcv-mcp` are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.8.0] — 2026-08-31

Findings from the first lens-distortion dogfood: the released pipeline run on
the real PlantCV fisheye tutorial photo, then compared against the same photo
undistorted through PlantCV's own checkerboard calibration.

### Added

- **`correct_lens_distortion` — the fourteenth tool, and the first that
  writes.** Measured on the fisheye photo, distortion inflated the
  centre-frame plant's area 2.13x (width 1.69x, height 1.22x), shifted shape
  traits (eccentricity 0.56 → 0.76), and gave the same pot three different
  pixel scales (rim 1.73x, face height 1.31x, base 1.59x) — an anisotropic
  error no `px_per_mm` can compensate, whatever it is calibrated on. The tool
  builds a calibration from a directory of checkerboard photos and writes
  `<image>_undistorted.png` next to the input (an explicit `output_path`
  refuses to overwrite an existing file). It reports what
  `pcv.transform.checkerboard_calib` computes and discards: every frame as
  used or skipped by name, the rms reprojection error, and the valid-pixel
  ROI — the corrected image is cropped to it, because the remap otherwise
  fabricates black voids that a value threshold measures as objects. Fewer
  than three detectable boards is a typed `LensCalibrationError` (PlantCV
  crashes with a raw cv2 error on that directory, and "calibrates" from one
  frame without comment). Calibrations are cached on the directory's content
  digest, so a batch re-uses them. Advisories: `lens_corrected` (measure the
  corrected file; re-calibrate any scale on it), `thin_calibration` (<5
  frames), `distortion_voids_remain` (no valid crop exists).

### Fixed

- **A far-away speck silently corrupted the extent traits.** On the fisheye
  photo, a 1,818-px sliver in the opposite corner from a 458,078-px plant
  made `measure()` report width 2040 px (the true plant: 940), with
  longest_path, ellipse and convex-hull traits equally wrong — and
  `warnings: []`: `multi_specimen` needs a comparably-sized object,
  `frame_clipping` deliberately ignores minor slivers (1.6.0), and
  `noisy_segmentation` needs dozens of specks. New
  `minor_components_inflate_extent` advisory on every segmenter and measurer
  when non-major components stretch the union extent >10% beyond the major
  components' own, naming the offender and the `keep_largest`/`fill_size`
  remedy. Judged against the MAJOR union, so a genuine second plant
  (multi_specimen territory) does not trip it.
- **`marker_touches_crop_edge` only suspected polarity.** On a scene with no
  dedicated marker, the pot+soil+plant is one contiguous dark object; every
  crop through it touched the crop edge, and four plausible pot crops
  returned px_per_mm from 10.8 to 18.1. The warning now also names this
  case: when the object continues beyond the crop there is no isolatable
  marker, and traits should stay in pixels.

## [1.7.0] — 2026-08-30

Findings from an independent panel audit of 1.6.0 (six judges in full with a
rebuttal round; one lost to our harness), every one reproduced against the code
— and the central one on the real beans photo — before it was fixed.

### Fixed

- **The card exclusion missed the card.** 1.6.0 padded the detected chip
  region by "the median chip extent" — which was PlantCV's fixed 20-px sample
  circle, ~41 px whatever the chips measure. On the real beans photo (chips
  ~200 px) 32,093 px of chip material lay outside the exclusion and five card
  components of up to 13,678 px were still measured as plant; the 1.6.0
  "eleven major objects" included one of them. The region is now a rotated
  rectangle around the detected chip lattice, one chip pitch (the median
  centre-to-centre distance) beyond the outer chips: on the same photo the
  residual is 0 px and no bean pixel is touched; the fixture card at 180-px
  chips is covered; and a card rotated 30° no longer takes 18% bench with it
  (an axis-aligned box zeroed a plant in its corner triangle).
- **An incomplete card was accepted.** PlantCV verifies that every chip it
  finds holds one grid centre, not that every centre has a chip, so erasing
  one interior chip "corrected" the image with every pixel shifted by ~19
  levels (max 127). `correct_color()` now runs the detect → sample → fit
  pipeline itself (one detection instead of two, which also closes the
  unwrapped second call) and reads each chip back against its reference
  after the fit: a chip more than 0.45 (0–1 RGB units) off refuses the image,
  naming the chip. Complete cards — the fixture, the CameraTrax card, the
  maize photo's card, the booth X-Rite — read ≤ 0.3; the erased chip 0.6.
- **`refine()` grew the plant back into the card.** The session kept only
  `color_correct`; a `dilate(ksize=101)` on a plant beside the card put 5,990
  px inside the card region and `measure()` sampled them as plant. Sessions
  now carry the card polygon and the exclusion is an invariant: `refine()`
  re-applies it (with a `color_card_excluded` note when an op grew into the
  card) and `measure()` repeats the advisory the trait table had lost.
- **Background islands measured as plants under a grid.** The ≥ 2-cell
  coverage demotion assumed an inverted background is one object spanning
  every cell; dark dividers cut it into one island per cell, 96% of the frame,
  and both rows measured. A cell whose object fills ≥ `CELL_BACKGROUND_COVERAGE
= 0.85` of it, in a mask that covers most of the frame, is withheld as
  `probable_background` (the fullest real cells are 0.19–0.51 of their cell,
  the dense fixture 0.72; the islands 0.90–0.96).
- **A fine grid laundered a noisy mask.** `components ≤ 4 × cells` scales with
  the grid: the calibrated noisy scene, refused under 1×2 and 2×2, was measured
  under 4×4 (13 "plants") and 10×10 (44). A cell holding several comparable
  specks in a mask that is texture overall falsifies the grid's claim that
  each cell is one plant: such cells are withheld as `noise_cluster` and the
  image is refused as `noisy_segmentation`. The late-germination plate and the
  split-plants tray have no such cell and still measure.

### Added

- `exclude_color_card` on `segment()` and `measure_images()`: keep the card out
  of the mask without correcting colours (raises when there is no card).
  Excluding the instrument and calibrating colours are independent choices,
  and an unattended batch that does not need comparable colours must not
  measure the card either.

### Documented, by design

- `frame_clipping` still misses a clipped specimen under 0.25× the largest
  interior one (and `multi_specimen` is silent for it too); the advisory is
  about specimen-scale clipping. A second, differently sized card that
  PlantCV's chip-size filter drops is not excluded. With neither flag the
  server never looks for a card.

### Tests

Eight new, each watched red on the 1.6.0 code (one via the old tuple API);
the card-region test now asserts a polygon. 319 tests.

## [1.6.0] — 2026-08-30

Findings from the first real-photograph run of `calibrate_scale_from_marker`
and colour correction (a CameraTrax 24ColorCard beside eleven kidney beans, an
X-Rite ColorChecker in an imaging booth, a card half-occluded by a cucumber
leaf), each reproduced against the code before it was fixed.

### Changed

- **The detected colour card is excluded from the measured mask.** On the real
  beans photo, `color_correct=true` detected the card, corrected the colours —
  and then measured the card: its warm chips merged into the largest object in
  the scene (254,571 px), which suppressed `multi_specimen` (the eleven beans
  all read as minor beside it) and dominated the group traits — `measure()`
  returned a 727,848-px "plant" spanning the frame with only `frame_clipping`
  warned, and the batch row came back `measured: true`. The card the
  correction just located is the instrument, not a specimen: its region
  (detected chip extent, padded by the median chip size) is now removed from
  the mask before any diagnostics, in `segment()` and `measure_images()` both,
  with a `color_card_excluded` advisory saying how many pixels went. On the
  same photo the eleven beans are now the eleven major objects and
  `multi_specimen` fires. A mask that is empty after exclusion is refused as
  `empty_mask` — everything the threshold selected was card. With
  `color_correct=false` the server never looks for a card; the guide now says
  to keep cards out of the frame or out of the measured regions there.
- **`frame_clipping` is judged per component, not on the mask as a whole.**
  Two 5-px-wide background slivers at the frame edge declared the eleven
  interior beans "cut off" (`in_bounds: false`). Clipping is a claim about the
  specimen, so it now requires an edge-touching component comparable to the
  largest (the `analyze_mask` major rule, ≥ 0.25×); a plant that is itself
  mostly out of frame still warns, because its visible sliver IS the largest
  object.
- `correct_color()` returns `(corrected image, card region)` instead of the
  image alone — the card's location is the by-product that makes exclusion
  possible.

### Verified on the real photos (no change needed)

- The scale tool is exact: booth chips ground-truthed at 121–127 px measured
  127 and 124; two CameraTrax chips agreed within 0.5%; a crop that clipped a
  chip fired `marker_touches_crop_edge` (guarding the 2.4% scale error it
  caused), and the wrong polarity fired it both times it was tried.
- Chip calibration (14.85 px/mm) through a 1×1 `rect_grid` cell returned a
  20.7×10.6 mm kidney bean — the full end-to-end mm chain.
- A half-occluded card refuses loudly, quoting PlantCV's inner reason, and
  per-image in a batch; a CameraTrax (non-X-Rite) card corrects fine.

### Tests

Six new or updated, each watched red first — five on the pre-fix code, one
(`test_correction_reports_where_the_card_is`) against the old single-return
API. 310 tests.

## [1.5.5] — 2026-08-30

Findings from an independent panel audit of 1.5.4 (big-pickle and or-deepseek
in full with a rebuttal round; codex timed out mid-run but its narration carried
three reproduced counterexamples), each reproduced against the code before it
was fixed, plus one crash class found while reproducing them.

### Fixed

- **A grid demoted `noisy_segmentation` unconditionally.** 1.5.2 let any grid
  turn the block into an advisory on the premise that the per-cell floor guards
  each cell; it guards near-empty cells only. The calibrated noisy scene (a
  90-px plant and 60 specks) is refused with no grid, but under `nrows=1,
ncols=2` came back as two measured plants of 1,620 and 2,592 px — clusters
  of specks — with `measured: 1, needs_review: 0`. A grid now explains a
  many-component mask only when it has about as many objects as cells (at
  most `NOISE_EXPLAINED_PER_CELL = 4` per cell); the late-germination plate
  (100 objects, 100 wells) still measures, and the refusal says which grid did
  not explain how many components.
- **A dense valid tray was refused for whole-frame coverage.** Two discs
  filling their 1×2 `rect_grid` cells are 72% of the frame, and the
  `implausible_coverage` block ran before the grid. With two or more cells it
  is now an advisory: an inverted mask is caught per cell (the background is
  one object spanning every cell), which needed the next two fixes to hold.
- **A cell measured a fragment of its neighbour's object.** Under that
  inverted tray PlantCV handed cell 1 the whole 400×200 background (caught as
  exceeding) and cell 0 a 544-px OUTLINE of the same object, 195×195 — inside
  the 1.25× ratio, above the floor, measured as a plant with `area=544`. A
  cell whose own object is under `OWNED_MATERIAL_FRACTION = 0.2` of the mask
  material inside it is refused as `object_claimed_by_neighbour`, naming the
  region that owns the rest. Calibrated on the real trays: the clean
  arabidopsis tray owns ≥ 0.999 of every cell; the misaligned X-Rite tray's
  intruded-upon cells own 0.35–0.39 and their own object is their plant
  (kept); the fragment owned 0.049.
- **An image with no measured row counted as measured.** One ellipse spanning
  both cells of a 1×2 grid: both rows `object_exceeds_region`, yet the entry
  was `measured=true` and the summary said `measured 1, needs_review 0`. It is
  now refused as `no_region_measured — 0 of N cells measured: …` with the
  per-cell reasons, and lands on `review_paths`.
- **`auto_grid` leaked sklearn and OpenCV errors.** It fits one mixture
  component per row and per column: one object under any grid raised
  `ValueError: Found array with 1 sample(s) … GaussianMixture` (the batch quoted
  it as the reason), and objects that do not spread into the rows asked (four
  discs in one row, 2×2 grid) gave NaN centres and `cv2.error` from
  `drawing.cpp`. Both are now `RegionSpecError: auto_grid could not infer a
RxC layout from the N object(s) …`, pointing at `rect_grid` or `measure()`.
- **Grid arguments without `nrows`/`ncols` were silently ignored.**
  `mode="rect_grid"`, `mode="bogus"`, `radius=-5` and full rectangle geometry
  each ran a whole-frame measurement with no error, because the recipe
  validator only looked at grid arguments once a grid was given. They are now
  refused before any image runs.
- **Batch duplicates were judged by the spelling.** `./a.png`, a symlink and
  the absolute path were three measurements of one image; the key is now the
  real path, and the summary still lists what was dropped as it was written.
- **The morphology inverted-mask refusal names its one legitimate case.** A
  macro shot of one leaf can genuinely fill the frame (the reason `measure()`
  only warns); morphology keeps refusing — the background's skeleton costs
  80 s and is never a plant — and now says to crop the photo.

### Rejected with evidence

- `vx == 0.0` in `_stem_line_leaves_int32` should be a tolerance (or-deepseek):
  a near-zero `vx` falls through to the int32 overflow check and returns True
  there; the proposed tolerance would only widen the swallow.

### Tests

- 15 new tests, each seen failing first — the owned-material guard by disabling
  `OWNED_MATERIAL_FRACTION` — with positive controls inside the same test
  (the late-germination tray, the dense tray upright, the X-Rite cells). 302
  tests.

## [1.5.4] — 2026-08-29

A remedy-convergence sweep over the leafy real photos (4 photos × 4 refine
chains × 4 prune sizes, after 1.5.3 made each call cheap), plus mutation round 7
over the 1.5.3 guards.

### Fixed

- **A third crash inside `segment_insertion_angle` is named.** PlantCV keeps one
  list of "pruned away" flags and another of computed angles; when an insertion
  segment vanishes for a reason it did not flag, the two desync and it pops an
  empty list — `IndexError` at its line 140 on a real 37-leaf photo. Only an
  `IndexError` raised inside that PlantCV module is turned into
  `insertion_angle_undefined` (the traceback is checked); the rest of the table
  is kept. An `IndexError` from anywhere else still raises.
- **The refusal remedies say what was measured, not what sounded right.** The
  "stem cannot be joined" remedy shipped in 1.5.3 suggested `closing` /
  `fill_holes`; on the real sorghum photo `closing` 7 and 15 both left the stem
  unjoinable — the break came from the refine chain (`opening 5` +
  `median_blur 11`), and a different chain measured the plant. Both that remedy
  and the "too many tips" one now name the chains that worked on the real
  photos (`median_blur 11`; `opening 9` + `median_blur 21`; `prune_size 100`
  warning-free on all three) and that `median_blur 5` was not enough.

### Tests

- Mutation round 7 (`docs/MUTATION-CHECKS.md`): nine 1.5.3 guards; three were
  removable — the palette reset on entering `isolated_pcv_outputs`, the restore
  of the host's palette on exit, and the refit that verifies a `cv2.error` is
  the vertical-stem case — and are now pinned.
- 287 tests.

## [1.5.3] — 2026-08-29

First run of `measure_morphology` on real photographs (eight of the tutorial
images, after `refine(keep_largest)`), plus a mutation round over every guard
shipped since 1.3.1. Each finding was reproduced on synthetic geometry before
being fixed, and re-checked on the photos afterwards.

### Fixed

- **Two raw tracebacks from inside PlantCV no longer reach the caller.** On a
  233-px seedling `measure_morphology` crashed with `IndexError` at some prune
  sizes and `cv2.error: Can't parse 'pt1'` at others. The first: PlantCV's
  `segment_*` functions share one process-global colour palette, and the
  prune-sensitivity pass (`segment_skeleton` at 2× `prune_size`) left a palette
  sized to _its_ segment count — one colour for two leaves. The palette is now
  owned by `isolated_pcv_outputs` (reset on entry, restored on exit) and reset
  again after the sensitivity pass. The second: `segment_insertion_angle` fits a
  line to the stem and draws it across the frame; a vertical stem extrapolates to
  y ≈ −4×10⁹ and OpenCV 4.11 rejects the point. The cause is verified by refitting
  the stem before anything is swallowed; insertion angles are then `null` with
  `insertion_angle_undefined`, and every other trait is kept.
- **Morphology on a big photo took minutes.** PlantCV's per-segment functions
  allocate a full-frame image per segment and prune iterates full-frame
  subtractions: on a 16 MP maize photo whose plant filled 5% of the frame,
  `segment_tangent_angle` alone took 354 s for 14 leaves (710 s per call under
  the profiler). Every morphology trait is invariant to where the plant sits, so
  the skeleton work now runs on the mask's bounding box plus a margin wider than
  any prune, tangent or stem-joining window; the overlay is assembled back into
  the frame. Traits are identical to the uncropped result (tested), the mask
  bounding boxes on the real photos are 6–75× smaller than their frames.
- **An inverted mask is refused before the skeleton is built.** A 94%-coverage
  mask (beans, thresholded the wrong way) was skeletonised for 80 s and then
  refused for "too many tips". `measure_morphology` now refuses
  `implausible_coverage` by name in under a second; `measure()` keeps warning
  only, because a macro shot of one leaf legitimately fills a frame and a
  skeleton of the background has no such case.
- **"Unable to combine stem objects" gets its own remedy.** That PlantCV error is
  a stem in pieces the skeleton cannot join (a gap in the mask, or several
  plants), and it arrived wrapped in the too-many-tips advice to raise
  `prune_size`. It now names the bridge (`closing`, `fill_holes`) or
  `measure_regions()`.

### Tests

- Mutation round 6 (`docs/MUTATION-CHECKS.md`): seventeen guards from 1.3.1–1.5.2
  disabled one at a time; two were removable with a green suite and are now
  pinned — the noise rule's largest-fraction clause, and the per-cell
  `multi_specimen` whole-object judgment, whose existing leaf re-entry test had a
  fixture PlantCV assigned to the other cell, so it could not fail.
- Five new morphology tests (palette starvation, vertical stem, inverted mask
  before skeletonise, unjoinable stem remedy, crop equivalence on a 2000×2400
  frame). 283 tests.

## [1.5.2] — 2026-08-29

Findings from an independent two-model panel audit of 1.5.1 (or-grok,
or-deepseek; codex was quota-refused), each reproduced against the code before
it was fixed.

### Fixed

- **A plant filling its cell was called an inverted mask.** `measure_regions`
  ran the whole-frame `implausible_coverage` check (50% threshold, RGB polarity
  remedy) on each cell's own pixels, so two discs filling tight `rect_grid`
  cells came back "72% of the frame … probably INVERTED … opposite
  object_type" on every region. Real trays hid it because `auto_grid` cells
  are ~15% filled. The per-cell coverage check is gone (the whole-frame check
  already ran at segmentation); what a cell CAN hide is several plants, so it
  now carries `multi_specimen` when its labelled object is several
  comparably-sized components — judged on the whole object, not the cell
  crop, because a leaf that loops out of the cell and back is two pieces in
  the crop and one plant in fact.
- **A late-germination plate was withheld before its grid could run.**
  `measure_images` applied `noisy_segmentation` (blocking) to the whole mask
  before partitioning; a 96-well plate with ten germinated wells and 86 late
  ones is 90 non-major components and was refused with `nrows=10, ncols=10`
  given. With a grid the code is now an advisory on the row; the per-cell
  degeneracy floor guards each well.
- **Batch grid rows the guard had already called a merge were returned
  measured.** `object_exceeds_region` rows in `measure_images` are now
  `measured=false` with the reason (interactive `measure_regions` keeps the
  number beside the numbered overlay). On the real X-Rite tray this withholds
  the 8 merged cells.
- **A small thermal plant erased by `fill_size` was blamed on the band.**
  `segment_thermal` raised the degenerate refusal before building
  `fill_erased_mask`, so a 150-px plant under the default `fill_size=200` was
  told to widen the band (which then selects background). The refusal now
  carries the fill_size sentence.
- **Batch recipe validation** now covers `mode`, `radius`, the 400-region
  cap, and requires BOTH `nrows` and `ncols` — `nrows=4` alone silently
  measured 4×1 row strips as plants.
- `suggest_segmentation` warns (`empty_mask`) when the recommended polarity
  selected nothing (a blank frame recommended 'dark' at 0.0% as unambiguous).
- `refine_dropped_object` says what it knows — "the last op that raised the
  component count" — instead of attributing every drop to that op; its
  closing sentence no longer claims every dropped object is a leaf (objects
  under 10% of the largest are not reported).
- `noisy_segmentation` is worded per modality (a thermal frame is told about
  its band, not the colourspace sheet).
- A cell refused as claimed-by-neighbour reports the `region_coverage` its
  reason describes instead of 0.0.
- An explicit `analyses=[]` is refused ("No analyses requested") at every
  tool instead of silently measuring size; the check runs before any grid is
  built.

### Tests

Guards that were removable with a green suite are now pinned:
`region_count_mismatch`, `marker_fills_crop`, the firing side of
`refine_large_change`, a not-noisy tray-shaped mask staying measurable, a
1.10× overhang not tripping `object_exceeds_region` (and 1.40× tripping it),
`mode`/`radius`/half-geometry recipe errors, `indices` on an RGB
`measure_regions`, thermal `fill_erased_mask`, the claimed-cell rule in
`grid_misaligned`, and a leaf re-entering its cell staying one plant.

## [1.5.1] — 2026-08-29

Documentation only; no code change. Released so the PyPI project page carries
the current README (PyPI captures it at upload).

### Changed

- **README is a front door** (172 lines): the two-image argument, install,
  configure, the tools table, one `segment()` response, what it refuses and
  why, security, licensing. Measured against nine popular MCP server READMEs,
  the previous page had the most top-level sections of the set (21) and the
  latest install section.
- **`docs/GUIDE.md`** carries everything else, in workflow order: each tool's
  parameters and guards, the measured facts behind them, a reference of all
  25 warning codes with which ones block the batch, worker isolation, read
  roots, the security boundary. Two new overlays (numbered tray regions,
  numbered morphology skeleton) rendered from the fixture; the original two
  regenerated with the cyan outline.
- The security section no longer claims read roots are unimplemented (they
  shipped in 0.9.0).

## [1.5.0] — 2026-08-29

Findings from the first run of the unattended batch path on real photographs
(12 tutorial images: trays, colour cards, side views).

### Added

- **`measure_images` measures per plant when given a grid.** Five of the nine
  images a real batch "measured" were trays, returned as one group area
  (`area=441,080` for 28 arabidopsis plants) with only a `multi_specimen`
  advisory. `nrows`/`ncols` (plus `rect_grid` geometry) now route each image
  through the same partition `measure_regions()` uses; the row carries
  `regions` and `regions_measured` instead of `traits`, and the group
  advisory is dropped because there is no group number.
- **Time budget and timing.** Real photographs took 9–11 s each under load,
  so the 200-image cap alone allowed a ~30-minute call with no output.
  `max_seconds` (default 300) bounds the call: images not started in time
  are returned as `not_run` with `summary.not_run_paths` for resubmission;
  every row reports `seconds` and the batch `elapsed_s`.
- **`noisy_segmentation` guard**, shared by `segment()` (advisory),
  `suggest_segmentation()` and the batch (blocking). A sorghum-in-chamber
  photo was measured as one 650,000-px plant at 118 components (522 at
  `fill_size=50`). Calibrated on the real set: ≥50 components that are not
  major objects and no object holding half the mask — which keeps a
  27-seedling tray (22 minor components) and the arabidopsis trays (14–15)
  measurable while withholding the sorghum (116 / 520).
- **Batch summary triage fields**: `unique`, `duplicates_dropped` (the same
  path twice is measured once), `with_advisories`, `advisory_counts`.

### Changed

- **A recipe error is raised once, before any image runs.** `channel="zz"`
  used to run the whole loop and return N identical `UnknownChannelError`
  rows; channel, method, object_type, analyses, `max_seconds` and grid
  geometry are validated up front.

## [1.4.0] — 2026-08-29

Findings from the first dogfood on REAL hyperspectral and thermal data
(Danforth Center tutorial cubes and a FLIR tray frame) after every earlier
round had used PlantCV's 43×31 test cube and synthetic frames.

### Added

- **`measure_regions()` works on thermal and hyperspectral sessions.** A
  FLIR frame of a 20-plant tray could only be measured as one mask, while
  its `multi_specimen` advisory sent the caller to a tool that refused the
  session. The per-region partition (grid, PlantCV labels, empty /
  claimed-by-neighbour / too-small refusals, `object_exceeds_region`) is now
  shared by every modality; thermal sessions return per-region temperature
  statistics through `analyze.thermal`, hyperspectral sessions per-region
  index statistics through `analyze.spectral_index` (`indices`, default the
  session's index, with the pinned white/dark calibration). Arguments that
  belong to another modality (`analyses`, `px_per_mm`, `include_histograms`
  on a typed session; `indices` on an RGB one) are refused by name, not
  ignored.
- **`threshold_outside_range` advisory.** On a real leaf cube (NDVI
  0.19–0.89) the default `threshold=0.2` sat below the minimum and selected
  every pixel; 0.95 sat above the maximum and selected none, and neither
  result blamed the threshold. `segment_hyperspectral()` now says which side
  of the index range the threshold falls on; `segment_thermal()` says when a
  band encloses the whole frame, and refuses outright a band that lies
  entirely outside it, naming the frame range.
- **Band-less `segment_thermal()` refuses with the frame's range and
  percentiles** (p5–p95) instead of "give min_c and/or max_c" alone, so the
  first call is no longer a blind guess.

### Changed

- **Refusals and advisories name the right knob for the modality.** Only
  the inverted-mask remedy had ever been written for thermal; `empty_mask`,
  degenerate-mask refusals from `measure_spectral()` / `measure_thermal()` /
  `segment_thermal()`, and `implausible_coverage` on cubes all said "re-run
  segment() with a different channel or method". A per-modality `Remedies`
  bundle now supplies the sentence for RGB, hyperspectral and thermal.

## [1.3.1] — 2026-08-28

The two LOW findings left open from the real-photo dogfood.

### Added

- **`refine()` says when an op throws away a leaf.** On a sorghum photo
  `opening` split a leaf off the plant and `keep_largest` discarded it; the
  17% change sat under the `refine_large_change` alarm and only the overlay
  showed it. The refinement is now traced op by op: any component ≥10% of the
  largest component present before an op that the op removes entirely is
  reported as `refine_dropped_object`, naming the op, the object's size, and
  the earlier op that split it off (the largest three are listed, the rest
  counted). `keep_largest`, `fill`, and an `erode` that eats a leaf all trip
  it; specks do not.
- **`suggest_segmentation` says when its recommendation is noise.** The
  polarity report's `ambiguous` flag only compares the two polarities'
  coverage, so on sorghum-in-a-chamber `a`/otsu recommended 'dark' — 118
  components of chamber-wall texture at 12.9% — with `ambiguous: false`. Each
  polarity now reports `largest_fraction` (largest component / masked pixels),
  and `polarity.warnings` carries `noisy_segmentation` when the recommended
  polarity has ≥20 components with the largest under half the mask, pointing
  at the colourspace sheet and at `refine()`.

## [1.3.0] — 2026-08-28

Findings from the first dogfood on REAL photographs (PlantCV tutorial images:
trays, colour cards, side views, seeds) after three rounds on synthetic
renders had run dry.

### Added

- **`measure_regions` says when a cell's object is not inside the cell.**
  PlantCV's `create_labels(roi_type="partial")` hands any object overlapping a
  cell to that cell WHOLE, and reports the cell it also overlapped as empty. On
  an X-Rite tray photo a misaligned `auto_grid` therefore returned objects
  620–785 px wide inside 369-px cells — two plants per row — with
  `regions_measured: 17` and no warning. Each region now carries
  `object_exceeds_region` when its object's bounding box is ≥1.25× the cell
  (measured: leaf-tip overhang on a tight clean-tray grid reaches 1.02×, merged
  neighbours 1.68–2.13×), and a cell whose material was assigned to a neighbour
  is reported as `object_claimed_by_neighbour` naming that region instead of
  "no plant material". Under `auto_grid`, both together raise a set-level
  `grid_misaligned` advisory pointing at `rect_grid`.
- **Overlays outline the mask in cyan** on its own boundary pixels (thickness
  scaled to the frame so it survives the client downscale), in addition to the
  red tint. A red tint on a red subject (beans, red colour-card
  chips) was invisible, so "look at the overlay" could not show what was
  selected. Unmasked pixels are still never touched.

### Changed

- **Morphology refusal stops recommending `prune_size` when it cannot help.**
  On a real sorghum photo the refusal said "raise prune_size" at 15, 30, 100
  and 200 while the segment count sat at 126; only `refine()` got the plant
  analysed. When doubling `prune_size` keeps ≥80% of the segments the message
  now says raising it does not help and points only at `refine()`.

## [1.2.1] — 2026-08-28

Second live dogfood, against the released 1.2.0 server: all six 1.2.0 fixes
verified over the wire; one new defect.

### Fixed

- **Non-colour images are refused by name at the load boundary.** A 1-channel
  PNG used to fail three different ways — a raw OpenCV `cvtColor … 'scn' is 1`
  traceback from `segment()` and from the batch's `refused_because`, and
  PlantCV's bare `Input image is not RGB!` from `suggest_segmentation()`.
  `imaging.decode_image` now raises `NotColorImageError` naming the file, its
  channel count, and `segment_thermal()` as the likely intent, so every RGB
  tool refuses the same way. An undecodable file's `Failed to open` message now
  says it is not a decodable image and points cubes and thermal files to their
  own segmenters.

## [1.2.0] — 2026-08-28

Findings from dogfooding the released server end-to-end over a live MCP
connection (all three modalities, real fixtures) — led by one critical defect
no code review had caught, because only a real client could see it.

### Fixed

- **Refusal messages reach the client again on mcp ≥ 2.1.** mcp 2.1 masks any
  exception that is not a `ToolError` to a bare `Error executing tool <name>` —
  the right contract for a network service, and exactly the wrong one for a
  local instrument whose refusal messages are the product. Verified live: the
  "this is a thermal session, use measure_thermal()" guidance reached the
  client as nothing at all, and so did every other guard. Every tool now
  converts exceptions to `ToolError` at the boundary, keeping the class name;
  the lockfile moves to mcp 2.1.1 so the tests exercise the real masking
  semantics.
- **Read roots travel with every worker request.** They were snapshotted at
  worker spawn, so a warm worker kept enforcing a policy the parent had since
  changed. Roots are request state, not process state.
- **The morphology overlay now carries PlantCV's segment id digits.** The docs
  promised "the number drawn on the picture"; the labeled image PlantCV draws
  the digits on was being discarded in favour of its unlabeled twin.
- **The thermal `implausible_coverage` advisory gives thermal advice.** It told
  users to flip `object_type`, a parameter `segment_thermal()` does not have;
  it now says to narrow the temperature band.

### Added

- **Tiny frames are upscaled so their overlays can be looked at.** A 31×43
  hyperspectral cube's overlay was an unreadable 1:1 thumbnail — for a product
  whose core discipline is "look at the overlay". Frames under 256 px on their
  longest edge are upscaled by an integer factor (nearest neighbour, crisp
  pixels), and `overlay_scale` reports it, as ever.
- **`list_methods()` reports `server_version`.** The engine version alone could
  not say which plantcv-mcp answered; a stale cached server was
  indistinguishable from current over the tool surface (found live: a 1.0.0
  uvx cache).
- **`implausible_longest_path` advisory.** PlantCV can report a longest_path of
  7 px for a 343 px tall object on a fragmented mask (observed live on the
  regions path) — an artefact that reads exactly like a measurement. Both
  measure() and measure_regions() now flag a longest_path shorter than a tenth
  of the bounding box's long side.

## [1.1.0] — 2026-08-28

Every confirmed finding from the second multi-judge panel audit (of 1.0.1,
2026-08-27): eight defects and four hardening items, each re-verified against
source before acceptance; ~10 other findings were rejected with evidence.

### Security

- **Read-root containment now lives at the read boundary itself.**
  `read_image_bytes()` — the one place bytes leave the disk — resolves and
  checks every path, and opens the resolved path it checked. Previously the
  check ran only at the tool layer, and the ENVI loader's derived sibling
  (`.hdr` ↔ `.raw`) was never checked at all: a symlinked `.raw` beside an
  in-root `.hdr` read bytes from outside the configured roots. The
  hyperspectral and thermal tool paths also discarded `check_readable`'s
  resolved path and re-read the original. Both mechanisms are gone.
- **CLI-configured read roots (`--root`) now reach the analysis worker.**
  `spawn` re-imports modules, so roots set via `set_roots()` did not exist in
  the worker process; the parent now passes its effective roots at worker
  start.

### Fixed

- **Calibration references are digest-pinned.** White/dark reference files are
  hashed at `segment_hyperspectral()` time and verified at `measure_spectral()`
  time; a reference that changed in between is refused
  (`CalibrationReferencesChangedError`) instead of silently changing every
  calibrated number. Reference files are also loaded in the server process now,
  never inside the worker.
- **A rotated marker no longer overstates the scale.** Marker length comes from
  the minimum-area rotated rectangle, not the axis-aligned bounding box: a
  square marker photographed at 45° previously calibrated `px_per_mm` ~41%
  high, and `marker_not_round` could not catch it (a rotated square's bbox is
  square). The roundness check now uses the rotated rect's own sides.
- **A calibration crop that hangs over the frame edge is refused.** It was
  silently clamped to fit; a marker cut by the clamped edge produced a
  plausible but wrong scale, and the edge-contact warning then blamed polarity.
- **Regions are held to the same degeneracy floor as whole-frame measure().**
  A few stray pixels in a grid cell previously came back as a full trait row
  indistinguishable from a real seedling; such cells are now refused by name
  (`measured=false` with the reason).
- **A non-finite `px_per_mm` is refused on every path.** The check moved into
  `convert_units()` itself; `measure_regions()` previously let `NaN` through
  (`NaN <= 0` is false) and returned NaN traits labelled `mm`.
- **A thermal `.npz` holding several arrays is refused naming them.** The
  loader silently used whichever array numpy listed first.

### Added

- **`nan_pixels` advisory and per-index `finite_pixel_count` on
  `measure_spectral()`.** min/max already skipped non-finite index values
  silently while `pixel_count` claimed the whole mask; the dropped evidence is
  now counted and named.
- **`warnings` on `measure()` results.** Mask-level advisories
  (`frame_clipping`, `multi_specimen`, `implausible_coverage`) are re-derived
  at measure time by the same code segment() uses, so the trait table — the
  artifact people keep — carries its own caveats.
- **Sessions require a digest, and hand out read-only masks.** The digest
  default (`""`) silently disabled the stale-image guard for any future caller
  that forgot it; the guards are now unconditional. The stored mask refuses
  in-place writes rather than trusting every future call site.
- **A worker that survives SIGKILL is reported, not ignored.** After the
  kill-and-join, a still-alive worker (uninterruptible kernel sleep) raises
  naming the pid instead of silently stacking a fresh worker on the zombie.

## [1.0.1] — 2026-08-27

The four findings confirmed by the 2026-08-27 multi-judge panel audit of 1.0.0
(every finding re-verified against source before acceptance; 12+ others were
rejected with evidence).

### Fixed

- **An unpicklable exception no longer kills the worker and loses the original
  error.** `_serve`'s error-send was outside any `try`: an exception whose
  pickle fails crashed the worker, and the parent reported a generic
  `WorkerCrashedError` instead of the real error. The err-send now falls back
  to a `RuntimeError` carrying the original exception's type and text, and the
  worker survives the call.
- **`refine(fill, size=0)` is refused.** PlantCV's `fill` no-ops on `size=0`,
  so the op would have been recorded in the session's lineage as if it had run
  — exactly the silent no-op this module's validation exists to prevent.
  `size` now requires `>= 1`.
- **The calibration-degenerate refusal names non-finite references.** A NaN in
  a white/dark reference correctly refused calibration, but the message said
  the span was "not positive"; it now says the reference contains invalid
  (NaN/Inf) pixels and where the count lies.

### Added

- **`nan_pixels` advisory on `segment_thermal`.** Non-finite pixels were
  excluded from the temperature band correctly but silently; the segmentation
  now reports how many pixels the band could never select.
- Regression tests pinning behavior the panel questioned: a worker crash
  landing on the `WORKER_MAX_TASKS` recycle boundary, and a proof that the
  calibration span check is exactly PlantCV's own `calibrate()` denominator
  (per column × band mean — a dead column refuses, a dead pixel does not).

## [1.0.0] — 2026-08-27

Sub-project E — the last of the post-0.5.0 roadmap
(`docs/superpowers/specs/2026-08-27-backlog-integration-plan-of-attack.md`): the
tool surface this server set out to cover is complete, hence 1.0. Design:
`docs/superpowers/specs/2026-08-27-hyperspectral-thermal-design.md`.

### Added

- **Typed sessions.** `Session.kind ∈ {rgb, hsi, thermal}`; every measurer
  refuses the wrong kind naming the right tool.
- **`segment_hyperspectral` / `measure_spectral`.** ENVI cubes (`.raw` + `.hdr`;
  either path accepted — PlantCV derives the header by stripping the raw file's
  extension, and handed the `.hdr` itself it reads the header as pixels) segment
  by any of PlantCV's 31 spectral indices and measure index statistics, with the
  full per-band spectrum opt-in. Guards, all measured on 4.11.3:
  - **An index computed on integer counts wraps around** (uint16 NDVI read 65.3
    on a [-1, 1] index). Cubes are calibrated to reflectance from white + dark
    references, or cast to float with `uncalibrated_cube` (indices are relative).
  - **A degenerate reference pair is refused** (`white − dark` not positive
    everywhere): PlantCV's own `calibrate()` silently clips that case to 1.0.
  - An index the wavelength range cannot support is refused by name.
  - Both ENVI files are hashed into the session digest (a changed header refuses
    at measure time exactly like a changed image).
- **`segment_thermal` / `measure_thermal`.** FLIR radiometric `.jpg` (flyr),
  `.csv`, and `.npz` frames segment by a °C band and measure max/min/mean/median
  temperature; a band selecting nothing is refused; masks are never borrowed
  from RGB sessions.
- **Real-data evals** on PlantCV's own MPL-2.0 test data, vendored with a NOTICE
  (`tests/fixtures/plantcv/`): the corn-kernel cube (31×43 px, 580 bands)
  segments and measures with thresholds taken from its measured NDVI
  distribution; `measure_thermal` reproduces PlantCV's own mean (33.509482) on
  its 480×640 frame + mask to 1e-6; a real FLIR JPEG (19–24 °C) segments by
  temperature. Synthetic known-value evals: a cube whose NDVI is 0.6/−0.2 by
  construction (recovered to ±0.01, calibrated and uncalibrated), a °C frame
  with a known warm disc.
- Both new analyses run in the isolation worker like every other analysis.

## [0.9.0] — 2026-08-27

Sub-project D of the roadmap. Design: `docs/superpowers/specs/2026-08-27-read-roots-design.md`.

### Added

- **Read-root allow-list.** `plantcv-mcp --root DIR` (repeatable) or
  `PLANTCV_MCP_ROOTS` (`os.pathsep`-separated) confines every path argument —
  `segment`, `suggest_segmentation`, `calibrate_scale_from_marker`, and every
  entry of `measure_images` — to the named directories. Paths are resolved with
  `os.path.realpath` **before** the containment check, so `..` and symlinks are
  followed first: a symlink inside a root that points outside is refused. A batch
  with one stray path is refused whole, before any file is read. Unset, behaviour
  is unchanged (the documented local trust boundary). `list_methods()` reports
  the policy as `read_roots`; `SECURITY.md` names the scope.

## [0.8.0] — 2026-08-27

Sub-project C of the roadmap. Design: `docs/superpowers/specs/2026-08-27-isolation-design.md`.

### Added

- **Every PlantCV analysis runs in a worker subprocess, by default.** `measure`,
  `measure_regions`, `measure_morphology`, `measure_images` and `refine` dispatch
  through `workers.dispatch()` to one warm `spawn`-context worker (never `fork`:
  the server runs anyio worker threads). A native crash inside PlantCV/OpenCV —
  the class 0.5.0 closed one instance of by validating geometry — is now a
  `WorkerCrashedError`/tool error naming the signal, and the next call starts a
  fresh worker; the server never executes native analysis code. Exceptions raised
  inside the worker come back as the same type, so a `MorphologyRefusedError` still
  reads as one at the tool layer. The worker is recycled after 200 calls.
- **Default on, on evidence.** The plan's gate was "opt-in if > 25 % overhead on
  the 3000×3000 fixture"; measured: in-process 1841 ms, isolated 1983 ms, **+7.7 %**.
  `plantcv-mcp --no-isolate` or `PLANTCV_MCP_ISOLATE=0` runs analyses in-process;
  the console script now parses arguments (`--isolate/--no-isolate`).
- `analysis.py` — the five analysis entry points with picklable arguments and
  results, used identically by both modes; `RegionSet`'s PlantCV ROI objects stay
  on the worker side, only bboxes cross.
- Tests: an `os.abort()` in the worker is a tool error the server survives (and
  a mutation that swallows the crash fails both tests); isolated and in-process
  results are equal for `measure`/`measure_regions`/`measure_morphology`;
  refusals keep their type; two threads through one worker each get their own
  numbers; the fresh-process tool-layer driver passes with isolation on.

### Changed

- "Drop `pcv.outputs` reliance" from the audit backlog is closed by isolation
  rather than removal: 15 of PlantCV's 19 morphology functions report only
  through `pcv.outputs`, so it cannot be dropped — but a worker's globals are
  nobody else's. The in-process lock stays for `--no-isolate`.

## [0.7.0] — 2026-08-27

Sub-project B of the roadmap. Design: `docs/superpowers/specs/2026-08-27-morphology-design.md`.

### Added

- **`measure_morphology(session_id, prune_size=15, tangent_size=25, px_per_mm)` —
  skeleton-based traits for one plant**, returning PlantCV's per-segment table
  (path/euclidean length, curvature, angle, tangent and insertion angle), per-plant
  stem height/length/angle, tip and branch-point counts, cycles and segment widths,
  together with the **numbered-segment overlay**: a segment `id` is the number on
  the picture. Runs entirely under `PCV_OUTPUTS_LOCK` — 15 of PlantCV's 19
  morphology functions report only through `pcv.outputs`.
  Every guard below was measured on PlantCV 4.11.3 against a synthetic plant of
  known geometry (stem + three leaves at 30/45/60°, lengths 90/80/70 px):
  - **A vertical stem yields `stem_angle = -14373°`** (PlantCV's slope-based
    estimate blows up). Returned as `null` with `stem_angle_undefined` instead of
    as a number.
  - **`tangent_size` default 25 is measured, not guessed.** Insertion-angle bias
    fell from 24.5° (10 px) to 5.9° (25) to 4.2° (30); PlantCV fits `size` pixels
    from each end, so `2 × size` longer than a leaf collapses its angle to `0.0`
    (`segment_tangent_angle.py:102`). `tangent_window_exceeds_segment` flags it.
  - **`prune_size_sensitive`** when the segment count changes by >30% at twice
    the prune size — then the table describes the parameter, not the plant.
  - **PlantCV's own abort on a fragmented skeleton** ("Too many tips found per
    segment, try pruning again") arrives as a refusal carrying the segment counts
    at `prune_size` and `2×prune_size`, not as a stack trace.
  - PlantCV's `segment_id` reuses a process-global cached colour palette
    (`params.saved_color_scale`) and indexes past it when a later plant has more
    segments; the cache is reset inside the lock.
  - Leaf-less skeletons (a bare stem, a ring) are reported with an empty table and
    `no_leaf_segments`, so cycles/tips/stem can still say why. Multi-plant masks
    are refused naming `measure_regions()` and `refine(keep_largest)`.
  - Known-geometry eval: leaves recovered in order, insertion angles within 12°,
    lengths within 15%; `stem_height` is PlantCV's base→topmost-junction height.
  - `'NA'` strings PlantCV emits for undefined traits become `null`.

## [0.6.0] — 2026-08-27

First step of the post-0.5.0 roadmap
(`docs/superpowers/specs/2026-08-27-backlog-integration-plan-of-attack.md`).

### Added

- **`refine(session_id, ops)` — mask refinement as a session→session operation.**
  Applies an ordered list of morphological ops (`fill_holes`, `fill`, `erode`,
  `dilate`, `opening`, `closing`, `median_blur`, and our own `keep_largest`) to a
  session's mask and mints a NEW session, returning the refined overlay plus
  before/after diagnostics and warnings — never traits. The original session is
  untouched, so a refinement that looks wrong is discarded rather than undone.
  - **Validation is all-or-nothing and stricter than PlantCV's.** PlantCV silently
    no-ops on `fill(size=-1)`, `erode(i=0)` and an even `median_blur` kernel
    (measured); a no-op recorded in the lineage as if it had run would be a lie,
    so every op is checked — name, required params, ranges, unknown params —
    before the first one touches the mask, and the error names the op index.
  - **A refinement that leaves no measurable plant is refused, not minted**, with
    before/after `mask_fraction`, `component_count` and `largest_area`.
  - `refine_large_change` advisory when the mask changed by more than 25%.
  - Known-value eval: a disc with a hole and salt noise measures wrong first
    (positive control), then `[fill_holes, keep_largest(1)]` recovers the area
    within 1%.
- **`lineage` on every trait table.** `measure()` and `measure_regions()` results
  carry the ops that produced their mask (`[]` for an unrefined session), so two
  tables made differently can be told apart. `Session` gains `lineage` and
  `parent_id`.
- `list_methods()` publishes `refine_ops` with each op's parameter constraints;
  the test suite applies every published op with its example so a documented op
  can never be a dead one.

## [0.5.0] — 2026-08-27

Findings of an adversarial audit run against 0.4.1 (Codex consult, every item
reproduced before it was fixed). Three were correctness defects in the shipped
runtime, not backlog.

### Fixed

- **Concurrent measurements corrupted each other through `pcv.outputs`.** The
  server's safety argument — synchronous tools run inline on the event loop, so
  two analyses can never interleave — was true for mcp 1.28.1 and became false
  when the dependency floor moved to `mcp>=2`: mcp 2.x runs synchronous tools on
  worker threads (`func_metadata.py`: `anyio.to_thread.run_sync`) and dispatches
  requests concurrently. Reproduced with two sessions and a widened window: one
  thread's `pcv.outputs.clear()` erased the other's observations (`Expected
observation group 'default_1' not found`), and one call returned the OTHER
  session's area as its own. One process-wide lock (`measurement.PCV_OUTPUTS_LOCK`,
  via `isolated_pcv_outputs`) now guards the snapshot → clear → analyze → read →
  restore section in both `measure()` and `measure_regions()`; the regression test
  drives a cross-path pair on two threads and fails without it.
- **Malformed `rect_grid` geometry crashed the whole server.** `coord=[-10, -10]`
  with a 50×50 cell on a 100×100 image passed `build_regions` and reached native
  OpenCV/PlantCV inside `measure_regions`, which died with SIGSEGV (exit 139) —
  taking every session with it, since this is a stdio server. `build_regions` now
  refuses non-positive `height`/`width`, non-positive `radius`, a missing
  `spacing` (required even for one cell: pinned PlantCV rejects it with a message
  that blames the wrong argument), and any cell that lies partly outside the
  image. An off-image cell previously reported "No plant material" — misleading,
  since the cell was off the frame, not empty.
- **Time-of-check/time-of-use hole in the stale-image guard, in both
  directions.** `segment()` decoded the file, built the mask, and only then hashed
  the path; `measure()` decoded and then hashed separately. A same-shape
  replacement landing in either window bound the OLD mask to the NEW file's hash,
  and the integrity check then passed. Both paths now read the bytes once, hash
  those bytes, and decode those bytes (`imaging.load_image_with_digest`); the
  regression swaps the file mid-segmentation and expects `measure()` to refuse.
- **`SessionStore` was not thread-safe.** `get()` is check-then-act and
  `create()` is insert-then-evict; under mcp 2.x worker threads an interleaving
  raised a bare `KeyError` for a session present a moment earlier. The store now
  carries its own lock (deliberately not the PlantCV analysis lock).
- **`pcv.outputs` restore was incomplete.** `clear()` resets `measurements`,
  `images`, `observations` and `metadata`; only `observations` was restored, so a
  host application using PlantCV directly had three tables silently emptied.

### Changed

- **`measure_images` now takes the whole recipe: `ksize`, `offset` and
  `color_correct`.** The tool tells users to settle a recipe with `segment()` and
  apply it, but could not accept the kernel parameters of the `mean`/`gaussian`
  methods or the colour-correction flag, so it silently ran a different threshold:
  1900 px vs 4536 px on the same file. The returned `recipe` records all three.
  An image that cannot be colour-corrected when asked is refused with the reason
  (`ColorCardNotFoundError`), never measured raw.
- **Every number-bearing result names its engine.** `engine: {name, version}`
  was added to `measure()` in 0.4.0 and is now on `measure_regions` and
  `measure_images` too; a stored trait table from any tool can say what produced
  it.
- **The server identifies its version at initialize.** `MCPServer` defaults
  `version` to `""`, and that empty string is what clients saw in `serverInfo`.
  Now `version=__version__`, checked through a real in-memory client handshake.
- The `plantcv==4.11.3` pin is described as PlantCV-version stability rather
  than determinism: `uvx plantcv-mcp` does not consume this repository's
  `uv.lock`, so the rest of the dependency set may vary within bounds.

### CI

- The sdist "exactly what we intend" step asserts every file in the allow-list
  (`CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` were not checked).
- The citation workflow also holds `CITATION.cff` `date-released` to the
  CHANGELOG date for the pyproject version; its error message had instructed
  updating both fields while checking one.
- The installed-wheel smoke test now runs the `plantcv-mcp` console script as a
  real stdio subprocess and completes an initialize handshake, instead of only
  importing the package and listing tools in-process.
- `tests/test_concurrency.py` (forced-interleave races), MCP-layer tests for
  `calibrate_scale_from_marker` and invalid geometry (`ToolError`, not a crash),
  and batch-parity tests through `call_tool`.

## [0.4.1] — 2026-08-02

### Added

- **Community-health and repo-hygiene files, matching the standard set by
  `data-aggregator-mcp` and `plant-genomics-mcp`.** An earlier parity audit
  compared this repo only against `ldraw-mcp`, which is itself thin on these, so
  the whole tier went unnoticed: `CONTRIBUTING.md`, `SECURITY.md`, issue forms
  (bug report + feature request + a config pointing security reports at private
  advisories), a pull-request template, `.editorconfig`, `.mcp.json`, `glama.json`,
  a CodeQL workflow, and a Dependabot config.

  **Dependabot uses the `uv` ecosystem, not `pip`.** This is a uv-locked project;
  the pip ecosystem would update `pyproject.toml` and leave `uv.lock` stale, which
  CI installs with `--frozen` and would fail on. Dependabot's native uv support
  reads both together.

  `CONTRIBUTING.md` and `SECURITY.md` were added to the sdist allow-list.
  hatchling's allow-list drops anything unlisted **silently** — verified with
  `tar tzf` on a real build rather than assumed, the same way a `NOTICE` was
  previously found missing.

  `SECURITY.md` documents this server's actual trust boundary and defers to the README's
  "Security and trust boundary" section as authoritative rather than restating it:
  this server reads image files anywhere the running user can read, deliberately and
  without a sandbox. It also states what is **in** scope — a path the caller did not
  ask for, non-image content disclosure, or an escape from the running user's own
  permissions — so "it read the file I asked it to read" is not filed as a
  vulnerability.

- **README gained a Glama badge, and its licence badge is now derived rather than
  hardcoded.** It read `badge/license-MIT-green`, a literal that would keep claiming
  MIT if the licence ever changed; it now reads `pypi/l/plantcv-mcp`. Verified to
  resolve to MIT before switching.

- **`server.json` is validated against the registry's own published schema.**
  `breedsim-mcp` v0.4.0 was tagged, uploaded to PyPI and GitHub-released before
  the MCP registry refused it with a 422: its description had grown past a
  100-character cap that nothing local measured. The publish workflow is the only
  thing that checks registry constraints, and it runs on tag push — after the
  version is already burned. This server's description is **91 characters**: it
  passes today with nine to spare, and nothing here was measuring it.

  Rather than copy the one constant, the check validates the whole document
  against the dated `$schema` `server.json` already declares, which is the
  registry's own statement of what it accepts. That covers the four other length
  caps and the required-field list as well. The schema is vendored at
  `tests/server.schema.json` rather than fetched, keeping the suite offline and
  deterministic, and a test asserts the vendored copy's `$id` still matches the
  declared `$schema` so the pin cannot drift silently.

  Verified by mutation rather than assumed: a 282-character description was
  written into the real `server.json` and the suite watched to fail on it, with a
  non-length failure (a missing required field) and an in-test positive control
  so a validator that raised on everything could not read as a working guard.

## [0.4.0] — 2026-08-01

### Added

- **`measure_regions()` — one row per plant.** `measure()` treats the whole frame
  as a single region, so a tray of seedlings was merged into one object and every
  size trait described the group; `multi_specimen` could warn about it but nothing
  could measure it. Regions come from `mode="auto_grid"` (PlantCV infers the
  layout from the mask; give only rows and columns) or `mode="rect_grid"`
  (explicit geometry). The response carries an overlay with every region outlined
  and numbered, because per-region numbers are unreadable without a picture
  saying which region is which.

  **An empty cell is refused by name, not reported as zero.** Measured on PlantCV
  4.11.3: for a 2×2 grid with one empty cell, `create_labels` returns n=4 and the
  analysis emits a complete trait set with `area = 0.0` for the empty one. That is
  the same failure `assert_not_degenerate` exists to stop in the single-region
  path. `np.unique(labeled)` is the discriminator, and such a region returns
  `measured: false` with a reason.

  That same measurement settled the index mapping: group `default_{i+1}`
  corresponds to region `i`, with **no shifting when a cell is empty**. Had
  empties been dropped, every trait after a gap would have been attributed to the
  neighbouring plant. The tests use plants of deliberately different sizes so a
  one-off shift is detectable at all — a fixture of identical plants could not
  fail that test.

- **Traits are now checked against shapes whose geometry is known in advance.**
  An 80×80 square must measure 6400 px, and it does. Every prior test compared
  this server against its own previous output, which cannot catch a change that
  is consistently wrong. This one can: an external oracle, arithmetic rather than
  a stored baseline.

- **`measure()` reports the engine that produced the numbers.** The result now
  carries `engine` with the PlantCV version. It was previously reachable only
  through `list_methods`, so a saved trait table could not be traced back to the
  version that measured it — and PlantCV's own trait definitions do change
  between versions.

- **A fresh-process test that drives the tool layer and nothing else.** Three
  sequential `call_tool` invocations in a subprocess that has never touched
  PlantCV directly, asserting the 6400 px oracle at the end.

  This exists because of a bug in a SIBLING server, not one found here: in
  breedsim-mcp, rpy2 published its conversion rules into a `ContextVar` at import
  time, the import happened inside the first request, and the rules were
  discarded when that request returned — while 27 tests passed throughout,
  because a test that imports the dependency into pytest's root context masks the
  whole class of failure. Nothing here is known to be broken; what was missing was
  the ability to notice. PlantCV has import-time global state of its own
  (`pcv.params.sample_label`, which this package reads), so a case covers a host
  process that sets it before importing us. The file carries its own positive
  control proving the driver can actually fail.

### Changed

- The `multi_specimen` warning pointed at "roi.auto_grid (phase 2)". Phase 2 is
  this release; it now names `measure_regions()`. A warning pointing at unbuilt
  work is worse than no pointer.
- `_load_session_image()` extracted so `measure()` and `measure_regions()` share
  one copy of the stale-image guards. A second entry point with its own copy is a
  copy that can drift, and a drifted staleness check measures a mask against
  pixels it was never drawn on.

## [0.3.2] — 2026-07-31

### Added

- **Zenodo archival.** This release exists to be archived: the Zenodo↔GitHub
  integration mints a DOI from the tag's tarball, and the previous tag predated
  `.zenodo.json` and `CITATION.cff` entirely — those files were added after it was
  cut. Zenodo archives the tag, not the default branch, so a release was the only
  way to get the metadata into an archived snapshot.

### Changed

- **`.zenodo.json` now uses Zenodo's lowercase licence identifier**
  (`mit` rather than `MIT`). That is the canonical spelling —
  `zenodo.org/api/vocabularies/licenses/<id>` returns 200 for the lowercase form
  and 404 for the SPDX-cased one. See the correction below: it fixed nothing.

### Correction — added after this release was published

This release was originally described here as **fixing** a defect in which the
SPDX casing "silently dropped the licence from the published record". **That was
wrong, and the entry is corrected rather than quietly deleted.**

Zenodo normalises the licence identifier on ingest. The sibling `ldraw-mcp`
archived with `"MIT"` still in place and its record reads `license: mit-license`;
this project's record reads `license: mit`. The licence was never dropped.

The apparent evidence was two of my own measurement errors, both the same
mistake — probing a proxy instead of the artifact:

1. Querying the licence **vocabulary endpoint** and treating a 404 there as what
   the ingest accepts. It is not; the ingest normalises casing.
2. Reading the **RDM-era field names** (`rights`, `subjects`,
   `creators[].person_or_org`) against an API endpoint that returns the **legacy**
   shape (`metadata.license`, `metadata.keywords`, `creators[].orcid`). Every
   field reported as absent was present throughout.

What remains true is the reason this release exists: the previous tag predated
`.zenodo.json` and `CITATION.cff`, and Zenodo archives the **tag**, not the
default branch. DOI: [10.5281/zenodo.21713869](https://doi.org/10.5281/zenodo.21713869).

### Notes

No functional change. Tools, guards and dependency pins are identical to 0.3.1.

## [0.3.1] — 2026-07-31

### Added

- **Published to the official MCP registry** (`io.github.musharna/plantcv-mcp`)
  via `server.json` and an OIDC workflow, so the server is discoverable from MCP
  clients and directories rather than only from PyPI.

  This needed a release rather than a docs commit. The registry proves PyPI
  ownership by finding an `mcp-name` marker in the package README **as published
  to PyPI**, and PyPI captures `long_description` at release time — so a marker
  sitting on `master` verifies nothing. That is the same mechanism that kept this
  package's "Not published to PyPI" line live on its project page after the fix
  had merged, and it is why 0.2.1 existed too.

- **`tests/test_registry_metadata.py`.** `server.json` states the version in
  three places and nothing else makes them agree with `pyproject.toml`; a stale
  one is rejected by the registry during a release, after the version is spent.
  The README marker is checked against the name `server.json` declares, since
  that exact string is what the registry greps for.

### Notes

No functional change. Tools, guards and dependency pins are identical to 0.3.0.

Running the real `mcp-publisher validate` against `server.json` is what caught a
100-character cap on `description` in the sibling `breedsim-mcp` — a constraint
no schema read surfaced, and one that would otherwise have failed the publish
after the version was already on PyPI. The workflow validates before
authenticating for that reason.

## [0.3.0] — 2026-07-30

### Changed

- **Migrated to `mcp` 2.x.** `FastMCP` was renamed, not removed: it is now
  `mcp.server.mcpserver.MCPServer`, and `Image` moved with it. The dependency
  **moves** to `mcp>=2,<3` rather than widening to `<3` — this package imports
  `mcp.server.mcpserver`, absent in 1.x, so a range spanning both majors could
  resolve to a version that cannot import the server. mcp 2.x requires Python

  > =3.10, below this package's >=3.11 floor, so the support matrix is unchanged.

- **`call_tool` returns a `CallToolResult`**, not a bare block sequence or a
  `(content, structured)` tuple. Tests read `.content` and `.structured_content`;
  the old `res[0] if isinstance(res, tuple) else res` shims are removed rather
  than extended, because under 2.x that sniff falls through and hands back the
  result object, moving the failure away from its cause.

- **`mcp.types` fields are snake_case** (`outputSchema` → `output_schema`,
  `readOnlyHint` → `read_only_hint`). Constructing `ToolAnnotations` still
  accepts camelCase via pydantic aliases, but reading the attributes does not —
  so construction would have stayed green while every read broke. Both spellings
  are now aligned, including in the comments that named the old field.

## [0.2.1] — 2026-07-30

A docs-and-metadata release. No behaviour changes; it exists because two claims
the package made about itself were wrong, and one of them could only be corrected
by publishing.

### Fixed

- **The README told users the package was unpublished.** The Install section read
  "Not published to PyPI. Install from the repository", so anyone following it
  built from a git checkout while `pip install plantcv-mcp` had worked since
  0.2.0. Corrected on `master` before this release — but PyPI renders the
  description captured at release time, so the project page kept serving the
  false claim regardless. **That is what this release is for.**

- **`plantcv_mcp.__version__` reported `0.1.0` from the published 0.2.0.** It was
  a literal sitting beside a `pyproject.toml` that said 0.2.0, with nothing
  enforcing agreement. It now reads from installed metadata — the source the
  packaging already enforces — so the two cannot drift again.

  `tests/test_version.py` asserted only that the version was a non-empty string,
  which is exactly what a wrong version is, so it passed throughout. It now
  compares the reported version against the one `pyproject.toml` declares, and
  was confirmed to fail on the old code before being kept.

### Changed

- Status is now derived rather than asserted: PyPI version and `pyversions`
  badges replace the hand-written publication claim and the static python badge.
  A hand-typed status line is only true until someone forgets, which is how the
  Install section went stale in the first place.

## [0.2.0] — 2026-07-28

First release published to PyPI. 0.1.0 was tagged in this changelog but never uploaded,
so everything below shipped together.

### Added

- **`calibrate_scale_from_marker`** — measures a marker of known real size and returns
  `px_per_mm`, closing the half of real-world units that was previously left to the caller.
  It does **not** wrap `pcv.report_size_marker_area`: that function takes an ROI, and against
  a synthetic disc of known 80 px diameter it returns `major_axis=79.1` with a whole-frame ROI
  but **348.0 with a tight ROI around the marker** — a silent 4.35× scale error under the most
  intuitive usage, because its ROI filter can select a background component that merely
  intersects the region. Here the region is **cropped before thresholding**, so nothing outside
  the box can be selected. That removes the mechanism instead of compensating for it.
  A `marker_touches_crop_edge` warning catches the wrong-polarity case: measured on a centred
  80 px disc in a 100×100 crop, the correct and inverted polarities both give `crop_fraction`
  0.50, so coverage cannot discriminate and edge contact is the guard that can.
- **Colour-card correction.** `segment(..., color_correct=true)` detects a ColorChecker and
  corrects to a standard reference; `measure()` re-applies it so traits are measured on the
  pixels the mask was drawn on. If no card is found it **raises** rather than silently
  measuring the uncorrected image. Previously deferred for want of a test fixture — a
  synthetic Macbeth chart driving the real detector, plus a known colour distortion, gives
  ground truth: mean absolute error to the undistorted original falls from 8.77 to 3.43.
- **`measure_images`** — one recipe across up to 200 images. Batch cannot honour "no number
  without the picture" literally, so the overlay is replaced by automated validation plus
  explicit refusal: every image runs the **same** guards as `segment()` via shared code, and
  any image tripping a blocking guard returns **no traits**, only a reason. That is weaker
  than a human reading a mask and is documented as such rather than implied.
- **Real-world units.** `measure(session_id, px_per_mm=…)` converts spatial traits to `mm`
  and `mm2`. Without it every size is in pixels, and pixel sizes are not comparable between
  images shot at different distance or zoom — the largest practical limitation the tool had.
  Lengths divide by `px_per_mm` and areas by its square, from an explicit table rather than
  from PlantCV's unit strings: PlantCV labels **both** `area` and `width` as `"pixels"`, so
  a unit-derived rule would leave every area wrong by exactly a factor of `px_per_mm`.
  Positions stay in pixels, having no meaning in mm without an origin.
- **Colour analysis.** `measure(session_id, analyses=["size", "color"])` adds hue,
  saturation and value statistics. Their three frequency histograms total 692 numbers and
  are withheld unless `include_histograms=true` — that is a context-window cost, not a
  feature. This takes the server from 1 to 2 of PlantCV's 11 `analyze` functions.
- **Server instructions.** The server now publishes MCP `instructions` telling the client to
  look at the overlay before trusting a number, and what each warning code means. The product
  is a discipline as much as a set of functions, and nothing was conveying that.
- **Tool metadata.** Every tool now publishes a human title and `ToolAnnotations`
  (`readOnlyHint`, `destructiveHint=false`, `idempotentHint`, `openWorldHint=false`) so a
  client can tell they only read and compute. `measure` and `list_methods` publish an
  `outputSchema` derived from typed returns, so callers get structured content instead of
  parsing JSON out of a text block — as do `calibrate_scale_from_marker` and
  `measure_images`. `segment` and `suggest_segmentation` return image blocks and so have
  no structured schema, by nature; a test asserts exactly that split so a new tool cannot
  quietly ship without one.

### Previously deferred, now resolved

Both of the items this section used to hold have shipped. Automatic scale is implemented
without wrapping the ROI-based API that produced the 4.35× error, and colour correction is
implemented now that a synthetic ColorChecker gives its happy path a ground-truth test. The
measured reasons are preserved in the entries above so neither is re-litigated from scratch.

### Documentation

- **The README now shows an overlay.** A tool whose entire argument is "you get the picture
  the numbers came from" had no pictures in it. The top of the page is now the same image
  segmented correctly (3.1% mask, `area=32427`) beside the same image segmented with the
  wrong polarity (96.1% mask — the background — `area=1007829`), which makes the case in
  one glance instead of three paragraphs. Assets render from the bundled fixture via
  committed code; each is a standalone image laid out with a table, not a stitched grid.
- **The seventeen traits are listed.** Previously the README named exactly one (`area`) and
  only as part of a failure example, so a reader could not tell whether the tool measured
  what they needed.
- Added a `segment` parameter table, a verbatim example response, a colourspace contact
  sheet, CI/Python/licence badges, and links to the changelog and the mutation-check log.
  Reordered so the reader sees what the tool produces before installation instructions, and
  promoted "Getting the polarity right" to a top-level section since it is the main
  correctness trap.

### Fixed

- **Mask validity is now two-sided.** `assert_not_degenerate` only ever rejected masks
  that were too _small_, which left inverting a threshold — the dominant failure of any
  threshold operation — outside the set of outcomes this system could express as a
  failure. A new `implausible_coverage` warning fires above 50% frame coverage.
  Measured on the fixture: plant masks land at 0.031–0.046 and inverted ones at
  0.959–0.967, so the boundary sits in a wide empty gap. It warns rather than raises,
  because a macro shot of one leaf can legitimately fill the frame.
- **`object_type` is exposed on `segment()`.** It was hardcoded to `"dark"` and
  unreachable, so channels `s` and `b` returned the **background** as the plant:
  `mask_fraction` 0.961, and `measure()` reported `area=1007829, width=1024,
height=1024` — the whole frame — with no error. `list_methods()` made this worse by
  recommending `'s'`. Its guidance is now correct and names the polarity each channel
  needs.
- **`suggest_segmentation` reports both polarities.** It now measures what `dark` and
  `light` each yield on your image and recommends one, flagging the case where the two
  are too close to call rather than guessing.
- **`fill_size`, `ksize` and `offset` are exposed.** `fill_size=200` was hardcoded and
  silently erased any specimen smaller than itself — a measured 144 px object became an
  empty mask. Thresholding and filling are now separate steps, so this reports
  `fill_erased_mask` naming `fill_size` and the size to drop below, instead of
  presenting as a bad channel choice.
- **`segment()` warns on an empty mask.** It previously returned `component_count=0`
  with no warning at all, so the failure surfaced only if `measure()` happened to be
  called afterwards.
- **`frame_clipping` is withheld when coverage is implausible.** It asserts that size
  traits are a lower bound, which presumes the mask is the plant; on an inverted mask
  that misleads. A genuinely clipped, plausibly-sized plant still reports it.
- **`measure()` no longer destroys the host's PlantCV state.** `pcv.outputs.clear()`
  wiped a process-global table shared with any application also using PlantCV directly.
  Observations are now snapshotted and restored in a `finally`.
- **The stale-image guard compares content, not just shape.** Swapping the file for a
  different image of identical dimensions previously passed, measuring a stale mask
  against new pixels. A SHA-256 of the file is now recorded and re-checked.

### Notes

- Tool functions remain deliberately **synchronous**. mcp 1.28.1 runs sync tools inline
  on the event loop, which serialises them and is what makes PlantCV's process-global
  `pcv.outputs` safe here. Making a tool `async` would allow two analyses to interleave
  on that global and needs a lock first.

## [0.1.0] — 2026-07-28

Phase 1. PlantCV exposed as an MCP **measurement instrument**: the model proposes a
segmentation, the server returns the numbers _and the overlay it measured_.

### Added

- Four MCP tools:
  - `list_methods()` — available segmentation channels and methods, plus the resolved
    PlantCV version and guidance on which channel suits which image.
  - `suggest_segmentation(image_path, channel="a")` — a colourspace contact sheet and an
    auto-threshold contact sheet, so the channel/method choice is informed rather than
    blind. The server never picks silently.
  - `segment(image_path, channel, method)` — mints a `session_id` and returns the overlay
    image alongside mask diagnostics (mask fraction, component count, major-object count,
    largest area) and any fired warnings. Returns **no traits**.
  - `measure(session_id)` — plant traits with units for a segmentation `segment` produced.
- **Degeneracy gate.** `measure` refuses to return traits on an empty or degenerate mask
  (zero connected components, zero largest-component area, or mask fraction below 0.1% of
  the frame), raising `DegenerateMaskError` with the mask fraction and a re-segmentation
  suggestion instead.
- **Multi-specimen warning** when two or more connected components each reach at least 25%
  of the largest component's area — a whole-image ROI would merge them into one "plant"
  and every size trait would describe the group.
- **Frame-clipping warning** when mask pixels touch any of the four frame edges, flagging
  that size traits are a lower bound rather than a measurement.
- **Session store** with LRU eviction (default 8 sessions), holding a copy of the mask and
  the image path rather than the decoded RGB image.
- `ImageChangedSinceSegmentationError` — `measure` re-reads the image from disk and refuses
  to proceed if its shape no longer matches the mask's, rather than silently applying a
  stale mask or letting PlantCV raise an opaque `IndexError`.
- Overlay and contact-sheet downscaling is always **reported** in the response
  (`overlay_scale`, `colorspace_sheet_scale`, `threshold_sheet_scale`), never silent.
- 47 tests, including a real-execution integration test over a tracked fixture render and
  determinism checks across repeat runs.
- `docs/MUTATION-CHECKS.md` — the mutants each guard test was confirmed to go red against.
- Continuous integration: lint, format check, and the full suite on Python 3.11, 3.12 and
  3.13; a packaging job that asserts the sdist ships `NOTICE`/`LICENSE`/`CHANGELOG.md` and
  leaks none of `.superpowers`, `docs/superpowers` or `.claude`, and that the built wheel
  imports and still registers exactly the expected tool surface; and an absolute-path check that runs its
  own negative control so a silently-disabled check fails the build.
- `NOTICE` — attribution for PlantCV (MPL-2.0), an explicit statement that this project is
  unofficial and unaffiliated with the Donald Danforth Plant Science Center, and the
  provenance of the test fixture.
- PyPI metadata: authors, keywords, classifiers, and project URLs.
- README sections covering MCP client configuration (Claude Code and Claude Desktop) and the
  server's security/trust boundary.

### Notes

- **`segment` and `measure` are deliberately separate tools.** A single `analyze(image)`
  call would hand the model 17 plausible numbers having never shown it the mask. Because
  `measure` requires a `session_id` only `segment` mints, and `segment` returns the overlay,
  the protocol structurally puts the visual evidence in the model's context before a number
  can be obtained.
- PlantCV's own `in_bounds` and `object_in_frame` are treated as _additional information_,
  never as validity signals: on an all-zero mask PlantCV reports both `True` while returning
  a full 17-trait set of zeros. They are bounds checks, not success checks, so this server
  computes its own degeneracy gate instead.
- The 25% multi-specimen threshold and the 0.1% degeneracy floor are calibrated starting
  values, not constants, and are expected to be re-tuned against a wider image set.
- **The server reads image files from anywhere the host user can read**, with no directory
  allow-list or sandbox, and returns them to the model as base64 images. This is the same
  trust boundary as any local filesystem MCP server, and it is documented rather than
  silently assumed. Restricting reads to a configured root is a candidate for a later release.
- `plantcv` is hard-pinned to `==4.11.3` on purpose: trait values can shift between PlantCV
  releases and determinism is a tested guarantee. `opencv-python` is deliberately left without
  an upper bound, because PlantCV already caps it and duplicating that ceiling would drift.

### Not included (phase 2)

`measure` uses the whole image as its region of interest; explicit and multi-plant ROIs
(`roi.rectangle`, `roi.circle`, `roi.auto_grid`) are not yet exposed — which is why the
multi-specimen warning can only advise, not correct. Also deferred: mask refinement
(`refine`), morphology traits (leaf insertion/tangent angles, stem, skeleton segments),
batch processing, and hyperspectral/thermal imaging.
