# Mutation checks

Each guard was actually disabled in the working tree, the full suite was
actually run against the mutated code, the results below were recorded from
that run, and the file was then restored to its original content (verified
with `git status --porcelain` and a `diff` against a pre-mutation backup
copy, both clean/identical) before moving to the next guard. A guard whose
test passes with the guard removed is not a test.

Baseline before mutation: `uv run pytest -q` → 41 passed.

| guard           | mutation applied                                                                                                                   | tests that went red                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | date       |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| degeneracy gate | body of `assert_not_degenerate` replaced with `pass` (both raise branches removed)                                                 | `tests/test_measurement.py::test_empty_mask_refuses_and_valid_mask_returns_traits` — `Failed: DID NOT RAISE DegenerateMaskError`; also (not named in the brief, but also red) `tests/test_diagnostics.py::test_degenerate_empty_mask_raises_and_valid_mask_does_not` and `tests/test_diagnostics.py::test_degenerate_below_min_fraction_raises_and_just_above_does_not`, same failure message. 3 failed, 38 passed.                                                                                                                 | 2026-07-28 |
| multi-specimen  | `if diag.major_object_count < 2:` changed to `< 99:` in `multi_specimen_warning`                                                   | `tests/test_diagnostics.py::test_multi_specimen_fires_on_measured_failure_and_not_on_single_plant` — `assert None is not None`; `tests/test_integration.py::test_real_render_fires_multi_specimen_warning_end_to_end` — `AssertionError: The multi-specimen guard did not fire on a known four-view render. diagnostics: 9 components, 4 major objects.`; `tests/test_server.py::test_segment_warnings_fire_for_real_failures_and_not_for_a_clean_plant` — `AssertionError: assert 'multi_specimen' in set()`. 3 failed, 38 passed. | 2026-07-28 |
| frame clipping  | `frame_clipping_warning` made to `return None` unconditionally as its first statement (rest of body left in place but unreachable) | `tests/test_diagnostics.py::test_frame_clipping_fires_when_touching_edge_and_not_when_interior` — `assert None is not None`; `tests/test_server.py::test_segment_warnings_fire_for_real_failures_and_not_for_a_clean_plant` — `AssertionError: assert 'frame_clipping' in set()`. 2 failed, 39 passed.                                                                                                                                                                                                                              | 2026-07-28 |

After each mutation, `src/plantcv_mcp/diagnostics.py` was restored from a
pre-mutation backup, `diff` against that backup was empty, `git status
--porcelain` reported only the untracked `tests/test_determinism.py`, and
`uv run pytest -q` returned to 41 passed before the next mutation was
applied.

No mutation produced zero reds — every guard listed here has at least one
test that depends on it.

Re-run these whenever a guard's logic changes.

## Round 2 — the two-sided mask-validity model (2026-07-28)

Baseline before mutation: `uv run pytest -q` → 60 passed. Each mutant was applied
to the working tree, the named test was run against the mutated code, and the file
was restored from a backup before the next mutant.

| guard                              | mutation applied                                                               | result               |
| ---------------------------------- | ------------------------------------------------------------------------------ | -------------------- |
| implausible coverage               | `if diag.mask_fraction <= max_fraction:` → `if True:` (always returns None)    | RED — test caught it |
| frame_clipping suppression         | `if not coverage:` → `if True:` in `_segment_impl`                             | RED — test caught it |
| fill-erasure diagnosis             | `if diag.component_count == 0 and pre_diag.component_count > 0:` → `if False:` | RED — test caught it |
| empty-mask warning                 | `if diag.component_count > 0:` → `if True:` (always returns None)              | RED — test caught it |
| `pcv.outputs` restore              | the `finally:` restore body replaced with `pass`                               | RED — test caught it |
| content-digest guard               | `if session.digest and file_digest(...) != session.digest:` → `if False:`      | RED — test caught it |
| `object_type` passthrough (server) | `object_type=object_type` → `object_type='dark'` in `_segment_impl`            | RED — test caught it |
| `fill_size` passthrough (server)   | `size=fill_size` → `size=200` in `_segment_impl`                               | RED — test caught it |

**The `object_type` passthrough mutant initially SURVIVED.** The first pass paired
it with a test that happened to use `object_type="dark"`, so hardcoding `'dark'`
changed nothing observable — and `test_object_type_is_reachable_and_changes_the_mask`
exercises `segment_mask` directly, never the server. The original P0 (a parameter
correct in the library but dropped at the server) was therefore untested.
`test_server_honours_object_type_end_to_end` and its `fill_size` counterpart were
added specifically to close that, and both mutants then went red. A surviving
mutant is a coverage report, not a nuisance.

## Round 3 — units, colour, and protocol metadata (2026-07-28)

Baseline: `uv run pytest -q` → 72 passed.

| guard                          | mutation applied                                          | result |
| ------------------------------ | --------------------------------------------------------- | ------ |
| area scales quadratically      | `value / (px_per_mm**2)` → `value / px_per_mm` (the trap) | RED    |
| `px_per_mm` passthrough (impl) | `px_per_mm=px_per_mm` → `None` inside `_measure_impl`     | RED    |
| `px_per_mm` passthrough (tool) | `px_per_mm=px_per_mm` → `None` inside the `measure` tool  | RED    |
| `analyses` passthrough         | `requested = tuple(analyses)…` → hardcoded `("size",)`    | RED    |
| histogram suppression          | `if not include_histograms:` → `if False:`                | RED    |
| colour analysis                | `if "color" in analyses:` → `if False:`                   | RED    |
| server instructions            | `FastMCP(name, instructions=…)` → `FastMCP(name)`         | RED    |
| tool annotations               | `annotations=READ_ONLY` removed from `measure`            | RED    |
| typed return → `outputSchema`  | `-> MeasureResult` → `-> dict`                            | RED    |
| unknown-analysis guard         | `if unknown:` → `if False:`                               | RED    |

The `px_per_mm` **tool-layer** mutant survived its first pairing, because that run named a
test which calls `_measure_impl` directly and so never crosses the tool boundary. Re-run
against `test_measure_over_the_real_mcp_layer_returns_structured_content` it goes red.
Same lesson as the `object_type` mutant above: **pair a mutant with a test that exercises
the layer the mutation is in.** When a mutant survives, check the pairing first — then check
the coverage.

## Round 4 — scale, colour correction, and batch (2026-07-28)

Baseline: `uv run pytest -q` → 84 passed.

| guard                            | mutation applied                                                   | result |
| -------------------------------- | ------------------------------------------------------------------ | ------ |
| crop-before-threshold            | `crop = img[y0:y1, x0:x1]` → `crop = img` (reverts the causal fix) | RED    |
| marker edge-touch guard          | `if touches:` → `if False:`                                        | RED    |
| colour correction applied        | `tr.auto_correct_color(rgb_img=img)` → `img` (no-op)               | RED    |
| missing-card raises              | early `return img` inserted before the raise (silent fallback)     | RED    |
| `BLOCKING_CODES` populated       | frozenset emptied                                                  | RED    |
| batch honours blocking guards    | `blocking = [...]` → `blocking = []`                               | RED    |
| batch size cap                   | `if len(image_paths) > MAX_BATCH:` → `if False:`                   | RED    |
| batch scale/analyses passthrough | `analyses=analyses, px_per_mm=px_per_mm` → `("size",), None`       | RED    |
| new tools' `outputSchema`        | `-> BatchResult` → `-> dict`                                       | RED    |
| new tools' `outputSchema`        | `-> ScaleResult` → `-> dict`                                       | RED    |

The last two are worth noting. `calibrate_scale_from_marker` and `measure_images` were first
written with a bare `-> dict`, so they returned a JSON string in a text block and published no
schema, while the older tools returned structured content — the exact regression the round-3
work had just fixed, reintroduced by new code. It was caught by calling the tool and looking at
what came back, not by any test. `test_every_structured_tool_publishes_an_output_schema` now
asserts the rule for **every** tool rather than a named list, and the two mutants above confirm
it fires. A per-tool assertion protects the tools that exist; a rule protects the ones that
have not been written yet.


## Round 5 — per-region measurement (`measure_regions`)

| mutant                              | change                                                              | result |
| ----------------------------------- | ------------------------------------------------------------------- | ------ |
| empty-region guard removed          | `if label not in present:` → `if False:`                             | RED    |
| label mapping shifted by one        | `label = i + 1` → `i + 2` for the first two regions                   | RED    |
| `_as_xy` slices instead of refusing | `if len(pair) != 2:` → `if False:`                                    | RED    |
| overlay draws all regions as measured | `colour = ... else EMPTY_BGR` → `colour = MEASURED_BGR`             | RED    |

The second mutant is the one that matters. Region indices and PlantCV's label
indices are separate sequences that happen to line up, and nothing but this test
holds them together; if they drift, every row after an empty cell describes the
neighbouring plant, with numbers that look entirely reasonable. It is only
detectable because the fixture's plants have deliberately different areas
(1257 / 5027 / 11310 px). Identical plants would make the mutant invisible.


## Round 6 — the 1.3.1–1.5.2 guards (2026-08-29)

Everything shipped between 1.3.1 and 1.5.2 (dropped-object naming, the noise rule,
per-cell `multi_specimen`, batch grid semantics, the time budget, recipe validation,
thermal `fill_erased_mask`, `analyses=[]`), disabled one at a time; full suite each
time (276 tests, ~80 s). Runner: a shell loop of literal `sed` mutations with
`git checkout` between them.

| mutant                                   | change                                                                          | result |
| ---------------------------------------- | ------------------------------------------------------------------------------- | ------ |
| noise rule never fires                   | `component_count - major_object_count >= NOISY_MINOR_COMPONENTS` → `False`      | RED (4) |
| noise rule ignores the dominant object   | `and largest_frac < NOISY_LARGEST_FRACTION` → `and True`                        | **GREEN → fixed** |
| per-cell `multi_specimen` removed        | `multi = multi_specimen_warning(whole_diag, scope="cell")` → `multi = None`     | RED    |
| per-cell `multi_specimen` on the crop    | `whole_diag = analyze_mask(labeled == label)` → same on the cell's bbox slice   | **GREEN → fixed** |
| `object_exceeds_region` always fires     | `if ratio < EXCEEDS_CELL_RATIO:` → `if True:`                                   | RED (3) |
| batch: noisy still blocks with a grid    | `blocking = [w … if w.code != "noisy_segmentation"]` → `blocking = blocking`    | RED    |
| batch: merged rows returned measured     | `if r["measured"] and spill:` → `if False:`                                     | RED    |
| batch: time budget off                   | `if max_seconds is not None and entries and elapsed > max_seconds:` → `if False:` | RED  |
| batch: `nrows` alone accepted            | `if grid["nrows"] is None or grid["ncols"] is None:` → `if False:`              | RED    |
| batch: `MAX_REGIONS` cap off             | `if grid["nrows"] * grid["ncols"] > MAX_REGIONS:` → `if False:`                 | RED    |
| batch: `mode` unchecked                  | `if grid["mode"] not in REGION_MODES:` → `if False:`                            | RED    |
| batch: `radius` unchecked                | `if grid["radius"] is not None and grid["radius"] <= 0:` → `if False:`          | RED    |
| batch: dedupe off                        | `(duplicates if path in unique else unique).append(path)` → `unique.append(path)` | RED  |
| thermal: `fill_erased_mask` text lost    | `if erased is not None:` → `if False:`                                          | RED    |
| refine: dropped-object floor 10% → 50%   | `a >= DROPPED_OBJECT_FRACTION * largest` → `a >= 0.5 * largest`                 | RED (2) |
| suggest: empty polarity unwarned         | `if per_polarity[recommended]["component_count"] == 0:` → `if False:`           | RED    |
| `analyses=[]` allowed                    | `if not analyses:` → `if False:` in `validate_analyses`                         | RED (2) |

Two of seventeen were green, and both were the same kind of hole: a rule with two
halves, tested only from the side that fires.

**The noise rule** is "≥50 minor components AND no object holds half the mask".
The only test was a speckled scene that IS noisy; nothing asserted that a plant
holding 70% of the mask with sixty specks around it is measurable — which is the
case the 1.5.0 calibration was done for (the 27-seedling tray that the first rule
refused). `test_is_noisy_needs_both_many_specks_and_no_dominant_object` and
`test_a_dominant_plant_with_many_specks_is_not_noisy_end_to_end` now pin both
sides; under the mutant they fail with `assert not True`.

**The crop mutant** is the 1.5.2 false positive itself (a 20,533-px arabidopsis
whose leaf left its cell and came back read as two plants). A test for it existed
— `test_a_leaf_that_leaves_and_reenters_its_cell_is_one_plant` — and could not
fail: it put the body in cell 0 and the loop in cell 1, and PlantCV's partial ROI
handed the whole object to cell 1, whose crop was one piece. The test asserted
the right thing on a fixture that never exercised the bug. It now checks its own
geometry (`connectedComponents` on the crop == 2, on the frame == 1, both pieces
major), asserts which cell owns the object, and carries a positive control (two
real plants in one cell DO fire). Under the mutant it fails on `multi_specimen`.

278 tests after this round.


## Round 7 — the 1.5.3 morphology guards (2026-08-29)

The five guards from the first real-photo morphology round, disabled one at a
time the day they shipped (283 tests, ~85 s each).

| mutant                                      | change                                                                          | result |
| ------------------------------------------- | ------------------------------------------------------------------------------- | ------ |
| palette not reset on entering the section   | `pcv.params.saved_color_scale = None` → `pass` in `isolated_pcv_outputs`        | **GREEN → fixed** |
| host's palette not restored on exit         | `pcv.params.saved_color_scale = saved_palette` → `pass`                         | **GREEN → fixed** |
| palette not reset after the 2× prune pass   | the reset in `measure_morphology` → `pass`                                      | RED (2) |
| crop margin zero                            | `crop_margin()` → `return 0`                                                    | RED (2) |
| crop off                                    | `_crop_bounds(…)` → the whole frame                                             | RED    |
| every cv2.error swallowed                   | `if not _stem_line_leaves_int32(…):` → `if False:`                              | **GREEN → fixed** |
| coverage refusal off                        | `if coverage:` → `if False:`                                                    | RED    |
| combine-stem remedy off                     | `if "combine stem" in str(exc).lower():` → `if False:`                          | RED    |
| overlay pasted at the origin                | `canvas[y0:y1, x0:x1] = id_img` → `canvas[0:h, 0:w] = id_img`                   | RED    |

Three of nine green, and this time the pattern is "a guard with a belt and
braces, tested only at the braces". The palette is reset in two places (on
entering the isolated section, and again after the 2× prune pass); the morphology
tests only ever hit the second, so the first — and the restore that hands the
host's palette back — were free to go. `test_isolated_section_starts_with_an_empty_palette_and_hands_the_hosts_back`
now pins both, on the success and the error path. The vertical-stem handler
refits the stem before it swallows a `cv2.error`; nothing asserted that a
*different* `cv2.error` still escapes, so the verification was removable —
`test_a_cv2_error_that_is_not_the_vertical_stem_still_raises` does. 285 tests.


## Round 8 — the 1.5.5 guards (2026-08-30)

The seven guards from the panel-audit round, plus the *other half* of every
rule that has one — the constants' values, not just their presence (302 tests,
~90 s each). Runner: the round-6 shell loop of literal `sed` mutations.

| mutant                          | change                                                              | result |
| ------------------------------- | ------------------------------------------------------------------- | ------ |
| grid always explains noise      | `if grid_explains_components(…):` → `if True:`                       | RED    |
| grid never explains noise       | → `if False:`                                                        | RED (2) |
| explained-per-cell budget = 1   | `NOISE_EXPLAINED_PER_CELL = 4` → `1`                                 | **GREEN → fixed** |
| explained-per-cell budget = 100 | → `100`                                                              | RED    |
| unexplained sentence lost       | `elif any(noisy in blocking):` → `elif False:`                       | RED    |
| coverage demotion off           | `if cells >= 2:` → `if False:`                                       | RED    |
| coverage demoted for 1×1 too    | → `if True:`                                                         | **GREEN → fixed** |
| no_region_measured off          | `if … not any(measured):` → `if False:`                              | RED (2) |
| refuses on ANY unmeasured row   | `not any(…)` → `not all(…)`                                          | **GREEN → fixed** |
| grid args unchecked             | `if given:` → `if False:`                                            | RED (4) |
| dedup by string                 | `key = os.path.realpath(path)` → `key = path`                        | RED    |
| auto_grid unwrapped             | `except (ValueError, cv2.error)` → `except ()`                       | RED (5) |
| auto_grid catches ValueError only | → `except (ValueError,)`                                           | RED (2) |
| auto_grid catches cv2.error only  | → `except (cv2.error,)`                                            | RED (3) |
| owned-material guard off        | `if in_cell and owned < 0.2 * in_cell:` → `if False:`                | RED (2) |
| owned fraction 0.2 → 0.5        | `OWNED_MATERIAL_FRACTION = 0.2` → `0.5`                              | **GREEN → fixed** |
| owned fraction 0.2 → 0.02       | → `0.02`                                                             | RED (2) |
| in-cell material = whole frame  | `mask[y:y+h, x:x+w]` → `mask`                                        | RED (9) |
| crop remedy reworded            | "crop the photo so the leaf" → "photograph it so the leaf"           | **GREEN → fixed** |

Five of nineteen green, and four of the five are the same species: a calibrated
threshold tested only from the side that fires. Every refusal had a test; no
test stood on the *keep* side of the line, so each constant could drift to its
strictest value and take the legitimate cases silently.

**The per-cell budget of 4** exists for plants whose leaves are disconnected
mask pieces; at 1 it survived because the late-germination plate is exactly one
component per well. `test_a_grid_explains_plants_that_are_several_pieces_each`
puts three pieces in every cell of a 6×6 tray (108 components, whole-mask
noisy) and asserts the grid explains it — with a fixture-honesty check that the
count sits in the band where the constant's value decides (36 < 108 ≤ 144).

**The ≥2-cells condition** on coverage demotion survived `if True:` because no
test gave a 1×1 grid an inverted mask — the one grid whose single cell cannot
catch the background (it fits). `test_a_single_cell_grid_does_not_excuse_an_inverted_mask`
pins it, with the right polarity measured under the same grid as the control.

**`not any` vs `not all`** is the difference between "refuse the image nothing
was measured in" and "refuse every partial tray". A late-germination tray with
an empty well is the normal case, and nothing asserted it stays measured —
`test_a_tray_with_an_empty_cell_is_still_measured` does.

**The owned-material fraction** was calibrated on real trays (intruded-upon
X-Rite cells own 0.35–0.39 and are kept; the fragment owned 0.049) but the
keep side lived only in a scratch script. `test_an_intruded_upon_cell_keeps_its_own_plant`
builds a cell that owns 0.32 of its material, asserts it stays measured, and
that its area is its own object rather than the intruder's.

**The crop remedy** was pinned by `match="crop"` — which "segment the crop"
elsewhere in the same message satisfies, so the actionable sentence could be
reworded away. The test now matches the sentence ("crop the photo", "under
half"), not the word. 306 tests.


## Round 9 — the 1.6.0 card-exclusion and clipping guards (2026-08-30)

The guards from the scale+colour dogfood, disabled one at a time the day they
shipped (310 tests, ~2 min each). Predictions were logged before the run:
`server_exclusion_off` was expected GREEN (both new exclusion tests drive
`measure_batch`) and `card_pad_zero` uncertain.

| mutant                        | change                                                   | result |
| ----------------------------- | -------------------------------------------------------- | ------ |
| server exclusion off          | `if card is not None:` → `if False:` (segment path)      | RED    |
| batch exclusion off           | same, batch path                                         | RED (3) |
| advisory suppressed           | `if not removed:` → `if True:`                           | RED (2) |
| exclusion counts, zeroes nothing | `if removed:` → `if False:` in exclude_card           | RED (2) |
| card padding zero             | `pad = int(np.median(extents))` → `pad = 0`              | RED (3) |
| clipping for any component    | `and area >= 0.25 * largest` → `and True`                | RED    |
| clipping never                | `if not touching_major:` → `if True:`                    | RED (7) |
| clipping bar 10× largest      | `0.25 * largest` → `10 * largest`                        | RED (7) |

Eight of eight red — the first fully-red round, and both pre-logged green
predictions were wrong in the right direction. The server-path exclusion is
held by an existing tool-layer test
(`test_measure_images_honours_color_correct_and_refuses_cardless_images`),
and zero padding fails the bbox-coverage assert in
`test_correction_reports_where_the_card_is` twice over: the sample-circle
extent stops short of the chip grid, and the un-excluded chip edges survive
`fill_size` to break the component count. Nothing to pin; no code changed.

## Round 10 — the 1.7.0 card-region and per-cell guards (2026-08-31)

The guards from the panel audit of 1.6.0, disabled one at a time (319 tests,
~2 min each). Predictions were logged before the run: `sparse_lattice_ok`,
`seg_exclude_flag_off`, `refine_erased_ok`, and `regrown_not_summed` were
expected GREEN from visible test gaps; all four were, and were pinned.

| mutant                    | change                                                        | result |
| ------------------------- | ------------------------------------------------------------- | ------ |
| pad zero                  | `pad = 2.0 * CARD_PAD_PITCHES * pitch` → `pad = 0.0`          | RED (6) |
| pad fixed 41 px           | → `pad = 41.0` (the 1.6.0 bug resurrected)                    | RED (6) |
| pitch fixed 20 px         | `pitch = median(NN distance)` → `pitch = 20.0`                | RED (6) |
| axis-aligned regression   | `boxPoints(..., angle)` → `boxPoints(..., 0.0)`               | RED (2) |
| sparse lattice accepted   | `if len(centres) < 2:` → `< 0:`                               | GREEN → pinned |
| residual check off        | `> CARD_CHIP_RESIDUAL_MAX` → `> 1e9`                          | RED    |
| residual loosened to 0.7  | `CARD_CHIP_RESIDUAL_MAX = 0.45` → `0.7` (erased chip is 0.6)  | RED    |
| residual tightened to 0.05| → `0.05` (complete cards read up to 0.3)                      | RED    |
| segment() exclude flag off| `elif exclude_color_card:` → `elif False:` (segment path)     | GREEN → pinned |
| measure advisory off      | `if session.card_region is not None:` → `if False:` (measure) | RED    |
| refine re-exclusion off   | same, refine path                                             | RED    |
| refine-erased refusal off | `if regrown and not (mask > 0).any():` → `if False:`          | GREEN → pinned |
| regrown advisory off      | `if regrown:` → `if False:`                                   | RED    |
| regrown not accumulated   | `card_excluded_px + regrown` → `card_excluded_px`             | GREEN → pinned |
| batch exclude flag off    | `elif exclude_color_card:` → `elif False:` (batch path)       | RED    |
| probable_background never | `CELL_BACKGROUND_COVERAGE = 0.85` → `1.01`                    | RED    |
| probable_background 0.5   | → `0.5` (dense fixture cells are 0.72)                        | RED (2) |
| coverage 1×1 eligible     | `coverage_demoted = cells >= 2 and any(` → `any(`             | GREEN — EQUIVALENT |
| noise_cluster cell off    | `elif noisy_demoted and any(` → `elif False and any(`         | RED (2) |
| noisy image refusal off   | `if clusters:` → `if False:`                                  | RED (2) |

Fifteen of twenty red. Four greens were the predicted test gaps, pinned by
`test_a_lattice_of_one_chip_refuses_a_card_region` (one centre has no pitch;
the mutant dies converting NaN to int instead of refusing),
`test_segment_excludes_the_card_without_correction` (the only exclude-flag
test drove `measure_batch`; under the mutant segment() measures the card and
reports only `multi_specimen`), `test_a_session_inside_the_card_cannot_refine_to_nothing`
(positive control inside the same test), and
`test_the_card_debt_accumulates_across_refines` (under the mutant measure()
reported 21,600 px where 21,600 + 3,267 were owed). The fifth green is an
EQUIVALENT mutant, kept: at the per-cell loop `implausible_coverage` in
`warnings` already implies `cells >= 2`, because the code is unconditionally
in `BLOCKING_CODES` and the 1×1 path refuses upstream at the blocking gate —
the conjunct is local documentation of the demotion trade, and no public-API
test can distinguish it. 323 tests after pinning.

## Round 11 — the 1.8.1 lens, containment, and per-axis guards (2026-09-01)

The guards from the panel audit of 1.8.0, disabled one at a time (349 tests,
`pytest -x` per mutant so reds return in under a minute). Predictions were
logged before the run: ten mutants were expected GREEN from visible test
gaps — `L2`, `L3`, `L6`, `L10`, `L12`, `L13`, `S7`, `S10`, `S11`, `D4`,
`D6` minus the one judged equivalent — and exactly those ten were; no
unpredicted green, no predicted green went red.

| mutant                          | change                                                                 | result |
| ------------------------------- | ---------------------------------------------------------------------- | ------ |
| L1 shape = first decoded        | majority `Counter` → `decoded[0][1].shape`                             | RED    |
| L2 size tie → smaller           | tie-break key `s[0] * s[1]` → `-s[0] * s[1]`                           | GREEN → pinned |
| L3 minimum views 1              | `MIN_CALIBRATION_FRAMES = 3` → `1`                                     | GREEN → pinned |
| L4 pose-diversity off           | `if spread < POSE_DIVERSITY_MIN_PX:` → `< 0.0`                         | RED    |
| L5 pose-diversity 1000 px       | `POSE_DIVERSITY_MIN_PX = 1.0` → `1000.0`                               | RED    |
| L6 rms gate off                 | `if not isfinite(rms) or rms > MAX:` → `if False:`                     | GREEN → pinned |
| L7 rms gate 0.01                | `MAX_CALIBRATION_RMS = 100.0` → `0.01`                                 | RED    |
| L8 resolution refusal off       | `if (h, w) != calib.shape:` → `if False:`                              | RED    |
| L9 stored shape swapped         | `shape=(shape[0], shape[1])` → `(shape[1], shape[0])`                  | RED    |
| L10 validity `> 0`              | white-frame remap `== 255` → `> 0`                                     | GREEN → pinned |
| L11 no crop                     | `rect = _largest_valid_rectangle(valid)` → `(0, 0, w, h)`              | RED    |
| L12 degenerate edge 0           | `_MIN_CROP_EDGE = 16` → `0`                                            | GREEN → pinned |
| L13 degenerate voids 0          | `"residual_void_px": int((~valid).sum())` → `0`                        | GREEN → pinned |
| L14 crop_fraction 0             | `1.0 - (rw * rh) / (w * h)` → `0.0`                                    | RED    |
| S1 member containment off       | per-member `check_readable(path)` removed                              | RED    |
| S2 unreadable member raises     | `except OSError: data = b""` → `raise`                                 | RED    |
| S3 digest unprefixed            | length-prefix `update`s removed (bare name‖bytes)                      | RED    |
| S4 cache key ignores digest     | `key = (digest, rows, cols)` → `("", rows, cols)`                      | RED    |
| S5 O_NOFOLLOW off               | `flags \|= O_NOFOLLOW` removed (imaging.write_image)                   | RED    |
| S6 O_EXCL off                   | `(O_EXCL if exclusive else O_TRUNC)` → `O_TRUNC`                       | RED    |
| S7 output_path uncontained      | `out = check_readable(output_path)` → `out = output_path`              | GREEN → pinned |
| S8 high_reprojection_error off  | `HIGH_REPROJECTION_RMS = 5.0` → `1e9`                                  | RED    |
| S9 thin_calibration off         | `if len(frames_used) < 5:` → `< 0`                                     | RED    |
| S10 voids advisory: degenerate only | `if roi_degenerate or residual_void_px > 0:` → `if roi_degenerate:` | GREEN — EQUIVALENT |
| S11 crop note never "cropped"   | `elif info["crop_fraction"] > 0:` → `elif False:`                      | GREEN → pinned |
| D1 diagonal regression          | per-axis `and` test → union/major diagonal ratio                       | RED    |
| D2 axes `or`                    | `union_w <= … and union_h <= …` → `or`                                 | RED    |
| D3 ratio 1.10                   | `EXTENT_INFLATION_RATIO = 1.25` → `1.10`                               | RED    |
| D4 ratio 2.0                    | → `2.0` (the real photo's sliver was 2.17x)                            | GREEN → pinned |
| D5 major threshold 0            | `major_threshold=0.25` → `0.0`                                         | RED    |
| D6 baseline = largest only      | `_extent(rows[is_major])` → extent of the single largest component     | GREEN → pinned |
| D7 offender by area             | `max(rows[~is_major], key=_overhang)` → key = area                     | RED    |
| D8 remedy ignores majors        | `if majors >= 2:` → `if False:`                                        | RED    |
| D9 mask_warnings wiring off     | `inflation = minor_extent_inflation_warning(mask, diag)` → `None`      | RED    |
| D10 refine wiring off           | same, in `refinement_warnings`                                         | RED    |
| C1 contiguous-object sentence   | `marker_touches_crop_edge` message truncated at "the marker."          | RED    |

Twenty-five of thirty-five red. Nine greens pinned, each watched red under
its own mutant on live code first:

- `test_a_size_tie_goes_to_the_larger_frames` (three thumbnails vs three
  full views; under the mutant the thumbnails win and are refused as one
  pose).
- `test_two_views_are_refused_by_the_literal_minimum` — the existing
  too-few test derives its fixture size AND its match string from
  `MIN_CALIBRATION_FRAMES`, so `= 1` wrote zero frames and matched the "1"
  in "Only 0 of 1". The pin says "at least 3" in so many words.
- `test_a_meaningless_fit_is_refused[inf|nan|1e12]` — no frame set the
  suite can build reaches the rms gate: half-MIRRORED views calibrate at
  rms 1.67 with fx=131 (true 400), another low-rms wrong model. The
  optimiser is stubbed to return the garbage the gate exists for (the one
  mock in this file), after a real-optimiser positive control.
- `test_every_pixel_of_the_crop_is_real_source_data` — `> 0` keeps the
  void-blended border: minimum pixel 8 on an all-127 source with zero
  exactly-black pixels, which the no-fabricated-pixels test cannot see.
- `test_a_frame_too_small_to_crop_is_returned_whole_with_its_voids_counted`
  — pins L12 and L13 together. Eighteen extreme models on 640x480 all
  keep a >=16-px rectangle (smallest 52x347); a 30x30 frame under f=25,
  k1=-0.8 has none: returned uncropped, 889 voids == its black pixels.
- `test_an_explicit_output_path_outside_the_roots_is_refused` (positive
  control: an explicit path inside the roots is written).
- `test_a_cropped_frame_says_so` — "cropped … 17%" and never "No void crop
  was needed" on a cropped frame.
- `test_material_half_a_plant_width_away_still_inflates_the_extent` —
  every firing fixture inflated an axis >=2.3x, so the threshold could
  drift to 2.0 unseen, past the real photo's 2.17x; a sliver 20 px past a
  60-px plant (1.47x) now pins the upper side.
- `test_a_crumb_beside_the_second_plant_is_measured_against_both_plants` —
  two plants far apart plus a crumb at the second plant's edge is silent
  against the major-union baseline and a 4x "inflation" against the
  largest plant alone.

S10 is an EQUIVALENT mutant, kept: `undistort_image` writes the literal
`0` for `residual_void_px` on the crop path (`lens.py:291`) and the void
count only on the degenerate path (`lens.py:283`), so for any info the
tool produces `residual_void_px > 0` implies `roi_degenerate`; the `or`
conjunct is defensive documentation and no public-API test can distinguish
it. 360 tests after pinning.
