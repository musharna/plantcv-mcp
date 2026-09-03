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

| mutant                                | change                                                  | result |
| ------------------------------------- | ------------------------------------------------------- | ------ |
| empty-region guard removed            | `if label not in present:` → `if False:`                | RED    |
| label mapping shifted by one          | `label = i + 1` → `i + 2` for the first two regions     | RED    |
| `_as_xy` slices instead of refusing   | `if len(pair) != 2:` → `if False:`                      | RED    |
| overlay draws all regions as measured | `colour = ... else EMPTY_BGR` → `colour = MEASURED_BGR` | RED    |

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

| mutant                                 | change                                                                            | result            |
| -------------------------------------- | --------------------------------------------------------------------------------- | ----------------- |
| noise rule never fires                 | `component_count - major_object_count >= NOISY_MINOR_COMPONENTS` → `False`        | RED (4)           |
| noise rule ignores the dominant object | `and largest_frac < NOISY_LARGEST_FRACTION` → `and True`                          | **GREEN → fixed** |
| per-cell `multi_specimen` removed      | `multi = multi_specimen_warning(whole_diag, scope="cell")` → `multi = None`       | RED               |
| per-cell `multi_specimen` on the crop  | `whole_diag = analyze_mask(labeled == label)` → same on the cell's bbox slice     | **GREEN → fixed** |
| `object_exceeds_region` always fires   | `if ratio < EXCEEDS_CELL_RATIO:` → `if True:`                                     | RED (3)           |
| batch: noisy still blocks with a grid  | `blocking = [w … if w.code != "noisy_segmentation"]` → `blocking = blocking`      | RED               |
| batch: merged rows returned measured   | `if r["measured"] and spill:` → `if False:`                                       | RED               |
| batch: time budget off                 | `if max_seconds is not None and entries and elapsed > max_seconds:` → `if False:` | RED               |
| batch: `nrows` alone accepted          | `if grid["nrows"] is None or grid["ncols"] is None:` → `if False:`                | RED               |
| batch: `MAX_REGIONS` cap off           | `if grid["nrows"] * grid["ncols"] > MAX_REGIONS:` → `if False:`                   | RED               |
| batch: `mode` unchecked                | `if grid["mode"] not in REGION_MODES:` → `if False:`                              | RED               |
| batch: `radius` unchecked              | `if grid["radius"] is not None and grid["radius"] <= 0:` → `if False:`            | RED               |
| batch: dedupe off                      | `(duplicates if path in unique else unique).append(path)` → `unique.append(path)` | RED               |
| thermal: `fill_erased_mask` text lost  | `if erased is not None:` → `if False:`                                            | RED               |
| refine: dropped-object floor 10% → 50% | `a >= DROPPED_OBJECT_FRACTION * largest` → `a >= 0.5 * largest`                   | RED (2)           |
| suggest: empty polarity unwarned       | `if per_polarity[recommended]["component_count"] == 0:` → `if False:`             | RED               |
| `analyses=[]` allowed                  | `if not analyses:` → `if False:` in `validate_analyses`                           | RED (2)           |

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

| mutant                                    | change                                                                   | result            |
| ----------------------------------------- | ------------------------------------------------------------------------ | ----------------- |
| palette not reset on entering the section | `pcv.params.saved_color_scale = None` → `pass` in `isolated_pcv_outputs` | **GREEN → fixed** |
| host's palette not restored on exit       | `pcv.params.saved_color_scale = saved_palette` → `pass`                  | **GREEN → fixed** |
| palette not reset after the 2× prune pass | the reset in `measure_morphology` → `pass`                               | RED (2)           |
| crop margin zero                          | `crop_margin()` → `return 0`                                             | RED (2)           |
| crop off                                  | `_crop_bounds(…)` → the whole frame                                      | RED               |
| every cv2.error swallowed                 | `if not _stem_line_leaves_int32(…):` → `if False:`                       | **GREEN → fixed** |
| coverage refusal off                      | `if coverage:` → `if False:`                                             | RED               |
| combine-stem remedy off                   | `if "combine stem" in str(exc).lower():` → `if False:`                   | RED               |
| overlay pasted at the origin              | `canvas[y0:y1, x0:x1] = id_img` → `canvas[0:h, 0:w] = id_img`            | RED               |

Three of nine green, and this time the pattern is "a guard with a belt and
braces, tested only at the braces". The palette is reset in two places (on
entering the isolated section, and again after the 2× prune pass); the morphology
tests only ever hit the second, so the first — and the restore that hands the
host's palette back — were free to go. `test_isolated_section_starts_with_an_empty_palette_and_hands_the_hosts_back`
now pins both, on the success and the error path. The vertical-stem handler
refits the stem before it swallows a `cv2.error`; nothing asserted that a
_different_ `cv2.error` still escapes, so the verification was removable —
`test_a_cv2_error_that_is_not_the_vertical_stem_still_raises` does. 285 tests.

## Round 8 — the 1.5.5 guards (2026-08-30)

The seven guards from the panel-audit round, plus the _other half_ of every
rule that has one — the constants' values, not just their presence (302 tests,
~90 s each). Runner: the round-6 shell loop of literal `sed` mutations.

| mutant                            | change                                                     | result            |
| --------------------------------- | ---------------------------------------------------------- | ----------------- |
| grid always explains noise        | `if grid_explains_components(…):` → `if True:`             | RED               |
| grid never explains noise         | → `if False:`                                              | RED (2)           |
| explained-per-cell budget = 1     | `NOISE_EXPLAINED_PER_CELL = 4` → `1`                       | **GREEN → fixed** |
| explained-per-cell budget = 100   | → `100`                                                    | RED               |
| unexplained sentence lost         | `elif any(noisy in blocking):` → `elif False:`             | RED               |
| coverage demotion off             | `if cells >= 2:` → `if False:`                             | RED               |
| coverage demoted for 1×1 too      | → `if True:`                                               | **GREEN → fixed** |
| no_region_measured off            | `if … not any(measured):` → `if False:`                    | RED (2)           |
| refuses on ANY unmeasured row     | `not any(…)` → `not all(…)`                                | **GREEN → fixed** |
| grid args unchecked               | `if given:` → `if False:`                                  | RED (4)           |
| dedup by string                   | `key = os.path.realpath(path)` → `key = path`              | RED               |
| auto_grid unwrapped               | `except (ValueError, cv2.error)` → `except ()`             | RED (5)           |
| auto_grid catches ValueError only | → `except (ValueError,)`                                   | RED (2)           |
| auto_grid catches cv2.error only  | → `except (cv2.error,)`                                    | RED (3)           |
| owned-material guard off          | `if in_cell and owned < 0.2 * in_cell:` → `if False:`      | RED (2)           |
| owned fraction 0.2 → 0.5          | `OWNED_MATERIAL_FRACTION = 0.2` → `0.5`                    | **GREEN → fixed** |
| owned fraction 0.2 → 0.02         | → `0.02`                                                   | RED (2)           |
| in-cell material = whole frame    | `mask[y:y+h, x:x+w]` → `mask`                              | RED (9)           |
| crop remedy reworded              | "crop the photo so the leaf" → "photograph it so the leaf" | **GREEN → fixed** |

Five of nineteen green, and four of the five are the same species: a calibrated
threshold tested only from the side that fires. Every refusal had a test; no
test stood on the _keep_ side of the line, so each constant could drift to its
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

| mutant                           | change                                              | result  |
| -------------------------------- | --------------------------------------------------- | ------- |
| server exclusion off             | `if card is not None:` → `if False:` (segment path) | RED     |
| batch exclusion off              | same, batch path                                    | RED (3) |
| advisory suppressed              | `if not removed:` → `if True:`                      | RED (2) |
| exclusion counts, zeroes nothing | `if removed:` → `if False:` in exclude_card         | RED (2) |
| card padding zero                | `pad = int(np.median(extents))` → `pad = 0`         | RED (3) |
| clipping for any component       | `and area >= 0.25 * largest` → `and True`           | RED     |
| clipping never                   | `if not touching_major:` → `if True:`               | RED (7) |
| clipping bar 10× largest         | `0.25 * largest` → `10 * largest`                   | RED (7) |

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

| mutant                     | change                                                        | result             |
| -------------------------- | ------------------------------------------------------------- | ------------------ |
| pad zero                   | `pad = 2.0 * CARD_PAD_PITCHES * pitch` → `pad = 0.0`          | RED (6)            |
| pad fixed 41 px            | → `pad = 41.0` (the 1.6.0 bug resurrected)                    | RED (6)            |
| pitch fixed 20 px          | `pitch = median(NN distance)` → `pitch = 20.0`                | RED (6)            |
| axis-aligned regression    | `boxPoints(..., angle)` → `boxPoints(..., 0.0)`               | RED (2)            |
| sparse lattice accepted    | `if len(centres) < 2:` → `< 0:`                               | GREEN → pinned     |
| residual check off         | `> CARD_CHIP_RESIDUAL_MAX` → `> 1e9`                          | RED                |
| residual loosened to 0.7   | `CARD_CHIP_RESIDUAL_MAX = 0.45` → `0.7` (erased chip is 0.6)  | RED                |
| residual tightened to 0.05 | → `0.05` (complete cards read up to 0.3)                      | RED                |
| segment() exclude flag off | `elif exclude_color_card:` → `elif False:` (segment path)     | GREEN → pinned     |
| measure advisory off       | `if session.card_region is not None:` → `if False:` (measure) | RED                |
| refine re-exclusion off    | same, refine path                                             | RED                |
| refine-erased refusal off  | `if regrown and not (mask > 0).any():` → `if False:`          | GREEN → pinned     |
| regrown advisory off       | `if regrown:` → `if False:`                                   | RED                |
| regrown not accumulated    | `card_excluded_px + regrown` → `card_excluded_px`             | GREEN → pinned     |
| batch exclude flag off     | `elif exclude_color_card:` → `elif False:` (batch path)       | RED                |
| probable_background never  | `CELL_BACKGROUND_COVERAGE = 0.85` → `1.01`                    | RED                |
| probable_background 0.5    | → `0.5` (dense fixture cells are 0.72)                        | RED (2)            |
| coverage 1×1 eligible      | `coverage_demoted = cells >= 2 and any(` → `any(`             | GREEN — EQUIVALENT |
| noise_cluster cell off     | `elif noisy_demoted and any(` → `elif False and any(`         | RED (2)            |
| noisy image refusal off    | `if clusters:` → `if False:`                                  | RED (2)            |

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

| mutant                              | change                                                              | result             |
| ----------------------------------- | ------------------------------------------------------------------- | ------------------ |
| L1 shape = first decoded            | majority `Counter` → `decoded[0][1].shape`                          | RED                |
| L2 size tie → smaller               | tie-break key `s[0] * s[1]` → `-s[0] * s[1]`                        | GREEN → pinned     |
| L3 minimum views 1                  | `MIN_CALIBRATION_FRAMES = 3` → `1`                                  | GREEN → pinned     |
| L4 pose-diversity off               | `if spread < POSE_DIVERSITY_MIN_PX:` → `< 0.0`                      | RED                |
| L5 pose-diversity 1000 px           | `POSE_DIVERSITY_MIN_PX = 1.0` → `1000.0`                            | RED                |
| L6 rms gate off                     | `if not isfinite(rms) or rms > MAX:` → `if False:`                  | GREEN → pinned     |
| L7 rms gate 0.01                    | `MAX_CALIBRATION_RMS = 100.0` → `0.01`                              | RED                |
| L8 resolution refusal off           | `if (h, w) != calib.shape:` → `if False:`                           | RED                |
| L9 stored shape swapped             | `shape=(shape[0], shape[1])` → `(shape[1], shape[0])`               | RED                |
| L10 validity `> 0`                  | white-frame remap `== 255` → `> 0`                                  | GREEN → pinned     |
| L11 no crop                         | `rect = _largest_valid_rectangle(valid)` → `(0, 0, w, h)`           | RED                |
| L12 degenerate edge 0               | `_MIN_CROP_EDGE = 16` → `0`                                         | GREEN → pinned     |
| L13 degenerate voids 0              | `"residual_void_px": int((~valid).sum())` → `0`                     | GREEN → pinned     |
| L14 crop_fraction 0                 | `1.0 - (rw * rh) / (w * h)` → `0.0`                                 | RED                |
| S1 member containment off           | per-member `check_readable(path)` removed                           | RED                |
| S2 unreadable member raises         | `except OSError: data = b""` → `raise`                              | RED                |
| S3 digest unprefixed                | length-prefix `update`s removed (bare name‖bytes)                   | RED                |
| S4 cache key ignores digest         | `key = (digest, rows, cols)` → `("", rows, cols)`                   | RED                |
| S5 O_NOFOLLOW off                   | `flags \|= O_NOFOLLOW` removed (imaging.write_image)                | RED                |
| S6 O_EXCL off                       | `(O_EXCL if exclusive else O_TRUNC)` → `O_TRUNC`                    | RED                |
| S7 output_path uncontained          | `out = check_readable(output_path)` → `out = output_path`           | GREEN → pinned     |
| S8 high_reprojection_error off      | `HIGH_REPROJECTION_RMS = 5.0` → `1e9`                               | RED                |
| S9 thin_calibration off             | `if len(frames_used) < 5:` → `< 0`                                  | RED                |
| S10 voids advisory: degenerate only | `if roi_degenerate or residual_void_px > 0:` → `if roi_degenerate:` | GREEN — EQUIVALENT |
| S11 crop note never "cropped"       | `elif info["crop_fraction"] > 0:` → `elif False:`                   | GREEN → pinned     |
| D1 diagonal regression              | per-axis `and` test → union/major diagonal ratio                    | RED                |
| D2 axes `or`                        | `union_w <= … and union_h <= …` → `or`                              | RED                |
| D3 ratio 1.10                       | `EXTENT_INFLATION_RATIO = 1.25` → `1.10`                            | RED                |
| D4 ratio 2.0                        | → `2.0` (the real photo's sliver was 2.17x)                         | GREEN → pinned     |
| D5 major threshold 0                | `major_threshold=0.25` → `0.0`                                      | RED                |
| D6 baseline = largest only          | `_extent(rows[is_major])` → extent of the single largest component  | GREEN → pinned     |
| D7 offender by area                 | `max(rows[~is_major], key=_overhang)` → key = area                  | RED                |
| D8 remedy ignores majors            | `if majors >= 2:` → `if False:`                                     | RED                |
| D9 mask_warnings wiring off         | `inflation = minor_extent_inflation_warning(mask, diag)` → `None`   | RED                |
| D10 refine wiring off               | same, in `refinement_warnings`                                      | RED                |
| C1 contiguous-object sentence       | `marker_touches_crop_edge` message truncated at "the marker."       | RED                |

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

## Round 12 — the 1.9.0 calibration, containment, and offender guards (2026-09-01)

The guards from the panel audit of 1.8.2, disabled one at a time (371 tests,
`pytest -x` per mutant). Predictions logged before the run: five greens
expected (`L6`, `S7`, `I3`, `I4`, `D3`); ten survived — the other five
(`L4`, `S5`, `I1`, `D1`, `D2`) were real gaps the predictions missed.

| mutant                          | change                                         | result                     |
| ------------------------------- | ---------------------------------------------- | -------------------------- |
| L1 minimum orientations 1       | `MIN_DISTINCT_ORIENTATIONS = 3` → `1`          | RED                        |
| L2 minimum orientations 2       | → `2`                                          | RED                        |
| L3 minimum orientations 4       | → `4`                                          | RED                        |
| L4 cluster angle 0.05°          | `ORIENTATION_DISTINCT_DEG = 5.0` → `0.05`      | GREEN → pinned             |
| L5 cluster angle 60°            | → `60.0`                                       | RED                        |
| L6 normal sign not folded       | `abs(dot)` → `dot` in `_distinct_orientations` | GREEN — EQUIVALENT         |
| L7 k3 free                      | `flags=cv2.CALIB_FIX_K3` removed               | RED                        |
| L8 rms gate off                 | `MAX_CALIBRATION_RMS_FRACTION = 0.03` → `1e12` | RED                        |
| L9 rms gate 1e-6                | → `1e-6`                                       | RED                        |
| L10 fraction is pixels          | `rms_fraction` → `return rms`                  | RED                        |
| L11 EXIF-honouring decode       | `IMREAD_UNCHANGED` → `IMREAD_GRAYSCALE`        | RED                        |
| L12 coverage always 1.0         | `coverage=float(hull.mean())` → `1.0`          | RED                        |
| L13 orientation gate `<=`       | `if orientations < MIN` → `<=`                 | RED                        |
| S1 high-rms advisory never      | `HIGH_REPROJECTION_FRACTION = 0.0015` → `1.0`  | RED                        |
| S2 high-rms advisory always     | → `1e-9`                                       | RED                        |
| S3 coverage advisory never      | `MIN_CALIBRATION_COVERAGE = 0.4` → `0.0`       | RED                        |
| S4 coverage advisory always     | → `0.99`                                       | RED                        |
| S5 member open by original name | `os.open(real, …)` → `os.open(path, …)`        | GREEN → pinned             |
| S6 member open follows links    | `O_NOFOLLOW` removed from the member open      | RED                        |
| S7 board_coverage not reported  | result key removed                             | GREEN → pinned             |
| I1 symlink refusal branch off   | `if stat.S_ISLNK(...)` → `if False`            | GREEN → pinned             |
| I2 hard-link refusal off        | `if st.st_nlink > 1` → `if False`              | RED                        |
| I3 exclusive path via replace   | `os.link(tmp, path)` → `os.replace(tmp, path)` | GREEN — EQUIVALENT by race |
| I4 partial file left behind     | `finally: os.unlink(tmp)` removed              | GREEN → pinned             |
| D1 width axis gate forced on    | `if over_w:` → `if True:`                      | GREEN → pinned             |
| D2 height axis gate forced on   | `if over_h:` → `if True:`                      | GREEN → pinned             |
| D3 absolute overhang            | `/ major_w` removed from the width score       | GREEN → pinned             |
| D4 trigger `and` not `or`       | `if not (over_w or over_h)` → `and`            | RED                        |

Eighteen of twenty-eight red. Eight greens pinned, each watched red under
its own mutant on live code first:

- `test_one_tilt_slid_around_the_frame_is_one_orientation` (L4): a board at
  one tilt slid to eight positions reads as ONE orientation at 5° and at
  0.5°, but as SIX at 0.05° — its recovered normals jitter by tenths of a
  degree — and calibrates to fx 354. Positive control: eight tilts about
  one axis calibrate to the true camera.
- `test_a_symlinked_member_inside_the_roots_is_read_not_skipped` (S5): the
  swap test starts from a regular file, so `real == path` and `O_NOFOLLOW`
  catches the swap either way; what the resolved path buys is a LEGITIMATE
  symlink to a frame inside the roots being read rather than skipped.
- `test_board_coverage_is_reported_in_the_result` (S7).
- `test_a_symlink_at_the_derived_output_is_refused` now matches "is a
  symlink" (I1): the old `match="symlink"` was satisfied by the tmp PATH —
  pytest names it after the test — so the mutant's "not a regular file"
  message passed. The refusal mechanism (`not S_ISREG`) survived the mutant;
  only the message was unpinned.
- `test_no_partial_file_is_left_beside_the_output` (I4): after `os.replace`
  the temp name is gone anyway; the explicit-path `os.link` branch leaves
  `.partial` behind without the cleanup.
- `test_the_offender_is_on_the_triggering_axis_even_when_another_overhangs_more`
  (D1, D2): relative scoring makes the axis gate redundant for a single
  extender (a non-triggering axis's extender is always under 0.25), but not
  when the trigger comes from two sides at 0.15 each — a 2-px blob at 0.20
  on the silent axis would be named. The first fixture catches the height
  gate (D2), its transpose the width gate (D1). A first draft placed the
  blob adjacent to the plant and it merged under 8-connectivity — three
  components, not four; the gap is one pixel.
- `test_offender_overhang_is_relative_to_each_axis_extent` (D3): with both
  axes over threshold, 300 px on a 1000-px width (0.30) must not outrank
  8 px on a 10-px height (0.80).

Two greens are EQUIVALENT and kept: L6 — every board normal in the fixture,
the all-mirrored and the half-mirrored sets has z-sign +1 (an opaque board
never presents a flipped normal), so folding the sign cannot change a
count on any physical input; the `abs()` stays as a guard against a
numerically flipped Rodrigues axis. I3 — `write_image` refuses any existing
name by `lstat` before staging, so `os.link`'s EEXIST atomicity is
observable only in the race between that check and the link; no
single-process test can distinguish it. 377 tests after pinning.

## Round 13 — the 1.10.0 conditioning, outlier, containment, and remedy guards (2026-09-01)

The guards from the panel audit of 1.9.0, disabled one at a time (388 tests,
`pytest -x` over the lens, imaging, diagnostics and paths files per mutant).
Predictions logged before the run: three greens expected (`C3`, `U1`, `W5`);
four survived — `D2` was a real gap the predictions missed.

| mutant                              | change                                              | result             |
| ----------------------------------- | --------------------------------------------------- | ------------------ |
| C1 conditioning gate off            | `MAX_FOCAL_CONDITIONING = 20.0` → `1e9`             | RED                |
| C2 conditioning threshold 60        | → `60.0` (the ±7° set measures 52)                  | RED                |
| C3 fy instead of fx / min not max   | `sd[0]` → `sd[1]`; then `max(sd fx, sd fy)` → `min` | GREEN (equivalent) |
| O1 outlier ratio off                | `OUTLIER_VIEW_RATIO = 3.0` → `1e9`                  | RED                |
| O2 outlier fraction floor off       | `OUTLIER_VIEW_FRACTION = 0.0025` → `0.0`            | RED                |
| O3 bar is the smaller of the two    | `bar = max(` → `min(`                               | RED                |
| O4 no frame is ever an outlier      | `if px > bar` → `if px > 1e9`                       | RED                |
| U1 cluster boundary open            | `if angle <= deg` → `<`                             | GREEN (equivalent) |
| U2 no transitive merge              | `parent[find(i)] = find(j)` → `pass`                | RED                |
| W1 directory identity unchecked     | `if actual != directory:` → `if False:`             | RED                |
| W2 no-hard-link fallback off        | `if exc.errno not in _NO_HARDLINKS:` → `if True:`   | RED                |
| W3 deterministic temp name          | random token → `<name>.<pid>.partial`               | RED                |
| W4 blocking open                    | `O_NONBLOCK` removed from `_READ_FLAGS`             | RED                |
| W5 regular-file check off           | `if not stat.S_ISREG(...)` → `if False:`            | GREEN (equivalent) |
| W6 opened file unchecked            | `check_open_fd(fd, path)` → `pass`                  | RED                |
| W7 opened file always inside        | `if _inside(real, roots):` → `if True:`             | RED                |
| S1 output symlink check off         | `if os.path.islink(output_path):` → `if False:`     | RED                |
| S2 outlier advisory off             | `if calib.frames_outliers:` → `if False:`           | RED                |
| D1 fill_size from the offender only | `fill_size = max(... extenders)` → `= area`         | RED                |
| D2 flush components count as far    | `_overhang(r) > 0` → `>= 0`                         | GREEN → pinned     |

The green that was a gap: `D2` — `_overhang` is negative for a component
inside the majors' extent and exactly zero for one FLUSH with its edge, so
`>= 0` counted a flush bystander as far material and would have raised
`fill_size` to its area. Pinned in
`test_the_fill_size_remedy_clears_every_extender_not_just_the_named_one`
with a 750-px blob flush with the plant's left edge: `fill_size above 400`
must stand. RED under the mutant.

Three greens are EQUIVALENT and kept: `C3` — on any square-pixel camera the
fx and fy uncertainties track each other (every fixture set, and the real
tutorial set, within a few percent), so which one is judged, or the larger
or smaller of the two, cannot change a verdict; the larger is kept as the
conservative reading. `U1` — no set in the suite or in nature has two
board normals at exactly 5.000°, so the open boundary is unobservable.
`W5` — without root there is no device node to plant at a member's name,
and a FIFO with a writer attached is a race (`O_NONBLOCK` returns `EAGAIN`
or the writer's bytes depending on timing); the regular-file check is
observable only by privilege, and stays. 388 tests after pinning.

## Round 14 — the 1.11.0 multi-start, uncertainty, drop-rule, packing, dedup, and containment guards (2026-09-02)

The guards from the panel audit of 1.10.1, disabled one at a time (402
tests; `pytest -x` over the lens, imaging, diagnostics, paths and isolation
files per mutant, 109 of them, 53 s green). Predictions logged before the
run: sixteen greens expected; fourteen survived. Three were not predicted
(`L14`, `L24`, `L25`) and five predicted greens went red (`L15`, `L21`,
`L23`, `L27`, `S6`).

| mutant                                 | change                                                                | result                        |
| -------------------------------------- | --------------------------------------------------------------------- | ----------------------------- |
| L1 multi-start off                     | the three warm starts removed from `_best_fit`                        | RED                           |
| L2 best fit is the worst               | `min(fits, key=rms)` → `max`                                          | RED                           |
| L3 guess flag dropped                  | `CALIB_USE_INTRINSIC_GUESS` never set (every start is the cold start) | RED                           |
| L4 start error propagates              | `except cv2.error: continue` → `raise`                                | GREEN → pinned                |
| L5 non-finite rms wins `min`           | `f.rms if isfinite else inf` → `f.rms`                                | GREEN → pinned                |
| L6 uncertainty gate off                | `MAX_FOCAL_UNCERTAINTY = 0.04` → `1e9`                                | RED                           |
| L7 uncertainty gate tight              | → `0.001`                                                             | RED                           |
| L8 non-finite uncertainty passes       | `not isfinite(uncertainty) or` removed                                | RED                           |
| L9 uncertainty advisory off            | `FOCAL_UNCERTAINTY_ADVISORY = 0.025` → `1e9`                          | RED                           |
| L10 uncertainty advisory always        | → `0.0`                                                               | RED                           |
| L11 smaller of fx/fy uncertainty       | `sd_f = max(` → `min(`                                                | GREEN — EQUIVALENT            |
| L12 residual ratio off                 | `OUTLIER_VIEW_RATIO = 3.0` → `1e9`                                    | RED                           |
| L13 residual fraction floor off        | `OUTLIER_VIEW_FRACTION = 0.0025` → `0.0`                              | RED                           |
| L14 residual drop from three views     | `MIN_VIEWS_FOR_RESIDUAL_DROP = 4` → `2`                               | GREEN — EQUIVALENT in verdict |
| L15 residual drop needs five           | → `5`                                                                 | RED                           |
| L16 median includes the view itself    | `np.median(np.delete(pv, k))` → `np.median(pv)`                       | RED                           |
| L17 influence off                      | `INFLUENCE_SHIFT = 0.03` → `1e9`                                      | RED                           |
| L18 influence shift floor off          | → `0.0`                                                               | GREEN — design question       |
| L19 influence sigma off                | `INFLUENCE_SIGMA = 4.0` → `0.0`                                       | RED                           |
| L20 influence from four views          | `MIN_VIEWS_FOR_INFLUENCE = 5` → `4`                                   | GREEN → pinned                |
| L21 influence from eleven views        | → `11`                                                                | RED                           |
| L22 leave-one-out fit cold             | `guess=fit.mtx` → `guess=None`                                        | RED                           |
| L23 sigma from the full fit's sd       | `loose` from `fit.sd / fx` instead of the fit without the view        | RED                           |
| L24 first flagged, not worst           | `score > worst[0]` → `worst is None`                                  | GREEN → pinned                |
| L25 refit after a drop cold            | `fit = _best_fit(...)` → `_fit(...)` in the drop loop                 | GREEN → pinned                |
| L26 no orientation count after drops   | `if outliers:` → `if False:`                                          | GREEN → pinned                |
| L27 packing in frame order             | `sorted(normals, key=canonical)` → `normals`                          | RED                           |
| L28 dedup off                          | `if key in seen:` → `if False:`                                       | RED                           |
| L29 the copy is still decoded          | `continue` after the duplicate is recorded removed                    | RED                           |
| L30 too-few message without duplicates | `if duplicates` → `if False`                                          | RED                           |
| S1 duplicate advisory off              | `if calib.frames_duplicates:` → `if False:`                           | RED                           |
| S2 uncertain advisory off              | `if calib.focal_uncertainty > ADVISORY:` → `if False:`                | RED                           |
| S3 reason not in the advisory          | `f"{n} {why}"` → `f"{n}"`                                             | GREEN → pinned                |
| S4 reason not in the response          | `"reason": why` removed from the entry                                | GREEN → pinned                |
| S5 non-regular member vanishes         | `frames.append((name, b""))` removed                                  | RED                           |
| S6 non-regular member not digested     | the three `digest.update` calls removed                               | RED                           |
| S7 directory output unchecked          | `if os.path.isdir(output_path):` → `if False:`                        | RED                           |
| S8 focal uncertainty not reported      | `"focal_uncertainty"` removed from the response                       | GREEN → pinned                |
| P1 roots re-resolved per call          | `if _env_snapshot is None or _env_snapshot[0] != raw:` → `if True:`   | RED                           |
| P2 snapshot never refreshed            | → `if _env_snapshot is None:`                                         | GREEN → pinned                |
| I1 fallback ignores roots              | `if configured_roots() is not None: raise` removed                    | RED                           |
| I2 fallback always refuses             | `return dfd` → `raise`                                                | RED                           |
| I3 no exclusive claim                  | `os.close(os.open(name, _CREATE_FLAGS, ...))` removed                 | GREEN → pinned                |
| D1 remedy count none                   | `removed = <count of minors under fill_size>` → `1`                   | RED                           |
| D2 remedy count off by one             | `{removed - 1} other` → `{removed} other`                             | RED                           |

Thirty-one of forty-five red. Eleven greens pinned in nine tests, each watched red
under its own mutant on live code first, for the stated reason:

- `test_a_start_that_fails_inside_opencv_is_skipped_not_fatal` (L4): one
  warm start raising `cv2.error` must not abort a calibration the other
  starts recover. Under the mutant the simulated error propagates.
- `test_a_start_with_a_non_finite_residual_never_wins` (L5): `min` over
  residuals with a NaN among them is order luck — nothing compares less
  than NaN. With the cold start's rms forced to NaN, the mutant keeps it and
  refuses the set as meaningless; the guard sorts it last and a finite start
  wins.
- `test_four_views_are_not_judged_by_a_three_view_leave_one_out` (L20):
  views 2, 8, 9 and 10 calibrate to 0.5% with the correction 17 px at
  worst; judged by three-view fits without each, one shifts fx 7% at 4.7σ
  and under the mutant is dropped, the remaining three 'calibrating' to
  fx 427 with the correction 111 px wrong and every gate quiet. Nine of the
  120 four-view subsets probed would lose a view the same way.
- `test_the_worst_view_is_dropped_first_not_the_first_flagged` (L24): view
  6 of eight rippled by 6 px bends the camera until the honest view 0 looks
  influential (24% at 4σ, only with view 1 rippled by 2.5 px to loosen the
  remainder); the mutant drops view 0 first, then view 6. The guard drops
  view 6 and view 0 is quiet.
- `test_the_refit_after_a_drop_starts_from_several_focal_lengths_too`
  (L25): view 12 spoiled and dropped leaves the thirteen views that a cold
  start fits to fx 780; the mutant's cold refit landed there and, from that
  camera, dropped the honest views 6 and 9 before recovering.
- `test_dropped_views_that_supplied_the_orientations_are_named` (L26):
  four views at one tilt plus two rippled views at other tilts — three
  orientations at the start, one after the drops; the guard refuses for the
  orientations naming the dropped views, the mutant as 'undetermined' (16%).
- `test_the_response_carries_each_reason_and_the_focal_uncertainty` (S3,
  S4, S8): each dropped frame's reason in the advisory text and as `reason`
  on the response entry, and `focal_uncertainty` in the response.
- `test_a_new_environment_roots_value_is_resolved` (P2): the snapshot is
  per value of the variable; a new value is resolved.
- `test_the_no_hard_link_fallback_claims_the_name_exclusively` (I3): a
  file that appears at the name after the existence check is left alone
  and the write refused as existing; the mutant replaces it.

One green is a DESIGN QUESTION, recorded and not pinned: `L18` — the 3%
shift floor under the influence rule. No fixture shows it protecting an
honest view (every honest view of every set measures ≤ 2.2σ, so the sigma
gate alone would keep them all). The one fixture where the floor acts is
view 0 of the full fourteen sheared by 5%: shift 2.5% at 4.8σ, KEPT by the
floor with the correction 28.7 px at worst (fx 407), dropped without it
for 8.1 px (fx 397). At 6.5% shear the shift passes 3% and both drop it.
Pinning that would pin the worse outcome; whether "a shift that matters"
should be 3% of the focal length or some multiple of the uncertainty
advisory is a question for the next audit, logged in memory.

**Decided in 1.11.1.** Measured: over 203 views of 22 sound sets the floor
protects none of them (not one reaches 4σ at a shift under 3%), while a
leave-one-out shift shrinks as views are added and its σ barely does (one
view sheared 5%: 15.9% at 6.7σ among eight views, 2.5% at 4.8σ among
fourteen), so the floor read "the set is small", not "the shift matters".
Run end to end over 52 sets at seven floors it decided one outcome, and
decided it wrong. `INFLUENCE_SHIFT` is now 0.005 — half a percent, about 2 px
of applied correction at the fixture's measured 0.45 px per 0.1% — and the
`L18` mutant to 0.0 stays green by design: the floor is a numerical guard,
not a second opinion on guilt. Pinned by
`test_a_bad_view_is_not_excused_by_the_size_of_the_set`.

Two greens are EQUIVALENT and kept: `L11` — on a square-pixel camera the fx
and fy uncertainties track each other (every fixture set within a few
percent), so the smaller or the larger cannot change a verdict; the larger
stays as the conservative reading (round 13's `C3` again, at the new site).
`L14` — from three views a drop leaves two, which the orientation gate
refuses in every case; probed with view 0 of the sound three-view set
rippled by 1.5 to 4 px, the guard refuses as 'undetermined' (4.5–9.3%)
wherever the mutant would have dropped the view and refused for the
orientations, and accepts the same 1.5-px case either way (nothing stands
3× above two others). Same verdict on every three-view set; the floor only
decides which refusal is worded, and the dropped-view wording would be the
more useful one — noted with L18 for the next audit. 411 tests after
pinning.

## Round 15 — the 1.11.1 floor, the 1.12.0 dogfood fixes, and the #80 drop-value pin (2026-09-02)

Baseline: `uv run pytest -q` → 416 passed in 181 s. Every mutant was applied to
the working tree, the FULL suite was run against it, and the file restored from a
backup before the next; `git status --porcelain` reported only the untracked
`tmp/` after each. The passed count is the count of catching tests (416 − passed),
confirmed afterwards by re-applying each mutant against only the test predicted
to catch it and recording the assertion that fired.

| guard                                    | mutation applied                                                        | result         | caught by                                                        |
| ---------------------------------------- | ----------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------- |
| W1 write refusal never fires             | the whole `if …realpath(intended) == realpath(real_dir):` → `if False:` | RED (1)        | `…not_written_into_the_checkerboard_directory` — DID NOT RAISE   |
| W2 only an explicit `output_path` judged | the derived `<image>_undistorted.png` branch → `os.devnull`             | RED (1)        | same test, the derived-name refusal at line 1984                 |
| W3 intended path not normalised          | `os.path.dirname(os.path.realpath(intended))` → `os.path.dirname(...)`  | RED (1)        | same test, the `..`-through-sibling path at line 2004            |
| W4 directory not normalised              | `== os.path.realpath(real_dir)` → `== real_dir`                         | GREEN → EQUIV  | —                                                                |
| W5 comparison inverted                   | `==` → `!=`                                                             | RED (15)       | the positive control, plus every lens test that corrects a photo |
| S1 skip advisory removed                 | `if calib.frames_skipped:` → `if False and …`                           | RED (1)        | `…skipped_frame_is_named_even_when_the_calibration_is_thick`     |
| S2 skip named only when thin             | `… and len(calib.frames_used) < 5:` (the pre-1.12.0 gate restored)      | RED (1)        | same test — `assert 0 == 1`                                      |
| S3 skipped names dropped                 | `{', '.join(calib.frames_skipped)}` removed from the message            | RED (1)        | same test — `'not_a_board.png' in …`                             |
| S4 INNER-corners hint dropped            | `(INNER corners, one less per side than squares)` removed               | RED (1)        | same test — `'INNER corners' in …`                               |
| S5 remaining count dropped               | `used the remaining {len(calib.frames_used)}` → `used the rest`         | GREEN → pinned | —                                                                |
| M1 corner counts transposed              | `{row_corners} x {col_corners}` → `{col_corners} x {row_corners}`       | RED (1)        | `…wrong_corner_counts_are_echoed_the_way_they_were_given`        |
| M2 as-given clarifier dropped            | `"(row_corners x col_corners, as given)"` → `""`                        | RED (1)        | same test                                                        |
| D1 influence rule off                    | `INFLUENCE_SHIFT = 0.005` → `1e9`                                       | RED (3)        | the floor test and the #80 pin, both by name                     |
| D2 influence thresholds removed          | `INFLUENCE_SHIFT` → `0.0` and `INFLUENCE_SIGMA` → `1e-9`                | RED (32)       | the floor test, and the lens suite at large                      |
| D3 floor back to the old 3%              | `INFLUENCE_SHIFT = 0.005` → `0.03`                                      | RED (1)        | `test_a_bad_view_is_not_excused_by_the_size_of_the_set`          |

Thirteen of fifteen red. The three drop-rule mutants fail at the same assertion
with three different signatures, which is what makes them worth keeping apart:
`D1` leaves `set()` where `{'view0.png'}` is expected (nothing dropped), `D2`
drops `view0`, `view4` and more (everything dropped), and `D3` leaves `set()`
again — the bad view excused by the size of its set, which is exactly the
1.11.1 decision holding. `D1` additionally fails the #80 pin on its own words,
"the rule is expected to fire on this set".

One green PINNED: `S5` — the skip advisory names how many files were left out
but nothing asserted it names how many frames the calibration actually USED.
That is the half of the message that says whether what survived is worth
trusting: one skip out of fourteen views is a note, the same skip out of five is
most of the set gone, and without the number the two read identically. Pinned
inside `test_a_skipped_frame_is_named_even_when_the_calibration_is_thick` and
tied to the response's own `frames_used`, so the message cannot drift from what
was fitted. Watched red under its own mutant first — `assert 'remaining 14' in
'1 file(s) … were left out …'` — and green on live code.

One green EQUIVALENT, proven from the source rather than from a fixture: `W4`.
`check_readable` returns `os.path.realpath(path)` (`src/plantcv_mcp/paths.py:76`)
and `real_dir` is its return value, so the guard's second `realpath` resolves an
already-resolved path. No input can distinguish the two forms; the call stays as
the local statement of what the comparison needs. 416 tests after pinning (S5
added an assertion to an existing test, not a new one).
