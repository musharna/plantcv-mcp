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
