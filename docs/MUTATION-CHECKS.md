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
