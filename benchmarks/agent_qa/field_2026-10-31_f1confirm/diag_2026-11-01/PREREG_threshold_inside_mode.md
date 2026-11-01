# Pre-registration: the `threshold_inside_mode` warning, 2026-11-01

Written and pushed before any held-out cell is opened. The cut-off was chosen on
the calibration crops in `README.md` (19 small-plant Otsu overcounts at 0.55 to
0.97, 26 other crops at 0.27 or less). It does not move after this file is
pushed.

## The warning

- **Signal:** `density = hs[t] / max(hs)`, where `hs` is the channel's 256-bin
  histogram smoothed by a 3-bin moving average and `t` is the threshold value the
  method applied. Computed on the whole grey channel that was thresholded.
- **Methods:** the single-value automatic thresholds, `otsu` and `triangle`. Not
  `mean`/`gaussian` (no single value) and not `rules`.
- **Fires when** `density >= 0.4`.
- **Emitted by** `segment` and `measure_images` (batch). It warns and does not
  block a batch.

## Held-out set

Every eligible plant (`field_2026-10-24/build.py`'s exclusion rules, unchanged)
in every annotated image of trays 031 (43 images) and 032 (13 images). Excluded:
the 40 (image, row, column) cells of the F1 questions (`f1c_*` and `f1_*`) on
which the cut-off was chosen.

Each plant is cropped to its pot cell, as `sweep.py` does, then run through the
real tools: `segment` (`a`, `otsu`, `dark`, `fill_size` 50) → `refine`
`keep_largest` → `measure` area. Its label is OVER if the area is above `px_hi`,
RIGHT if inside `[px_lo, px_hi]` and UNDER if below `px_lo`.

## Pass criteria (both required)

- **P1, sensitivity:** the warning fires on at least 90% of OVER cells.
- **P2, false alarms:** it fires on at most 10% of RIGHT cells.

Failure on either: the cut-off is not moved and the warning does not ship as
calibrated; the result is reported as it is. Fewer than 10 OVER cells: OPEN
(too few to judge).

Reported without a gate: the firing rate on UNDER cells; results by tray and by
plant fraction of the cell; and on cells where it fires and Otsu is OVER, how
often `triangle` is RIGHT (the remedy the message names).

## Controls

- **Positive control first:** the evaluation script reproduces `sweep.json`'s
  `otsu50` area on all 40 calibration cells exactly, and the warning fires on
  exactly the calibration cells whose density is at least 0.4. Any mismatch
  stops the run.
- **Caveat on independence:** the held-out cells include the calibration plants
  on other dates, and cells in one image share its lighting. They are not
  independent draws; results are also given per tray.
