# Pre-registration: `threshold_inside_mode` for `segment` with `rules`, 2026-11-01

Written and pushed before any held-out cell is run with rules. The signal and
its 0.4 cut-off are unchanged from `PREREG_threshold_inside_mode.md`.

## The rule

- `or`, or a lone rule: one warning for each `otsu`/`triangle` rule whose cut
  has density ≥ 0.4 (rules sharing a channel and method share one cut and are
  named once).
- `and`: the same warnings, but only when every rule is an `otsu`/`triangle`
  rule and every one fires. Otherwise none.

The `and` form was chosen on the 40 calibration cells (`rules_pilot.py`,
`rules_pilot.json`, a/otsu/dark with b/otsu/light, `fill_size` 50, then
`keep_largest`). With a warning on any firing rule, `and` flagged 3 of 19
RIGHT cells and 2 UNDER cells, where the b rule cut cleanly and the
intersection was fine. With every rule required: 8 of 9 OVER, 0 of 19 RIGHT.
Under `or`, any firing rule: 14 of 14 OVER, 0 of 22 RIGHT.

## Held-out set and labels

The 739 cells of `heldout.json` (the 40 calibration cells excluded), cropped
as `heldout.py` did. Each is run through the real tools: `segment` with rules
`[{a, otsu, dark}, {b, otsu, light}]` and `combine` `or`, then again with
`and`; each then `refine` `keep_largest` and `measure` area, labelled OVER /
RIGHT / UNDER against the cell's `[px_lo, px_hi]` as before.

Positive control, run first: the 40 calibration cells must reproduce
`rules_pilot.json`'s labels under both combines, and the warning must fire
exactly where the rule above, applied to the pilot's densities, says.

## Pass criteria

- **R1 (`or`):** fires on at least 90% of OVER cells.
- **R2 (`or`):** fires on at most 10% of RIGHT cells.
- **R3 (`and`):** fires on at least 90% of OVER cells. Fewer than 10 OVER
  cells under `and`: OPEN.
- **R4 (`and`):** fires on at most 10% of RIGHT cells.

If R1 or R2 fails, the `or` warning does not ship as designed and the result
is reported as it is. If R3 or R4 fails (OPEN alone does not fail), `and` gets
no warning: the function returns nothing for `and`. Neither the cut-off nor the
rule moves after this file is pushed.

Reported without a gate: firing on UNDER cells under each combine.
