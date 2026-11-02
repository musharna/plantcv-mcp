# Result: `threshold_inside_mode` for `segment` with `rules`, 2026-11-01

Pre-registered in `PREREG_rules_inside_mode.md` (`738814b`); run on `738814b`
with `rules_check.py`, 4 shards. Each shard first passed the 40/40 positive
control (labels equal to `rules_pilot.json` under both combines; the warning
fired exactly where the rule, applied to the pilot's densities, says). Rows in
`rules_check.json`.

Rules `[{a, otsu, dark}, {b, otsu, light}]` on the 739 held-out pot cells:

| combine | OVER fired      | RIGHT fired   | UNDER fired | verdict              |
| ------- | --------------- | ------------- | ----------- | -------------------- |
| `or`    | 355/359 (98.9%) | 2/362 (0.6%)  | 0/18        | R1 PASS, R2 PASS     |
| `and`   | 227/247 (91.9%) | 0/320         | 0/172       | R3 PASS, R4 PASS     |

**All four criteria pass.** The misses differ in kind:

- `or`: the 4 missed OVER cells are all within 1% of the window's top.
- `and`: the 20 missed OVER cells are large overcounts (median 4.4 times the
  window's top, up to 10.6). In every one the `a` rule fires (0.69 to 0.98)
  and the `b` rule sits just under the cut-off (0.35 to 0.40), so `and`, which
  needs every rule to fire, stays quiet. The 0.4 cut-off was chosen on channel
  `a`; `b` may need its own. It was not moved here: the pre-registration froze
  it, and R3 passed.

Not this warning's subject, but measured on the way: `a AND b` undercounts 172
of the 739 cells against 18 for `a OR b`. The GUIDE's example pairs them with
`or`.
