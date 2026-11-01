# `threshold_inside_mode` on held-out cells: PASS

Run 2026-11-01 on `71a19bd` under [`PREREG_threshold_inside_mode.md`](PREREG_threshold_inside_mode.md)
(pushed as `31d5fb2` first). Four shards. Each one first reproduced the 40
calibration cells exactly: their cell bounds and px windows matched the questions,
their Otsu areas matched `sweep.json`, and the warning fired exactly where the
density is at least 0.4. Scoring checked that all 739 held-out cells were
present, each once.

## The pre-registered result

The held-out set is every eligible plant on trays 031 and 032 except the 40
calibration cells: 739 of 779. Each is a pot cell, segmented with Otsu on `a`.

| Otsu's area is | cells | warning fires | criterion       |
| -------------- | ----- | ------------- | --------------- |
| OVER the window | 351  | 351 (100%)    | P1 ≥ 90%: pass  |
| RIGHT           | 354  | 6 (1.7%)      | P2 ≤ 10%: pass  |
| UNDER the window | 34  | 0             | reported        |

By tray: 031 fires on 266 of 266 OVER and 5 of 282 RIGHT; 032 on 85 of 85 and 1
of 72.

## Reported without a gate

- **Otsu on a pot crop overcounts almost half of these plants.** All 351 OVER
  cells are plants under 6.4% of their cell (median 1.8%). They are 13% to
  9,967% too large, with a median of +718%.
- **The six false alarms are near-misses.** Each has density 0.40 to 0.42 and a
  plant 5.9% to 6.6% of its cell, and Otsu's area is within 1% of the top of the
  window (for example 14,088 px against a ceiling of 14,093). The lowest OVER
  density is 0.42, and the highest density among RIGHT cells that stay quiet is
  0.397: the margin is thin exactly where the label itself is.
- **The remedy works about half the time.** On the 351 cells where the warning
  fires and Otsu is OVER, `triangle` is right on 171, under on 98 and still over
  on 82. The message says so: the overlay decides, not the method name.

## Limits

- The held-out cells include the calibration plants on other dates, and cells in
  one image share its lighting, so they are not independent draws.
- One dataset, one camera, one channel (`a`). The cut-off is untested on other
  backgrounds.
- The warning is raised by `segment` and `measure_images`; `measure` does not
  repeat it, because it needs the image's histogram, not only the mask.
