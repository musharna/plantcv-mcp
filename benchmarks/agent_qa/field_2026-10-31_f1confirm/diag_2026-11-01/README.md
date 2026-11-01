# Tracing the server's young-plant overcount, 2026-11-01

Exploratory, after the confirmatory result. Behind the "traced" paragraph of
[`../RESULTS.md`](../RESULTS.md).

- `sweep.py OUTDIR`: crops each F1 plant's pot cell (the 30 tray 031 questions
  and the pilot's 10) from the raw image, then runs `segment` (`a`, dark) through
  the real tools: Otsu with `fill_size` 0, 50 and 300, and `triangle` at 50, each
  then `keep_largest` and `measure`. Run from the repo root with the datasets
  fetched by `../../field_2026-10-24/fetch.py`.
- `sweep.json`: its output on plantcv-mcp 1.18.2, where `triangle` was still
  PlantCV's and so Otsu (`tri50` equals `otsu50` in every row).
- `sweep_tri.json`: the same sweep with `triangle` as OpenCV's `THRESH_TRIANGLE`,
  as plantcv-mcp #180 makes it.
- `valley.py OUTDIR`: for each cell crop (and the five crops the agent used on its
  young-plant misses), the histogram's height at the Otsu threshold as a share of
  its peak. Every small-plant overcount is 0.55 or more, every other crop 0.27 or
  less.

Plants under about 5% of their pot cell: Otsu wrong on 14 of 14 (+42% to
+5,232%), triangle right on 9. All 40: Otsu 21 right, triangle 29.
