# Field set 2026-10-24: real trays, hand-annotated truth

Two families of 10 questions on raw top-view tray photographs of Arabidopsis:
**F1**, the area of one plant in pixels and in mm²; **F2**, its leaf count.
Pre-registered in [`../PREREG_2026-10-24_field-families.md`](../PREREG_2026-10-24_field-families.md).

**Data.** Aberystwyth Leaf Evaluation Dataset: Bell J, Dee HM (2016), Zenodo,
<https://doi.org/10.5281/zenodo.168158>, CC BY 4.0. The images stay outside
the repo (`~/plantcv-mcp-artifacts/field_2026-10-24/aberystwyth`, sha256 in its
`MANIFEST.json`). `fetch.py` reads only the needed members from the 61.6 GB
archive with HTTP range requests.

**Tray 032 only, and it is not unseen.** Only trays 031 and 032 have hand
annotations; 033 and 034 have none (dataset PDF §2.2; the archive's `gt/`
listing has 0 files for each). Tray 031 is excluded because `count_leaves` was
designed on it (docs/EVAL.md). Tray 032 is the server's own held-out reporting
tray: its figures were seen during development, though nothing was chosen from
them. This set is reported separately from the rendered set for that reason.

## Truth and tolerance

Nothing here calls plantcv-mcp.

- **Plant.** The hand-painted pixels that connect after a 25 px dilation (the
  docs/EVAL.md rule), matched to a pot by the tray grid. A pot is occupied
  unless the dataset's harvest sheet says the plant was gone by that date.
  Plants whose group touches a neighbour's are excluded. That rule kept 171 of
  the 210 annotated plants; all 39 exclusions are touching neighbours, from
  2016-01-05 on. Ten distinct images per family, one plant each, seeded.
- **F1 px:** the mask's pixel count. The window is the mask eroded and dilated
  by 1 px (3 × 3). That is −3% to −32% and +3% to +35% across the sample,
  widest on the youngest plants.
- **F1 mm²:** the lens is calibrated from the 14 of 23 shipped checkerboards
  OpenCV detects (12 × 9 inner corners, 26 mm squares, RMS 0.56 px). Each mask
  pixel is weighted by the area it covers after undistortion. The scale is
  60 mm over the rim size of the 6 interior pots in corrected coordinates:
  7.429 px/mm.
  - The window is the ±1 px band, widened by the spread across lens models
    (OpenCV's three, the dataset's own one-frame file, and leaving out each
    board in turn: at most 0.8%), plus 2 × 1.96 × the scale's standard error
    (0.8%).
  - **The scale is from the pot rim, not the pitch.** The pots sit about 1 mm
    apart in a holder, so the pitch has no known mm value.
  - **Interior pots only.** Edge pots measure about 2% wider after correction
    (some wall shows above the holder off-axis), and the top-right pot carries
    a label.
  - **The mm² is in the rim plane**, a few mm above the compost. It
    overstates the rosette's true area by a few percent, but the question
    names the convention.
- **F2:** distinct annotation colours covering at least 8 px of the plant (one
  colour per leaf, dataset PDF §2.3.1). Mostly hidden leaves keep their colour,
  so the question asks for them. At 1, 30 or 100 px instead of 8, a sampled
  truth moves by at most 2 leaves. Score: exact; |DiC| ≤ 1 is secondary
  (inter-observer |DiC| 0.46 ± 0.62, Giuffrida 2018).

## Controls, measured before any arm ran

**Barrel distortion.** It moves the image corners about 65 px, and an area
there grows by up to 14% when corrected.

**Uncorrected mm²** (raw px over the raw interior pots' px/mm):

- It reads edge plants low by a median 3.7% (range −0.9% to −6.7%) and interior
  plants by 0.2%.
- That is inside the ±1 px band for most plants. It fails on **3 of 171
  eligible plants**, all at the edge, and on **1 of the 10 sampled** (f1_10, a
  223,448 px rosette at row 1, column 5).
- So F1's mm² part tests segmentation far more than lens correction.

**Age guess for F2.** The median count of the plants of that date is exact on
60 of 171 and within one on 121. On the sample it is exact on 6 of 10. Young
rosettes vary little.

**Other controls:**

- **Positive control:** `reference.py` passes all 20 questions. It uses the
  dataset's shipped calibration, a resampled mask, its own pot scale and
  SciPy labelling.
- **Leaks:** the alt arm has no web tools in this set (`"alt_web": false`).
  Zenodo, CVPPP and dataset file names are leak markers in every set.
- **Recall (added 2026-10-31, report-only):** the leak scan reads tool traffic, so a
  model that only names the dataset in its own text or reply is not caught by it.
  Each run now also records `text_mentions`, the same markers in the model's own
  words; it never changes validity, which the PREREG fixed. Scanned afterwards, the
  50 saved runs of this set (pilot and the 10-26 F1 re-run) contain none. 47 of them
  have text to scan, and a marker planted into one of them is found.

## Files

- `fetch.py`: range-reads the members.
- `build.py`: writes `questions.json`, plus `build_report.json` in the
  artifacts directory. Needs `uv run --with openpyxl==3.1.5` for the harvest
  sheet.
- `field_lib.py`: the truth.
- `reference.py`: the positive control.

Tests: `tests/test_agent_qa_field.py`.
