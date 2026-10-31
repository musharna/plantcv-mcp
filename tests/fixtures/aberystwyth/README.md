# Real Arabidopsis rosettes with hand-annotated leaves

Two crops from the **Aberystwyth Leaf Evaluation Dataset** (Bell, Jonathan; Dee,
Hannah M.; 2016), <https://doi.org/10.5281/zenodo.168158>, licensed
**CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>). Top-down
visible-light images of *Arabidopsis thaliana* trays with leaf-level ground
truth drawn by hand.

**Changes made:** each file is a crop of one plant (the plant's annotated
extent plus 60 px) from a tray image; the `_gt.png` crops have every pixel that
belongs to another plant set to black. Pixels are otherwise untouched (PNG,
lossless).

| file                                 | source image (in `images_and_annotations.zip`)           | crop `y0:y1, x0:x1` | annotated leaves |
| ------------------------------------ | -------------------------------------------------------- | ------------------- | ---------------- |
| `tray032_2015-12-21_plant0.png`      | `PSI_Tray032/tv/PSI_Tray032_2015-12-21--14-11-41_top.png` | `173:445, 715:966`  | 11               |
| `tray032_2016-01-05_plant0.png`      | `PSI_Tray032/tv/PSI_Tray032_2016-01-05--14-08-11_top.png` | `34:544, 130:643`   | 25               |

`*_gt.png` is the matching crop of `tv/gt/<name>_gt.png`: each leaf of the plant
is one colour, so the leaf count is the number of distinct non-black colours
(`tests/test_leaves.py` recounts it rather than trusting this table).
`tests/test_sam_leaves_model.py` runs Segment Anything on the same two crops.

**How they were chosen:** plant 0 (first in raster order) of each of the two
tray images used as the held-out set in `docs/EVAL.md` — fixed before any
count was looked at, so they are not the plants the tool does best on. The
annotation counts every leaf a person could identify, including small central
leaves and leaves mostly hidden under others; a mask-based method cannot see
the hidden ones.

These files ship in the sdist with the tests and are not part of the wheel.

## Tray mask crops (`*_mask_crop.png`)

Binary masks derived from two tray images of the same dataset (CC BY 4.0, as
above). `tests/test_labels.py` checks the region labeller against PlantCV's
`create_labels` on them: real mask structure, with hundreds of holes, objects
crossing cell boundaries, and one component filling most of a crop.

**Changes made:** each is the mask this server's own tools produced for the
2026-10-24 field benchmark (question id in brackets), then cropped. Pixels are
0/255 (PNG, lossless).

| file                               | source image (`PSI_Tray032/tv/`)                   | mask recipe                                                                  | crop `y0:y1, x0:x1` |
| ---------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------- |
| `tray032_2015-12-18_mask_crop.png` | `PSI_Tray032_2015-12-18--14-08-19_top.png` (f1_03) | `segment` channel a, otsu, dark, fill_size 150; `refine` opening 3, fill 150 | `95:515, 400:850`   |
| `tray032_2016-01-17_mask_crop.png` | `PSI_Tray032_2016-01-17--14-07-22_top.png` (f1_10) | `segment` channel a, otsu, dark (fill_size 200)                              | `62:262, 380:780`   |

The 2015-12-18 crop spans the boundary between the first two cells of the 4x5
grid the benchmark agent drew there (cells 420 px wide, 23 px apart). The
2016-01-17 crop spans the gap between the first two circles `auto_grid` drew.
Crops are small because the reference, PlantCV's own `create_labels`, costs
ROIs x contours x frame.

## Whole pot cells (`*_potcell_*.png`)

Two crops of one pot each, from the same dataset (CC BY 4.0, as above), taken
exactly as the 2026-10-24 field benchmark's F2 questions name them: the cell of
the 4x5 grid at row/col (1-based), bounds rounded. Pixels are untouched (PNG,
lossless). They are the two ways a whole pot broke the server in the F2
diagnosis (`benchmarks/agent_qa/field_2026-10-24/f2_counterfactual/` on the
`bench/field-families` branch): a grown rosette fills its pot, so "the plant
covers less of the frame" picks the soil (`tests/test_suggest.py`); a seedling
is about 2% of its pot, so otsu splits soil from the pot rim and `fill_holes`
fills the whole pot inside the rim (`tests/test_refine.py`).

| file                                  | source image (`PSI_Tray032/tv/`)                    | cell                   | crop `y0:y1, x0:x1`  |
| ------------------------------------- | --------------------------------------------------- | ---------------------- | -------------------- |
| `tray032_2016-01-17_potcell_r1c5.png` | `PSI_Tray032_2016-01-17--14-07-22_top.png` (f2_10)  | row 1, col 5, 29 leaves | `70:520, 1967:2417`  |
| `tray032_2015-12-14_potcell_r3c3.png` | `PSI_Tray032_2015-12-14--12-54-51_top.png` (f2_01)  | row 3, col 3, 6 leaves  | `968:1419, 1069:1521` |
