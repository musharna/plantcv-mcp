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

**How they were chosen:** plant 0 (first in raster order) of each of the two
tray images used as the held-out set in `docs/EVAL.md` — fixed before any
count was looked at, so they are not the plants the tool does best on. The
annotation counts every leaf a person could identify, including small central
leaves and leaves mostly hidden under others; a mask-based method cannot see
the hidden ones.

These files ship in the sdist with the tests and are not part of the wheel.
