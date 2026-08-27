# Vendored test data from PlantCV

The files in this directory are copied unmodified from the PlantCV repository
(<https://github.com/danforthcenter/plantcv>, `tests/testdata/`, retrieved
2026-08-27) and are distributed under PlantCV's licence, the **Mozilla Public
License 2.0** (<https://github.com/danforthcenter/plantcv/blob/main/LICENSE>).
They are used only as evaluation fixtures for the hyperspectral and thermal tools.

| file | what it is |
|---|---|
| `corn-kernel-hyperspectral` + `.hdr` | ENVI cube, 31×43 px, 580 bands (366–1048 nm), uint16 counts; the raw file is named without an extension because PlantCV derives `<raw>.hdr` |
| `FLIR_test.jpg` | FLIR radiometric JPEG, 480×640, decoded to °C by `flyr` |
| `thermal_img.npz` | 480×640 float64 °C array |
| `thermal_img_mask.png` | the plant mask PlantCV's own tests use with it |

plantcv-mcp itself is MIT-licensed; these fixtures are not part of the package
wheel (the sdist carries `tests/**`, and this notice travels with it).
