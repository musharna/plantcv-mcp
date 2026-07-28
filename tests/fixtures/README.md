# Test fixtures

`multi_specimen.png` — copied from `~/bio3d-arena/data/assets/renders/736_multi4.png` (our own render, no third-party rights).

## Why this fixture matters

It is a **FOUR-VIEW panel**: one image containing four separate views of a plant. Segmenting it with a whole-image ROI merges all four plants into one connected object. This is a known-bad case that demonstrates why the `multi_specimen` guard is essential.

When this fixture runs through `_segment_impl` with channel="a" and method="otsu", it produces multiple connected components (one per plant view) rather than a single object. The `multi_specimen` warning must fire to alert the user that size traits computed on the merged object will be meaningless.

## Observed measurements

Measured with the current pipeline:

- **Connected components**: 9
- **Major objects** (after filtering): ≥2
- **Component areas**: varies by processing, but the top components correspond to individual plant views

## Copyright

Do **NOT** add images from `~/orchid-data/` — those carry third-party copyright watermarks and must not enter this repository, its tests, its examples, or its README.
