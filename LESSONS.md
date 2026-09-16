# Lessons

One line per miss: date, the class of miss, and the mechanism that now catches it.

- 2026-09-16: Input validation covered the shapes I imagined, not the shapes the boundary can receive (empty/truncated files, JSON of the wrong type, floats whose SQUARE leaves the range, ints past int32 reached native code before the geometry check). Caught now by: decoders wrapped at the loader (thermal, hyperspectral), a str check before dict membership (refine), a guard on the derived factor rather than the input (measurement), and the grid predicted in Python ints before any native call (regions); the weekly hypothesis fuzz job (nightly-guardrails.yml) is the mechanism that finds the next such shape. (#96)
