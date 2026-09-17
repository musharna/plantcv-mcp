# Lessons

One line per miss: date, the class of miss, and the mechanism that now catches it.

- 2026-09-16: Input validation covered the shapes I imagined, not the shapes the boundary can receive (empty/truncated files, JSON of the wrong type, floats whose SQUARE leaves the range, ints past int32 reached native code before the geometry check). Caught now by: decoders wrapped at the loader (thermal, hyperspectral), a str check before dict membership (refine), a guard on the derived factor rather than the input (measurement), and the grid predicted in Python ints before any native call (regions); the weekly hypothesis fuzz job (nightly-guardrails.yml) is the mechanism that finds the next such shape. (#96)
- 2026-09-16: A leaked child process is a class the test runner cannot see, because its own exit hides it; the harness that runs the suite in-process (mutmut's stats pass) inherits the child and crashes when it dies. Now: `tests/conftest.py` shuts the worker down at session end and fails the session on any surviving spawn child, and the nightly gate treats a mutmut exit that is neither 0 nor the cap as an abort.
