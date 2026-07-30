import tomllib
from importlib.metadata import version
from pathlib import Path

from plantcv_mcp import __version__, plantcv_version

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_version_is_a_string():
    assert isinstance(__version__, str)
    assert __version__


def test_reported_version_matches_the_one_the_project_declares():
    """The version the package REPORTS must equal the version it SHIPS AS.

    0.2.0 was published with `__version__ = "0.1.0"` hardcoded beside a
    pyproject saying 0.2.0. The test above passed the whole time — a non-empty
    string is exactly what a wrong version is — so it could not fail on the one
    defect this file exists to catch.

    Comparing against pyproject rather than against `importlib.metadata` is the
    point. Metadata is where `__version__` now reads FROM, so asserting the two
    agree would be comparing a value to itself; only pyproject is an independent
    source that a reintroduced literal would disagree with.
    """
    assert PYPROJECT.is_file(), f"expected pyproject.toml at {PYPROJECT}"
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert __version__ == declared, (
        f"package reports {__version__!r} but pyproject declares {declared!r}"
    )
    # Installed metadata must agree too, or the checkout is stale relative to
    # what is importable — which would make the assertion above meaningless.
    assert version("plantcv-mcp") == declared


def test_plantcv_version_is_pinned_4_11_3():
    # plantcv.__version__ does NOT exist; we must read package metadata.
    assert plantcv_version() == "4.11.3"
