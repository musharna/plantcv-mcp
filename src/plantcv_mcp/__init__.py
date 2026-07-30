"""plantcv-mcp — PlantCV as an MCP measurement instrument."""

from importlib.metadata import version

# Read from installed metadata, not restated. This was hardcoded "0.1.0" while
# pyproject said 0.2.0, so the published 0.2.0 reported the wrong version to
# anyone who asked it. Two hand-maintained copies with nothing enforcing
# agreement is the bug; deriving from the one the packaging already enforces
# removes it rather than resynchronising the copies.
#
# No PackageNotFoundError fallback, deliberately: plantcv_version() below has
# the same failure mode for the same reason, and a sentinel like "0.0.0+unknown"
# would answer a version question with a lie instead of an error.
__version__ = version("plantcv-mcp")


def plantcv_version() -> str:
    """Return the installed PlantCV version.

    plantcv.__version__ does not exist — metadata is the only source.
    """
    return version("plantcv")
