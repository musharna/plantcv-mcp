"""plantcv-mcp — PlantCV as an MCP measurement instrument."""

from importlib.metadata import version

__version__ = "0.1.0"


def plantcv_version() -> str:
    """Return the installed PlantCV version.

    plantcv.__version__ does not exist — metadata is the only source.
    """
    return version("plantcv")
