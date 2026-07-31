"""`server.json` must agree with the package it claims to describe.

The MCP registry verifies PyPI ownership by looking for an `mcp-name: <name>`
marker in the package README **as published to PyPI**, and rejects a submission
whose declared version does not exist on PyPI. Both of those failures surface
only during a release, in a workflow log, after the version has already been
burned -- so they are asserted here instead.

`server.json` carries the version in three places and nothing but this file
makes them agree. That is the same drift that shipped 0.2.0 of this package
reporting `__version__ == "0.1.0"`.
"""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_JSON = json.loads((ROOT / "server.json").read_text())
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
README = (ROOT / "README.md").read_text()


def test_every_version_in_server_json_matches_pyproject():
    """Three fields, one source of truth."""
    declared = PYPROJECT["project"]["version"]
    found = {
        "server.version": SERVER_JSON["version"],
        "packages[0].version": SERVER_JSON["packages"][0]["version"],
        "_meta.publisher-provided.version": SERVER_JSON["_meta"][
            "io.modelcontextprotocol.registry/publisher-provided"
        ]["version"],
    }
    for where, value in found.items():
        assert value == declared, (
            f"server.json {where} is {value!r} but pyproject declares {declared!r} -- "
            "the registry rejects a version that is not on PyPI"
        )


def test_readme_marker_matches_the_declared_server_name():
    """This exact string is what the registry greps for to prove ownership.

    It must also survive into the PyPI description, which is why it lives in the
    README rather than anywhere else -- `readme = "README.md"` makes the README
    the `long_description`.
    """
    name = SERVER_JSON["name"]
    assert PYPROJECT["project"]["readme"] == "README.md", (
        "the marker only reaches PyPI if the README is the long_description"
    )

    marker = re.search(r"mcp-name:\s*(\S+?)\s*(?:-->|$)", README, re.MULTILINE)
    assert marker is not None, "README carries no mcp-name marker"
    assert marker.group(1) == name, (
        f"README marker names {marker.group(1)!r}, server.json names {name!r}"
    )

    # Negative control: the assertion above can fail. A name the README does not
    # contain must NOT match, or a regex that captured too much would pass here
    # regardless of what the README actually says.
    assert marker.group(1) != "io.github.musharna/not-this-server"


def test_the_package_identifier_is_the_distribution_actually_published():
    pkg = SERVER_JSON["packages"][0]
    assert pkg["identifier"] == PYPROJECT["project"]["name"]
    assert pkg["registryType"] == "pypi"
    assert pkg["transport"]["type"] == "stdio", (
        "this server is stdio; declaring otherwise would send clients down a "
        "transport it does not implement"
    )
