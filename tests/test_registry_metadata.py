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

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_JSON = json.loads((ROOT / "server.json").read_text())
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
README = (ROOT / "README.md").read_text()

# The registry's own statement of what it accepts, vendored.
#
# `server.json` declares a dated `$schema`, and that document encodes constraints
# nothing local was measuring: a 100-character cap on `description`, further caps
# on `name`, `title` and `version`, and the required-field list. The registry
# enforces them server-side and answers HTTP 422.
#
# breedsim-mcp v0.4.0 was tagged, uploaded to PyPI and GitHub-released before the
# registry refused it -- an audit fix had grown its `description` to 315
# characters. The publish workflow runs on tag push, so the one check that would
# have caught it ran after the version was already burned. This package's own
# description is 91 characters: it passes today with nine to spare.
#
# Asserting a hand-copied constant would cover only the constraint that happened
# to bite. Validating against the schema covers the class. The copy is vendored
# rather than fetched so the suite stays offline and deterministic.
SCHEMA = json.loads((ROOT / "tests" / "server.schema.json").read_text())
DESCRIPTION_MAX = SCHEMA["definitions"]["ServerDetail"]["properties"]["description"][
    "maxLength"
]


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


def test_the_vendored_schema_is_the_one_server_json_declares():
    """A pin that can drift silently is not a pin.

    If `$schema` is bumped to a newer dated revision, the vendored copy stops
    describing what the registry will actually apply, and every check below it
    goes quietly out of date rather than failing.
    """
    assert SCHEMA["$id"] == SERVER_JSON["$schema"], (
        f"server.json declares {SERVER_JSON['$schema']} but the vendored copy is "
        f"{SCHEMA['$id']} -- re-vendor tests/server.schema.json from the declared URL"
    )


def test_server_json_satisfies_the_registry_schema():
    """The whole document, against the whole schema."""
    jsonschema.validate(SERVER_JSON, SCHEMA)


def test_the_description_fits_what_the_registry_accepts():
    """Called out by name because this is the one that has actually shipped broken.

    The cap is read from the vendored schema rather than restated, so there is
    one source of truth and it is the registry's, not ours.
    """
    description = SERVER_JSON["description"]
    assert len(description) <= DESCRIPTION_MAX, (
        f"server.json description is {len(description)} characters; the registry "
        f"caps it at {DESCRIPTION_MAX} and rejects the submission with a 422. "
        "Long-form install or usage detail belongs in the README."
    )
    assert description, "server.json has no description at all"


def test_the_schema_check_rejects_what_the_registry_rejected():
    """Never trust a test you have not seen fail.

    The 315-character description that 422'd breedsim-mcp v0.4.0, run back
    through this validator. If it passes, the checks above are inert and would
    wave the identical mistake through again.
    """
    burned = SERVER_JSON | {"description": "x" * 315}
    with pytest.raises(jsonschema.ValidationError) as caught:
        jsonschema.validate(burned, SCHEMA)
    assert "too long" in str(caught.value)

    # And a failure mode with nothing to do with length, so this file is not
    # merely a length check wearing a schema costume.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {k: v for k, v in SERVER_JSON.items() if k != "version"}, SCHEMA
        )

    # Positive control, deliberately inside the same test: the untouched
    # document must still validate. Without it, an unloadable schema or a
    # validator that raised on everything would read as "the guard works".
    jsonschema.validate(SERVER_JSON, SCHEMA)
