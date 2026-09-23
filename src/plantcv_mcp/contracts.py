"""Closed output contracts.

mcp builds each structured tool's `outputSchema` from its return TypedDict and
validates the result through pydantic before it goes on the wire. pydantic's
default for a TypedDict is extra="ignore": a key the producer returns but the
TypedDict does not declare is DROPPED from the structured channel, silently,
while the text channel still carries it. That is how measure_images() came to
publish `traits: null` and no `regions` for every tray in a grid batch (audit of
2026-09-22, H1): the schema was written once and the batch grew past it.

`closed` makes a TypedDict forbid undeclared keys. Drift in either direction is
then a failed call — a missing required key or an undeclared extra — never a
quieter result, and the published schema says `additionalProperties: false`, so
it describes exactly what arrives. tests/test_tool_boundary.py checks every
structured tool's published schema is closed.
"""

from pydantic import ConfigDict, with_config

# typing.TypedDict breaks pydantic schema generation below Python 3.12.
from typing_extensions import TypedDict

closed = with_config(ConfigDict(extra="forbid"))


@closed
class WarningItem(TypedDict):
    """One advisory attached to a result."""

    code: str
    message: str
