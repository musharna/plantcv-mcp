#!/usr/bin/env bash
# Wait until PyPI serves a just-uploaded version, then check that its
# description carries the MCP registry's ownership marker.
#
#   wait_for_pypi.sh <project> <version> <marker>
#
# For about a minute after an upload PyPI's answers for the new version
# disagree: a 200 can be followed by a 404. So one answer is not "live". This
# waits for NEED answers in a row, and reads the marker from the body of the
# last one instead of asking again: a second request can be the one that 404s,
# and the registry asks PyPI itself right after this step.
set -euo pipefail

[ "$#" -eq 3 ] || { echo "usage: wait_for_pypi.sh <project> <version> <marker>" >&2; exit 2; }
project=$1 version=$2 marker=$3
url="https://pypi.org/pypi/$project/$version/json"
tries=${PYPI_WAIT_TRIES:-30}
pause=${PYPI_WAIT_PAUSE:-10}
NEED=3

body=$(mktemp)
trap 'rm -f "$body"' EXIT

streak=0
for i in $(seq 1 "$tries"); do
  if curl -fsSL -o "$body" "$url"; then
    streak=$((streak + 1))
    if [ "$streak" -ge "$NEED" ]; then break; fi
  else
    streak=0
  fi
  echo "waiting for PyPI $project $version ($i/$tries; $streak of $NEED answers in a row)"
  sleep "$pause"
done
if [ "$streak" -lt "$NEED" ]; then
  echo "::error::PyPI did not serve $project $version $NEED times in a row within $tries tries ($url)"
  exit 1
fi
echo "PyPI $project $version is live ($NEED answers in a row)"

python3 - "$body" "$marker" <<'PY'
import json
import sys

body, marker = sys.argv[1], sys.argv[2]
description = json.load(open(body, encoding="utf-8"))["info"]["description"] or ""
if marker not in description:
    print(f"::error::the PyPI description does not carry {marker!r}; the registry will refuse the package")
    sys.exit(1)
print("marker present: True")
PY
