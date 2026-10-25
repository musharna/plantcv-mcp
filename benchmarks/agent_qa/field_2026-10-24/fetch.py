"""Fetch the Aberystwyth Leaf Evaluation Dataset members this set needs.

Bell J, Dee HM (2016) Aberystwyth Leaf Evaluation Dataset. Zenodo.
https://doi.org/10.5281/zenodo.168158 (CC BY 4.0). The images archive is
61.6 GB, so single members are read out of it with HTTP range requests
(Zenodo answers 206); the whole archive is never downloaded. zipfile checks
every member's CRC-32 as it is read, and each file's sha256 is written to
MANIFEST.json beside it.

What is fetched, and nothing else:

- every tray 032 hand annotation (``gt/``, 13 images, 210 plants) and the raw
  image each was painted on;
- the camera-correction material: the 23 checkerboard photographs and the
  shipped ``psi_camera_data.xml``.

Tray 031 is NOT fetched: the server's leaf counting was designed on it
(docs/EVAL.md). Trays 033/034 have no hand annotations.

    python benchmarks/agent_qa/field_2026-10-24/fetch.py [DEST]

DEST defaults to ~/plantcv-mcp-artifacts/field_2026-10-24/aberystwyth, outside
the repo: the images are 8.7 MB each and the set's leak markers name it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/168158/files/{}/content"
# Sizes and md5 from the Zenodo API record, read 2026-10-24. A size that
# differs means the record changed: stop rather than read a different archive.
ARCHIVES = {
    "images_and_annotations.zip": (61592445493, "d0ab2ab9c631953be8e27f12af35ffe7"),
    "software.zip": (226721675, "0dea54d6c682c93f2f21e8f2137ca1f2"),
}
TRAY = "032"
DEFAULT_DEST = (
    Path.home() / "plantcv-mcp-artifacts" / "field_2026-10-24" / "aberystwyth"
)


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file through Range requests."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + off
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[no-untyped-def]
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        # A dropped connection is retried (3 tries, then the error is raised);
        # a wrong status or length is not: it means a different file.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:  # nosec B310: fixed https Zenodo URL, traced: RECORD constant
                    if r.status != 206:
                        raise RuntimeError(
                            f"{self.url}: HTTP {r.status}, want 206 (range read)"
                        )
                    data = r.read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        if len(data) != n:
            raise RuntimeError(
                f"{self.url}: asked {n} bytes at {self.pos}, got {len(data)}"
            )
        b[:n] = data
        self.pos += n
        return n


def open_archive(name: str) -> zipfile.ZipFile:
    size, _ = ARCHIVES[name]
    head = urllib.request.Request(RECORD.format(name), method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as r:  # nosec B310: fixed https Zenodo URL, traced: RECORD constant
        got = int(r.headers["Content-Length"])
    if got != size:
        raise RuntimeError(f"{name}: Zenodo serves {got} bytes, the pin is {size}")
    raw = io.BufferedReader(RangeFile(RECORD.format(name), size), buffer_size=1 << 20)
    return zipfile.ZipFile(raw)


def wanted(
    images: zipfile.ZipFile, software: zipfile.ZipFile
) -> list[tuple[zipfile.ZipFile, str]]:
    gt_prefix = f"images_and_annotations/PSI_Tray{TRAY}/tv/gt/"
    gts = sorted(
        n for n in images.namelist() if n.startswith(gt_prefix) and n.endswith(".png")
    )
    if len(gts) != 13:
        raise RuntimeError(
            f"tray {TRAY}: {len(gts)} annotations, the dataset PDF says 13"
        )
    out: list[tuple[zipfile.ZipFile, str]] = []
    names = set(images.namelist())
    for g in gts:
        stem = (
            Path(g)
            .name.replace("_top_gt.png", "_top.png")
            .replace("_top-gt.png", "_top.png")
        )
        raw = f"images_and_annotations/PSI_Tray{TRAY}/tv/{stem}"
        if raw not in names:
            raise RuntimeError(f"{g}: no raw image {raw}")
        out += [(images, g), (images, raw)]
    cc = "software/camera_correction/"
    out += [
        (software, n)
        for n in sorted(software.namelist())
        if n.startswith(cc + "images/")
        and n.endswith(".png")
        or n == cc + "psi_camera_data.xml"
    ]
    return out


def main(argv: list[str]) -> int:
    dest = Path(argv[0]) if argv else DEFAULT_DEST
    dest.mkdir(parents=True, exist_ok=True)
    images = open_archive("images_and_annotations.zip")
    software = open_archive("software.zip")
    manifest = {}
    for z, name in wanted(images, software):
        rel = name.split("/", 1)[1]  # drop the archive's top directory
        path = dest / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)  # raises BadZipFile on a CRC mismatch
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            print(f"fetched {rel} ({len(data)} B)", flush=True)
        manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dest / "MANIFEST.json").write_text(json.dumps(
        {"source": "https://doi.org/10.5281/zenodo.168158", "licence": "CC BY 4.0",
         "citation": "Bell J, Dee HM (2016) Aberystwyth Leaf Evaluation Dataset. Zenodo.",
         "archives": {k: {"size": v[0], "md5": v[1]} for k, v in ARCHIVES.items()},
         "sha256": manifest}, indent=1) + "\n")  # fmt: skip
    print(f"{len(manifest)} files in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
