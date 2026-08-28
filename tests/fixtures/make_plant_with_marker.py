"""Regenerate plant_with_marker.png — a frame holding BOTH a plant and a size marker.

Deterministic, so the fixture is reproducible from this file rather than being an
opaque binary: a light-grey background (200), a green 200x100 px rectangle standing
in for the plant (exact area, exact width/height — see test_eval_known_geometry),
and a black disc of radius 50 (100 px footprint incl. the +1 pixel convention) as
the marker. At marker_length_mm=20 that is exactly 5 px/mm, so the plant must
measure 40 x 20 mm and 800 mm2.

Run: python tests/fixtures/make_plant_with_marker.py
"""

from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480
PLANT = (120, 140, 200, 100)  # x, y, w, h
MARKER_CENTRE, MARKER_RADIUS = (500, 380), 50
MARKER_CROP = (430, 310, 140, 140)  # x, y, w, h — a margin around the disc
MARKER_LENGTH_MM = 20.0


def render() -> np.ndarray:
    img = np.full((H, W, 3), 200, np.uint8)
    x, y, w, h = PLANT
    img[y : y + h, x : x + w] = (40, 160, 40)  # BGR green
    cv2.circle(img, MARKER_CENTRE, MARKER_RADIUS, (0, 0, 0), -1)
    return img


if __name__ == "__main__":
    out = Path(__file__).with_name("plant_with_marker.png")
    cv2.imwrite(str(out), render())
    print(out)
