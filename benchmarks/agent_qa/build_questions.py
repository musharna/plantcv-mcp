"""Render the agent benchmark's images and write their answer key.

Every image is synthetic, so every answer is known exactly: the key is the
plant mask this script drew, counted with numpy, never a measurement. The
pictures are made to look like plant photos, so the answer cannot be had by
counting the one colour in a flat image. They have:

- mottled, grainy soil with pebbles, moss specks and the plant's shadow;
- pot rims, a black seedling tray, and a wall and terracotta pot in side views;
- leaves of different shades, each with a lighter midrib and a darker edge,
  that overlap one another (rosette leaves overlap well beyond the centre);
- a printed size-marker card for the mm² questions.

None of that changes the key. Plant pixels are painted from the mask with
crisp edges, and every plant pixel stays green (or yellow) under the noise
(asserted), so "which pixels are the plant" has one answer: the mask.

This script never imports plantcv_mcp or PlantCV, so a bug in the server
cannot also be in the key. OpenCV is used for texture (blurring noise) and
for one diagnostic, ``contour_area`` (what ``cv2.contourArea`` gives for the
drawn mask), which lets the scorer say "this answer is the contour area"
after checking it instead of guessing why an area was low.

Geometry, colours and textures come from seeded generators, not choices.
The images are build outputs, not committed (noisy PNGs are ~0.5 MB each):
questions.json records each image's sha256, and run.py refuses to ask a
question whose image does not match it.

    uv run python benchmarks/agent_qa/build_questions.py   # writes questions.json + images/
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SEED = 20261016
HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"

SOIL = (96, 70, 52)  # RGB
PEBBLE = (128, 122, 112)
MOSS = (82, 118, 48)
POT_RIM = (52, 52, 56)
TRAY = (34, 34, 37)
WALL = (206, 200, 188)
TERRACOTTA = (172, 92, 60)
CARD = (236, 235, 228)
MARKER = (20, 20, 22)
ALGAE = (92, 80, 54)  # olive film on damp soil
STRAW = (178, 150, 84)
# Leaves run from bright to dull olive; the dull ones are close to algae-
# tinted soil in hue, value and saturation, apart in LAB a* (asserted).
GREEN_LEAF = {"hue": (78, 140), "sat": (0.4, 0.68), "val": (0.30, 0.66)}
YELLOW_LEAF = {"hue": (50, 58), "sat": (0.6, 0.75), "val": (0.72, 0.85)}

# --- geometry (the masks are the truth) ---------------------------------------


def grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:h, :w]
    return yy, xx


def ellipse_leaf(shape, cx, cy, a, b, theta) -> dict[str, np.ndarray]:
    """A leaf: pixel centres inside the ellipse, plus its midrib (the band
    along the major axis), which is used for shading only."""
    yy, xx = grid(*shape)
    dx, dy = xx - cx, yy - cy
    c, s = math.cos(theta), math.sin(theta)
    u, v = dx * c + dy * s, -dx * s + dy * c
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    return {"mask": mask, "midrib": mask & (np.abs(v) <= 1.0) & (np.abs(u) < 0.85 * a)}


def disc(shape, cx, cy, r) -> np.ndarray:
    yy, xx = grid(*shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def centre(shape, cx, cy, r) -> dict[str, np.ndarray]:
    return {"mask": disc(shape, cx, cy, r), "midrib": np.zeros(shape, bool)}


def square(shape, cx, cy, side, angle) -> np.ndarray:
    """Pixel centres inside a square of `side` px turned by `angle`."""
    yy, xx = grid(*shape)
    c, s = math.cos(angle), math.sin(angle)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
    return (np.abs(u) <= side / 2) & (np.abs(v) <= side / 2)


def union(parts: list[dict[str, np.ndarray]]) -> np.ndarray:
    m = np.zeros(parts[0]["mask"].shape, bool)
    for p in parts:
        m |= p["mask"]
    return m


def blob_parts(rng: random.Random, shape, cx, cy, size) -> list[dict[str, np.ndarray]]:
    """A top-view plant: a centre and 4-7 overlapping leaves around it."""
    parts = [centre(shape, cx, cy, size * 0.3)]
    n = rng.randint(4, 7)
    start = rng.uniform(0, 2 * math.pi)
    for k in range(n):
        t = start + 2 * math.pi * k / n + rng.uniform(-0.35, 0.35)
        a = size * rng.uniform(0.45, 0.75)
        b = a * rng.uniform(0.38, 0.55)
        d = a * 0.8
        parts.append(
            ellipse_leaf(shape, cx + math.cos(t) * d, cy + math.sin(t) * d, a, b, t)
        )
    return parts


def rosette_parts(rng: random.Random, shape, n: int) -> list[dict[str, np.ndarray]]:
    """n leaves radiating from a small centre, wide enough to overlap their
    neighbours well beyond the centre, as on a grown rosette. Older (larger)
    leaves lie under younger (smaller) ones and the crown sits on top, as on
    a plant. Redrawn until every leaf shows at least a quarter of itself, so
    n is countable. The crown is the LAST part here."""
    h, w = shape
    for _ in range(500):
        cx, cy = w / 2 + rng.uniform(-10, 10), h / 2 + rng.uniform(-10, 10)
        start = rng.uniform(0, 2 * math.pi)
        sized = []
        for k in range(n):
            t = start + 2 * math.pi * k / n + rng.uniform(-0.3, 0.3)
            a = rng.uniform(42, 80)
            b = min(a * rng.uniform(0.34, 0.5), 1.1 * math.pi * a / n + 8)
            d = a * rng.uniform(0.7, 0.85)
            sized.append(
                (
                    a,
                    ellipse_leaf(
                        shape, cx + math.cos(t) * d, cy + math.sin(t) * d, a, b, t
                    ),
                )
            )
        leaves = [leaf for _, leaf in sorted(sized, key=lambda x: -x[0])]
        parts = [*leaves, centre(shape, cx, cy, 10)]
        on_top = np.full(shape, -1)
        for j, p in enumerate(parts):
            on_top[p["mask"]] = j
        if all(
            (on_top == j).sum() >= 0.25 * p["mask"].sum() for j, p in enumerate(leaves)
        ):
            return parts
    raise RuntimeError(
        f"no {n}-leaf rosette with every leaf a quarter visible in 500 draws"
    )


def side_view_parts(rng: random.Random, shape, sx: int, base: int):
    """A stem rising from the soil line at y=base, with 2-5 leaves; low
    leaves may droop in front of the pot (only the stem stops at the rim)."""
    top = rng.randint(40, base - 160)
    sw = rng.randint(6, 10)
    stem = np.zeros(shape, bool)
    stem[top:base, sx - sw // 2 : sx - sw // 2 + sw] = True
    parts = [{"mask": stem, "midrib": np.zeros(shape, bool)}]
    for _ in range(rng.randint(2, 5)):
        y = rng.uniform(top + 15, base - 50)
        side = rng.choice((-1, 1))
        a = rng.uniform(35, 70)
        b = a * rng.uniform(0.25, 0.4)
        t = math.radians(rng.uniform(-50, 40))
        theta = t if side > 0 else math.pi - t
        parts.append(
            ellipse_leaf(
                shape, sx + side * math.cos(t) * a, y + math.sin(t) * a, a, b, theta
            )
        )
    return parts


def contour_area(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return float(sum(cv2.contourArea(c) for c in contours))


def extent(mask: np.ndarray) -> tuple[int, int]:
    """(width, height) of the mask's bounding box, both ends inclusive."""
    ys, xs = np.nonzero(mask)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def one_component(mask: np.ndarray) -> bool:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return n == 2  # background + one


# --- texture (never touches the truth) ---------------------------------------


class Painter:
    def __init__(self, seed: int, shape: tuple[int, int]) -> None:
        self.n = np.random.default_rng(seed)
        self.r = random.Random(seed)
        self.shape = shape

    def surface(self, rgb, grain: float = 7.0, mottle: float = 12.0) -> np.ndarray:
        img = np.empty((*self.shape, 3), np.float32)
        img[:] = rgb
        low = cv2.GaussianBlur(
            self.n.normal(0, 1, self.shape).astype(np.float32), (0, 0), 14
        )
        img += (low / (low.std() + 1e-9) * mottle)[..., None]
        # Grain is mostly luminance, as in a camera at base ISO, plus a little
        # colour noise.
        img += self.n.normal(0, grain, self.shape).astype(np.float32)[..., None]
        img += self.n.normal(0, 2, (*self.shape, 3)).astype(np.float32)
        return img

    def specks(
        self, img, rgb, count: int, rmin: float, rmax: float, avoid, gap: int
    ) -> None:
        """Small ellipses (pebbles, moss) kept `gap` px clear of `avoid`."""
        h, w = self.shape
        k = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
        keep_out = cv2.dilate(avoid.astype(np.uint8), k).astype(bool)
        for _ in range(count * 20):
            if count == 0:
                return
            a = self.r.uniform(rmin, rmax)
            m = ellipse_leaf(
                self.shape,
                self.r.uniform(0, w),
                self.r.uniform(0, h),
                a,
                a * self.r.uniform(0.5, 1.0),
                self.r.uniform(0, math.pi),
            )["mask"]
            if not m.any() or (m & keep_out).any():
                continue
            tint = self.n.normal(0, 8) + self.n.normal(0, 2, 3)  # mostly lightness
            img[m] = np.array(rgb, np.float32) + tint.astype(np.float32)
            count -= 1

    def shadow(self, img, mask, dx: int = 9, dy: int = 7, depth: float = 0.72) -> None:
        sh = np.roll(np.roll(mask.astype(np.float32), dy, 0), dx, 1)
        sh = cv2.GaussianBlur(sh, (0, 0), 5)
        img *= (1 - (1 - depth) * np.clip(sh * 1.4, 0, 1))[..., None]

    def algae(self, img, cover: float = 0.35) -> None:
        """Olive film over part of the soil, in soft-edged patches."""
        f = cv2.GaussianBlur(
            self.n.normal(0, 1, self.shape).astype(np.float32), (0, 0), 20
        )
        f = (f - np.quantile(f, 1 - cover)) / (f.std() + 1e-9)
        wgt = np.clip(f * 1.5, 0, 0.8)[..., None]
        img[:] = img * (1 - wgt) + np.array(ALGAE, np.float32) * wgt

    def ring(self, img, cx, cy, r, width, rgb) -> None:
        yy, xx = grid(*self.shape)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img[d <= r - width] *= 0.88  # damp potting soil is darker than the bed
        rim = (d <= r) & (d > r - width)
        n_px = int(rim.sum())
        img[rim] = (
            np.array(rgb, np.float32)
            + self.n.normal(0, 6, (n_px, 1)).astype(np.float32)
            + self.n.normal(0, 2, (n_px, 3)).astype(np.float32)
        )

    def leaf_rgb(self, hue, sat, val) -> np.ndarray:
        rgb = colorsys.hsv_to_rgb(
            self.r.uniform(*hue) / 360, self.r.uniform(*sat), self.r.uniform(*val)
        )
        return np.array(rgb, np.float32) * 255

    def paint_plant(self, img, parts, colours=None, edge: float = 0.8, **leaf) -> None:
        """Paint leaves in order (a later one covers an earlier one), each its
        own shade, with a lighter midrib, a darker one-pixel edge and grain."""
        style = {**GREEN_LEAF, **leaf}
        for i, p in enumerate(parts):
            m = p["mask"]
            rgb = colours[i] if colours is not None else self.leaf_rgb(**style)
            img[m] = rgb
            inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
                bool
            )
            img[m & ~inner] = rgb * edge
            img[p["midrib"]] = np.minimum(rgb * 1.18 + 8, 255)
        plant = union(parts)
        n_px = int(plant.sum())
        img[plant] += self.n.normal(0, 4, (n_px, 1)).astype(np.float32)
        img[plant] += self.n.normal(0, 1.5, (n_px, 3)).astype(np.float32)

    @staticmethod
    def finish(img) -> np.ndarray:
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def hsv_hue_deg(rgb: np.ndarray) -> np.ndarray:
    """HSV hue in degrees for an (N, 3) uint8 array, as colorsys computes it."""
    x = rgb.astype(np.float64) / 255
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx, mn = x.max(1), x.min(1)
    span = np.where(mx - mn == 0, 1, mx - mn)
    h = np.where(
        mx == r,
        ((g - b) / span) % 6,
        np.where(mx == g, (b - r) / span + 2, (r - g) / span + 4),
    )
    return np.where(mx - mn == 0, 0, h) * 60.0


def assert_plant_colour(img: np.ndarray, mask: np.ndarray) -> None:
    """Every plant pixel, after shading and noise, is still green or yellow
    (hue 40-160): shading never pushes plant tissue into the background's
    colours. (Moss specks are green too; they are small and kept clear of
    the plant, and the questions ask about the plant.)"""
    h = hsv_hue_deg(img[mask])
    assert ((h > 40) & (h < 160)).all(), "a plant pixel left the plant hues"


def assert_separable(img: np.ndarray, mask: np.ndarray, band: int = 4) -> None:
    """A colour rule separates the plant from everything within `band` px of
    it (soil, algae, shadow, pot), so the drawn mask is the exact
    segmentation, however dull a leaf or olive the soil. The rule is "LAB a*
    below T1 (green) or b* above T2 (yellow)", with T1 the lowest a* and T2
    the highest b* in the band; green-only plants need only the first half."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(int)
    a, b = lab[..., 1], lab[..., 2]
    ring = cv2.dilate(
        mask.astype(np.uint8), np.ones((2 * band + 1,) * 2, np.uint8)
    ).astype(bool)
    ring &= ~mask
    t1, t2 = a[ring].min(), b[ring].max()
    left = mask & (a >= t1)
    assert not left.any() or b[left].min() > t2, (t1, t2, a[left].max(), b[left].min())


def save(img: np.ndarray, name: str) -> tuple[str, str]:
    IMAGES.mkdir(exist_ok=True)
    path = IMAGES / name
    Image.fromarray(img).save(path, optimize=True)
    return name, hashlib.sha256(path.read_bytes()).hexdigest()


def moss(p: Painter, img, avoid, count: int) -> None:
    """Moss clumps of 30-150 px, each its own olive shade, clear of the plant."""
    for _ in range(count):
        rgb = p.leaf_rgb(hue=(70, 95), sat=(0.45, 0.65), val=(0.35, 0.5))
        p.specks(img, rgb, 1, 3.0, 7.0, avoid, gap=10)


def potted_top_view(
    seed: int,
    shape,
    parts,
    cx,
    cy,
    pot_r,
    paint_extra=None,
    colours=None,
    straw=False,
    **leaf,
):
    """Soil bed with algae, a pot rim, pebbles, moss (and straw) clear of the
    plant, the plant's shadow, then the plant. `paint_extra` paints anything else
    (a marker card) before the plant and returns its mask, which specks also
    avoid."""
    p = Painter(seed, shape)
    plant = union(parts)
    img = p.surface(SOIL)
    p.algae(img)
    p.ring(img, cx, cy, pot_r, p.r.randint(8, 12), POT_RIM)
    avoid = plant | (paint_extra(p, img) if paint_extra else np.zeros(shape, bool))
    p.specks(img, PEBBLE, p.r.randint(12, 30), 2, 5, avoid, gap=4)
    moss(p, img, avoid, p.r.randint(6, 14))
    if straw:
        for _ in range(p.r.randint(6, 12)):
            p.specks(img, STRAW, 1, 6.0, 14.0, avoid, gap=10)
    p.shadow(img, plant)
    p.paint_plant(img, parts, colours=colours, **leaf)
    out = p.finish(img)
    assert_plant_colour(out, plant)
    assert_separable(out, plant)
    return out


# --- the questions -------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)
    seeds = iter(range(SEED, SEED + 1000))
    questions: list[dict[str, Any]] = []

    def add(category, saved, question, fmt, truth, note) -> None:
        name, sha = saved
        questions.append(
            {
                "category": category,
                "image": name,
                "sha256": sha,
                "question": question,
                "answer_format": fmt,
                "truth": truth,
                "how_built": note,
            }
        )

    # 1. area (8): a potted plant seen from above.
    for k in range(8):
        shape = (rng.randint(400, 480), rng.randint(500, 640))
        h, w = shape
        cx, cy = w / 2 + rng.uniform(-30, 30), h / 2 + rng.uniform(-20, 20)
        size = rng.uniform(70, 115)
        parts = blob_parts(rng, shape, cx, cy, size)
        m = union(parts)
        assert one_component(m)
        seed = next(seeds)
        colours, note = None, ""
        if k >= 4:
            # Half the set: 1-2 old leaves have yellowed and straw lies on the
            # soil, so "pixels greener than they are red" misses plant tissue
            # and "green or yellow" picks up straw. Own RNG: later draws unchanged.
            pick, local = Painter(seed + 500, shape), random.Random(seed)
            n_leaves = len(parts) - 1
            yellow_idx = set(local.sample(range(n_leaves), local.randint(1, 2)))
            colours = [pick.leaf_rgb(**GREEN_LEAF)] + [
                pick.leaf_rgb(**(YELLOW_LEAF if j in yellow_idx else GREEN_LEAF))
                for j in range(n_leaves)
            ]
            note = f", {len(yellow_idx)} of them yellowed, straw on the soil"
        img = potted_top_view(
            seed,
            shape,
            parts,
            cx,
            cy,
            size * 1.25 + 20,
            colours=colours,
            straw=colours is not None,
        )
        add(
            "area",
            save(img, f"area_{k + 1}.png"),
            "This is a top-down photo of a potted plant. What is the plant's projected "
            "area in pixels, i.e. how many pixels does the plant cover?",
            "an integer number of pixels, e.g. 12345",
            {"area_px": int(m.sum()), "contour_area": contour_area(m)},
            f"{len(parts) - 1} overlapping leaves{note}; truth = plant pixels drawn",
        )

    # 2. extent (8): side view, a plant rising out of a terracotta pot.
    for k in range(8):
        shape = (rng.randint(460, 540), rng.randint(440, 600))
        h, w = shape
        sx = rng.randint(w // 3, 2 * w // 3)
        pot_h, pot_w = rng.randint(90, 120), rng.randint(130, 170)
        base = h - pot_h - 10
        parts = side_view_parts(rng, shape, sx, base)
        yy, xx = grid(*shape)
        parts[0]["mask"] &= yy < base  # the stem enters the soil at the rim
        m = union(parts)
        assert one_component(m)
        pt = Painter(next(seeds), shape)
        img = pt.surface(WALL, grain=5, mottle=6)
        img *= np.linspace(1.04, 0.9, h, dtype=np.float32)[
            :, None, None
        ]  # light from above
        half = pot_w / 2 * (1 - 0.18 * (yy - base) / pot_h)
        pot = (yy >= base) & (yy < base + pot_h) & (np.abs(xx - sx) <= half)
        img[pot] = np.array(TERRACOTTA, np.float32) + pt.n.normal(
            0, 6, (int(pot.sum()), 3)
        )
        rim = (yy >= base) & (yy < base + 8) & (np.abs(xx - sx) <= pot_w / 2 + 6)
        img[rim] = np.array(TERRACOTTA, np.float32) * 0.85
        pt.paint_plant(img, parts)
        img = pt.finish(img)
        assert_plant_colour(img, m)
        assert_separable(img, m)
        w_, h_ = extent(m)
        axis = "height" if k % 2 == 0 else "width"
        what = (
            "height in pixels, from its lowest point to its highest point"
            if axis == "height"
            else "width in pixels, from its leftmost to its rightmost point"
        )
        add(
            "extent",
            save(img, f"extent_{k + 1}.png"),
            f"This is a side-view photo of a potted plant against a wall. What is the "
            f"plant's {what}?",
            "an integer number of pixels, e.g. 321",
            {
                "axis": axis,
                "px": h_ if axis == "height" else w_,
                "other_px": w_ if axis == "height" else h_,
            },
            "stem from the pot rim, leaves may hang in front of the pot; "
            "truth = bounding box of the plant pixels drawn",
        )

    # 3. area_mm (6): a printed marker card in the top-left corner.
    for k in range(6):
        side_px = rng.choice((60, 72, 80, 90, 100, 120))
        side_mm = rng.choice((10, 15, 20, 25))
        shape = (rng.randint(440, 520), rng.randint(600, 700))
        h, w = shape
        cx, cy = w * 0.62, h * 0.55
        size = rng.uniform(70, 100)
        parts = blob_parts(rng, shape, cx, cy, size)
        m = union(parts)
        assert one_component(m)
        # The card lies at an angle, so the square's side is not its bounding box.
        margin, angle = 14, math.radians(rng.uniform(10, 35))
        reach = (side_px / 2 + margin) * (math.cos(angle) + math.sin(angle))
        card = square(shape, 18 + reach, 18 + reach, side_px + 2 * margin, angle)
        mark = square(shape, 18 + reach, 18 + reach, side_px, angle)
        assert not (card & m).any()

        def paint_card(p, img, card=card, mark=mark):
            p.shadow(img, card, dx=5, dy=5, depth=0.8)
            img[card] = np.array(CARD, np.float32) + p.n.normal(
                0, 3, (int(card.sum()), 3)
            )
            img[mark] = np.array(MARKER, np.float32) + p.n.normal(
                0, 3, (int(mark.sum()), 3)
            )
            return card

        img = potted_top_view(
            next(seeds), shape, parts, cx, cy, size * 1.25 + 20, paint_extra=paint_card
        )
        ppm = side_px / side_mm
        add(
            "area_mm",
            save(img, f"area_mm_{k + 1}.png"),
            f"This is a top-down photo of a potted plant. The black square printed on the "
            f"white card in the top-left corner is a size reference, exactly {side_mm} mm x "
            f"{side_mm} mm. What is the plant's projected area in square millimetres?",
            "a number in mm², e.g. 1234.5",
            {
                "area_mm2": int(m.sum()) / ppm**2,
                "area_px": int(m.sum()),
                "px_per_mm": ppm,
            },
            f"marker {side_px} px = {side_mm} mm, turned {math.degrees(angle):.0f} deg; "
            "truth = plant pixels drawn / px_per_mm²",
        )

    # 4. tray (8): a black tray of square pots, some empty, moss in some.
    for k in range(8):
        rows, cols = rng.choice(((2, 3), (3, 4), (4, 5), (3, 5), (4, 6)))
        cell = rng.randint(96, 124)
        shape = (rows * cell + 40, cols * cell + 40)
        n_empty = rng.randint(1, max(1, rows * cols // 4))
        empty = set(rng.sample(range(rows * cols), n_empty))
        areas: dict[str, int] = {}
        contours: dict[str, float] = {}
        all_parts: list[dict[str, np.ndarray]] = []
        full = np.zeros(shape, bool)
        for i in range(rows * cols):
            if i in empty:
                continue
            r, c = divmod(i, cols)
            cx = 20 + c * cell + cell / 2 + rng.uniform(-8, 8)
            cy = 20 + r * cell + cell / 2 + rng.uniform(-8, 8)
            parts = blob_parts(
                rng, shape, cx, cy, rng.uniform(cell * 0.14, cell * 0.32)
            )
            m = union(parts)
            assert one_component(m) and not (m & full).any()
            full |= m
            all_parts += parts
            areas[f"{r + 1},{c + 1}"] = int(m.sum())
            contours[f"{r + 1},{c + 1}"] = contour_area(m)
        pt = Painter(next(seeds), shape)
        img = pt.surface(TRAY, grain=4, mottle=3)
        soil = pt.surface(SOIL)
        pt.algae(soil)
        soil *= 0.92
        yy, xx = grid(*shape)
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            x0, y0 = 20 + c * cell + 6, 20 + r * cell + 6
            pot = (
                (xx >= x0) & (xx < x0 + cell - 12) & (yy >= y0) & (yy < y0 + cell - 12)
            )
            img[pot] = soil[pot]
        moss(pt, img, full, pt.r.randint(8, 18))
        pt.shadow(img, full)
        pt.paint_plant(img, all_parts)
        img = pt.finish(img)
        assert_plant_colour(img, full)
        assert_separable(img, full)
        saved = save(img, f"tray_{k + 1}.png")
        intro = (
            f"This is a top-down photo of a seedling tray: {rows} rows x {cols} columns of "
            f"pots, at most one seedling per pot."
        )
        if k % 2 == 0:
            add(
                "tray",
                saved,
                f"{intro} How many of the {rows * cols} pots hold a seedling?",
                "an integer, e.g. 7",
                {"kind": "count", "count": len(areas)},
                f"{rows}x{cols}, {n_empty} empty; truth = seedlings drawn (moss specks are not)",
            )
        else:
            target = rng.choice(sorted(areas))
            r, c = target.split(",")
            add(
                "tray",
                saved,
                f"{intro} What is the projected area, in pixels, of the seedling in row {r}, "
                f"column {c} (row 1 is the top row, column 1 the leftmost)?",
                "an integer number of pixels, e.g. 12345",
                {
                    "kind": "area",
                    "area_px": areas[target],
                    "contour_area": contours[target],
                },
                f"{rows}x{cols}, {n_empty} empty; truth = pixels drawn for that seedling",
            )

    # 5. leaf_count (8): rosettes of 5-14 overlapping leaves of nearly one
    # shade with faint edges, as leaves of one plant are.
    for k, n in enumerate(rng.sample(range(5, 15), 8)):
        shape = (380, 380)
        parts = rosette_parts(rng, shape, n)
        m = union(parts)
        assert one_component(m)
        hc, sc, vc = (
            rng.uniform(85, 125),
            rng.uniform(0.45, 0.6),
            rng.uniform(0.42, 0.58),
        )
        img = potted_top_view(
            next(seeds), shape, parts, 190, 190, 182, edge=0.93,
            hue=(hc - 4, hc + 4), sat=(sc - 0.03, sc + 0.03), val=(vc - 0.04, vc + 0.04),
        )  # fmt: skip
        add(
            "leaf_count",
            save(img, f"leaves_{k + 1}.png"),
            "This is a top-down photo of a potted rosette plant. How many leaves does it have?",
            "an integer, e.g. 6",
            {"count": n},
            f"{n} overlapping leaves of one shade around a centre; truth = leaves drawn",
        )

    # 6. color (8): mean hue of a plant; share of yellow (chlorotic) leaves.
    for k in range(8):
        shape = (rng.randint(400, 470), rng.randint(500, 600))
        h, w = shape
        cx, cy, size = w / 2, h / 2, rng.uniform(85, 110)
        parts = blob_parts(rng, shape, cx, cy, size)
        m = union(parts)
        assert one_component(m)
        pot_r = size * 1.25 + 20
        if k % 2 == 0:
            mid = rng.uniform(80, 140)
            img = potted_top_view(
                next(seeds), shape, parts, cx, cy, pot_r, hue=(mid - 8, mid + 8)
            )
            hues = hsv_hue_deg(img[m])
            rad = np.radians(hues)
            circ = (
                math.degrees(math.atan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360
            )
            assert abs(circ - hues.mean()) < 0.5  # no wrap: the two means agree
            add(
                "color",
                save(img, f"color_{k + 1}.png"),
                "This is a top-down photo of a potted plant. What is the mean hue of the "
                "plant's pixels, in degrees on the 0-360 colour wheel (HSV hue)?",
                "a number of degrees, e.g. 104.5",
                {
                    "kind": "hue",
                    "hue_deg": circ,
                    "arithmetic_mean_deg": float(hues.mean()),
                },
                "leaves of varied green shades; truth = circular mean HSV hue of plant pixels",
            )
        else:
            seed = next(seeds)
            pick = Painter(seed + 500, shape)
            n_leaves = len(parts) - 1
            yellow_idx = set(rng.sample(range(n_leaves), rng.randint(1, n_leaves // 2)))
            colours = [pick.leaf_rgb(**GREEN_LEAF)] + [
                pick.leaf_rgb(**(YELLOW_LEAF if j in yellow_idx else GREEN_LEAF))
                for j in range(n_leaves)
            ]
            img = potted_top_view(
                seed, shape, parts, cx, cy, pot_r, colours=colours, straw=True
            )
            # What shows is decided by paint order: a later leaf covers an earlier one.
            visible = np.full(shape, -1)
            for j, p_ in enumerate(parts):
                visible[p_["mask"]] = j
            yellow = np.isin(visible, [j + 1 for j in yellow_idx])
            pct = 100.0 * int(yellow.sum()) / int(m.sum())
            # score.py reads a bare value <= 1 as a fraction; that is only
            # unambiguous while every truth is well above 1%.
            assert 10 < pct < 60, pct
            add(
                "color",
                save(img, f"color_{k + 1}.png"),
                "This is a top-down photo of a potted plant. Some of its leaves are yellow "
                "(chlorotic), the rest green. What percentage of the plant's area is yellow?",
                "a percentage, e.g. 23.4",
                {
                    "kind": "yellow_pct",
                    "pct": pct,
                    "yellow_px": int(yellow.sum()),
                    "plant_px": int(m.sum()),
                },
                "whole leaves yellow; truth = visible yellow-leaf pixels / plant pixels",
            )

    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i:02d}"
    out = {
        "built": datetime.now(UTC).date().isoformat(),
        "seed": SEED,
        "questions": questions,
    }
    (HERE / "questions.json").write_text(json.dumps(out, indent=1) + "\n")
    n: dict[str, int] = {}
    for q in questions:
        n[q["category"]] = n.get(q["category"], 0) + 1
    print(len(questions), n)


if __name__ == "__main__":
    main()
