"""Skeleton-based morphology traits.

Everything here is measured against a synthetic plant whose geometry is KNOWN:
a vertical stem with leaves drawn at chosen angles and lengths. PlantCV 4.11.3
has systematic behaviours that the tolerances below absorb deliberately (all
measured 2026-08-27, see docs/superpowers/specs/2026-08-27-morphology-design.md):

* leaf path lengths come out ~10 px short (junction trimming), ordering kept;
* insertion angles come out 5-9 degrees high, ordering kept;
* the stem tip above the last junction is classed as a "leaf";
* a perfectly vertical stem yields stem_angle = -14373 degrees (slope blow-up).

The last one is the reason this module exists as more than a pass-through: a
number that is not an angle must not be returned as one.
"""

import math

import cv2
import numpy as np
import pytest

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.morphology import (
    MorphologyRefusedError,
    measure_morphology,
)

SIZE = 400
STEM_TOP, STEM_BASE = 120, 380  # stem height 260 px


def _plant(leaves=((30, 90), (45, 80), (60, 70)), stem_tilt_deg=0.0):
    """Vertical (or tilted) stem with leaves at (angle_from_stem, length)."""
    img = np.full((SIZE, SIZE, 3), 200, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    dx = math.tan(math.radians(stem_tilt_deg)) * (STEM_BASE - STEM_TOP)
    cv2.line(mask, (200, STEM_BASE), (int(200 + dx), STEM_TOP), 255, 9)
    bases = [320, 250, 180][: len(leaves)]
    for i, ((ang, length), by) in enumerate(zip(leaves, bases, strict=True)):
        bx = int(200 + dx * (STEM_BASE - by) / (STEM_BASE - STEM_TOP))
        sgn = 1 if i % 2 == 0 else -1
        ex = int(bx + sgn * length * math.sin(math.radians(ang)))
        ey = int(by - length * math.cos(math.radians(ang)))
        cv2.line(mask, (bx, by), (ex, ey), 255, 7)
    img[mask > 0] = (40, 150, 40)
    return img, mask


def _leafiest(segments, n):
    """The n segments with the largest insertion angles = the drawn leaves; the
    stem tip PlantCV also calls a leaf has insertion angle ~0."""
    return sorted(segments, key=lambda s: s["insertion_angle"], reverse=True)[:n]


def test_recovers_known_leaf_angles_and_lengths_in_order():
    truth = [(30, 90), (45, 80), (60, 70)]
    img, mask = _plant(truth)
    res = measure_morphology(img, mask)

    assert res.plant["num_cycles"] == 0
    assert res.plant["leaf_count"] >= 3
    # PlantCV's stem is the skeleton BELOW the topmost junction (the stem tip
    # above it is sorted as a leaf), so stem_height runs base -> last junction:
    # 380 - 180 = 200 px, not the 260 px drawn. Measured: 202.
    assert abs(res.plant["stem_height"] - (STEM_BASE - 180)) / 200 < 0.10

    leaves = _leafiest(res.segments, 3)
    # Ordering: the leaf with the largest insertion angle is the shortest one.
    angles = [s["insertion_angle"] for s in leaves]
    lengths = [s["path_length"] for s in leaves]
    assert angles == sorted(angles, reverse=True)
    assert lengths == sorted(lengths), (angles, lengths)
    for (true_ang, true_len), seg in zip(
        sorted(truth, reverse=True), leaves, strict=True
    ):
        assert abs(seg["insertion_angle"] - true_ang) <= 12, (seg, true_ang)
        assert abs(seg["path_length"] - true_len) / true_len <= 0.15, (seg, true_len)

    # Positive control: a different plant gives a different answer.
    two = measure_morphology(*_plant(((30, 90), (60, 70))))
    assert len(two.segments) < len(res.segments)

    # The overlay is the numbered-segment picture, same size as the input.
    assert res.overlay.shape == img.shape
    assert not np.array_equal(res.overlay, img)


def test_vertical_stem_angle_is_null_with_a_named_warning():
    res = measure_morphology(*_plant(stem_tilt_deg=0.0))
    assert res.plant["stem_angle"] is None
    assert "stem_angle_undefined" in [w.code for w in res.warnings]

    # Positive control: a tilted stem has a finite angle and no such warning.
    tilted = measure_morphology(*_plant(stem_tilt_deg=20.0))
    assert tilted.plant["stem_angle"] is not None
    assert -180 <= tilted.plant["stem_angle"] <= 180
    assert "stem_angle_undefined" not in [w.code for w in tilted.warnings]


def test_ring_shaped_mask_reports_skeleton_cycles():
    img = np.full((SIZE, SIZE, 3), 200, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(mask, (200, 200), 80, 255, 12)  # a ring: one cycle
    img[mask > 0] = (40, 150, 40)
    res = measure_morphology(img, mask)
    assert res.plant["num_cycles"] >= 1
    assert "skeleton_has_cycles" in [w.code for w in res.warnings]

    # Positive control: the plant has no cycles and no such warning.
    plain = measure_morphology(*_plant())
    assert "skeleton_has_cycles" not in [w.code for w in plain.warnings]


def test_multi_specimen_and_empty_masks_are_refused_by_name():
    img, mask = _plant()
    two_plants = mask.copy()
    two_plants[:, :] = np.roll(mask, 150, axis=1) | np.roll(mask, -150, axis=1)
    with pytest.raises(MorphologyRefusedError, match="measure_regions"):
        measure_morphology(img, two_plants)
    with pytest.raises(DegenerateMaskError):
        measure_morphology(img, np.zeros_like(mask))
    # Positive control: the single plant measures.
    assert measure_morphology(img, mask).segments


def _hairy(mask, spur_len):
    """The plant with short spurs drawn off it every 40th mask pixel."""
    rng = np.random.default_rng(3)
    hairy = mask.copy()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys[::40], xs[::40], strict=True):
        ang = rng.uniform(0, 2 * np.pi)
        cv2.line(
            hairy,
            (int(x), int(y)),
            (int(x + spur_len * np.cos(ang)), int(y + spur_len * np.sin(ang))),
            255,
            2,
        )
    return hairy


def test_prune_size_sensitivity_is_reported_when_segment_count_is_unstable():
    """Measured segment counts: the clean plant is 7 at every prune size (PlantCV
    never prunes a tip-terminated leaf); 12-px spurs give 70 at prune 15 and 46
    at 30, a 34% change. Both directions are asserted so the advisory can
    discriminate."""
    img, mask = _plant()
    stable = measure_morphology(img, mask, prune_size=15)
    assert "prune_size_sensitive" not in [w.code for w in stable.warnings]

    res = measure_morphology(img, _hairy(mask, 12), prune_size=15)
    assert "prune_size_sensitive" in [w.code for w in res.warnings]


def test_a_skeleton_plantcv_cannot_analyse_is_refused_with_the_prune_counts():
    """On a badly fragmented skeleton PlantCV's segment_tangent_angle calls
    fatal_error ("Too many tips found per segment, try pruning again"). That
    must arrive as a refusal that says what to do, carrying the counts."""
    img, mask = _plant()
    with pytest.raises(MorphologyRefusedError, match="prune_size=8 leaves"):
        measure_morphology(img, _hairy(mask, 6), prune_size=8)
    # Positive control: pruned harder, the same mask analyses.
    assert measure_morphology(img, _hairy(mask, 6), prune_size=20).plant


def test_px_per_mm_scales_lengths_but_never_angles():
    img, mask = _plant()
    px = measure_morphology(img, mask)
    mm = measure_morphology(img, mask, px_per_mm=10.0)
    s_px, s_mm = _leafiest(px.segments, 1)[0], _leafiest(mm.segments, 1)[0]
    assert s_mm["path_length"] == pytest.approx(s_px["path_length"] / 10.0)
    assert s_mm["insertion_angle"] == s_px["insertion_angle"]
    assert mm.plant["stem_height"] == pytest.approx(px.plant["stem_height"] / 10.0)
    assert mm.units["path_length"] == "mm" and px.units["path_length"] == "pixels"
    assert mm.units["insertion_angle"] == "degrees"


def test_tangent_window_longer_than_a_leaf_is_flagged_not_reported_as_zero():
    """Measured: insertion-angle bias shrinks as tangent_size grows (24.5 deg at
    10, 5.9 at 25, 4.2 at 30) until the window exceeds a leaf's length, when the
    angle silently collapses to 0.0 (size 40 on a 70 px leaf). A zero that is an
    artefact of the window must not read as a measurement."""
    img, mask = _plant()
    res = measure_morphology(img, mask, tangent_size=40)
    assert "tangent_window_exceeds_segment" in [w.code for w in res.warnings]
    # Positive control: the default window is shorter than every leaf.
    ok = measure_morphology(img, mask)
    assert "tangent_window_exceeds_segment" not in [w.code for w in ok.warnings]


def test_the_overlay_carries_plantcvs_segment_id_text(monkeypatch):
    """The docs promise 'segment id is the number drawn on the picture', but
    segment_id() returns (colored, labeled) and the labeled one — the one with
    the digits — was discarded. Spy on the real call and assert its text pixels
    reach the returned overlay verbatim."""
    from plantcv import plantcv as pcv

    captured = {}
    real = pcv.morphology.segment_id

    def spy(skel_img, objects, mask=None):
        seg, labeled = real(skel_img=skel_img, objects=objects, mask=mask)
        captured["seg"], captured["labeled"] = seg.copy(), labeled.copy()
        return seg, labeled

    monkeypatch.setattr(pcv.morphology, "segment_id", spy)
    img, mask = _plant()
    res = measure_morphology(img, mask)
    text_only = captured["labeled"].any(axis=2) & ~captured["seg"].any(axis=2)
    assert text_only.sum() > 0, "PlantCV drew no id text at all (fixture issue)"
    assert (res.overlay[text_only] == captured["labeled"][text_only]).all(), (
        "the id digits PlantCV drew must appear in the overlay"
    )


def test_refusal_stops_recommending_prune_size_when_doubling_it_does_not_help(
    monkeypatch,
):
    """On a real sorghum photo the refusal said 'raise prune_size' at 15, 30,
    100 and 200 while the segment count sat at 126; only refine() got the
    plant analysed. When doubling prune_size keeps >=80% of the segments,
    raising it is not the remedy and the message must not offer it."""
    from plantcv_mcp import morphology as m

    img, mask = _plant()
    monkeypatch.setattr(m, "_segment_count", lambda *a, **k: 10**6)
    with pytest.raises(MorphologyRefusedError) as exc:
        measure_morphology(img, _hairy(mask, 6), prune_size=8)
    msg = str(exc.value)
    assert "refine()" in msg
    assert "raise prune_size" not in msg
    assert "does not" in msg or "will not" in msg


def test_leaves_that_vanish_at_double_prune_do_not_starve_plantcvs_palette():
    """PlantCV's segment_* functions share ONE process-global colour palette
    (params.saved_color_scale, filled by whichever call ran first). The
    prune-sensitivity pass runs segment_skeleton at 2x prune_size; when that
    removes every leaf it leaves a 1-colour palette behind, and the per-segment
    functions then index past it — IndexError from inside PlantCV on a
    233-px real seedling at prune_size=15. Leaves of 10-14 px reproduce it:
    prune 8 keeps them, prune 16 does not."""
    img, mask = _plant(leaves=((40, 14), (50, 12), (60, 10)))
    res = measure_morphology(img, mask, prune_size=8)
    assert res.plant["tip_count"] >= 2
    assert len(res.segments) == len({s["id"] for s in res.segments})
    # Positive control: at prune 30 the leaves are gone and that is reported.
    bare = measure_morphology(img, mask, prune_size=30)
    assert "no_leaf_segments" in [w.code for w in bare.warnings]


def test_a_vertical_stem_does_not_crash_insertion_angles():
    """segment_insertion_angle fits a line to the stem and draws it across the
    frame; a vertical stem makes the extrapolated y ~ -4e9 and OpenCV 4.11
    rejects the point (cv2.error, 'Can't parse pt1') — a raw traceback on a
    real 233-px seedling at prune_size=5. The angles that need that line are
    undefined; say so by name and keep everything else."""
    short = ((40, 30), (50, 25))  # the default leaves happen not to trip it
    res = measure_morphology(*_plant(leaves=short, stem_tilt_deg=0.0), prune_size=5)
    assert "insertion_angle_undefined" in [w.code for w in res.warnings]
    assert res.segments, "the per-segment table survives"
    assert all(s["insertion_angle"] is None for s in res.segments)
    assert all(s["path_length"] is not None for s in res.segments)
    assert res.plant["stem_height"] is not None
    # Positive control: a tilted stem at the same prune has finite insertion
    # angles and no such warning.
    tilted = measure_morphology(*_plant(leaves=short, stem_tilt_deg=20.0), prune_size=5)
    assert "insertion_angle_undefined" not in [w.code for w in tilted.warnings]
    assert any(s["insertion_angle"] is not None for s in tilted.segments)


def test_an_inverted_mask_is_refused_before_the_skeleton_is_built(monkeypatch):
    """A 94%-coverage mask (a real photo of beans thresholded the wrong way)
    was skeletonised for 80 s and then refused for 'too many tips'. The
    session already carried implausible_coverage; morphology must refuse it
    by name first, without touching the skeletoniser."""
    from plantcv import plantcv as pcv

    img, plant = _plant()
    inverted = 255 - plant

    def no_skeleton(*a, **k):
        raise AssertionError("skeletonize must not run on an inverted mask")

    monkeypatch.setattr(pcv.morphology, "skeletonize", no_skeleton)
    with pytest.raises(MorphologyRefusedError, match="implausible_coverage"):
        measure_morphology(img, inverted)
    # Positive control: the plant itself still reaches the skeletoniser.
    with pytest.raises(AssertionError, match="skeletonize must not run"):
        measure_morphology(img, plant)


def test_unjoinable_stem_pieces_are_refused_with_what_actually_worked(monkeypatch):
    """PlantCV's 'Unable to combine stem objects.' is a stem in pieces. The
    first remedy shipped for it — closing/fill_holes to bridge the gap — was
    a guess, and on the real sorghum photo closing 7 and 15 both left the stem
    unjoinable; the break came from the refine chain itself (opening 5 +
    median_blur 11), and a different chain measured the plant. Say what was
    measured, not what sounds right."""
    from plantcv import plantcv as pcv

    def unjoinable(*a, **k):
        raise RuntimeError("Unable to combine stem objects.")

    monkeypatch.setattr(pcv.morphology, "segment_insertion_angle", unjoinable)
    with pytest.raises(MorphologyRefusedError) as exc:
        measure_morphology(*_plant())
    msg = str(exc.value)
    assert "stem" in msg and "closing did not" in msg
    assert "opening 9" in msg and "median_blur 21" in msg and "median_blur 11" in msg
    assert "overlay" in msg and "measure_regions" in msg
    assert "prune_size" not in msg


def test_a_small_plant_in_a_huge_frame_is_analysed_on_its_crop(monkeypatch):
    """PlantCV's per-segment functions allocate a full-frame image per segment:
    on a 16 MP real photo whose plant filled 5% of the frame, segment_tangent_angle
    alone took 354 s for 14 leaves. Every morphology trait is invariant to where
    the plant sits, so the skeleton work runs on the mask's bounding box plus a
    margin wider than any prune or tangent window, and the overlay is put back
    into the frame."""
    from plantcv import plantcv as pcv

    img, mask = _plant()
    big_img = np.full((2000, 2400, 3), 200, np.uint8)
    big_mask = np.zeros((2000, 2400), np.uint8)
    big_img[1200:1600, 1500:1900] = img
    big_mask[1200:1600, 1500:1900] = mask

    seen: list[tuple[int, ...]] = []
    real = pcv.morphology.skeletonize
    monkeypatch.setattr(
        pcv.morphology,
        "skeletonize",
        lambda mask: (seen.append(mask.shape), real(mask=mask))[1],
    )
    small = measure_morphology(img, mask)
    big = measure_morphology(big_img, big_mask)
    assert seen[-1][0] < 1000 and seen[-1][1] < 1000, seen[-1]  # not the frame
    assert big.overlay.shape == big_img.shape
    assert big.overlay[1200:1600, 1500:1900].any()
    assert not big.overlay[:1100].any() or (big.overlay[:1100] == 200).all()
    assert big.plant == small.plant
    assert big.segments == small.segments
    assert [w.code for w in big.warnings] == [w.code for w in small.warnings]


def test_a_cv2_error_that_is_not_the_vertical_stem_still_raises(monkeypatch):
    """The vertical-stem handler verifies its cause by refitting the stem. A
    cv2.error from segment_insertion_angle on a plant whose stem line is
    drawable is something else and must not be turned into a tidy warning."""
    from plantcv import plantcv as pcv

    def other_failure(*a, **k):
        raise cv2.error("something else entirely")

    monkeypatch.setattr(pcv.morphology, "segment_insertion_angle", other_failure)
    with pytest.raises(cv2.error, match="something else"):
        measure_morphology(*_plant(stem_tilt_deg=20.0), prune_size=5)


def _raiser_inside_plantcvs_insertion_angle(exc_type, text):
    """A function whose code object claims PlantCV's segment_insertion_angle.py
    as its file, so a traceback through it looks like PlantCV raised."""
    ns = {}
    src = f"def f(*a, **k):\n    raise {exc_type.__name__}({text!r})\n"
    exec(  # noqa: S102 — a test double that must claim PlantCV's file name
        compile(
            src,
            "/site-packages/plantcv/plantcv/morphology/segment_insertion_angle.py",
            "exec",
        ),
        {exc_type.__name__: exc_type},
        ns,
    )
    return ns["f"]


def test_plantcvs_lost_insertion_segment_bookkeeping_is_a_named_warning(monkeypatch):
    """segment_insertion_angle keeps one list of 'pruned away' flags and another
    of computed angles; when a segment vanishes for a reason it did not flag
    the two desync and it pops an empty list — IndexError at its line 140 on a
    real 37-leaf photo (merged_VS, opening 5 + median_blur 11). Only an
    IndexError raised INSIDE that PlantCV module is turned into
    insertion_angle_undefined; the rest of the table is kept."""
    from plantcv import plantcv as pcv

    monkeypatch.setattr(
        pcv.morphology,
        "segment_insertion_angle",
        _raiser_inside_plantcvs_insertion_angle(IndexError, "list index out of range"),
    )
    res = measure_morphology(*_plant(stem_tilt_deg=20.0))
    assert "insertion_angle_undefined" in [w.code for w in res.warnings]
    assert res.segments and all(s["insertion_angle"] is None for s in res.segments)
    assert all(s["path_length"] is not None for s in res.segments)
    msg = next(w.message for w in res.warnings if w.code == "insertion_angle_undefined")
    assert "PlantCV" in msg and "insertion" in msg


def test_an_index_error_from_our_own_code_is_not_disguised(monkeypatch):
    """The same exception type raised anywhere else is a bug and must surface."""
    from plantcv import plantcv as pcv

    def ours(*a, **k):
        raise IndexError("list index out of range")

    monkeypatch.setattr(pcv.morphology, "segment_insertion_angle", ours)
    with pytest.raises(IndexError):
        measure_morphology(*_plant(stem_tilt_deg=20.0))


def test_the_inverted_mask_refusal_names_the_one_legitimate_case():
    """A macro shot of one leaf can genuinely fill the frame; measure() only
    warns for that reason. Morphology keeps refusing (the background's
    skeleton costs 80 s and is never a plant), so the message must tell the
    user with a real full-frame leaf what to do instead of only 'fix the
    segmentation'."""
    img, plant = _plant()
    with pytest.raises(MorphologyRefusedError, match="crop the photo") as exc:
        measure_morphology(img, 255 - plant)
    assert "one leaf" in str(exc.value)
    assert "under half" in str(exc.value)  # match="crop" alone was satisfied
    # by "segment the crop" with the actionable sentence gone (round 8)
