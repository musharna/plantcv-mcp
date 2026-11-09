"""measure_regions, refine, count_leaves, measure and the session-kind guard at
their edges: what each hands its analysis, the refusals word for word, and the
whole envelope (mutation burn-down, #97, after tests/test_server_edges.py).

The same two shapes as there: a recording `dispatch` (calling through to the
real one) asserts the handoff an analysis would otherwise make up for, and
envelopes and refusal texts are compared whole.
"""

import asyncio
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from mcp.types import ToolAnnotations

from plantcv_mcp import plantcv_version, server, workers
from plantcv_mcp.color import color_card_excluded_advisory, exclude_card
from plantcv_mcp.diagnostics import (
    analyze_mask,
    implausible_longest_path_warning,
    mask_warnings,
)
from plantcv_mcp.hyperspectral import IndexUnavailableError, load_cube
from plantcv_mcp.imaging import (
    downscale,
    encode_png,
    load_image,
    load_image_with_digest,
    render_overlay,
    render_region_overlay,
)
from plantcv_mcp.leaves import SeedSpecError
from plantcv_mcp.refine import (
    RefinementErasedMaskError,
    RefineSpecError,
    refinement_warnings,
)
from plantcv_mcp.regions import RegionSpecError
from plantcv_mcp.server import (
    UnknownTraitError,
    WrongSessionKindError,
    _count_leaves_impl,
    _measure_impl,
    _measure_regions_impl,
    _measure_spectral_impl,
    _measure_thermal_impl,
    _refine_impl,
    _require_trait_names,
    _segment_hsi_impl,
    _segment_impl,
    _segment_thermal_impl,
    _store,
    build_server,
)

sys.path.insert(0, os.path.dirname(__file__))
from test_server_edges import _counts_cube, _hsi_helpers, _thermal_npz

ENGINE = {"name": "PlantCV", "version": plantcv_version()}
GREEN = (60, 180, 60)


@pytest.fixture
def calls(monkeypatch):
    """Every dispatch, recorded and run for real: (name, args, kwargs, result)."""
    seen: list = []
    real = workers.dispatch

    def recording(name, *args, **kwargs):
        result = real(name, *args, **kwargs)
        seen.append(SimpleNamespace(name=name, args=args, kwargs=kwargs, result=result))
        return result

    workers.set_isolation(False)
    monkeypatch.setattr(server, "dispatch", recording)
    yield seen
    workers.set_isolation(None)


def _png(tmp_path, img, name="img.png"):
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


def _two_discs(tmp_path):
    """Two comparable discs, apart: a multi_specimen mask with two major objects."""
    img = np.full((160, 240, 3), 128, np.uint8)
    cv2.circle(img, (60, 80), 35, GREEN, -1)
    cv2.circle(img, (180, 80), 30, GREEN, -1)
    return _png(tmp_path, img, "discs.png")


def _rgb_session(tmp_path):
    return _segment_impl(_two_discs(tmp_path), "a", "otsu")["session_id"]


# auto_grid needs one object per row and column; these frames hold one, so
# their regions are laid down explicitly: two cells side by side.
THERMAL_GRID = {
    "mode": "rect_grid",
    "nrows": 1,
    "ncols": 2,
    "coord": [0, 0],
    "height": 120,
    "width": 80,
    "spacing": [80, 0],
}
HSI_GRID = {
    "mode": "rect_grid",
    "nrows": 1,
    "ncols": 2,
    "coord": [0, 0],
    "height": 60,
    "width": 40,
    "spacing": [40, 0],
}


def _dicts(items):
    return [{"code": w.code, "message": w.message} for w in items]


# --- traits: the shape check ---------------------------------------------------


@pytest.mark.parametrize(
    "traits",
    ["area", b"area", ("area",), [], ["area", 1]],
    ids=["str", "bytes", "tuple", "empty", "non-str-item"],
)
def test_a_malformed_traits_selection_is_refused_word_for_word(traits):
    with pytest.raises(UnknownTraitError) as exc:
        _require_trait_names(traits)
    assert str(exc.value) == (
        f"traits must be a non-empty list of trait names, got {traits!r}."
    )


def test_a_well_formed_traits_selection_passes_the_shape_check():
    _require_trait_names(None)
    _require_trait_names(["area", "perimeter"])


def test_measure_and_measure_regions_check_traits_before_measuring(tmp_path, calls):
    sid = _rgb_session(tmp_path)
    calls.clear()
    for fn in (_measure_impl, _measure_regions_impl):
        with pytest.raises(UnknownTraitError) as exc:
            fn(sid, traits="area")
        assert str(exc.value) == (
            "traits must be a non-empty list of trait names, got 'area'."
        )
    assert calls == []


# --- measure() -----------------------------------------------------------------


def test_measure_hands_over_its_histogram_flag(tmp_path, calls):
    sid = _rgb_session(tmp_path)
    calls.clear()
    _measure_impl(sid)
    _measure_impl(sid, include_histograms=True)
    flags = [c.args[-1] for c in calls if c.name == "measure"]
    assert flags == [False, True]


def test_measure_reports_an_implausible_longest_path(tmp_path, monkeypatch):
    sid = _rgb_session(tmp_path)
    table = {
        "area": {"value": 7000, "label": "pixels"},
        "longest_path": {"value": 3, "label": "pixels"},
        "width": {"value": 190, "label": "pixels"},
        "height": {"value": 70, "label": "pixels"},
    }
    monkeypatch.setattr(server, "dispatch", lambda name, *a, **k: table)
    out = _measure_impl(sid)
    lp = implausible_longest_path_warning(table)
    assert lp is not None, "premise: 3 px against a 190 px box is implausible"
    assert {"code": lp.code, "message": lp.message} in out["warnings"]
    assert out["warnings"][-1] == {"code": lp.code, "message": lp.message}


# --- the session-kind guard ----------------------------------------------------


def test_a_session_of_the_wrong_kind_names_its_tool_and_the_right_ones(tmp_path):
    rgb = _rgb_session(tmp_path)
    thermal = _segment_thermal_impl(_thermal_npz(tmp_path), min_c=25.0, max_c=35.0)[
        "session_id"
    ]
    hsi = _segment_hsi_impl(_counts_cube(tmp_path), index="ndvi", threshold=0.2)[
        "session_id"
    ]
    cases = [
        (
            lambda: _measure_thermal_impl(rgb),
            (
                f"Session {rgb!r} is a rgb session (from segment()), not thermal. Use "
                "measure(), measure_regions(), measure_morphology(), count_leaves(), "
                "segment_leaves_sam() or refine() for it."
            ),
        ),
        (
            lambda: _measure_impl(thermal),
            (
                f"Session {thermal!r} is a thermal session (from segment_thermal()), "
                "not rgb. Use measure_thermal() or measure_regions() for it."
            ),
        ),
        (
            lambda: _measure_impl(hsi),
            (
                f"Session {hsi!r} is a hsi session (from segment_hyperspectral()), "
                "not rgb. Use measure_spectral() or measure_regions() for it."
            ),
        ),
    ]
    for call, message in cases:
        with pytest.raises(WrongSessionKindError) as exc:
            call()
        assert str(exc.value) == message
    # Positive control: each kind is accepted by its own tool.
    _measure_thermal_impl(thermal)
    _measure_spectral_impl(hsi)


# --- refine() ------------------------------------------------------------------


def test_a_kernel_past_the_mask_is_refused_before_the_worker_is_asked(
    tmp_path, monkeypatch
):
    sid = _rgb_session(tmp_path)
    asked = []
    real = workers.dispatch

    def spy(name, *a, **k):
        asked.append(name)  # BEFORE running: a refusal inside the worker is too late
        return real(name, *a, **k)

    workers.set_isolation(False)
    monkeypatch.setattr(server, "dispatch", spy)
    try:
        with pytest.raises(RefineSpecError) as exc:
            _refine_impl(sid, [{"op": "dilate", "ksize": 241}])
        assert str(exc.value) == (
            "op 0 (dilate): ksize=241 applied 1 time(s) reaches 241 px, wider than "
            "the mask's longest edge (240 px). A kernel past the mask's own extent "
            "changes nothing further; use a smaller ksize or fewer iterations."
        )
        assert asked == []
        # Positive control: one pixel less fits, and the worker runs it.
        _refine_impl(sid, [{"op": "dilate", "ksize": 240}])
        assert asked == ["refine"]
    finally:
        workers.set_isolation(None)


def test_refine_envelope_is_exact_and_names_what_it_dropped(tmp_path, calls):
    seg = _segment_impl(_two_discs(tmp_path), "a", "otsu")
    parent = _store.get(seg["session_id"])
    calls.clear()
    out = _refine_impl(parent.session_id, [{"op": "keep_largest", "n": 1}])
    (call,) = [c for c in calls if c.name == "refine"]
    mask, dropped, floods = call.result
    assert dropped, "premise: keep_largest dropped the second disc"
    before, after = analyze_mask(parent.mask), analyze_mask(mask)
    child = _store.get(out["session_id"])
    small, scale = downscale(render_overlay(load_image(parent.image_path), mask))
    png = encode_png(small)

    def summary(d):
        return {
            "mask_fraction": d.mask_fraction,
            "component_count": d.component_count,
            "largest_area": d.largest_area,
        }

    assert out == {
        "session_id": child.session_id,
        "parent_session_id": parent.session_id,
        "ops": [{"op": "keep_largest", "n": 1}],
        "lineage": [{"op": "keep_largest", "n": 1}],
        "before": summary(before),
        "after": summary(after),
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "warnings": _dicts(refinement_warnings(mask, before, after, dropped, floods)),
        "engine": ENGINE,
        "_png": png,
    }
    assert any(w["code"] == "refine_dropped_object" for w in out["warnings"]), out[
        "warnings"
    ]
    # The child carries the parent's provenance, not defaults.
    assert (
        (child.channel, child.method)
        == (parent.channel, parent.method)
        == (
            "a",
            "otsu",
        )
    )
    assert child.parent_id == parent.session_id
    assert child.color_correct is parent.color_correct


def _card_session(tmp_path, color_correct=True, inside_card=False):
    img = np.full((100, 140, 3), 128, np.uint8)
    path = _png(tmp_path, img, "card.png")
    _, digest = load_image_with_digest(path)
    card = [[80, 20], [130, 20], [130, 80], [80, 80]]
    mask = np.zeros((100, 140), np.uint8)
    if inside_card:
        mask[35:65, 95:120] = 255  # a blob lying wholly on the card
    else:
        mask[30:70, 20:74] = 255  # the plant, ending 6 px short of the card
    return _store.create(
        path,
        mask,
        "a",
        "otsu",
        digest=digest,
        color_correct=color_correct,
        card_region=card,
        card_excluded_px=40,
    )


def test_refine_that_grows_into_the_card_cuts_it_out_again_and_says_so(tmp_path, calls):
    session = _card_session(tmp_path, color_correct=False)
    calls.clear()
    out = _refine_impl(session.session_id, [{"op": "dilate", "ksize": 21}])
    (call,) = [c for c in calls if c.name == "refine"]
    grown, dropped, floods = call.result
    mask, regrown = exclude_card(grown, session.card_region)
    assert regrown > 0, "premise: the dilation reached the card"
    note = color_card_excluded_advisory(
        regrown,
        session.card_region,
        note=" An op grew the mask into the card region; those pixels were "
        "removed again.",
    )
    assert out["warnings"] == _dicts(
        [
            *refinement_warnings(
                mask, analyze_mask(session.mask), analyze_mask(mask), dropped, floods
            ),
            note,
        ]
    )
    child = _store.get(out["session_id"])
    assert child.card_excluded_px == 40 + regrown
    assert not np.any(child.mask[20:81, 80:131])


def test_refine_carries_a_colour_corrected_parents_flag(tmp_path, monkeypatch):
    session = _card_session(tmp_path, color_correct=True)
    # The flag is what is under test, not the correction: read the pixels as-is.
    monkeypatch.setattr(
        server, "_load_session_image", lambda s: load_image(s.image_path)
    )
    child = _store.get(
        _refine_impl(session.session_id, [{"op": "fill_holes"}])["session_id"]
    )
    assert child.color_correct is True
    assert child.parent_id == session.session_id


def test_refine_that_leaves_only_card_pixels_is_refused_word_for_word(tmp_path):
    session = _card_session(tmp_path, inside_card=True)
    with pytest.raises(RefinementErasedMaskError) as exc:
        _refine_impl(session.session_id, [{"op": "fill_holes"}])
    assert str(exc.value) == (
        "Every pixel the refinement produced lay inside the colour card; nothing "
        "measurable is left. The card is the instrument, not a specimen — drop "
        "the op that grew the mask into it."
    )


# --- count_leaves() ------------------------------------------------------------


def test_seeds_with_a_min_distance_are_refused_word_for_word(tmp_path):
    sid = _rgb_session(tmp_path)
    with pytest.raises(SeedSpecError) as exc:
        _count_leaves_impl(sid, min_distance=9, seeds=[[60, 80], [180, 80]])
    assert str(exc.value) == (
        "min_distance applies to the automatic split only; with seeds the markers "
        "are the seeds."
    )


def _carried(mask):
    return [
        w for w in mask_warnings(mask, analyze_mask(mask)) if w.code != "multi_specimen"
    ]


def test_seeded_count_envelope_is_exact(tmp_path, calls):
    sid = _rgb_session(tmp_path)
    session = _store.get(sid)
    assert "multi_specimen" in [
        w.code for w in mask_warnings(session.mask, analyze_mask(session.mask))
    ], "premise: two comparable discs read as several specimens"
    calls.clear()
    out = _count_leaves_impl(sid, px_per_mm=2.0, seeds=[[60, 80], [180, 80]])
    (call,) = calls
    assert call.name == "leaves_seeded"
    _img, _mask, seeds, px_per_mm = call.args
    assert (seeds, px_per_mm) == ([[60, 80], [180, 80]], 2.0)
    res = call.result
    small, scale = downscale(res.overlay)
    assert out == {
        "session_id": sid,
        "lineage": [],
        "method": "seeded_distance_watershed",
        "min_distance": None,
        "px_per_mm": 2.0,
        "leaf_count": 2,
        "count_at_other_min_distance": None,
        "instances": res.instances,
        "units": res.units,
        "warnings": _dicts([*res.warnings, *_carried(session.mask)]),
        "overlay_scale": scale,
        "engine": ENGINE,
        "_png": encode_png(small),
    }
    assert "multi_specimen" not in [w["code"] for w in out["warnings"]]


def test_automatic_count_envelope_is_exact(tmp_path, calls):
    sid = _rgb_session(tmp_path)
    session = _store.get(sid)
    calls.clear()
    out = _count_leaves_impl(sid, min_distance=12)
    (call,) = calls
    assert call.name == "leaves"
    res = call.result
    small, scale = downscale(res.overlay)
    assert out == {
        "session_id": sid,
        "lineage": [],
        "method": "distance_transform_watershed",
        "min_distance": 12,
        "px_per_mm": None,
        "leaf_count": res.leaf_count,
        "count_at_other_min_distance": res.sensitivity,
        "instances": res.instances,
        "units": res.units,
        "warnings": _dicts([*res.warnings, *_carried(session.mask)]),
        "overlay_scale": scale,
        "engine": ENGINE,
        "_png": encode_png(small),
    }
    assert res.sensitivity is not None


# --- measure_regions(): rgb ------------------------------------------------------


def test_measure_regions_defaults_reach_the_worker(tmp_path, calls):
    sid = _rgb_session(tmp_path)
    calls.clear()
    _measure_regions_impl(sid)
    (call,) = calls
    assert call.name == "regions"
    assert call.kwargs == {
        "mode": "auto_grid",
        "nrows": 1,
        "ncols": 1,
        "coord": None,
        "height": None,
        "width": None,
        "spacing": None,
        "radius": None,
        "analyses": ("size",),
        "px_per_mm": None,
        "include_histograms": False,
    }


def test_measure_regions_rgb_envelope_is_exact(tmp_path, calls):
    path = _two_discs(tmp_path)
    seg = _segment_impl(path, "a", "otsu")
    sid = _refine_impl(seg["session_id"], [{"op": "fill_holes"}])["session_id"]
    session = _store.get(sid)
    calls.clear()
    # The right-hand cell of a 1x3 grid over the frame holds no plant pixels.
    out = _measure_regions_impl(
        sid,
        mode="rect_grid",
        nrows=1,
        ncols=3,
        coord=[0, 0],
        width=80,
        height=160,
        spacing=[80, 0],
        px_per_mm=2.0,
        include_histograms=True,
        traits=["area"],
    )
    (call,) = calls
    assert call.kwargs["px_per_mm"] == 2.0
    assert call.kwargs["include_histograms"] is True
    assert call.kwargs["coord"] == (0, 0) and call.kwargs["spacing"] == (80, 0)
    regions = call.result
    ms = regions["measurements"]
    flags = [bool(m["measured"]) for m in ms]
    assert True in flags and False in flags, f"premise: some cell is empty ({flags})"
    for m in ms:
        if m.get("traits") is not None:
            assert list(m["traits"]) == ["area"]
    overlay = render_region_overlay(
        load_image(path), session.mask, regions["bboxes"], flags
    )
    small, scale = downscale(overlay)
    assert out == {
        "session_id": sid,
        "kind": "rgb",
        "mode": regions["mode"],
        "nrows": regions["nrows"],
        "ncols": regions["ncols"],
        "regions_total": len(ms),
        "regions_measured": flags.count(True),
        "regions_empty": flags.count(False),
        "analyses": ["size"],
        "px_per_mm": 2.0,
        "lineage": [{"op": "fill_holes"}],
        "regions": ms,
        "warnings": [{"code": c, "message": m} for c, m in regions["warnings"]],
        "overlay_scale": scale,
        "engine": ENGINE,
        "_png": encode_png(small),
    }


def test_measure_regions_reports_a_grid_warning_by_code_and_message(
    tmp_path, monkeypatch
):
    # A worker that constructs fewer regions than asked returns this warning;
    # forcing PlantCV's auto_grid to do that is fragile, so it is appended to
    # the real worker's result instead.
    sid = _rgb_session(tmp_path)
    real = workers.dispatch

    def short_grid(name, *args, **kwargs):
        result = real(name, *args, **kwargs)
        if name == "regions":
            result["warnings"] = [
                *result["warnings"],
                ("region_count_mismatch", "Asked for 1x2 = 2 regions but 1 were."),
            ]
        return result

    workers.set_isolation(False)
    monkeypatch.setattr(server, "dispatch", short_grid)
    try:
        out = _measure_regions_impl(sid)
    finally:
        workers.set_isolation(None)
    assert out["warnings"][-1] == {
        "code": "region_count_mismatch",
        "message": "Asked for 1x2 = 2 regions but 1 were.",
    }


@pytest.mark.parametrize("name", ["coord", "spacing"])
def test_a_bad_pair_is_refused_by_its_own_name(tmp_path, name):
    sid = _rgb_session(tmp_path)
    with pytest.raises(RegionSpecError) as exc:
        _measure_regions_impl(sid, mode="grid", **{name: [1, 2, 3]})
    assert str(exc.value) == (
        f"{name} must be exactly [x, y]; got 3 values: [1, 2, 3]."
    )


# --- measure_regions(): the arguments of another modality ------------------------


def test_rgb_arguments_on_another_kind_are_refused_by_name(tmp_path):
    thermal = _segment_thermal_impl(_thermal_npz(tmp_path), min_c=25.0, max_c=35.0)[
        "session_id"
    ]
    hsi = _segment_hsi_impl(_counts_cube(tmp_path), index="ndvi", threshold=0.2)[
        "session_id"
    ]
    for sid, kind, stats in (
        (thermal, "thermal", "temperature"),
        (hsi, "hsi", "index"),
    ):
        for kw, name in (
            ({"analyses": ["size"]}, "analyses"),
            ({"px_per_mm": 2.0}, "px_per_mm"),
            ({"include_histograms": True}, "include_histograms"),
            ({"traits": ["area"]}, "traits"),
        ):
            with pytest.raises(WrongSessionKindError) as exc:
                _measure_regions_impl(sid, **kw)
            assert str(exc.value) == (
                f"{name} applies to rgb sessions; session {sid!r} is {kind}. Its "
                f"regions report {stats} statistics."
            )
    # Positive control: the same calls without those arguments succeed.
    _measure_regions_impl(thermal, **THERMAL_GRID)
    _measure_regions_impl(hsi, **HSI_GRID)


def test_an_unknown_session_kind_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(_store, "get", lambda sid: SimpleNamespace(kind="lidar"))
    with pytest.raises(WrongSessionKindError) as exc:
        _measure_regions_impl("s")
    assert str(exc.value) == "Unknown session kind 'lidar'."


# --- measure_regions(): hsi ----------------------------------------------------


def test_measure_regions_hsi_rereads_the_refs_and_reports_them(tmp_path, calls):
    th = _hsi_helpers()
    path = _counts_cube(tmp_path)
    white, dark = th._refs(tmp_path)
    sid = _segment_hsi_impl(
        path, index="sr", threshold=1.5, white_reference=white, dark_reference=dark
    )["session_id"]
    session = _store.get(sid)
    calls.clear()
    out = _measure_regions_impl(sid, **HSI_GRID)
    (call,) = calls
    assert call.name == "hsi_regions"
    load, mask = call.args
    assert load.raw_path == path and mask is session.mask
    kw = dict(call.kwargs)
    white_load, dark_load = kw.pop("white_load"), kw.pop("dark_load")
    assert white_load.raw_path == white and dark_load.raw_path == dark
    assert white_load.digest == load_cube(white).digest
    assert dark_load.digest == load_cube(dark).digest
    assert kw["indices"] == ("sr",)
    assert kw["calibration"] == session.extra["calibration_args"]
    assert kw["calibration"].get("white_reference") == white
    regions = call.result
    ms = regions["measurements"]
    small, scale = downscale(
        render_region_overlay(
            np.ascontiguousarray(load.cube.pseudo_rgb),
            session.mask,
            regions["bboxes"],
            [bool(m["measured"]) for m in ms],
        )
    )
    assert out == {
        "session_id": sid,
        "kind": "hsi",
        "mode": regions["mode"],
        "nrows": regions["nrows"],
        "ncols": regions["ncols"],
        "regions_total": len(ms),
        "regions_measured": sum(bool(m["measured"]) for m in ms),
        "regions_empty": sum(not m["measured"] for m in ms),
        "indices": ["sr"],
        "calibration": regions["calibration"],
        "regions": ms,
        "warnings": [{"code": c, "message": m} for c, m in regions["warnings"]],
        "overlay_scale": scale,
        "engine": ENGINE,
        "_png": encode_png(small),
    }


def test_an_empty_indices_list_is_refused_naming_the_sessions_index(tmp_path):
    sid = _segment_hsi_impl(_counts_cube(tmp_path), index="sr", threshold=1.5)[
        "session_id"
    ]
    with pytest.raises(IndexUnavailableError) as exc:
        _measure_regions_impl(sid, indices=[])
    assert str(exc.value) == (
        "No indices requested; omit indices to measure the session's index ('sr')."
    )


def test_a_session_without_an_index_falls_back_to_ndvi(tmp_path, calls):
    path = _counts_cube(tmp_path)
    load = load_cube(path)
    mask = np.zeros(load.cube.pseudo_rgb.shape[:2], np.uint8)
    mask[2:6, 2:6] = 255
    session = _store.create(
        path, mask, "index:ndvi", "threshold:0.2", digest=load.digest, kind="hsi"
    )
    with pytest.raises(IndexUnavailableError) as exc:
        _measure_regions_impl(session.session_id, indices=[])
    assert str(exc.value) == (
        "No indices requested; omit indices to measure the session's index ('ndvi')."
    )
    calls.clear()
    out = _measure_regions_impl(session.session_id, **HSI_GRID)
    (call,) = calls
    assert call.kwargs["indices"] == ("ndvi",)
    # No calibration recorded: neither reference is loaded.
    assert call.kwargs["white_load"] is None and call.kwargs["dark_load"] is None
    assert out["indices"] == ["ndvi"]


# --- build_server(): the read-only contract --------------------------------------


def test_every_read_only_tool_says_exactly_read_only():
    tools = asyncio.run(build_server().list_tools())
    exact = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
    read_only = [t for t in tools if t.annotations.read_only_hint]
    assert len(read_only) == 15, [t.name for t in read_only]
    for t in read_only:
        assert t.annotations == exact, (t.name, t.annotations)
