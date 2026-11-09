"""segment() and the session re-read at their edges (mutation burn-down, #97).

Each envelope is rebuilt independently from the helpers segment() is meant to
call (threshold, card exclusion, fill, warnings, overlay), so a dropped or
re-wired argument shows as a difference rather than passing on its own output.
Every refusal is checked word for word and on a path that does not exist, so
it is also shown to fire before any pixel is read.
"""

import os
import sys

import cv2
import numpy as np
import pytest
from plantcv import plantcv as pcv

from plantcv_mcp.color import (
    color_card_excluded_advisory,
    correct_color,
    detect_card_region,
    exclude_card,
)
from plantcv_mcp.diagnostics import (
    analyze_mask,
    segmentation_warnings,
    threshold_inside_mode_warning,
)
from plantcv_mcp.imaging import (
    downscale,
    encode_png,
    load_image,
    load_image_with_digest,
    render_overlay,
)
from plantcv_mcp.limits import ParameterRangeError
from plantcv_mcp.segmentation import (
    SegmentRuleError,
    describe_rules,
    rules_mask,
    threshold_mask,
    to_gray,
    validate_rules,
)
from plantcv_mcp.server import (
    ImageChangedSinceSegmentationError,
    _load_session_image,
    _segment_impl,
    _store,
    rules_inside_mode_warnings,
)

sys.path.insert(0, os.path.dirname(__file__))
from test_diagnostics import _disc_on_soil
from test_scale_color_batch import _card_and_two_plants

GREEN = (60, 180, 60)


def _png(tmp_path, img, name="img.png"):
    path = str(tmp_path / name)
    assert cv2.imwrite(path, img)
    return path


def _two_discs(h=160, w=240):
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.circle(img, (60, 80), 35, GREEN, -1)
    cv2.circle(img, (180, 80), 30, GREEN, -1)
    return img


def _dicts(warnings):
    return [{"code": w.code, "message": w.message} for w in warnings]


def _envelope(img, mask, warnings, recipe, fill_size=200, color_correct=False):
    """segment()'s documented envelope, built from its parts."""
    diag = analyze_mask(mask)
    small, scale = downscale(render_overlay(img, mask))
    png = encode_png(small)
    return {
        **recipe,
        "fill_size": fill_size,
        "color_correct": color_correct,
        "mask_fraction": diag.mask_fraction,
        "component_count": diag.component_count,
        "major_object_count": diag.major_object_count,
        "largest_area": diag.largest_area,
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "_png": png,
        "warnings": _dicts(warnings),
    }


def _assert_session(out, path, mask, channel, method, cc=False, card=None, removed=0):
    s = _store.get(out["session_id"])
    assert (s.image_path, s.channel, s.method) == (path, channel, method)
    assert s.digest == load_image_with_digest(path)[1]
    assert (s.color_correct, s.card_region, s.card_excluded_px) == (cc, card, removed)
    assert (s.kind, s.lineage, s.parent_id) == ("rgb", [], None)
    assert np.array_equal(s.mask, mask)


# --- refusals, word for word and before any pixel is read --------------------

MISSING = "/nonexistent/never-read.png"
RULES = [{"channel": "a", "method": "otsu", "object_type": "dark"}]


@pytest.mark.parametrize(
    ("extra", "given"),
    [
        ({"channel": "a"}, ["channel"]),
        ({"method": "otsu"}, ["method"]),
        ({"object_type": "light"}, ["object_type"]),
        ({"ksize": 13}, ["ksize"]),
        ({"offset": 3}, ["offset"]),
        (
            {
                "channel": "a",
                "method": "otsu",
                "object_type": "dark",
                "ksize": 10,
                "offset": 1,
            },
            ["channel", "method", "object_type", "ksize", "offset"],
        ),
    ],
)
def test_single_channel_arguments_beside_rules_are_named(extra, given):
    with pytest.raises(SegmentRuleError) as exc:
        _segment_impl(MISSING, rules=RULES, **extra)
    assert str(exc.value) == (
        f"{given} cannot be combined with rules: each rule carries its own "
        "channel and threshold. Give either channel + method (one channel) or "
        "rules (several), not both."
    )


def test_rules_with_the_default_kernel_and_offset_spelled_out_are_accepted(tmp_path):
    path = _png(tmp_path, _two_discs())
    out = _segment_impl(path, rules=RULES, ksize=11, offset=2)
    assert out["largest_area"] > 0


def test_combine_without_rules_is_refused_word_for_word():
    with pytest.raises(SegmentRuleError) as exc:
        _segment_impl(MISSING, "a", "otsu", combine="or")
    assert str(exc.value) == "combine applies to rules only."


@pytest.mark.parametrize("given", [{"channel": "a"}, {"method": "otsu"}, {}])
def test_a_missing_channel_or_method_is_refused_word_for_word(given):
    with pytest.raises(SegmentRuleError) as exc:
        _segment_impl(MISSING, **given)
    assert str(exc.value) == (
        "Give channel and method (one channel), or rules (several channels or "
        "explicit value ranges, joined by combine)."
    )


def test_malformed_rules_are_refused_before_reading():
    with pytest.raises(SegmentRuleError, match="combine is required"):
        _segment_impl(
            MISSING, rules=[{"channel": "a", "min": 10}, {"channel": "b", "min": 140}]
        )


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"channel": "a", "method": "otsu", "fill_size": -1}, "fill_size"),
        ({"channel": "a", "method": "gaussian", "ksize": 0}, "ksize"),
        ({"channel": "a", "method": "mean", "offset": 10_000}, "offset"),
        ({"rules": RULES, "fill_size": -1}, "fill_size"),
    ],
)
def test_out_of_range_numbers_are_refused_before_reading(kwargs, name):
    with pytest.raises(ParameterRangeError) as exc:
        _segment_impl(MISSING, **kwargs)
    assert name in str(exc.value)


def test_a_global_method_ignores_the_adaptive_parameters(tmp_path):
    # Positive control for the check above: otsu never reads ksize or offset.
    path = _png(tmp_path, _two_discs())
    out = _segment_impl(path, "a", "otsu", ksize=0, offset=10_000)
    assert out["largest_area"] > 0


# --- envelopes ---------------------------------------------------------------


def test_single_channel_envelope_is_exact_with_the_inside_mode_warning(tmp_path):
    path = _png(tmp_path, _disc_on_soil(14))
    out = _segment_impl(path, "a", "otsu")
    img = load_image(path)
    pre = threshold_mask(img, "a", "otsu", object_type="dark", ksize=11, offset=2)
    mask = pcv.fill(bin_img=pre, size=200)
    inside = threshold_inside_mode_warning(to_gray(img, "a"), "otsu", "a")
    assert inside is not None, "premise: otsu cuts inside the soil's mode"
    warnings = [
        *segmentation_warnings(mask, analyze_mask(mask), analyze_mask(pre), 200),
        inside,
    ]
    recipe = {"channel": "a", "method": "otsu", "object_type": "dark"}
    assert out == {
        "session_id": out["session_id"],
        **_envelope(img, mask, warnings, recipe),
    }
    _assert_session(out, path, mask, "a", "otsu")


def test_a_fill_that_erases_the_mask_names_its_own_size(tmp_path):
    path = _png(tmp_path, _two_discs())
    out = _segment_impl(path, "a", "otsu", fill_size=100_000)
    img = load_image(path)
    pre = threshold_mask(img, "a", "otsu")
    mask = pcv.fill(bin_img=pre, size=100_000)
    warnings = segmentation_warnings(
        mask, analyze_mask(mask), analyze_mask(pre), 100_000
    )
    assert "fill_size=100000 removed every one" in _dicts(warnings)[0]["message"], (
        "premise: the fill erased a mask the threshold had found"
    )
    assert out["warnings"] == _dicts(warnings)
    assert out["fill_size"] == 100_000 and out["largest_area"] == 0


def test_an_adaptive_threshold_gets_every_parameter(tmp_path):
    path = _png(tmp_path, _disc_on_soil(40))
    out = _segment_impl(
        path, "a", "gaussian", object_type="light", fill_size=5, ksize=31, offset=-3
    )
    img = load_image(path)
    pre = threshold_mask(img, "a", "gaussian", object_type="light", ksize=31, offset=-3)
    # Premise: each parameter changes the result, so dropping one shows.
    others = [
        threshold_mask(img, "a", "gaussian", object_type="light", offset=-3),
        threshold_mask(img, "a", "gaussian", object_type="light", ksize=31),
        threshold_mask(img, "a", "gaussian", object_type="dark", ksize=31, offset=-3),
    ]
    assert not any(np.array_equal(pre, o) for o in others)
    mask = pcv.fill(bin_img=pre, size=5)
    assert not np.array_equal(mask, pcv.fill(bin_img=pre, size=200))
    warnings = segmentation_warnings(mask, analyze_mask(mask), analyze_mask(pre), 5)
    recipe = {"channel": "a", "method": "gaussian", "object_type": "light"}
    assert out == {
        "session_id": out["session_id"],
        **_envelope(img, mask, warnings, recipe, fill_size=5),
    }
    _assert_session(out, path, mask, "a", "gaussian")


def test_rules_envelope_is_exact(tmp_path):
    path = _png(tmp_path, _disc_on_soil(14))
    rules = [
        {"channel": "a", "method": "otsu", "object_type": "dark"},
        {"channel": "a", "max": 118},  # min filled in as 0
    ]
    out = _segment_impl(path, rules=rules, combine="or", fill_size=150)
    img = load_image(path)
    valid = validate_rules(rules, "or")
    assert valid != rules, "premise: validation fills in defaults"
    pre, coverage = rules_mask(img, valid, "or")
    mask = pcv.fill(bin_img=pre, size=150)
    inside = rules_inside_mode_warnings(img, valid, "or")
    assert inside, "premise: the otsu rule cuts inside the soil's mode"
    warnings = [
        *segmentation_warnings(mask, analyze_mask(mask), analyze_mask(pre), 150),
        *inside,
    ]
    recipe = {"rules": valid, "combine": "or", "rule_coverage": coverage}
    assert out == {
        "session_id": out["session_id"],
        **_envelope(img, mask, warnings, recipe, fill_size=150),
    }
    _assert_session(out, path, mask, describe_rules(valid, "or"), "rules")


def _card_case(img, pre_fill, card, fill_size=200):
    pre, removed = exclude_card(pre_fill, card)
    assert removed > 0, "premise: the threshold reached the card"
    mask = pcv.fill(bin_img=pre, size=fill_size)
    warnings = segmentation_warnings(
        mask, analyze_mask(mask), analyze_mask(pre), fill_size
    )
    inside = threshold_inside_mode_warning(to_gray(img, "a"), "otsu", "a")
    warnings = [*warnings, *([inside] if inside else [])]
    return mask, removed, [*warnings, color_card_excluded_advisory(removed, card)]


def test_colour_correction_envelope_is_exact_and_cuts_the_card_out(tmp_path):
    path = _png(tmp_path, _card_and_two_plants())
    out = _segment_impl(path, "a", "otsu", object_type="light", color_correct=True)
    raw = load_image(path)
    img, card = correct_color(raw)
    assert not np.array_equal(img, raw), "premise: correction changes the pixels"
    pre = threshold_mask(img, "a", "otsu", object_type="light")
    mask, removed, warnings = _card_case(img, pre, card)
    recipe = {"channel": "a", "method": "otsu", "object_type": "light"}
    assert out == {
        "session_id": out["session_id"],
        **_envelope(img, mask, warnings, recipe, color_correct=True),
    }
    _assert_session(out, path, mask, "a", "otsu", cc=True, card=card, removed=removed)


def test_excluding_the_card_without_correction_keeps_the_colours(tmp_path):
    path = _png(tmp_path, _card_and_two_plants())
    out = _segment_impl(path, "a", "otsu", object_type="light", exclude_color_card=True)
    img = load_image(path)
    card = detect_card_region(img)
    pre = threshold_mask(img, "a", "otsu", object_type="light")
    mask, removed, warnings = _card_case(img, pre, card)
    recipe = {"channel": "a", "method": "otsu", "object_type": "light"}
    assert out == {
        "session_id": out["session_id"],
        **_envelope(img, mask, warnings, recipe),
    }
    _assert_session(out, path, mask, "a", "otsu", card=card, removed=removed)


# --- _load_session_image -----------------------------------------------------


def _session_on(tmp_path, img, **kw):
    path = _png(tmp_path, img)
    return path, _store.get(_segment_impl(path, "a", "otsu", **kw)["session_id"])


def test_a_resized_file_is_refused_word_for_word(tmp_path):
    path, session = _session_on(tmp_path, _two_discs())
    _png(tmp_path, _two_discs(h=170))
    with pytest.raises(ImageChangedSinceSegmentationError) as exc:
        _load_session_image(session)
    assert str(exc.value) == (
        f"Image at {path!r} is now (170, 240) but was (160, 240) when segment() "
        "produced this session's mask. The file changed on disk since "
        "segmentation, so the mask no longer corresponds to its current content. "
        "Re-run segment() on the current file before measuring it."
    )


def test_a_same_shape_content_change_is_refused_word_for_word(tmp_path):
    path, session = _session_on(tmp_path, _two_discs())
    changed = _two_discs()
    changed[0, 0] = (0, 0, 0)
    _png(tmp_path, changed)
    with pytest.raises(ImageChangedSinceSegmentationError) as exc:
        _load_session_image(session)
    assert str(exc.value) == (
        f"Image at {path!r} still has shape (160, 240), but its CONTENT changed "
        "since segment() produced this session's mask (SHA-256 mismatch). The "
        "mask describes pixels that are no longer there. Re-run segment() on the "
        "current file."
    )


def test_an_unchanged_file_is_returned_as_read(tmp_path):
    path, session = _session_on(tmp_path, _two_discs())
    assert np.array_equal(_load_session_image(session), load_image(path))


def test_a_corrected_session_is_reread_corrected(tmp_path):
    path, session = _session_on(
        tmp_path, _card_and_two_plants(), object_type="light", color_correct=True
    )
    raw = load_image(path)
    corrected = correct_color(raw)[0]
    assert not np.array_equal(corrected, raw)
    assert np.array_equal(_load_session_image(session), corrected)


def test_the_integrity_guards_run_before_the_correction(tmp_path):
    # The file now has no card: correcting first would raise ColorCardNotFound
    # and hide that the image itself changed.
    _path, session = _session_on(
        tmp_path, _card_and_two_plants(), object_type="light", color_correct=True
    )
    _png(tmp_path, np.full((900, 760, 3), 200, np.uint8))
    with pytest.raises(ImageChangedSinceSegmentationError, match="CONTENT changed"):
        _load_session_image(session)
