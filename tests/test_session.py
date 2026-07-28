import numpy as np
import pytest

from plantcv_mcp.session import SessionStore, UnknownSessionError


def _store_with(n, max_sessions=8):
    store = SessionStore(max_sessions=max_sessions)
    ids = [
        store.create("/img.png", np.zeros((4, 4), np.uint8), "a", "otsu").session_id
        for _ in range(n)
    ]
    return store, ids


def test_create_then_get_roundtrips():
    store, ids = _store_with(1)
    assert store.get(ids[0]).channel == "a"


def test_unknown_session_id_names_what_was_passed():
    store, _ = _store_with(1)
    with pytest.raises(UnknownSessionError) as exc:
        store.get("bogus-id")
    assert "bogus-id" in str(exc.value)


def test_lru_evicts_oldest_beyond_max():
    store, ids = _store_with(9, max_sessions=8)
    assert len(store) == 8
    with pytest.raises(UnknownSessionError):
        store.get(ids[0])  # oldest evicted
    assert store.get(ids[-1])  # newest survives (positive control)


def test_get_refreshes_recency_so_used_sessions_survive():
    store, ids = _store_with(8, max_sessions=8)
    store.get(ids[0])  # touch the oldest
    store.create("/img.png", np.zeros((4, 4), np.uint8), "a", "otsu")
    assert store.get(ids[0])  # survived because touched
    with pytest.raises(UnknownSessionError):
        store.get(ids[1])  # the new oldest went instead
