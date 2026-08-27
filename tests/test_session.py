import threading
import time
from collections import OrderedDict

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


def test_mask_is_copied_isolated_from_caller_mutations():
    store = SessionStore(max_sessions=8)
    mask = np.zeros((4, 4), np.uint8)
    sess = store.create("/img.png", mask, "a", "otsu")

    # Mutate the caller's array after create()
    mask[0, 0] = 255

    # Stored mask must be unchanged (positive control)
    assert store.get(sess.session_id).mask[0, 0] == 0
    # And the stored mask must equal what was originally passed
    assert np.array_equal(store.get(sess.session_id).mask, np.zeros((4, 4), np.uint8))


def test_max_sessions_less_than_one_raises_valueerror():
    with pytest.raises(ValueError) as exc:
        SessionStore(max_sessions=0)
    assert "max_sessions" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        SessionStore(max_sessions=-1)
    assert "max_sessions" in str(exc.value)


# --- thread safety ---
#
# mcp 2.x runs synchronous tools on worker threads, so segment() and measure()
# can hit the store at the same time. get() is check-then-act (`in`, then
# move_to_end) and create() is insert-then-evict; interleaved, get() raises a
# bare KeyError for a session that was present a microsecond earlier.


class _SlowContains(OrderedDict):
    """An OrderedDict whose membership test yields to other threads afterwards,
    widening the check-then-act window so the race is reproducible rather than
    one-in-a-million."""

    def __contains__(self, key):
        present = super().__contains__(key)
        time.sleep(0.001)
        return present


def test_concurrent_create_and_get_never_raise_keyerror():
    store = SessionStore(max_sessions=2)
    store._sessions = _SlowContains()
    unexpected: list[BaseException] = []

    def worker():
        mask = np.zeros((4, 4), np.uint8)
        for _ in range(50):
            sid = store.create("/img.png", mask, "a", "otsu").session_id
            try:
                store.get(sid)
            except UnknownSessionError:
                pass  # evicted by a neighbour: the documented outcome
            except BaseException as exc:  # noqa: BLE001 — the bug under test
                unexpected.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not unexpected, f"store raised {unexpected[0]!r} under concurrency"
    # Positive control: the store is intact and still bounded afterwards.
    assert len(store) == 2
    newest = store.create("/img.png", np.zeros((4, 4), np.uint8), "a", "otsu")
    assert store.get(newest.session_id) is newest
