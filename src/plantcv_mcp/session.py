"""In-memory session store.

Sessions hold the mask (uint8 HxW) but NOT the RGB image — that is re-read from
disk on demand, keeping memory bounded when several sessions are live.
"""

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


class UnknownSessionError(Exception):
    """Raised when a session_id is not in the store."""


@dataclass
class Session:
    session_id: str
    image_path: str
    mask: np.ndarray
    channel: str
    method: str
    shape: tuple[int, int]
    # SHA-256 of the source file at segmentation time. Shape alone cannot detect
    # a same-dimension content swap, which would silently measure a stale mask
    # against new pixels. REQUIRED and non-empty: the guards compare it
    # unconditionally, so a session without one cannot exist.
    digest: str
    # Whether the image was colour-corrected before segmentation. measure()
    # re-reads from disk, so it must re-apply the same transform or it would
    # measure different pixels than the ones the mask was drawn on.
    color_correct: bool = False
    # The detected colour card's polygon, when one was excluded from the mask,
    # and how many mask pixels it took. refine() re-applies the exclusion (a
    # dilation grew a plant into the card and measure() sampled card pixels)
    # and measure() re-emits the advisory the trait table would otherwise lose.
    card_region: list[list[int]] | None = None
    card_excluded_px: int = 0
    # How this mask was made, beyond the threshold: the refine() ops applied to
    # reach it, in order, cumulative across chained refinements. Echoed on every
    # trait table so a stored result can say what produced its mask.
    lineage: list[dict] = field(default_factory=list)
    parent_id: str | None = None
    # Modality. "rgb" sessions come from segment()/refine(); "hsi" and
    # "thermal" from their own segmenters. Tools refuse the wrong kind by name.
    kind: str = "rgb"
    # Modality-specific record needed to re-derive the analysis at measure time
    # (e.g. calibration references for a cube). Never the pixels themselves.
    extra: dict = field(default_factory=dict)


class SessionStore:
    def __init__(self, max_sessions: int = 8) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}")
        self._max = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        # mcp 2.x runs tools on worker threads. get() is check-then-act and
        # create() is insert-then-evict; interleaved, get() raised a bare
        # KeyError for a session that was present a moment earlier. This lock
        # is the store's own — it must not be the PlantCV analysis lock, which
        # would serialise cheap bookkeeping behind slow measurement.
        self._lock = threading.Lock()

    def create(
        self,
        image_path: str,
        mask: np.ndarray,
        channel: str,
        method: str,
        *,
        digest: str,
        color_correct: bool = False,
        lineage: list[dict] | None = None,
        parent_id: str | None = None,
        kind: str = "rgb",
        extra: dict | None = None,
        card_region: list[list[int]] | None = None,
        card_excluded_px: int = 0,
    ) -> Session:
        if not digest:
            raise ValueError(
                "A session requires the source file's digest: without it the "
                "stale-image guard would be silently skipped."
            )
        stored_mask = np.array(mask, copy=True)
        # get() hands out this very array; a caller's in-place write would
        # corrupt every later measurement of the session. Refuse writes at the
        # array itself instead of trusting every future call site.
        stored_mask.setflags(write=False)
        session = Session(
            session_id=str(uuid.uuid4()),
            image_path=image_path,
            mask=stored_mask,
            channel=channel,
            method=method,
            shape=(int(mask.shape[0]), int(mask.shape[1])),
            digest=digest,
            color_correct=color_correct,
            card_region=[list(p) for p in card_region] if card_region else None,
            card_excluded_px=int(card_excluded_px),
            lineage=[dict(op) for op in (lineage or [])],
            parent_id=parent_id,
            kind=kind,
            extra=dict(extra or {}),
        )
        with self._lock:
            self._sessions[session.session_id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)  # evict least-recently-used
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise UnknownSessionError(
                    f"Unknown session_id {session_id!r}. Sessions are in-memory "
                    f"and capped at {self._max}; the oldest are evicted. Re-run "
                    "segment()."
                )
            self._sessions.move_to_end(session_id)  # refresh recency
            return self._sessions[session_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
