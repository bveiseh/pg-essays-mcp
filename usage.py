"""Aggregate usage counters that never include request content."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("pg_usage")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_started_at = time.time()
_counts: dict[str, int] = {}
_lock = threading.Lock()


def record(name: str) -> None:
    with _lock:
        _counts[name] = _counts.get(name, 0) + 1


def snapshot() -> dict[str, object]:
    with _lock:
        counts = dict(sorted(_counts.items()))
    return {"uptime_seconds": round(time.time() - _started_at, 1), "counts": counts}


def report() -> None:
    logger.info("usage %s", snapshot())
