"""Redis connection + the RQ queue. MOG_SYNC_JOBS=1 runs jobs inline (tests, single-process dev)."""

from __future__ import annotations

import os
from functools import lru_cache

from redis import Redis
from rq import Queue

from mog.config import settings

QUEUE_NAME = "mog"


@lru_cache(maxsize=1)
def redis_conn() -> Redis:
    if os.environ.get("MOG_FAKE_REDIS") == "1":
        import fakeredis

        return fakeredis.FakeRedis()
    return Redis.from_url(settings().redis_url)


def queue() -> Queue:
    sync = os.environ.get("MOG_SYNC_JOBS") == "1"
    return Queue(QUEUE_NAME, connection=redis_conn(), is_async=not sync, default_timeout=60 * 30)
