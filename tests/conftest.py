"""Shared pytest fixtures.

Provides a fakeredis-backed override for the `get_redis` dependency so any
endpoint wired with a Phase 14 rate limiter can run without a live Redis server.
"""

import fakeredis
import pytest

from app.api.v1 import deps
from app.main import app


@pytest.fixture(autouse=True)
def _redis_dependency_override():
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeStrictRedis(server=server)

    def override_redis():
        return redis

    app.dependency_overrides[deps.get_redis] = override_redis
    yield
    app.dependency_overrides.clear()
