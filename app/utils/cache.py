from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover
    Redis = None  # type: ignore[assignment]


@dataclass
class MemoryCache:
    storage: dict[str, str] = field(default_factory=dict)

    async def get(self, key: str) -> str | None:
        return self.storage.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.storage[key] = value


class CacheClient:
    def __init__(self) -> None:
        self._memory = MemoryCache()
        self._redis: Redis | None = None
        if settings.cache_enabled and Redis is not None:
            try:
                self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("redis_init_failed", extra={"error": str(exc)})
                self._redis = None

    async def get_json(self, key: str) -> Any | None:
        backend = self._redis or self._memory
        payload = await backend.get(key)
        if payload is None:
            return None
        return json.loads(payload)

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        backend = self._redis or self._memory
        await backend.set(key, json.dumps(value, default=str), ex=ttl_seconds)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()


cache = CacheClient()


def ttl_hours(hours: int) -> int:
    return int(timedelta(hours=hours).total_seconds())
