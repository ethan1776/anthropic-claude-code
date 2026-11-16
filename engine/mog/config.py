"""Settings from environment variables (all optional, sensible local defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("MOG_DATA", "./data")).resolve())
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    max_clips: int = field(default_factory=lambda: _int("MOG_MAX_CLIPS", 10))
    max_upload_bytes: int = field(default_factory=lambda: _int("MOG_MAX_UPLOAD_BYTES", 2 * 1024**3))   # all clips
    max_audio_bytes: int = field(default_factory=lambda: _int("MOG_MAX_AUDIO_BYTES", 300 * 1024**2))
    retention_hours: int = field(default_factory=lambda: _int("MOG_RETENTION_HOURS", 24))
    jobs_per_hour: int = field(default_factory=lambda: _int("MOG_JOBS_PER_HOUR", 60))   # per client IP
    # Behind the web app's /api proxy every request comes from the proxy; trust the client IP it
    # forwards (X-Forwarded-For) so rate limits are per user, not global. Only enable when the
    # API isn't reachable directly from the internet.
    trust_proxy: bool = field(default_factory=lambda: os.environ.get("MOG_TRUST_PROXY") == "1")
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o for o in os.environ.get("MOG_CORS_ORIGINS", "http://localhost:3000").split(",") if o))


def settings() -> Settings:
    return Settings()
