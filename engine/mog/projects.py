"""Projects: uploads, analyses and edits, stored as small JSON files.

Layout (storage keys):
  projects/<pid>/project.json                     written by the API
  projects/<pid>/clips/<cid>/meta.json + file     API (upload)
  projects/<pid>/clips/<cid>/analysis.json        worker
  projects/<pid>/audio/meta.json + file           API (upload)
  projects/<pid>/audio/analysis.json              worker
  projects/<pid>/audio/playback.m4a               worker (browser playback copy)
  projects/<pid>/edits/<eid>/edit.json            API creates, worker updates
  projects/<pid>/edits/<eid>/timeline.json        worker / API (user edits)
  projects/<pid>/edits/<eid>/preview.mp4, export.mp4   worker

Each file has a single writer at a time, so no locking is needed.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Literal

from pydantic import BaseModel, Field

from mog.errors import MogError
from mog.storage import Storage

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    return (keep.lstrip(".") or "file")[:80]


class Project(BaseModel):
    id: str
    created_at: float
    name: str = "Untitled edit"
    saved: bool = Field(False, description="Saved projects are kept; others are deleted after 24 h")


class ClipMeta(BaseModel):
    id: str
    filename: str
    size: int
    uploaded_at: float


class AudioMeta(BaseModel):
    filename: str
    size: int
    uploaded_at: float


class EditSettings(BaseModel):
    preset: str = "mog"
    length: float = Field(15, ge=5, le=60)
    section_start: float | None = None
    intensity: float = Field(0.6, ge=0, le=1)
    seed: int = 1
    text: str | None = Field(None, max_length=120)
    fps: Literal[30, 60] = 30


class Edit(BaseModel):
    id: str
    created_at: float
    settings: EditSettings
    status: Literal["queued", "running", "done", "failed"] = "queued"
    job_id: str | None = None
    error: str | None = None
    warnings: list[str] = []
    preview: str | None = None      # storage key
    export: str | None = None       # storage key
    export_status: Literal["none", "queued", "running", "done", "failed"] = "none"
    export_job_id: str | None = None


class ProjectStore:
    def __init__(self, storage: Storage):
        self.s = storage

    # ---- keys
    @staticmethod
    def base(pid: str) -> str:
        if not pid.startswith("prj_") or not pid[4:].isalnum():
            raise MogError("Invalid project id.")
        return f"projects/{pid}"

    def _json(self, key: str, model):
        if not self.s.exists(key):
            raise MogError(f"Not found: {key.split('/')[-2] if '/' in key else key}")
        return model.model_validate_json(self.s.get_bytes(key))

    def _write(self, key: str, obj: BaseModel) -> None:
        self.s.put_bytes(key, obj.model_dump_json(indent=1).encode())

    # ---- projects
    def create(self, name: str | None = None) -> Project:
        p = Project(id=new_id("prj"), created_at=time.time(), name=name or "Untitled edit")
        self._write(f"{self.base(p.id)}/project.json", p)
        return p

    def get(self, pid: str) -> Project:
        return self._json(f"{self.base(pid)}/project.json", Project)

    def update(self, p: Project) -> None:
        self._write(f"{self.base(p.id)}/project.json", p)

    def list(self) -> list[Project]:
        out = []
        for pid in self.s.list("projects"):
            try:
                out.append(self.get(pid))
            except (MogError, ValueError):
                continue
        return sorted(out, key=lambda p: -p.created_at)

    def delete(self, pid: str) -> None:
        self.s.delete_prefix(self.base(pid))

    # ---- clips
    def clips(self, pid: str) -> list[ClipMeta]:
        out = []
        for cid in self.s.list(f"{self.base(pid)}/clips"):
            key = f"{self.base(pid)}/clips/{cid}/meta.json"
            if self.s.exists(key):
                out.append(self._json(key, ClipMeta))
        return sorted(out, key=lambda c: c.uploaded_at)

    def clip_key(self, pid: str, c: ClipMeta) -> str:
        return f"{self.base(pid)}/clips/{c.id}/{c.filename}"

    def clip_analysis(self, pid: str, cid: str) -> dict | None:
        key = f"{self.base(pid)}/clips/{cid}/analysis.json"
        return json.loads(self.s.get_bytes(key)) if self.s.exists(key) else None

    # ---- audio
    def audio(self, pid: str) -> AudioMeta | None:
        key = f"{self.base(pid)}/audio/meta.json"
        return self._json(key, AudioMeta) if self.s.exists(key) else None

    def audio_key(self, pid: str, a: AudioMeta) -> str:
        return f"{self.base(pid)}/audio/{a.filename}"

    def audio_analysis(self, pid: str) -> dict | None:
        key = f"{self.base(pid)}/audio/analysis.json"
        return json.loads(self.s.get_bytes(key)) if self.s.exists(key) else None

    # ---- edits
    def edit_key(self, pid: str, eid: str) -> str:
        if not eid.startswith("edt_") or not eid[4:].isalnum():
            raise MogError("Invalid edit id.")
        return f"{self.base(pid)}/edits/{eid}"

    def edits(self, pid: str) -> list[Edit]:
        out = []
        for eid in self.s.list(f"{self.base(pid)}/edits"):
            key = f"{self.base(pid)}/edits/{eid}/edit.json"
            if self.s.exists(key):
                out.append(self._json(key, Edit))
        return sorted(out, key=lambda e: -e.created_at)

    def get_edit(self, pid: str, eid: str) -> Edit:
        return self._json(f"{self.edit_key(pid, eid)}/edit.json", Edit)

    def save_edit(self, pid: str, e: Edit) -> None:
        self._write(f"{self.edit_key(pid, e.id)}/edit.json", e)

    def timeline(self, pid: str, eid: str) -> dict | None:
        key = f"{self.edit_key(pid, eid)}/timeline.json"
        return json.loads(self.s.get_bytes(key)) if self.s.exists(key) else None

    # ---- retention
    def expired(self, hours: float, now: float | None = None) -> list[str]:
        now = now or time.time()
        return [p.id for p in self.list() if not p.saved and now - p.created_at > hours * 3600]
