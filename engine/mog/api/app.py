"""HTTP API (FastAPI). Thin: validates input, stores uploads, enqueues jobs, serves results.

Run:  uvicorn mog.api.app:app --port 8000
All heavy work happens in the worker (`rq worker mog`), so requests return immediately and
the UI polls /api/jobs/{id} for stage + progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from mog.config import settings
from mog.errors import MogError
from mog.planner.presets import list_presets
from mog.projects import AUDIO_EXT, VIDEO_EXT, AudioMeta, ClipMeta, Edit, EditSettings, ProjectStore, new_id, safe_name
from mog.storage import Storage, default_storage
from mog.timeline.models import Timeline
from mog.worker import jobs
from mog.worker.queue import queue, redis_conn

log = logging.getLogger("mog.api")
_storage: Storage | None = None


def storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = default_storage()
    return _storage


def store() -> ProjectStore:
    return ProjectStore(storage())


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


async def _cleanup_loop() -> None:
    """Every hour: delete unsaved projects older than the retention window (24 h)."""
    while True:
        try:
            res = await asyncio.to_thread(jobs.cleanup_job, storage())
            if res["projects_removed"]:
                log.info("cleanup: %s", res)
        except Exception:  # noqa: BLE001 - cleanup must never take the API down
            log.exception("cleanup failed")
        await asyncio.sleep(3600)


app = FastAPI(title="MOG", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings().cors_origins), allow_methods=["*"],
                   allow_headers=["*"])


@app.exception_handler(MogError)
async def _mog_error(_req: Request, exc: MogError):
    status = 404 if str(exc).startswith("Not found") else 400
    return JSONResponse({"error": str(exc)}, status_code=status)


@app.exception_handler(ValidationError)
async def _validation_error(_req: Request, exc: ValidationError):
    msgs = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]]
    return JSONResponse({"error": "Invalid data: " + "; ".join(msgs)}, status_code=422)


# ----------------------------------------------------------------------------- rate limiting

def rate_limit(request: Request) -> None:
    """At most N heavy jobs per client IP per hour (simple fixed window in Redis)."""
    ip = request.client.host if request.client else "unknown"
    key = f"mog:rl:{ip}:{int(time.time() // 3600)}"
    r = redis_conn()
    n = r.incr(key)
    if n == 1:
        r.expire(key, 3700)
    if n > settings().jobs_per_hour:
        raise _TooMany()


class _TooMany(Exception):
    pass


@app.exception_handler(_TooMany)
async def _too_many(_req: Request, _exc: _TooMany):
    return JSONResponse({"error": "Too many renders this hour. Please wait a bit and try again."},
                        status_code=429)


def _enqueue(fn, *args, job_id: str | None = None) -> str:
    job = queue().enqueue(fn, *args, job_id=job_id, job_timeout=60 * 30, result_ttl=24 * 3600,
                          failure_ttl=24 * 3600)
    return job.id


def _job_id() -> str:
    # Chosen BEFORE enqueueing and saved with the edit first: the worker can finish the job
    # before the API's next write, which would otherwise overwrite its result.
    return new_id("job")


def url(key: str | None) -> str | None:
    return f"/api/files/{key}" if key else None


# ----------------------------------------------------------------------------- basics

@app.get("/api/health")
def health() -> dict:
    try:
        redis_ok = bool(redis_conn().ping())
    except Exception:  # noqa: BLE001
        redis_ok = False
    return {"ok": True, "redis": redis_ok}


@app.get("/api/presets")
def presets() -> list[dict]:
    return [{"id": p.id, "name": p.name, "description": p.description,
             "preview": f"/presets/{p.id}.mp4" if p.preview is None else p.preview} for p in list_presets()]


@app.get("/api/limits")
def limits() -> dict:
    s = settings()
    return {"max_clips": s.max_clips, "max_upload_bytes": s.max_upload_bytes, "max_audio_bytes": s.max_audio_bytes,
            "video_types": sorted(VIDEO_EXT), "audio_types": sorted(AUDIO_EXT | VIDEO_EXT),
            "retention_hours": s.retention_hours}


# ----------------------------------------------------------------------------- projects

class NewProject(BaseModel):
    name: str | None = Field(None, max_length=80)


@app.post("/api/projects")
def create_project(body: NewProject | None = None) -> dict:
    return store().create(body.name if body else None).model_dump()


@app.get("/api/projects")
def recent_projects() -> list[dict]:
    ps = store()
    out = []
    for p in ps.list()[:30]:
        edits = ps.edits(p.id)
        thumb = next((c.id for c in ps.clips(p.id)), None)
        out.append({**p.model_dump(), "edits": len(edits),
                    "thumb": url(f"{ps.base(p.id)}/clips/{thumb}/thumb.jpg") if thumb else None,
                    "latest_preview": url(next((e.preview for e in edits if e.preview), None))})
    return out


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    ps = store()
    p = ps.get(pid)
    clips = []
    for c in ps.clips(pid):
        an = ps.clip_analysis(pid, c.id)
        clips.append({**c.model_dump(), "url": url(ps.clip_key(pid, c)),
                      "thumb": url(f"{ps.base(pid)}/clips/{c.id}/thumb.jpg") if an else None,
                      "analysed": an is not None,
                      "duration": an["duration"] if an else None,
                      "shots": len(an["shots"]) if an else None,
                      "warnings": an["warnings"] if an else []})
    audio = ps.audio(pid)
    adata = ps.audio_analysis(pid)
    return {**p.model_dump(), "clips": clips,
            "audio": ({**audio.model_dump(), "analysed": adata is not None,
                       "playback": url(adata["playback"]) if adata else None} if audio else None),
            "edits": [_edit_out(pid, e) for e in ps.edits(pid)]}


@app.post("/api/projects/{pid}/save")
def save_project(pid: str) -> dict:
    ps = store()
    p = ps.get(pid)
    p.saved = True
    ps.update(p)
    return p.model_dump()


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict:
    ps = store()
    ps.get(pid)
    ps.delete(pid)
    return {"deleted": pid}


# ----------------------------------------------------------------------------- uploads

@app.post("/api/projects/{pid}/clips")
def upload_clip(pid: str, file: UploadFile = File(...), _rl: None = Depends(rate_limit)) -> dict:
    ps, s, cfg = store(), storage(), settings()
    ps.get(pid)
    name = safe_name(file.filename or "clip.mp4")
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise MogError(f"{name}: unsupported file type. Use MP4 or MOV.")
    existing = ps.clips(pid)
    if len(existing) >= cfg.max_clips:
        raise MogError(f"You can add up to {cfg.max_clips} clips.")
    room = cfg.max_upload_bytes - sum(c.size for c in existing)
    if room <= 0:
        raise MogError(f"Clips are limited to {cfg.max_upload_bytes / 1024**3:.0f} GB in total.")
    cid = new_id("clp")
    key = f"{ps.base(pid)}/clips/{cid}/{name}"
    try:
        size = s.put_stream(key, file.file, room)
    except MogError as e:
        s.delete_prefix(f"{ps.base(pid)}/clips/{cid}")
        raise MogError(f"{name}: {e} Total clip limit is {cfg.max_upload_bytes / 1024**3:.0f} GB.") from None
    meta = ClipMeta(id=cid, filename=name, size=size, uploaded_at=time.time())
    s.put_bytes(f"{ps.base(pid)}/clips/{cid}/meta.json", meta.model_dump_json().encode())
    job = _enqueue(jobs.scout_clip_job, pid, cid)
    return {**meta.model_dump(), "job_id": job}


@app.delete("/api/projects/{pid}/clips/{cid}")
def delete_clip(pid: str, cid: str) -> dict:
    ps = store()
    if not cid.startswith("clp_") or not cid[4:].isalnum():
        raise MogError("Invalid clip id.")
    storage().delete_prefix(f"{ps.base(pid)}/clips/{cid}")
    return {"deleted": cid}


@app.post("/api/projects/{pid}/audio")
def upload_audio(pid: str, file: UploadFile = File(...), _rl: None = Depends(rate_limit)) -> dict:
    """A song (MP3/WAV/M4A...) or a video whose soundtrack should be used."""
    ps, s, cfg = store(), storage(), settings()
    ps.get(pid)
    name = safe_name(file.filename or "track.mp3")
    if Path(name).suffix.lower() not in AUDIO_EXT | VIDEO_EXT:
        raise MogError(f"{name}: unsupported file type. Use MP3, WAV or M4A (or a video to take its sound).")
    s.delete_prefix(f"{ps.base(pid)}/audio")
    size = s.put_stream(f"{ps.base(pid)}/audio/{name}", file.file, cfg.max_audio_bytes)
    meta = AudioMeta(filename=name, size=size, uploaded_at=time.time())
    s.put_bytes(f"{ps.base(pid)}/audio/meta.json", meta.model_dump_json().encode())
    job = _enqueue(jobs.analyze_audio_job, pid)
    return {**meta.model_dump(), "job_id": job}


@app.get("/api/projects/{pid}/audio/analysis")
def audio_analysis(pid: str) -> dict:
    ps = store()
    data = ps.audio_analysis(pid)
    if data is None:
        raise MogError("Not found: the song hasn't been analysed yet.")
    an = data["analysis"]
    return {"duration": data["duration"], "bpm": an["bpm"], "beats": an["beats"], "downbeats": an["downbeats"],
            "drop": an["drop"], "sections": data["sections"], "waveform": data["waveform"],
            "playback": url(data["playback"]), "warnings": an["warnings"]}


# ----------------------------------------------------------------------------- edits

def _edit_out(pid: str, e: Edit) -> dict:
    return {**e.model_dump(), "preview_url": url(e.preview), "export_url": url(e.export)}


@app.post("/api/projects/{pid}/edits")
def create_edit(pid: str, body: EditSettings, _rl: None = Depends(rate_limit)) -> dict:
    ps = store()
    ps.get(pid)
    if ps.audio(pid) is None:
        raise MogError("Upload a song first.")
    if not ps.clips(pid):
        raise MogError("Upload at least one video clip.")
    from mog.planner.presets import load_preset

    load_preset(body.preset)  # fail fast on an unknown preset
    e = Edit(id=new_id("edt"), created_at=time.time(), settings=body, job_id=_job_id())
    ps.save_edit(pid, e)
    _enqueue(jobs.make_edit_job, pid, e.id, job_id=e.job_id)
    return _edit_out(pid, ps.get_edit(pid, e.id))


@app.get("/api/projects/{pid}/edits/{eid}")
def get_edit(pid: str, eid: str) -> dict:
    ps = store()
    return {**_edit_out(pid, ps.get_edit(pid, eid)), "timeline": ps.timeline(pid, eid)}


@app.put("/api/projects/{pid}/edits/{eid}/timeline")
def update_timeline(pid: str, eid: str, tl: Timeline, _rl: None = Depends(rate_limit)) -> dict:
    """The editor saves a changed timeline; the preview is re-rendered from it."""
    ps, s = store(), storage()
    e = ps.get_edit(pid, eid)
    base = s.local_path(ps.base(pid))
    missing = sorted({seg.source for seg in tl.segments if not (base / seg.source).exists()})
    if missing or not (base / tl.audio.file).exists():
        raise MogError(f"The timeline refers to files that aren't in this project: {', '.join(missing) or tl.audio.file}")
    s.put_bytes(f"{ps.edit_key(pid, eid)}/timeline.json", tl.model_dump_json(indent=1).encode())
    e.status, e.error, e.job_id = "queued", None, _job_id()
    ps.save_edit(pid, e)
    _enqueue(jobs.render_preview_job, pid, eid, job_id=e.job_id)
    return _edit_out(pid, ps.get_edit(pid, eid))


class ExportBody(BaseModel):
    fps: int = Field(30, description="30 or 60")
    quality: str = Field("high", description="fast or high")


@app.post("/api/projects/{pid}/edits/{eid}/export")
def export(pid: str, eid: str, body: ExportBody, _rl: None = Depends(rate_limit)) -> dict:
    if body.fps not in (30, 60) or body.quality not in ("fast", "high"):
        raise MogError("Export settings: fps must be 30 or 60, quality fast or high.")
    ps = store()
    e = ps.get_edit(pid, eid)
    if ps.timeline(pid, eid) is None:
        raise MogError("This edit hasn't been generated yet.")
    e.export_status, e.export, e.export_job_id = "queued", None, _job_id()
    ps.save_edit(pid, e)
    _enqueue(jobs.export_job, pid, eid, body.fps, body.quality, job_id=e.export_job_id)
    return _edit_out(pid, ps.get_edit(pid, eid))


# ----------------------------------------------------------------------------- jobs + files

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    try:
        job = Job.fetch(job_id, connection=redis_conn())
    except NoSuchJobError:
        raise MogError("Not found: job.") from None
    status = job.get_status(refresh=True)
    status = getattr(status, "value", status)
    meta = job.meta or {}
    out = {"id": job.id, "status": status, "stage": meta.get("stage", "Queued"),
           "progress": meta.get("progress", 0.0), "error": meta.get("error")}
    if status == "failed" and not out["error"]:
        out["error"] = "The job failed unexpectedly."
    if status == "finished":
        out["progress"] = 1.0
    return out


@app.get("/api/files/{key:path}")
def files(key: str):
    if not key.startswith("projects/"):
        raise MogError("Not found: file.")
    p = storage().local_path(key)
    if not p.is_file():
        raise MogError("Not found: file.")
    media = {".mp4": "video/mp4", ".m4a": "audio/mp4", ".jpg": "image/jpeg", ".json": "application/json"}
    return FileResponse(p, media_type=media.get(p.suffix.lower()), headers={"Cache-Control": "private, max-age=60"})


# ----------------------------------------------------------------------------- editor

def _project_inputs(pid: str):
    """Audio analysis + clip analyses with project-relative sources (no recomputation)."""
    from mog.audio.models import AudioAnalysis
    from mog.video.models import VideoAnalysis

    ps = store()
    adata = ps.audio_analysis(pid)
    if adata is None:
        raise MogError("The song is still being analysed.")
    clips = []
    for c in ps.clips(pid):
        a = ps.clip_analysis(pid, c.id)
        if a is None:
            raise MogError(f"{c.filename} is still being analysed.")
        clips.append(VideoAnalysis.model_validate(a).model_copy(update={"source": f"clips/{c.id}/{c.filename}"}))
    return AudioAnalysis.model_validate(adata["analysis"]), clips


class EditOp(BaseModel):
    op: str = Field(description="swap | speed | effect | nudge | text")
    segment: str | None = None
    source: str | None = None        # swap: clip source path, e.g. clips/<cid>/<file>
    shot: int | None = None          # swap: shot index
    mode: str | None = None          # speed: normal | slow | fast | ramp
    effect: str | None = None        # effect: flash | zoom_punch | shake | rgb_split | motion_blur
    on: bool = True
    edge: str | None = None          # nudge: in | out
    beats: int = 1                   # nudge: +-1
    content: str | None = None       # text


@app.post("/api/projects/{pid}/edits/{eid}/ops")
def edit_op(pid: str, eid: str, body: EditOp) -> dict:
    """Apply one editor action to the timeline and save it (no render; see /render)."""
    from mog.planner import edit_ops as ops
    from mog.planner.presets import load_preset

    ps, s = store(), storage()
    e = ps.get_edit(pid, eid)
    raw = ps.timeline(pid, eid)
    if raw is None:
        raise MogError("This edit hasn't been generated yet.")
    tl = Timeline.model_validate(raw)
    audio, clips = _project_inputs(pid)
    preset = load_preset(e.settings.preset)
    need_seg = body.op in ("swap", "speed", "effect", "nudge")
    if need_seg and not body.segment:
        raise MogError("Pick a segment first.")
    if body.op == "swap":
        tl = ops.swap_shot(tl, body.segment, clips, body.source, body.shot)
    elif body.op == "speed":
        if body.mode not in ("normal", "slow", "fast", "ramp"):
            raise MogError("Speed must be normal, slow, fast or ramp.")
        tl = ops.set_speed(tl, body.segment, body.mode, clips, preset)
    elif body.op == "effect":
        if body.effect not in ("flash", "zoom_punch", "shake", "rgb_split", "motion_blur"):
            raise MogError("Unknown effect.")
        tl = ops.toggle_effect(tl, body.segment, body.effect, body.on, preset)
    elif body.op == "nudge":
        if body.edge not in ("in", "out") or body.beats not in (-1, 1):
            raise MogError("Nudge a cut by one beat at a time (edge in/out, beats -1/+1).")
        tl = ops.nudge_cut(tl, body.segment, body.edge, body.beats, audio, clips)
    elif body.op == "text":
        drop = None
        if audio.drop is not None:
            drop = audio.drop.t - tl.audio.start
        tl = ops.set_text(tl, body.content, preset, drop)
    else:
        raise MogError("Unknown edit action.")
    s.put_bytes(f"{ps.edit_key(pid, eid)}/timeline.json", tl.model_dump_json(indent=1).encode())
    return {"timeline": tl.model_dump()}


@app.post("/api/projects/{pid}/edits/{eid}/render")
def rerender(pid: str, eid: str, _rl: None = Depends(rate_limit)) -> dict:
    """Render a new preview from the (edited) timeline."""
    ps = store()
    e = ps.get_edit(pid, eid)
    if ps.timeline(pid, eid) is None:
        raise MogError("This edit hasn't been generated yet.")
    e.status, e.error, e.job_id = "queued", None, _job_id()
    ps.save_edit(pid, e)
    _enqueue(jobs.render_preview_job, pid, eid, job_id=e.job_id)
    return _edit_out(pid, ps.get_edit(pid, eid))


@app.post("/api/projects/{pid}/edits/{eid}/reroll")
def reroll(pid: str, eid: str, _rl: None = Depends(rate_limit)) -> dict:
    """Same settings, new seed: a different edit."""
    ps = store()
    old = ps.get_edit(pid, eid)
    settings_ = old.settings.model_copy(update={"seed": old.settings.seed + 1})
    e = Edit(id=new_id("edt"), created_at=time.time(), settings=settings_, job_id=_job_id())
    ps.save_edit(pid, e)
    _enqueue(jobs.make_edit_job, pid, e.id, job_id=e.job_id)
    return _edit_out(pid, ps.get_edit(pid, e.id))


@app.get("/api/projects/{pid}/shots")
def shots(pid: str) -> list[dict]:
    """Every shot of every clip (for 'swap shot')."""
    ps = store()
    out = []
    for c in ps.clips(pid):
        a = ps.clip_analysis(pid, c.id)
        if a is None:
            continue
        for sh in a["shots"]:
            out.append({"source": f"clips/{c.id}/{c.filename}", "clip": c.filename, "shot": sh["index"],
                        "start": sh["start"], "end": sh["end"], "stare": sh["stare_score"],
                        "action": sh["action_score"], "thumb": url(f"{ps.base(pid)}/clips/{c.id}/thumb.jpg")})
    return out


@app.get("/api/projects/{pid}/looks")
def looks(pid: str) -> list[dict]:
    """Each preset's grade applied to a still from the user's own footage (for the style picker)."""
    import subprocess

    import cv2

    from mog.color.grade_lut import bake
    from mog.ffmpeg.audio_io import ffmpeg_bin
    from mog.render.pixel_fx import Glow, Vignette

    ps, s = store(), storage()
    clips = ps.clips(pid)
    thumb_key = next((f"{ps.base(pid)}/clips/{c.id}/thumb.jpg" for c in clips
                      if s.exists(f"{ps.base(pid)}/clips/{c.id}/thumb.jpg")), None)
    out = []
    for p in list_presets():
        key = f"{ps.base(pid)}/looks/{p.id}.jpg"
        if thumb_key and not s.exists(key):
            src = s.local_path(thumb_key)
            dst = s.local_path(key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            if p.grade.glow > 0:
                img = Glow(w, h, p.grade.glow)(img)
            if p.grade.vignette > 0:
                img = Vignette(w, h, p.grade.vignette)(img)
            tmp = dst.with_suffix(".pre.png")
            cv2.imwrite(str(tmp), img)
            lut = bake(p.grade)
            vf = f"lut3d=file={lut}" if lut else "null"
            subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-i", str(tmp), "-vf", vf, "-q:v", "4", str(dst)],
                           capture_output=True)
            tmp.unlink(missing_ok=True)
        out.append({"id": p.id, "name": p.name, "description": p.description,
                    "image": url(key) if s.exists(key) else None, "preview": f"/presets/{p.id}.mp4"})
    return out
