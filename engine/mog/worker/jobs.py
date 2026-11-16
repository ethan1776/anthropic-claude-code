"""Background jobs run by the RQ worker. Each reports {stage, progress} in its job meta (the
UI polls it) and stores a plain-English error message on failure (never a raw traceback)."""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import time
import traceback

import numpy as np

from mog.audio import analyze_audio, pick_section
from mog.audio.models import AudioAnalysis
from mog.config import settings
from mog.errors import MogError
from mog.ffmpeg.audio_io import decode_audio, ffmpeg_bin
from mog.pipeline import _manual_section
from mog.planner.planner import PlanOptions, plan_edit
from mog.planner.presets import load_preset
from mog.projects import ProjectStore
from mog.render.renderer import render
from mog.storage import Storage, default_storage
from mog.timeline.models import Timeline
from mog.video.analyze import analyze_video
from mog.video.models import VideoAnalysis

log = logging.getLogger(__name__)
WAVEFORM_BINS = 1200
SECTION_LENGTHS = (10, 15, 20, 30)


def _report(stage: str, progress: float) -> None:
    from rq import get_current_job

    job = get_current_job()
    if job is not None:
        job.meta.update({"stage": stage, "progress": round(float(progress), 4)})
        job.save_meta()


def _job(fn):
    """Turn exceptions into a clear message in job meta, then re-raise so RQ marks it failed."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            from rq import get_current_job

            msg = str(exc) if isinstance(exc, MogError) else f"Something went wrong: {type(exc).__name__}: {exc}"
            if not isinstance(exc, MogError):
                log.error("job %s failed:\n%s", fn.__name__, traceback.format_exc())
            job = get_current_job()
            if job is not None:
                job.meta.update({"error": msg, "stage": "Failed"})
                job.save_meta()
            raise
    return wrapper


def _store(storage: Storage | None) -> tuple[Storage, ProjectStore]:
    s = storage or default_storage()
    return s, ProjectStore(s)


# ----------------------------------------------------------------------------- audio

@_job
def analyze_audio_job(pid: str, storage: Storage | None = None) -> dict:
    s, ps = _store(storage)
    meta = ps.audio(pid)
    if meta is None:
        raise MogError("Upload a song first.")
    path = s.local_path(ps.audio_key(pid, meta))
    _report("Finding the beat", 0.1)
    an = analyze_audio(path)
    _report("Finding the beat", 0.7)
    sections = {str(L): pick_section(an, L).model_dump() for L in SECTION_LENGTHS}
    y = np.abs(decode_audio(path, sr=8000))
    bins = np.array_split(y, WAVEFORM_BINS)
    peaks = [round(float(b.max()) if len(b) else 0.0, 3) for b in bins]
    base = ps.base(pid)
    # Browser-friendly playback copy (works for audio extracted from a video upload too)
    playback = s.local_path(f"{base}/audio/playback.m4a")
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-i", str(path), "-vn", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(playback)], check=True, capture_output=True)
    result = {"analysis": an.model_dump(), "sections": sections, "waveform": peaks,
              "duration": an.duration, "playback": f"{base}/audio/playback.m4a"}
    s.put_bytes(f"{base}/audio/analysis.json", json.dumps(result).encode())
    _report("Done", 1.0)
    return {"bpm": an.bpm, "drop": an.drop.t if an.drop else None}


# ----------------------------------------------------------------------------- clips

@_job
def scout_clip_job(pid: str, cid: str, storage: Storage | None = None) -> dict:
    s, ps = _store(storage)
    clip = next((c for c in ps.clips(pid) if c.id == cid), None)
    if clip is None:
        raise MogError("Clip not found.")
    path = s.local_path(ps.clip_key(pid, clip))
    _report("Scouting shots", 0.1)
    va = analyze_video(path)
    base = f"{ps.base(pid)}/clips/{cid}"
    # Thumbnail from the best shot (by stare or action score)
    best = max(va.shots, key=lambda sh: max(sh.stare_score, sh.action_score, 0.01) * sh.quality + 1e-3)
    thumb = s.local_path(f"{base}/thumb.jpg")
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{(best.start + best.end) / 2:.3f}", "-i", str(path),
                    "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(thumb)], capture_output=True)
    s.put_bytes(f"{base}/analysis.json", va.model_dump_json().encode())
    _report("Done", 1.0)
    return {"shots": len(va.shots), "duration": va.duration}


# ----------------------------------------------------------------------------- edits

def _load_inputs(s: Storage, ps: ProjectStore, pid: str) -> tuple[AudioAnalysis, dict, list[VideoAnalysis]]:
    meta = ps.audio(pid)
    if meta is None:
        raise MogError("Upload a song first.")
    if ps.audio_analysis(pid) is None:
        analyze_audio_job.__wrapped__(pid, s)
    adata = ps.audio_analysis(pid)
    an = AudioAnalysis.model_validate(adata["analysis"])
    clips = ps.clips(pid)
    if not clips:
        raise MogError("Upload at least one video clip.")
    analyses = []
    for i, c in enumerate(clips):
        if ps.clip_analysis(pid, c.id) is None:
            _report("Scouting shots", 0.1 + 0.15 * i / len(clips))
            scout_clip_job.__wrapped__(pid, c.id, s)
        va = VideoAnalysis.model_validate(ps.clip_analysis(pid, c.id))
        # Timeline sources are paths relative to the project folder
        analyses.append(va.model_copy(update={"source": f"clips/{c.id}/{c.filename}"}))
    return an, adata, analyses


@_job
def make_edit_job(pid: str, eid: str, storage: Storage | None = None) -> dict:
    s, ps = _store(storage)
    edit = ps.get_edit(pid, eid)
    edit.status = "running"
    ps.save_edit(pid, edit)
    try:
        st = edit.settings
        _report("Finding the beat", 0.02)
        an, adata, analyses = _load_inputs(s, ps, pid)
        sec = (_manual_section(an, st.section_start, st.length) if st.section_start is not None
               else pick_section(an, st.length))
        _report("Directing", 0.27)
        meta = ps.audio(pid)
        tl, warnings = plan_edit(an, sec, analyses, load_preset(st.preset),
                                 PlanOptions(seed=st.seed, intensity=st.intensity, fps=st.fps, text=st.text,
                                             audio_file=f"audio/{meta.filename}"))
        key = ps.edit_key(pid, eid)
        s.put_bytes(f"{key}/timeline.json", tl.model_dump_json(indent=1).encode())
        res = _render(s, ps, pid, tl, f"{key}/preview.mp4", preview=True, start=0.3)
        edit = ps.get_edit(pid, eid)
        edit.status, edit.preview = "done", f"{key}/preview.mp4"
        edit.warnings = list(dict.fromkeys(warnings + res.warnings))
        ps.save_edit(pid, edit)
        _report("Done", 1.0)
        return {"preview": edit.preview}
    except Exception as exc:
        edit = ps.get_edit(pid, eid)
        edit.status, edit.error = "failed", str(exc) if isinstance(exc, MogError) else "Rendering failed."
        ps.save_edit(pid, edit)
        raise


@_job
def render_preview_job(pid: str, eid: str, storage: Storage | None = None) -> dict:
    """Re-render the preview after the user changed the timeline in the editor."""
    s, ps = _store(storage)
    edit = ps.get_edit(pid, eid)
    edit.status = "running"
    ps.save_edit(pid, edit)
    try:
        tl = Timeline.model_validate(ps.timeline(pid, eid))
        key = ps.edit_key(pid, eid)
        res = _render(s, ps, pid, tl, f"{key}/preview.mp4", preview=True, start=0.0)
        edit = ps.get_edit(pid, eid)
        edit.status, edit.preview, edit.error = "done", f"{key}/preview.mp4", None
        edit.export, edit.export_status = None, "none"      # the old export no longer matches
        edit.warnings = res.warnings
        ps.save_edit(pid, edit)
        _report("Done", 1.0)
        return {"preview": edit.preview}
    except Exception as exc:
        edit = ps.get_edit(pid, eid)
        edit.status, edit.error = "failed", str(exc) if isinstance(exc, MogError) else "Rendering failed."
        ps.save_edit(pid, edit)
        raise


@_job
def export_job(pid: str, eid: str, fps: int = 30, quality: str = "high", storage: Storage | None = None) -> dict:
    s, ps = _store(storage)
    edit = ps.get_edit(pid, eid)
    edit.export_status = "running"
    ps.save_edit(pid, edit)
    try:
        tl = Timeline.model_validate(ps.timeline(pid, eid))
        tl = tl.model_copy(update={"output": tl.output.model_copy(update={"fps": fps})})
        key = ps.edit_key(pid, eid)
        _render(s, ps, pid, tl, f"{key}/export.mp4", preview=False, start=0.0, quality=quality)
        edit = ps.get_edit(pid, eid)
        edit.export_status, edit.export = "done", f"{key}/export.mp4"
        ps.save_edit(pid, edit)
        _report("Done", 1.0)
        return {"export": edit.export}
    except Exception:
        edit = ps.get_edit(pid, eid)
        edit.export_status = "failed"
        ps.save_edit(pid, edit)
        raise


def _render(s: Storage, ps: ProjectStore, pid: str, tl: Timeline, out_key: str, preview: bool, start: float,
            quality: str | None = None):
    media = s.local_path(ps.base(pid))
    out = s.local_path(out_key)

    def prog(stage: str, frac: float) -> None:
        if stage == "Rendering":
            _report("Rendering", start + (1 - start) * 0.97 * frac)
        elif stage == "Encoding":
            _report("Rendering", start + (1 - start) * 0.98)
        else:
            _report("Preparing clips", start)

    return render(tl, out, media, preview=preview, progress=prog, quality=quality)


# ----------------------------------------------------------------------------- cleanup

def cleanup_job(storage: Storage | None = None, now: float | None = None) -> dict:
    """Delete unsaved projects older than the retention window, and stale cache files."""
    from mog.video.ingest import cache_dir

    s, ps = _store(storage)
    removed = ps.expired(settings().retention_hours, now)
    for pid in removed:
        ps.delete(pid)
    cutoff = (now or time.time()) - 48 * 3600
    stale = 0
    root = cache_dir().parent
    for sub in ("ingest", "analysis", "text"):
        d = root / sub
        if d.is_dir():
            for f in d.iterdir():
                try:
                    if f.is_file() and f.stat().st_atime < cutoff and f.stat().st_mtime < cutoff:
                        f.unlink()
                        stale += 1
                except OSError:
                    pass
    return {"projects_removed": removed, "cache_files_removed": stale}
