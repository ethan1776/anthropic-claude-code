"""API + worker end to end (in-memory Redis, jobs run inline)."""

import importlib
import io
import time

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from mog.testing.clips import make_demo_clips
from mog.testing.synth import make_track


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    mp = pytest.MonkeyPatch()
    mp.setenv("MOG_DATA", str(data))
    mp.setenv("MOG_FAKE_REDIS", "1")
    mp.setenv("MOG_SYNC_JOBS", "1")
    mp.setenv("MOG_MAX_CLIPS", "4")
    mp.setenv("MOG_JOBS_PER_HOUR", "40")
    import mog.api.app as app_mod
    import mog.worker.queue as q

    q.redis_conn.cache_clear()
    importlib.reload(app_mod)
    with TestClient(app_mod.app) as c:
        yield c, app_mod
    mp.undo()


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("in")
    clips = make_demo_clips(d)
    y, _ = make_track(bpm=145, sr=44100)
    sf.write(d / "song.wav", y, 44100)
    return clips, d / "song.wav"


def test_full_flow(client, media):
    c, _ = client
    clips, song = media
    assert c.get("/api/health").json()["redis"] is True
    assert {p["id"] for p in c.get("/api/presets").json()} >= {"mog", "dark_triad", "velocity"}
    pid = c.post("/api/projects", json={"name": "test"}).json()["id"]

    for p in clips[:3]:
        r = c.post(f"/api/projects/{pid}/clips", files={"file": (p.name, p.read_bytes(), "video/mp4")})
        assert r.status_code == 200, r.text
        assert c.get(f"/api/jobs/{r.json()['job_id']}").json()["status"] == "finished"
    r = c.post(f"/api/projects/{pid}/audio", files={"file": ("song.wav", song.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text

    proj = c.get(f"/api/projects/{pid}").json()
    assert len(proj["clips"]) == 3 and all(cl["analysed"] for cl in proj["clips"])
    assert c.get(proj["clips"][0]["thumb"]).status_code == 200
    an = c.get(f"/api/projects/{pid}/audio/analysis").json()
    assert 140 < an["bpm"] < 150 and len(an["waveform"]) == 1200 and an["sections"]["15"]["drop"]
    assert c.get(an["playback"]).headers["content-type"] == "audio/mp4"

    edit = c.post(f"/api/projects/{pid}/edits", json={"preset": "velocity", "length": 6, "seed": 2,
                                                      "text": "test"}).json()
    job = c.get(f"/api/jobs/{edit['job_id']}").json()
    assert job["status"] == "finished" and job["progress"] == 1.0, job
    full = c.get(f"/api/projects/{pid}/edits/{edit['id']}").json()
    assert full["status"] == "done" and full["timeline"]["segments"]
    # video streams with HTTP range requests (needed for seeking in the browser)
    r = c.get(full["preview_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100

    # editor: remove all effects and change a speed, then re-render
    tl = full["timeline"]
    for s in tl["segments"]:
        s["effects"] = []
    tl["segments"][0]["speed_curve"] = [[0, 0.5]]
    r = c.put(f"/api/projects/{pid}/edits/{edit['id']}/timeline", json=tl)
    assert r.status_code == 200, r.text
    again = c.get(f"/api/projects/{pid}/edits/{edit['id']}").json()
    assert again["status"] == "done" and all(not s["effects"] for s in again["timeline"]["segments"])

    r = c.post(f"/api/projects/{pid}/edits/{edit['id']}/export", json={"fps": 30, "quality": "fast"})
    assert r.status_code == 200, r.text
    done = c.get(f"/api/projects/{pid}/edits/{edit['id']}").json()
    assert done["export_status"] == "done"
    head = c.get(done["export_url"]).content[:4096]
    assert b"ftyp" in head and b"moov" in head  # +faststart: index at the front


def test_errors_are_clear(client, media):
    c, _ = client
    clips, song = media
    pid = c.post("/api/projects").json()["id"]
    r = c.post(f"/api/projects/{pid}/clips", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "unsupported file type" in r.json()["error"]
    r = c.post(f"/api/projects/{pid}/edits", json={"preset": "mog"})
    assert r.status_code == 400 and "song" in r.json()["error"]
    c.post(f"/api/projects/{pid}/audio", files={"file": ("song.wav", song.read_bytes(), "audio/wav")})
    for p in clips[:4]:
        c.post(f"/api/projects/{pid}/clips", files={"file": (p.name, p.read_bytes(), "video/mp4")})
    r = c.post(f"/api/projects/{pid}/clips", files={"file": (clips[0].name, clips[0].read_bytes(), "video/mp4")})
    assert r.status_code == 400 and "up to 4 clips" in r.json()["error"]
    r = c.post(f"/api/projects/{pid}/edits", json={"preset": "nope"})
    assert r.status_code == 400 and "Unknown preset" in r.json()["error"]
    r = c.post(f"/api/projects/{pid}/edits", json={"preset": "mog", "length": 2})
    assert r.status_code == 422
    # a broken file named .mp4 is accepted, then its analysis job fails with a readable message
    pid2 = c.post("/api/projects").json()["id"]
    r = c.post(f"/api/projects/{pid2}/clips", files={"file": ("x.mp4", b"not a video", "video/mp4")})
    st = c.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert st["status"] == "failed" and "doesn't look like a video" in st["error"]


def test_failed_job_reports_message(client):
    c, app_mod = client
    pid = c.post("/api/projects").json()["id"]
    from mog.worker import jobs
    from mog.worker.queue import queue

    job = queue().enqueue(jobs.analyze_audio_job, pid)  # no song uploaded
    st = c.get(f"/api/jobs/{job.id}").json()
    assert st["status"] == "failed" and st["error"] == "Upload a song first."


def test_paths_cannot_escape(client):
    c, _ = client
    assert c.get("/api/files/projects/../../etc/passwd").status_code in (400, 404)
    assert c.get("/api/files/secret.txt").status_code == 404
    assert c.get("/api/projects/prj_..%2F..").status_code in (400, 404)


def test_rate_limit(client):
    c, app_mod = client
    from mog.worker.queue import redis_conn

    pid = c.post("/api/projects").json()["id"]
    key = f"mog:rl:testclient:{int(time.time() // 3600)}"
    redis_conn().set(key, 10_000)
    r = c.post(f"/api/projects/{pid}/edits", json={"preset": "mog"})
    assert r.status_code == 429
    redis_conn().delete(key)


def test_cleanup_removes_only_old_unsaved_projects(client):
    c, app_mod = client
    from mog.worker.jobs import cleanup_job

    old = c.post("/api/projects").json()["id"]
    kept = c.post("/api/projects").json()["id"]
    c.post(f"/api/projects/{kept}/save")
    res = cleanup_job(app_mod.storage(), now=time.time() + 25 * 3600)
    assert old in res["projects_removed"] and kept not in res["projects_removed"]
    assert c.get(f"/api/projects/{old}").status_code == 404
    assert c.get(f"/api/projects/{kept}").status_code == 200


def test_upload_size_limit(client, monkeypatch):
    c, app_mod = client
    monkeypatch.setenv("MOG_MAX_AUDIO_BYTES", "1000")
    pid = c.post("/api/projects").json()["id"]
    r = c.post(f"/api/projects/{pid}/audio", files={"file": ("big.wav", io.BytesIO(b"0" * 5000), "audio/wav")})
    assert r.status_code == 400 and "too large" in r.json()["error"]
    assert not app_mod.store().audio(pid)
