# MOG: auto phonk / velocity edits

Upload a few clips and a phonk track, pick a style, and get a beat-synced, vertical (9:16)
edit ready for TikTok, Reels and Shorts. MOG finds the beat and the drop, scouts your
footage for faces and motion, directs the edit (cuts on the beat, velocity ramps, slow-mo
stare on the drop, flashes, punches, shake), grades it and renders it.

> Only use clips and music you have the rights to post. MOG only edits files you upload;
> it never downloads from other sites. Unsaved projects are deleted after 24 hours.

## Run it (one command)

Needs [Docker](https://docs.docker.com/get-docker/) with Compose.

```bash
docker compose up --build
```

Then open **http://localhost:3000**. The first build takes a few minutes. Data lives in the
`data` Docker volume.

| Service | What it does | Port |
|---|---|---|
| `web` | Next.js app (what you open in the browser) | 3000 |
| `api` | FastAPI: uploads, projects, jobs | 8000 |
| `worker` | Analyses audio/video and renders edits | – |
| `redis` | Job queue | – |

More render capacity: `docker compose up --scale worker=2`.
More accurate beat/downbeat tracking (adds PyTorch, ~700 MB):
`docker compose build --build-arg INSTALL_BEATS=1`.

## Command line (no Docker)

Python 3.11+ and FFmpeg 6+.

```bash
cd engine
pip install -e ".[dev,server]"

# A whole edit in one go
python -m mog make --clips a.mp4 b.mov c.mp4 --audio track.mp3 --preset mog --length 15 \
                   --text "he doesn't chase." --out edit.mp4
python -m mog presets                      # dark_triad, mog, velocity, drift_phonk, clean_sigma

# The steps on their own
python -m mog debug track.mp3 --out out/    # beat/drop plot + click track to check by ear
python -m mog scout clip.mp4 --debug out/   # shots, faces, scores, crop preview
python -m mog reframe clip.mp4 --out v.mp4  # auto-reframe a horizontal clip to 9:16
python -m mog render timeline.json --media DIR --out v.mp4 [--preview]
```

Options for `make`:
- `--seed N`: a different edit with the same settings (the app's "Re-roll").
- `--intensity 0..1`: more or fewer effects.
- `--start S`: choose the song section yourself.
- `--fps 30|60`, `--preview`: half resolution, fast.
- `--plan-only`: writes just the timeline JSON.

### Developing the app without Docker

```bash
redis-server &                                         # or: docker run -p 6379:6379 redis:7-alpine
cd engine && uvicorn mog.api.app:app --port 8000 &     # API
cd engine && rq worker mog &                           # worker
cd web && npm install && npm run dev                   # http://localhost:3000 (API_URL defaults to :8000)
```

## How it works

```
upload ─► ingest (CFR, rotation, HDR→SDR) ─► audio analysis (beats, downbeats, drums, drop, section)
                                         └─► video analysis (shots, faces, motion, content area, scores)
       ─► planner ─► timeline JSON ─► renderer (OpenCV warps + effects ─► FFmpeg grade/text/grain/encode)
```

The **timeline JSON** is the single source of truth. The planner only writes it, the
renderer only reads it, and the editor in the app changes it. Its schema is in
`engine/mog/timeline/models.py`, with TypeScript types in `web/lib/timeline.ts`.
**Presets** are plain JSON in `presets/`; add a style by adding a file.

```
engine/mog/
  audio/      beat + downbeat tracking, drum onsets, drop, section picker, debug plots
  video/      ingest, shot detection, faces (YuNet), content area, tracking, scores, reframe
  planner/    the edit planner, presets, editor operations
  render/     renderer, geometry (crop/zoom/shake), slow-mo interpolation, effects, text, finish
  timeline/   the timeline schema + speed-curve maths
  api/        FastAPI app        worker/  RQ jobs        storage.py  local disk (S3-ready interface)
web/          Next.js app (upload → song → style → generate → edit → export)
presets/      the 5 styles        assets/  fonts (OFL) + face model (MIT)
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MOG_DATA` | `./data` | Where uploads and renders are stored |
| `MOG_CACHE` | `~/.cache/mog` | Converted clips, analyses, LUTs (old entries pruned after 48 h) |
| `REDIS_URL` | `redis://localhost:6379/0` | Job queue |
| `MOG_MAX_CLIPS` | `10` | Clips per project |
| `MOG_MAX_UPLOAD_BYTES` | 2 GB | Total clip size per project |
| `MOG_MAX_AUDIO_BYTES` | 300 MB | Song size |
| `MOG_RETENTION_HOURS` | `24` | Unsaved projects are deleted after this |
| `MOG_JOBS_PER_HOUR` | `60` | Render/upload jobs per client IP per hour |
| `MOG_TRUST_PROXY` | off | Use `X-Forwarded-For` for rate limits (set in Compose; the web app proxies `/api`) |
| `API_URL` | `http://localhost:8000` | (web) where the Python API is |

## Tests

```bash
cd engine && python -m pytest          # ~180 tests, ~4 min
cd web && npm run typecheck && npm run build
# Browser end-to-end (needs the stack running and Playwright installed):
cd web && CLIPS=a.mp4,b.mov AUDIO=song.mp3 node e2e/smoke.mjs
```

See [PROGRESS.md](PROGRESS.md) for what's done, measured speed and accuracy, and known limitations.
