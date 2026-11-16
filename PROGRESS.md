# Progress

## Phase 1: Audio brain (CLI). Done, accepted on your 5 tracks

### What's done
- `analyze_audio(path) -> AudioAnalysis` (`engine/mog/audio/`). It returns:
  - **BPM and beat grid.** BPM uses a comb-filter estimate limited to 100–180 BPM. Beats come from the librosa tracker, then are pinned onto the actual drum hits (about 1–3 ms error on test tracks).
  - **Downbeats.** With `beat_this` they come from the neural net. With librosa they're estimated from kick strength, chord/808 note changes, and loudness changes at bar lines.
  - **Drum hits by type**, each with a strength: kick (below 150 Hz), snare/clap (150 Hz–4 kHz, noisy), cowbell (same band, tonal) and hi-hat (6–11 kHz).
  - **Energy curves.** Full-band and bass loudness, smoothed over about 0.5 s, sampled every 50 ms.
  - **Drop.** The downbeat with the biggest bass jump compared with the 2 bars before it, plus a 0–1 confidence.
  - **Bars**, each labelled low / mid / high energy relative to the song.
  - **Beat-grid cleanup** (`grid.py`). Snaps beats to the producer's straight tempo grid, which fixes tracker slips (one real track had 14 beats slipped by 100 ms). It keeps genuine jumps where the audio is spliced and reports them as `grid_breaks`. It also checks whether the grid should be shifted half a beat, and warns when on-beat vs off-beat is a near tie.
  - **Drop as an anchor.** The drop is searched on every beat, then pinned to the sharpest unsmoothed bass jump. A confident drop also decides where beat 1 is.
- `pick_section(analysis, 10|15|20|30)`. Starts 2–4 bars before the drop and snaps both ends to downbeats. Handles a drop near the start or end, no drop, and a song shorter than the target (with warnings).
- **CLI:** `python -m mog analyze | debug | synth`. `--offbeat` shifts the grid by half a beat as a manual fix.
- **Debug outputs:** annotated PNG, full click track and section click track.
- **Audio input:** everything goes through FFmpeg, so MP3, WAV, M4A and audio inside MP4/MOV all work.
- **Error messages:** clear messages for missing, unreadable, silent or too-short audio.
- **77 tests** (`cd engine && python -m pytest`, about 8 s):
  - Synthetic phonk tracks with exact ground truth, at 108, 115, 130, 140, 145, 155 and 160 BPM.
  - Clean versions, plus "dirty master" versions with reverb, heavy saturation and noise.
  - Every beat, downbeat and drop is checked to within ±1 frame at 60 fps (16.7 ms).
- **Speed:** a 3-minute MP3 analyzes in about 3.5 s on 4 CPU cores.

### Accuracy on synthetic tracks (±1 frame @ 60 fps)
| | clean | dirty master |
|---|---|---|
| BPM | within 0.3% | within 0.3% |
| Beats / downbeats | 100% recall, mean error 1–3 ms | 100% recall, mean error 2–3 ms |
| Drop | within 5 ms | within 5 ms |
| Kick / snare / hi-hat / cowbell | 96–100% | kick 83–100%, snare 91–100%, hi-hat 98–100%, cowbell about 78% |

### Your 5 tracks (screen-recorded MP3s)
| Track | BPM | Drop (confidence) | Notes |
|---|---|---|---|
| 1 | 130.0 | 29.66 s (0.73) | Drums on beats *and* off-beats: on-beat vs off-beat is a near tie. Listen for this one. |
| 2 | 142.5 | 5.90 s (0.79) | Beat jumps near the end (100–109 s), likely splices in the recording. |
| 3 | 154.1 | 22.30 s (0.77) | Beat jumps in the outro (85–96 s). |
| 4 | 128.2 | 8.71 s (0.74) | Very busy drums. |
| 5 | 179.1 | 96.40 s (0.81) | Possibly about 90 BPM with a double-time feel; counted at 179. Jumps at 0.6 s and 117.5 s. |

Fixes made from these tracks:
- tracker slips snapped back to the tempo grid;
- splice handling;
- off-beat check;
- the drop was being placed one bar early (it is now found on any beat and anchors beat 1);
- drop confidence no longer drops just because a song repeats its drop.

### Known limitations
- **Not yet tested on real music.** Synthetic tracks are a good start but not the real thing. The acceptance test (your 5 phonk tracks, listened to with the click track) is still open.
- **`beat_this` is not exercised.** This cloud session's network blocks PyTorch's download server and the host of the model weights. The code path exists and falls back to librosa automatically, but it hasn't run here. It will be installed in the Phase 6 Docker image, and it's worth trying locally with `pip install -e ".[beats]"`.
- **Downbeats without `beat_this` are a heuristic.** On tracks with no clear bar-level changes they can land on beat 3. The CLI warns when it's unsure.
- **4/4 time is assumed.** That's fine for phonk.
- **Half-time-feel tracks are counted at double speed.** For example, a 75 BPM track is reported as 150 BPM, which is what cutting on every beat needs.

## Phase 2: Basic renderer (CLI). Done

### What's done
- **Timeline schema** (`engine/mog/timeline/models.py`):
  - Pydantic models with strict validation. Unknown fields, gaps or overlaps between segments, bad speeds, effects after a segment ends, and audio shorter than the video are all rejected with clear messages.
  - Exported to JSON Schema and TypeScript (`web/lib/timeline.ts`, type-checked with `tsc --strict`). A test fails if they go stale.
  - Times inside a segment (speed keyframes, effects, crop paths) are segment-local, so segments can be moved in the editor.
- **Ingest** (`video/ingest.py`). Each clip is converted once, then cached:
  - constant frame rate at the clip's own standard rate;
  - phone rotation applied;
  - iPhone HDR (HLG / HDR10) tone-mapped to Rec.709;
  - shrunk only when much larger than needed;
  - a keyframe every second for fast seeking.
- **Frame-exact reading** (`ffmpeg/frames.py`). Verified with barcode test clips at 24, 30 and 60 fps and rotated: every frame read is exactly the frame asked for.
- **Renderer** (`render/renderer.py`):
  - cuts (land on the nearest frame), speed curves with linear / smoothstep / cubic easing, 9:16 crop with zoom and moving crop paths;
  - grade: LUT + strength, contrast, brightness, gamma, saturation, black crush, glow, vignette, film grain;
  - audio section with fade-in and fade-out;
  - progress reporting;
  - output checked after encoding (size, frame count, audio). A failed render leaves no half-written file.
- **Colour:**
  - 5 built-in LUTs, generated from maths so there's nothing to license: `bw_crush`, `teal_cold`, `velocity_punch`, `night_drift`, `clean_film`.
  - Users' own `.cube` files also work.
  - The whole per-pixel grade is baked into one LUT for speed.
- **Output:**
  - H.264 High, yuv420p, Rec.709 tags, 30 or 60 fps CFR, about 15.6 Mbps, AAC 320 kbps 48 kHz, `+faststart`.
  - Preview mode: 540x960, fast preset.
- **No black edges, by construction:** the crop window is clamped inside the source (tested with 1,200 random crops and zooms).
- **Same timeline, same output:** two renders produce byte-identical frames (tested).
- **132 tests** in about 20 s.

### Sample
`samples/phase2/sample_timeline.json` uses 4 synthetic clips (landscape 30 fps, vertical 60 fps, 4:3 24 fps, rotated 29.97 fps phone clip) and the synthetic phonk track. `samples/phase2/sample_frames.png` shows frames from the render.

### Speed (this cloud machine: 4 CPU cores, no GPU)
| Render | Time for the 14.9 s sample |
|---|---|
| Export, 1080x1920 @ 60 fps | 61 s (was 93 s before optimising) |
| Preview, 540x960 @ 60 fps | 14 s |

The remaining bottleneck is the 3D LUT inside FFmpeg (about 70 fps on its own) competing with the x264 encoder for 4 cores. Phase 3's effects and slow-motion interpolation will add cost. If the 90 s target gets tight, the planned fix is rendering chunks of the timeline in parallel and joining them.

### Tested on your real clips (5 files from Google Drive; kept out of the repo)
- **Ingest handled everything:** HEVC, full-range colour, variable frame rate, sizes from 386x484 to 1320x2868. It now prefers a clip's declared frame rate (a variable-rate phone clip declared at 30 fps stays 30), and warns about low-resolution clips (3 of the 5 need over 2x enlargement).
- **The looks were too dark on low-light footage.** Dark Triad and Mog crushed a dimly lit podcast shot nearly to black. Fixes:
  - a new per-segment `exposure` setting (in stops) corrects each shot before the grade; Phase 4 will set it automatically from face brightness;
  - the black & white LUT's built-in shadow crush is gentler.
- **Findings for Phase 4:**
  - screen recordings include the phone's status bar and app buttons, and some clips have letterbox bars or watermarks; auto-crop must remove these;
  - one clip has a screenshot cut into the middle, which should become a separate low-scoring shot.

### Known limitations
- Slow motion currently repeats frames (choppy). Smooth interpolated slow-mo is Phase 3.
- Per-segment effects (zoom punch, shake, flash, RGB split, motion blur) and text overlays are in the schema but rejected by the renderer until Phase 3.
- The demo clips are synthetic test patterns. Judging the grades properly needs real footage.

## Phase 3: Effects. Done

### What's done
- **Smooth slow motion** (`render/source.py`).
  - Frames between recorded frames are made with optical flow (OpenCV DIS): motion is estimated between frame pairs, both frames are warped to the in-between moment and blended.
  - "fast" (previews) and "high" (export) quality settings; `interpolation="off"` repeats frames.
  - **Change from your spec:** I used OpenCV optical flow instead of FFmpeg `minterpolate`. It handles any speed curve per output frame rather than needing a fixed target frame rate, it's faster, and it avoids minterpolate's block artifacts.
  - **RIFE is not included:** this machine has no GPU to build or test it on.
- **Zoom punch**: instant scale-in, ease-out-cubic back over `decay`. Centred on the crop/face point.
- **Shake**: smooth seeded wobble (offset plus up to about 1.5° of rotation) with a decaying envelope. The frame is automatically scaled up just enough that the rotated, shifted window never leaves the source, so no black edges appear (tested with 1,200 random cases and full renders).
- **Flash**: full strength on the beat frame, fading over N frames, any colour.
- **RGB split**: red and blue channels pushed apart, closing over N frames.
- **Motion blur**: virtual shutter averaging in-between frames across 1.5 frames of motion, used during fast sections.
- **Text**:
  - drawn once with Pillow, with a soft outline for legibility;
  - "pop" (scales 1.35 to 1.0 on the beat), "fade" and "none" animations;
  - bundled OFL fonts (Cinzel, Anton, Inter; licences in `assets/fonts`);
  - composited after the colour grade, so white stays white under any look.
- **Frame-accurate timing:** every effect starts on output frame `round((segment start + t) * fps)`. A test checks the flash is white on exactly the beat frame.
- **Deterministic shake:** the seed comes from a checksum, not Python's per-process `hash()`, so the same timeline shakes the same way every time.
- **136 tests.**

### Demos (on your clips)
`samples/phase3/make_demos.py` writes one timeline plus video per effect: `slowmo` (side by side: repeated frames vs smooth), `velocity_ramp` (3x snapping to 0.3x on the beat, with blur), `zoom_punch`, `shake`, `flash`, `rgb_split`, `text`, `motion_blur` (side by side). The videos use your clips, so they stay out of the repo; the timelines are in the script.

### Speed
Optical flow and motion blur cost time. On this 4-core machine at 1080x1920 30 fps, a 3.3 s clip takes about 9 s with plain effects, 20 s with slow-mo or blur throughout. Real edits use slow-mo and blur for only part of the length; Phase 5 measures a full edit against the 90 s target.

### Next: Phase 4, video analysis and auto-reframe
