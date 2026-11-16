# Progress

## Phase 1: Audio brain (CLI). Built; acceptance test in progress (your 5 tracks)

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

### Next: Phase 2, basic renderer
Timeline JSON schema (Pydantic + TypeScript types); renderer for cuts, constant speed, grade, grain and vignette; a hand-written sample timeline rendered to 1080x1920 MP4.
