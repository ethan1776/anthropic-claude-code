"""The edit planner ("director brain"): music + footage + style -> Timeline JSON.

Structure of every edit (times are on the output timeline, 0 = section start):

  [ build-up: cuts on bars/half-bars, medium-energy shots, last bar speeds up ]
  [ DROP: flash + zoom punch + shake, the best "stare" shot in slow motion for 1-2 bars ]
  [ after the drop: cuts on every beat (fewer in calm bars), velocity ramps that snap from
    fast to slow exactly on downbeats, action shots, effects on kicks/snares/cowbells ]

Rules enforced here (all numbers come from the preset):
  * every cut lands on a beat (never off-beat)
  * at most one flash per beat; at most one bass shake per bar
  * the same cut transition is never used 3 times in a row
  * no source range is used twice, unless there isn't enough footage (then it is reused
    with a different crop and a warning)
  * same inputs + same seed => the same edit (all randomness comes from one seeded RNG)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mog.audio.models import AudioAnalysis, Section
from mog.errors import MogError
from mog.planner.presets import Preset
from mog.timeline.models import Segment, Timeline
from mog.timeline.speed import footage_needed, source_offsets
from mog.video.models import Shot, VideoAnalysis

MIN_SHOT = 0.25
FOOTAGE_MARGIN = 0.05


@dataclass
class PlanOptions:
    seed: int = 1
    intensity: float = 0.6        # 0..1: scales effect density and strength
    fps: int = 30
    width: int = 1080
    height: int = 1920
    text: str | None = None
    audio_file: str = "track.mp3"


@dataclass
class _Cut:
    t0: float
    t1: float
    role: str                      # build | build_ramp | drop | ramp | post
    speed_curve: list[tuple[float, float]]
    ease: str = "smoothstep"
    snap: float | None = None      # segment-local time of a velocity snap
    effects: list[dict] = field(default_factory=list)


@dataclass
class _Pick:
    clip: VideoAnalysis
    shot: Shot
    src_in: float
    reused: bool


class Planner:
    def __init__(self, audio: AudioAnalysis, section: Section, clips: list[VideoAnalysis], preset: Preset,
                 opts: PlanOptions):
        if not clips:
            raise MogError("Add at least one video clip.")
        self.a, self.sec, self.p, self.o = audio, section, preset, opts
        self.clips = sorted(clips, key=lambda c: c.source)       # order-independent => deterministic
        self.rng = np.random.default_rng(opts.seed)
        self.warnings: list[str] = []
        self.D = section.end - section.start
        rel = lambda ts: np.array([t - section.start for t in ts if section.start - 1e-3 <= t <= section.end + 1e-3])  # noqa: E731
        self.beats = rel(audio.beats)
        self.downs = rel(audio.downbeats)
        if len(self.beats) < 4:
            raise MogError("The chosen song section has too few beats to edit to.")
        self.beat = float(np.median(np.diff(self.beats)))
        self.bar = 4 * self.beat
        self.drop = (section.drop - section.start) if section.drop is not None else None
        if self.drop is not None and not (0 <= self.drop < self.D - self.beat):
            self.drop = None
        self.onsets = {k: rel([o.t for o in audio.onsets if o.kind == k]) for k in ("kick", "snare", "hihat", "cowbell")}
        self.strength = {(o.kind, round(o.t - section.start, 3)): o.strength for o in audio.onsets}
        self.used: dict[str, list[tuple[float, float]]] = {c.source: [] for c in self.clips}
        self.last_source: str | None = None
        self.uses: dict[str, int] = {}
        self.transitions: list[str] = []
        self.intensity = float(np.clip(opts.intensity, 0, 1))

    # ------------------------------------------------------------------ grid helpers
    def _grid(self, unit: str, a: float, b: float) -> list[float]:
        """Cut points in [a, b) on the beat grid for a given unit."""
        if unit == "bar":
            pts = [d for d in self.downs if a - 1e-3 <= d < b - 1e-3]
        elif unit == "half_bar":
            pts = [t for i, t in enumerate(self.beats) if a - 1e-3 <= t < b - 1e-3 and self._beat_in_bar(t) % 2 == 0]
        elif unit == "beat":
            pts = [t for t in self.beats if a - 1e-3 <= t < b - 1e-3]
        else:  # half_beat
            pts = []
            for t in self.beats:
                if a - 1e-3 <= t < b - 1e-3:
                    pts.append(t)
                    if t + self.beat / 2 < b - 1e-3:
                        pts.append(t + self.beat / 2)
        if not pts or pts[0] > a + 1e-3:
            pts = [a] + pts
        return sorted(set(round(p, 4) for p in pts))

    def _beat_in_bar(self, t: float) -> int:
        prev = self.downs[self.downs <= t + 1e-3]
        if len(prev) == 0:
            return int(round(t / self.beat)) % 4
        return int(round((t - prev[-1]) / self.beat)) % 4

    def _energy(self, t: float) -> float:
        return self.a.energy.at(t + self.sec.start)

    # ------------------------------------------------------------------ 1. structure
    def structure(self) -> list[_Cut]:
        p, D = self.p, self.D
        cuts: list[_Cut] = []
        drop = self.drop
        pre_end = drop if drop is not None else 0.0
        if drop is not None and drop > 1e-3:
            pts = self._grid(p.cuts.pre_drop, 0.0, drop) + [drop]
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                last = i == len(pts) - 2
                v = p.speed.pre_drop
                if last and p.speed.build_ramp > 1.01 and b - a > self.beat * 0.9:
                    # tension: speed up through the last stretch before the drop
                    d = b - a
                    cuts.append(_Cut(a, b, "build_ramp", [(0.0, v), (d * 0.5, v), (d, p.speed.build_ramp)], "cubic"))
                else:
                    cuts.append(_Cut(a, b, "build", [(0.0, v)]))
        post_start = 0.0
        if drop is not None:
            stare_end = min(D, drop + p.speed.stare_bars * self.bar)
            stare_end = self._snap_to_beat(stare_end)
            if stare_end - drop < self.beat * 0.9:
                stare_end = min(D, drop + self.bar)
            cuts.append(_Cut(drop, stare_end, "drop", [(0.0, p.speed.stare_speed)]))
            post_start = stare_end
        else:
            post_start = pre_end
        cuts += self._post_drop(post_start, D)
        # sanity: tile exactly
        cuts = [c for c in cuts if c.t1 - c.t0 > 1e-3]
        cuts[0].t0 = 0.0
        cuts[-1].t1 = D
        return cuts

    def _snap_to_beat(self, t: float) -> float:
        cand = np.concatenate([self.beats, [self.D]])
        return float(cand[np.argmin(np.abs(cand - t))])

    def _post_drop(self, a: float, b: float) -> list[_Cut]:
        p = self.p
        if b - a < 1e-3:
            return []
        b_ = self.beat
        # Ramp anchors: downbeats to snap on. A ramp starts on the ACTUAL beat before the anchor
        # (real beats are not perfectly evenly spaced, so "anchor - average beat" can miss it).
        starts: dict[float, float] = {}
        if p.speed.velocity_ramps:
            downs = [d for d in self.downs if a + 0.5 * b_ <= d <= b - 0.5 * b_]
            for d in downs[::p.speed.ramp_every_bars]:
                prev = self.beats[self.beats < d - 0.5 * b_]
                if len(prev) and prev[-1] >= a - 1e-3:
                    starts[round(float(prev[-1]), 4)] = float(d)
        out: list[_Cut] = []
        t = a
        while t < b - 1e-3:
            key = next((k for k in starts if abs(k - t) < 0.02), None)
            if key is not None:
                d = starts.pop(key)
                after = self.beats[self.beats > d + 0.5 * b_]
                t1 = min(float(after[0]) if len(after) else b, b)
                snap = d - t
                ease_w = min(0.06, snap * 0.3)
                curve = [(0.0, p.speed.ramp_fast), (snap - ease_w, p.speed.ramp_fast), (snap, p.speed.ramp_slow),
                         (t1 - t, p.speed.ramp_slow)]
                out.append(_Cut(t, t1, "ramp", curve, "cubic", snap=snap))
                t = t1
                continue
            calm = self._energy(t) < 0.45
            unit = p.cuts.calm_post_drop if calm else p.cuts.post_drop
            nxt = min((k for k in starts if k > t + 1e-3), default=b)
            pts = [x for x in self._grid(unit, t, nxt) if x > t + 1e-3]
            t1 = min(pts[0] if pts else nxt, nxt, b)
            out.append(_Cut(t, t1, "post", [(0.0, 1.0)]))
            t = t1
        return _merge_slivers(out, 0.4 * b_)

    # ------------------------------------------------------------------ 2. shots
    def _role_score(self, role: str, s: Shot) -> float:
        q = 0.4 + 0.6 * s.quality
        if role == "drop":
            return (0.15 + s.stare_score) * q
        if role in ("ramp", "post"):
            return (0.2 + s.action_score + 0.4 * s.face_score) * q
        # build-up: medium energy, faces welcome
        return (0.3 + 0.6 * s.face_score + 0.5 * (1 - abs(s.motion_score - 0.35))) * q

    def _free(self, clip: str, a: float, b: float) -> list[tuple[float, float]]:
        free = [(a, b)]
        for u0, u1 in sorted(self.used[clip]):
            nxt = []
            for f0, f1 in free:
                if u1 <= f0 or u0 >= f1:
                    nxt.append((f0, f1))
                    continue
                if u0 > f0:
                    nxt.append((f0, u0))
                if u1 < f1:
                    nxt.append((u1, f1))
            free = nxt
        return free

    def pick(self, role: str, need: float) -> _Pick:
        need_m = need + FOOTAGE_MARGIN
        cands = []
        for c in self.clips:
            for s in c.shots:
                if s.duration < max(MIN_SHOT, need_m) or s.quality < 0.08:
                    continue
                spans = [f for f in self._free(c.source, s.start, s.end) if f[1] - f[0] >= need_m]
                if not spans:
                    continue
                score = self._role_score(role, s)
                if c.source == self.last_source:
                    score *= 0.55   # variety: avoid the same clip twice in a row
                # balance: each earlier use of this clip makes it less likely again, so one
                # long, sharp clip doesn't take over the whole edit
                score /= 1.0 + 0.45 * self.uses.get(c.source, 0)
                cands.append((score, c, s, spans))
        reused = False
        if not cands:
            # Not enough fresh footage: reuse the best long-enough shot (with a different crop)
            for c in self.clips:
                for s in c.shots:
                    if s.duration >= need_m:
                        cands.append((self._role_score(role, s), c, s, [(s.start, s.end)]))
            reused = True
        if not cands:
            raise _NeedTooLong(need)
        cands.sort(key=lambda x: -x[0])
        top = cands[:4]
        w = np.array([max(x[0], 1e-3) ** 2 for x in top])
        score, clip, shot, spans = top[int(self.rng.choice(len(top), p=w / w.sum()))]
        span = spans[int(self.rng.integers(len(spans)))]
        slack = span[1] - span[0] - need_m
        src_in = span[0] + float(self.rng.uniform(0, max(slack, 0)))
        if role == "drop" and shot.face_path:
            # stare: start where the face is most frontal-ish... approximate with the shot's middle
            mid = (shot.start + shot.end) / 2 - need / 2
            src_in = float(np.clip(mid, span[0], span[1] - need_m))
        self.used[clip.source].append((src_in, src_in + need))
        self.uses[clip.source] = self.uses.get(clip.source, 0) + 1
        self.last_source = clip.source
        return _Pick(clip, shot, round(src_in, 4), reused)

    # ------------------------------------------------------------------ 3. effects
    def _p(self, base: float) -> float:
        return float(np.clip(base * (0.4 + 1.2 * self.intensity), 0, 1))

    def effects(self, cuts: list[_Cut]) -> None:
        e = self.p.effects
        k = 0.6 + 0.8 * self.intensity
        flashes: list[float] = []
        shake_bars: set[int] = set()

        def can_flash(t):
            return all(abs(t - f) > 0.9 * self.beat for f in flashes)

        for i, c in enumerate(cuts):
            dur = c.t1 - c.t0
            local = lambda t: round(t - c.t0, 4)  # noqa: E731
            transition = "none"
            if c.role == "drop":
                if e.drop_flash_frames > 0:
                    c.effects.append({"type": "flash", "t": 0.0, "frames": e.drop_flash_frames, "color": e.flash_color})
                    flashes.append(c.t0)
                if e.drop_punch > 1.0:
                    c.effects.append({"type": "zoom_punch", "t": 0.0, "strength": round(1 + (e.drop_punch - 1) * k, 3),
                                      "decay": 0.45, "anchor": "face"})
                if e.drop_shake > 0:
                    c.effects.append({"type": "shake", "t": 0.0, "intensity": round(e.drop_shake * k, 1), "decay": 0.5})
                    shake_bars.add(int(c.t0 // self.bar))
                transition = "drop"
            elif i > 0:
                on_snare = self._near(self.onsets["snare"], c.t0) or self._beat_in_bar(c.t0) == 0
                choices = []
                if e.flash_on_cuts > 0 and on_snare and can_flash(c.t0) and self.rng.random() < self._p(e.flash_on_cuts):
                    choices.append("flash")
                if e.rgb_on == "downbeat" and self._beat_in_bar(c.t0) == 0 and self.rng.random() < self._p(e.rgb_prob):
                    choices.append("rgb")
                if c.role == "ramp" and e.punch_on != "none" and self.rng.random() < self._p(e.punch_prob):
                    choices.append("punch_snap")
                # variety: never the same transition 3 times in a row
                choices = [x for x in choices if self.transitions[-2:] != [x, x]]
                if choices:
                    transition = choices[0]
                if transition == "flash":
                    c.effects.append({"type": "flash", "t": 0.0, "frames": e.flash_frames, "color": e.flash_color})
                    flashes.append(c.t0)
                elif transition == "rgb":
                    c.effects.append({"type": "rgb_split", "t": 0.0, "frames": 3, "offset": round(10 * k, 1)})
            self.transitions.append(transition)

            # velocity snap: punch on the snap (and blur during the fast part)
            if c.snap is not None:
                if transition == "punch_snap" or (e.punch_on != "none" and self.rng.random() < self._p(e.punch_prob)):
                    c.effects.append({"type": "zoom_punch", "t": round(c.snap, 4),
                                      "strength": round(1 + (e.punch_strength - 1) * k, 3), "decay": 0.3,
                                      "anchor": "face"})
                if e.motion_blur > 0:
                    c.effects.append({"type": "motion_blur", "t_in": 0.0, "t_out": round(c.snap, 4),
                                      "strength": e.motion_blur})
            if c.role == "build_ramp" and e.motion_blur > 0:
                c.effects.append({"type": "motion_blur", "t_in": round(dur * 0.5, 4), "t_out": round(dur, 4),
                                  "strength": e.motion_blur * 0.7})

            if c.role in ("post", "ramp"):
                energy = self._energy(c.t0)
                # zoom punches on kicks / downbeats inside the segment
                if e.punch_on in ("kick", "downbeat") and c.snap is None:
                    times = self.onsets["kick"] if e.punch_on == "kick" else self.downs
                    for t in times:
                        if c.t0 + 0.02 < t < c.t1 - 0.05 and self.rng.random() < self._p(e.punch_prob) * energy:
                            c.effects.append({"type": "zoom_punch", "t": local(t),
                                              "strength": round(1 + (e.punch_strength - 1) * k, 3),
                                              "decay": min(0.3, c.t1 - t), "anchor": "face"})
                            break
                # bass shake: at most one per bar, never on consecutive cuts
                if e.shake_on_bass > 0 and e.shake_intensity > 0:
                    bar_i = int(c.t0 // self.bar)
                    kicks = [t for t in self.onsets["kick"] if c.t0 - 1e-3 <= t < c.t1 - 0.05]
                    if kicks and bar_i not in shake_bars and (bar_i - 1) not in shake_bars \
                            and self.rng.random() < self._p(e.shake_on_bass) * energy:
                        strongest = max(kicks, key=lambda t: self.strength.get(("kick", round(t, 3)), 0.5))
                        c.effects.append({"type": "shake", "t": local(strongest),
                                          "intensity": round(e.shake_intensity * k, 1), "decay": 0.3})
                        shake_bars.add(bar_i)
                # rgb split on cowbells / snares
                if e.rgb_on in ("cowbell", "snare"):
                    for t in self.onsets[e.rgb_on]:
                        if c.t0 + 0.02 < t < c.t1 - 0.05 and self.rng.random() < self._p(e.rgb_prob) * 0.6:
                            c.effects.append({"type": "rgb_split", "t": local(t), "frames": 3,
                                              "offset": round(8 * k, 1)})

    def _near(self, arr: np.ndarray, t: float, tol: float = 0.035) -> bool:
        return bool(len(arr)) and float(np.min(np.abs(arr - t))) < tol

    # ------------------------------------------------------------------ 4. assemble
    def plan(self) -> Timeline:
        cuts = self.structure()
        self.effects(cuts)
        segs = []
        reused = 0
        for i, c in enumerate(cuts):
            seg = self._segment(i, c)
            need = footage_needed(seg)
            try:
                pk = self.pick(c.role, need)
            except _NeedTooLong:
                # No shot is long enough for this speed: slow the segment down to fit the longest shot
                longest = max(s.duration for cl in self.clips for s in cl.shots)
                f = max((longest - FOOTAGE_MARGIN) / need, 0.05)
                c.speed_curve = [(t, max(v * f, 0.05)) for t, v in c.speed_curve]
                seg = self._segment(i, c)
                need = footage_needed(seg)
                self.warnings.append(f"Clips are short: slowed segment {i + 1} to fit the footage.")
                pk = self.pick(c.role, need)
            reused += pk.reused
            segs.append(self._fill(seg, pk, c.effects))
        if reused:
            self.warnings.append(f"Not enough footage for a {self.D:.0f}s edit without repeats: reused {reused} "
                                 f"shot(s) with a different crop. Add more clips for a better edit.")
        text = []
        if self.o.text:
            t_in = self.drop + self.bar if self.drop is not None else self.bar
            t_in = self._snap_to_beat(min(t_in, max(self.D - self.bar, 0)))
            text.append({"content": self.o.text, "t_in": round(t_in, 4),
                         "t_out": round(min(self.D - 0.15, t_in + 3 * self.bar), 4), "font": self.p.text_font,
                         "anim": self.p.text_anim, "y": 0.70, "size": 0.05})
        tl = {
            "version": 1, "seed": self.o.seed,
            "output": {"width": self.o.width, "height": self.o.height, "fps": self.o.fps, "duration": round(self.D, 4)},
            "audio": {"file": self.o.audio_file, "start": round(self.sec.start, 4), "end": round(self.sec.end, 4),
                      "fade_in": 0.01, "fade_out": 0.4},
            "grade": self.p.grade.model_dump(),
            "segments": segs, "text": text,
        }
        return Timeline.model_validate(tl)

    def _segment(self, i: int, c: _Cut) -> Segment:
        return Segment.model_validate({"id": f"s{i + 1}", "source": "x", "src_in": 0, "timeline_in": round(c.t0, 4),
                                       "timeline_out": round(c.t1, 4), "speed_curve": c.speed_curve, "ease": c.ease})

    def _fill(self, seg: Segment, pk: _Pick, effects: list[dict]) -> dict:
        shot, clip = pk.shot, pk.clip
        # Crop path in SEGMENT-LOCAL time: map output time -> source time through the speed curve
        taus = np.arange(0.0, seg.duration + 1e-9, 0.1)
        src = pk.src_in + source_offsets(seg.speed_curve, taus, seg.ease)
        cp = np.array(shot.crop_path)
        rel = src - shot.start
        xs = np.interp(rel, cp[:, 0], cp[:, 1])
        ys = np.interp(rel, cp[:, 0], cp[:, 2])
        zoom = 1.0
        if pk.reused:
            zoom = round(float(self.rng.uniform(1.15, 1.35)), 3)
            xs = np.clip(xs + float(self.rng.uniform(-0.05, 0.05)), 0, 1)
        crop = {"mode": "face_track" if shot.face_path else "manual", "zoom": zoom, "bounds": list(clip.content),
                "path": [[round(float(t), 3), round(float(x), 4), round(float(y), 4)] for t, x, y in zip(taus, xs, ys)]}
        return {"id": seg.id, "source": clip.source, "src_in": pk.src_in, "timeline_in": seg.timeline_in,
                "timeline_out": seg.timeline_out, "speed_curve": [list(k) for k in seg.speed_curve], "ease": seg.ease,
                "exposure": shot.exposure if self.p.auto_exposure else 0.0, "crop": crop, "effects": effects}


def _merge_slivers(cuts: list[_Cut], min_len: float) -> list[_Cut]:
    """Fold any segment shorter than min_len into its neighbour (a plain cut absorbs it)."""
    out: list[_Cut] = []
    for c in cuts:
        if out and c.t1 - c.t0 < min_len and c.role == "post":
            out[-1].t1 = c.t1
            if out[-1].role == "ramp":
                out[-1].speed_curve[-1] = (out[-1].t1 - out[-1].t0, out[-1].speed_curve[-1][1])
            continue
        if out and out[-1].t1 - out[-1].t0 < min_len and out[-1].role == "post":
            c.t0 = out.pop().t0 if c.role == "post" else c.t0
        out.append(c)
    return out


class _NeedTooLong(Exception):
    def __init__(self, need: float):
        self.need = need


def plan_edit(audio: AudioAnalysis, section: Section, clips: list[VideoAnalysis], preset: Preset,
              opts: PlanOptions) -> tuple[Timeline, list[str]]:
    pl = Planner(audio, section, clips, preset, opts)
    tl = pl.plan()
    return tl, pl.warnings + list(section.warnings) + [w for c in clips for w in c.warnings]


def describe(tl: Timeline) -> str:
    """Human-readable outline of a plan."""
    lines = []
    for s in tl.segments:
        sp = "/".join(f"{v:g}x" for _, v in s.speed_curve)
        fx = ", ".join(e.type for e in s.effects)
        lines.append(f"  {s.id:>4} {s.timeline_in:6.2f}-{s.timeline_out:6.2f}  {s.source:<14} @{s.src_in:6.2f}s  "
                     f"{sp:<14} {fx}")
    return "\n".join(lines)
