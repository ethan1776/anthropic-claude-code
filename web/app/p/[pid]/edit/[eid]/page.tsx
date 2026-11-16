"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bar, ErrorNote, Header, Warnings } from "@/components/ui";
import { api, waitForJob, type Edit, type ShotInfo, type SongAnalysis } from "@/lib/api";
import type { Segment, Timeline } from "@/lib/timeline";

type FullEdit = Edit & { timeline: Timeline | null };
const EFFECTS: { type: string; label: string; icon: string }[] = [
  { type: "flash", label: "Flash", icon: "✦" },
  { type: "zoom_punch", label: "Zoom punch", icon: "⊕" },
  { type: "shake", label: "Shake", icon: "≋" },
  { type: "rgb_split", label: "RGB split", icon: "◐" },
  { type: "motion_blur", label: "Motion blur", icon: "≫" },
];
const SPEEDS = ["normal", "slow", "fast", "ramp"] as const;

function speedMode(s: Segment): (typeof SPEEDS)[number] {
  const vs = s.speed_curve.map((k) => k[1]);
  if (vs.length >= 3 && Math.max(...vs) > 1.2 && Math.min(...vs) < 0.9) return "ramp";
  if (Math.max(...vs) < 0.95) return "slow";
  if (Math.min(...vs) > 1.2) return "fast";
  return "normal";
}

function hue(src: string) {
  let h = 0;
  for (const c of src) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}

export default function EditorPage({ params }: { params: { pid: string; eid: string } }) {
  const { pid, eid } = params;
  const router = useRouter();
  const [edit, setEdit] = useState<FullEdit | null>(null);
  const [tl, setTl] = useState<Timeline | null>(null);
  const [song, setSong] = useState<SongAnalysis | null>(null);
  const [shots, setShots] = useState<ShotInfo[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [showExport, setShowExport] = useState(false);
  const [version, setVersion] = useState(0);
  const video = useRef<HTMLVideoElement>(null);

  const load = useCallback(async () => {
    const e = await api.edit(pid, eid);
    setEdit(e);
    setTl(e.timeline);
    setCaption(e.timeline?.text[0]?.content ?? "");
  }, [pid, eid]);

  useEffect(() => {
    load().catch((e) => setErr(e.message));
    api.song(pid).then(setSong).catch(() => undefined);
    api.shots(pid).then(setShots).catch(() => undefined);
  }, [pid, load]);

  async function op(body: Record<string, unknown>, label: string) {
    setBusy(label);
    setErr(null);
    try {
      const r = await api.op(pid, eid, body);
      setTl(r.timeline);
      setDirty(true);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function rerender() {
    setBusy("Rendering preview");
    setErr(null);
    setProgress(0);
    try {
      const e = await api.render(pid, eid);
      const j = await waitForJob(e.job_id!, (u) => setProgress(u.progress));
      if (j.status !== "finished") throw new Error(j.error || "Rendering failed.");
      await load();
      setDirty(false);
      setVersion((v) => v + 1);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function reroll() {
    setBusy("Re-rolling");
    try {
      const e = await api.reroll(pid, eid);
      router.push(`/p/${pid}/generate/${e.id}`);
    } catch (e) {
      setErr((e as Error).message);
      setBusy(null);
    }
  }

  const seg = tl?.segments.find((s) => s.id === sel) ?? null;
  const D = tl?.output.duration ?? 1;
  const beats = useMemo(
    () => (song && tl ? song.beats.map((b) => b - tl.audio.start).filter((b) => b >= 0 && b <= D) : []),
    [song, tl, D],
  );
  const drop = song?.drop && tl ? song.drop.t - tl.audio.start : null;

  function select(s: Segment) {
    setSel(s.id);
    if (video.current) {
      video.current.pause();
      video.current.currentTime = s.timeline_in + 0.01;
    }
  }

  const preview = edit?.preview_url ? `${edit.preview_url}?v=${version}` : null;

  return (
    <>
      <Header pid={pid} step="edit" />
      <main className="mx-auto max-w-3xl space-y-4 px-4 pb-32 pt-4">
        {!edit && !err && <p className="text-sm text-ink-400">Loading…</p>}
        {edit && (
          <div className="mx-auto w-full max-w-[300px]">
            <div className="relative aspect-[9/16] overflow-hidden rounded-2xl bg-ink-900">
              {preview && (
                <video ref={video} key={preview} src={preview} controls playsInline loop className="h-full w-full object-cover" data-testid="preview" />
              )}
              {dirty && !busy && (
                <div className="pointer-events-none absolute inset-x-0 top-0 bg-ink-950/75 p-2 text-center text-[11px]">
                  Changes not in the preview yet
                </div>
              )}
            </div>
          </div>
        )}

        {tl && (
          <section aria-label="Timeline">
            <div className="label mb-2 flex justify-between">
              <span>Timeline · tap a cut</span>
              <span>{tl.segments.length} cuts</span>
            </div>
            <div className="overflow-x-auto pb-2">
              <div className="relative h-20" style={{ width: `max(100%, ${D * 70}px)` }}>
                {tl.segments.map((s) => {
                  const fx = s.effects.map((e) => EFFECTS.find((x) => x.type === e.type)?.icon).join("");
                  const mode = speedMode(s);
                  return (
                    <button
                      key={s.id}
                      data-testid={`seg-${s.id}`}
                      onClick={() => select(s)}
                      className={`absolute top-0 flex h-16 flex-col justify-between overflow-hidden rounded-md border px-1 py-1 text-left text-[10px] leading-none ${
                        sel === s.id ? "z-10 border-bone ring-2 ring-bone" : "border-ink-950"
                      }`}
                      style={{
                        left: `${(s.timeline_in / D) * 100}%`,
                        width: `calc(${((s.timeline_out - s.timeline_in) / D) * 100}% - 2px)`,
                        background: `hsl(${hue(s.source)} 30% 22%)`,
                      }}
                      title={`${s.id}: ${s.source.split("/").pop()} ${mode}`}
                    >
                      <span className="truncate text-bone/80">{mode !== "normal" ? mode : ""}</span>
                      <span className="truncate text-sm leading-none text-bone">{fx}</span>
                    </button>
                  );
                })}
                {/* beat ticks + drop */}
                <div className="pointer-events-none absolute inset-x-0 bottom-0 h-3">
                  {beats.map((b) => (
                    <span key={b} className="absolute bottom-0 h-2 w-px bg-ink-400/60" style={{ left: `${(b / D) * 100}%` }} />
                  ))}
                  {drop !== null && drop >= 0 && drop <= D && (
                    <span className="absolute bottom-0 h-3 w-0.5 bg-blood" style={{ left: `${(drop / D) * 100}%` }} />
                  )}
                </div>
              </div>
            </div>
          </section>
        )}

        {seg && (
          <section className="card space-y-4 p-4" aria-label="Selected cut">
            <div className="flex items-center justify-between">
              <span className="label">
                Cut {seg.id.slice(1)} · {seg.timeline_in.toFixed(2)}–{seg.timeline_out.toFixed(2)}s
              </span>
              <button className="text-xs text-ink-400" onClick={() => setSel(null)}>
                Close
              </button>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">Shot</span>
                <button className="chip" disabled={!!busy} onClick={() => op({ op: "swap", segment: seg.id }, "Swapping")}>
                  ↻ Swap shot
                </button>
              </div>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {shots.map((sh) => {
                  const on = sh.source === seg.source && seg.src_in >= sh.start && seg.src_in < sh.end;
                  return (
                    <button
                      key={`${sh.source}-${sh.shot}`}
                      disabled={!!busy}
                      onClick={() => op({ op: "swap", segment: seg.id, source: sh.source, shot: sh.shot }, "Swapping")}
                      className={`w-16 shrink-0 overflow-hidden rounded-md border text-left ${on ? "border-bone" : "border-ink-700"}`}
                    >
                      <div className="aspect-[9/16] bg-ink-800">
                        {sh.thumb && (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img src={sh.thumb} alt="" className="h-full w-full object-cover" />
                        )}
                      </div>
                      <div className="truncate px-1 py-0.5 text-[9px] text-bone/70">
                        {sh.start.toFixed(1)}–{sh.end.toFixed(1)}s
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-semibold">Speed</span>
              <div className="flex flex-wrap gap-2">
                {SPEEDS.map((m) => (
                  <button
                    key={m}
                    disabled={!!busy}
                    className={speedMode(seg) === m ? "chip-on" : "chip"}
                    onClick={() => op({ op: "speed", segment: seg.id, mode: m }, "Changing speed")}
                  >
                    {m === "ramp" ? "fast → slow" : m}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-semibold">Effects</span>
              <div className="flex flex-wrap gap-2">
                {EFFECTS.map((e) => {
                  const on = seg.effects.some((x) => x.type === e.type);
                  return (
                    <button
                      key={e.type}
                      disabled={!!busy}
                      aria-pressed={on}
                      className={on ? "chip-on" : "chip"}
                      onClick={() => op({ op: "effect", segment: seg.id, effect: e.type, on: !on }, "Updating effects")}
                    >
                      {e.icon} {e.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-sm font-semibold">Move the cut (one beat)</span>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex gap-1">
                  <button className="chip flex-1" disabled={!!busy} onClick={() => op({ op: "nudge", segment: seg.id, edge: "in", beats: -1 }, "Moving cut")}>
                    ◀ start
                  </button>
                  <button className="chip flex-1" disabled={!!busy} onClick={() => op({ op: "nudge", segment: seg.id, edge: "in", beats: 1 }, "Moving cut")}>
                    start ▶
                  </button>
                </div>
                <div className="flex gap-1">
                  <button className="chip flex-1" disabled={!!busy} onClick={() => op({ op: "nudge", segment: seg.id, edge: "out", beats: -1 }, "Moving cut")}>
                    ◀ end
                  </button>
                  <button className="chip flex-1" disabled={!!busy} onClick={() => op({ op: "nudge", segment: seg.id, edge: "out", beats: 1 }, "Moving cut")}>
                    end ▶
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}

        {tl && (
          <section className="card space-y-2 p-4">
            <label className="label" htmlFor="cap">Caption</label>
            <div className="flex gap-2">
              <input
                id="cap"
                maxLength={120}
                value={caption}
                placeholder="he doesn't chase."
                onChange={(e) => setCaption(e.target.value)}
                className="min-w-0 flex-1 rounded-lg border border-ink-600 bg-ink-950 px-3 py-2 text-sm outline-none focus:border-bone"
              />
              <button className="btn-ghost px-4 py-2" disabled={!!busy} onClick={() => op({ op: "text", content: caption.trim() || null }, "Updating caption")}>
                Apply
              </button>
            </div>
          </section>
        )}

        {busy && (
          <div className="space-y-2" aria-live="polite">
            <div className="text-xs text-ink-400">{busy}…</div>
            {busy === "Rendering preview" && <Bar value={progress} />}
          </div>
        )}
        <ErrorNote msg={err} />
        {edit && <Warnings items={edit.warnings.filter((w) => !w.includes("trimmed"))} />}
      </main>

      <footer className="fixed inset-x-0 bottom-0 z-30 border-t border-ink-800 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center gap-2 px-4 py-3">
          <button className="btn-ghost px-3" disabled={!!busy} onClick={reroll} title="A different edit with the same settings">
            ↻ Re-roll
          </button>
          <button className="btn-ghost flex-1" disabled={!dirty || !!busy} onClick={rerender} data-testid="update-preview">
            Update preview
          </button>
          <button className="btn-blood flex-1" disabled={!!busy || dirty} onClick={() => setShowExport(true)} data-testid="open-export">
            Export
          </button>
        </div>
      </footer>
      {showExport && edit && <ExportSheet pid={pid} edit={edit} onClose={() => setShowExport(false)} />}
    </>
  );
}

function ExportSheet({ pid, edit, onClose }: { pid: string; edit: Edit; onClose: () => void }) {
  const [fps, setFps] = useState<30 | 60>(30);
  const [quality, setQuality] = useState<"fast" | "high">("high");
  const [state, setState] = useState<"idle" | "running" | "done" | "failed">(edit.export_url ? "done" : "idle");
  const [url, setUrl] = useState<string | null>(edit.export_url);
  const [progress, setProgress] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function go() {
    setState("running");
    setErr(null);
    try {
      const e = await api.exportEdit(pid, edit.id, fps, quality);
      const j = await waitForJob(e.export_job_id!, (u) => setProgress(u.progress));
      if (j.status !== "finished") throw new Error(j.error || "Export failed.");
      const full = await api.edit(pid, edit.id);
      setUrl(full.export_url);
      setState("done");
    } catch (e) {
      setErr((e as Error).message);
      setState("failed");
    }
  }

  async function share() {
    if (!url) return;
    try {
      const blob = await (await fetch(url)).blob();
      const file = new File([blob], "mog-edit.mp4", { type: "video/mp4" });
      if (navigator.canShare?.({ files: [file] })) await navigator.share({ files: [file], title: "MOG edit" });
      else window.location.href = url;
    } catch {
      /* user cancelled the share sheet */
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 sm:items-center" onClick={onClose}>
      <div role="dialog" aria-label="Export" className="card w-full max-w-md space-y-5 rounded-b-none p-5 sm:rounded-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="font-display text-2xl uppercase tracking-wide">Export</h2>
          <button className="text-sm text-ink-400" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="space-y-2">
          <span className="label">Frame rate</span>
          <div className="flex gap-2">
            {([30, 60] as const).map((f) => (
              <button key={f} className={fps === f ? "chip-on" : "chip"} onClick={() => setFps(f)} disabled={state === "running"}>
                {f} fps
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <span className="label">Quality</span>
          <div className="flex gap-2">
            {(["fast", "high"] as const).map((q) => (
              <button key={q} className={quality === q ? "chip-on" : "chip"} onClick={() => setQuality(q)} disabled={state === "running"}>
                {q === "fast" ? "Fast" : "High"}
              </button>
            ))}
          </div>
          <p className="text-xs text-ink-400">High: smoother slow-mo and a sharper encode, takes longer.</p>
        </div>
        {state === "running" && (
          <div className="space-y-2">
            <div className="text-xs text-ink-400">Rendering 1080×1920… {Math.round(progress * 100)}%</div>
            <Bar value={progress} />
          </div>
        )}
        <ErrorNote msg={err} />
        {state === "done" && url ? (
          <div className="grid grid-cols-2 gap-2">
            <a className="btn-primary" href={url} download="mog-edit.mp4" data-testid="download">
              Download
            </a>
            <button className="btn-ghost" onClick={share}>
              Share
            </button>
            <button
              className="btn-ghost col-span-2 text-xs"
              disabled={saved}
              onClick={() => api.saveProject(pid).then(() => setSaved(true))}
            >
              {saved ? "Saved ✓ (kept after 24 h)" : "Save project (otherwise deleted after 24 h)"}
            </button>
            <button className="col-span-2 text-xs text-ink-400 underline" onClick={go}>
              Export again with these settings
            </button>
          </div>
        ) : (
          <button className="btn-blood w-full" onClick={go} disabled={state === "running"} data-testid="export">
            {state === "running" ? "Exporting…" : "Export video"}
          </button>
        )}
      </div>
    </div>
  );
}
