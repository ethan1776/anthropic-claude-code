"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Bar, ErrorNote, Header, Warnings, fmtBytes } from "@/components/ui";
import { api, upload, waitForJob, type Limits, type ProjectFull } from "@/lib/api";

type Pending = { key: string; name: string; size: number; progress: number; phase: "uploading" | "analysing" | "failed"; error?: string };

export default function UploadPage({ params }: { params: { pid: string } }) {
  const { pid } = params;
  const router = useRouter();
  const [project, setProject] = useState<ProjectFull | null>(null);
  const [limits, setLimits] = useState<Limits | null>(null);
  const [pending, setPending] = useState<Pending[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(() => api.project(pid).then(setProject).catch((e) => setErr(e.message)), [pid]);
  useEffect(() => {
    refresh();
    api.limits().then(setLimits).catch(() => undefined);
  }, [refresh]);

  const setP = (key: string, patch: Partial<Pending>) =>
    setPending((ps) => ps.map((p) => (p.key === key ? { ...p, ...patch } : p)));

  async function send(file: File, kind: "clips" | "audio") {
    const key = `${kind}-${file.name}-${file.size}-${Date.now()}`;
    setPending((ps) => [...ps, { key, name: file.name, size: file.size, progress: 0, phase: "uploading" }]);
    try {
      const res = await upload(`/api/projects/${pid}/${kind}`, file, (f) => setP(key, { progress: f }));
      setP(key, { phase: "analysing", progress: 0 });
      const job = await waitForJob(res.job_id, (j) => setP(key, { progress: j.progress }));
      if (job.status !== "finished") throw new Error(job.error || "Analysis failed.");
      setPending((ps) => ps.filter((p) => p.key !== key));
      refresh();
    } catch (e) {
      setP(key, { phase: "failed", error: (e as Error).message });
    }
  }

  function addClips(files: FileList | File[]) {
    setErr(null);
    const list = Array.from(files);
    const room = (limits?.max_clips ?? 10) - (project?.clips.length ?? 0) - pending.filter((p) => p.key.startsWith("clips")).length;
    if (list.length > room) setErr(`You can add ${Math.max(room, 0)} more clip${room === 1 ? "" : "s"} (max ${limits?.max_clips ?? 10}).`);
    list.slice(0, Math.max(room, 0)).forEach((f) => send(f, "clips"));
  }

  const clipsReady = (project?.clips.filter((c) => c.analysed).length ?? 0) > 0;
  const songReady = !!project?.audio?.analysed;
  const busy = pending.some((p) => p.phase !== "failed");

  return (
    <>
      <Header pid={pid} step="upload" />
      <main className="mx-auto max-w-3xl space-y-6 px-4 pb-28 pt-6">
        <p className="rounded-xl border border-ink-700 bg-ink-900 px-4 py-3 text-xs text-bone/70">
          Only use clips and music you have the rights to post.
        </p>

        <Drop
          title="Clips"
          hint={`MP4 or MOV · up to ${limits?.max_clips ?? 10} clips, ${fmtBytes(limits?.max_upload_bytes ?? 2 * 1024 ** 3)} total`}
          accept="video/mp4,video/quicktime,.mp4,.mov,.m4v"
          multiple
          onFiles={addClips}
          testId="drop-clips"
        />
        <ul className="grid grid-cols-3 gap-2 sm:grid-cols-5">
          {project?.clips.map((c) => (
            <li key={c.id} className="card relative overflow-hidden">
              <div className="aspect-[9/16] bg-ink-800">
                {c.thumb && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={c.thumb} alt="" className="h-full w-full object-cover" />
                )}
              </div>
              <div className="p-1.5 text-[10px] leading-tight text-bone/70">
                <div className="truncate">{c.filename}</div>
                {c.analysed ? (
                  <div className="text-ink-400">
                    {c.duration?.toFixed(1)}s · {c.shots} shot{c.shots === 1 ? "" : "s"}
                  </div>
                ) : (
                  <div className="text-ink-400">analysing…</div>
                )}
              </div>
              <button
                aria-label={`Remove ${c.filename}`}
                className="absolute right-1 top-1 rounded-full bg-ink-950/80 px-1.5 text-xs"
                onClick={() => api.deleteClip(pid, c.id).then(refresh)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
        <Warnings items={(project?.clips ?? []).flatMap((c) => c.warnings).filter((w) => w.includes("low resolution"))} />

        <Drop
          title="Song"
          hint="MP3, WAV or M4A, or a video to use its sound"
          accept="audio/*,video/mp4,video/quicktime,.mp3,.wav,.m4a,.mp4,.mov"
          onFiles={(f) => f[0] && send(f[0], "audio")}
          testId="drop-audio"
        />
        {project?.audio && (
          <div className="card flex items-center justify-between px-4 py-3 text-sm">
            <span className="truncate">{project.audio.filename}</span>
            <span className="text-xs text-ink-400">{project.audio.analysed ? "beat found ✓" : "finding the beat…"}</span>
          </div>
        )}

        {pending.length > 0 && (
          <ul className="space-y-2">
            {pending.map((p) => (
              <li key={p.key} className="card space-y-2 px-4 py-3">
                <div className="flex justify-between gap-3 text-xs">
                  <span className="truncate">{p.name}</span>
                  <span className="shrink-0 text-ink-400">
                    {p.phase === "uploading" ? `uploading ${fmtBytes(p.size)}` : p.phase === "analysing" ? "analysing" : "failed"}
                  </span>
                </div>
                {p.phase === "failed" ? <ErrorNote msg={p.error ?? "Failed."} /> : <Bar value={p.progress} />}
              </li>
            ))}
          </ul>
        )}
        <ErrorNote msg={err} />
      </main>

      <footer className="fixed inset-x-0 bottom-0 z-30 border-t border-ink-800 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3 px-4 py-3">
          <span className="text-xs text-ink-400">
            {clipsReady ? "✓ clips" : "add clips"} · {songReady ? "✓ song" : "add a song"}
          </span>
          <button className="btn-primary" disabled={!clipsReady || !songReady || busy} onClick={() => router.push(`/p/${pid}/song`)}>
            Continue
          </button>
        </div>
      </footer>
    </>
  );
}

function Drop(props: {
  title: string;
  hint: string;
  accept: string;
  multiple?: boolean;
  onFiles: (f: FileList | File[]) => void;
  testId: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        props.onFiles(e.dataTransfer.files);
      }}
      onClick={() => input.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-8 text-center transition ${
        over ? "border-bone bg-ink-800" : "border-ink-600 bg-ink-900 hover:border-ink-400"
      }`}
    >
      <span className="font-display text-2xl uppercase tracking-wide">{props.title}</span>
      <span className="mt-1 text-xs text-ink-400">Tap to choose or drop here · {props.hint}</span>
      <input
        ref={input}
        data-testid={props.testId}
        type="file"
        accept={props.accept}
        multiple={props.multiple}
        className="hidden"
        onChange={(e) => {
          if (e.target.files) props.onFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
