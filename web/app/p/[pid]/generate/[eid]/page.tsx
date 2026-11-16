"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Bar, ErrorNote, Header } from "@/components/ui";
import { api, waitForJob, type Job } from "@/lib/api";

const STAGES = ["Finding the beat", "Scouting shots", "Directing", "Rendering"];
const NICE: Record<string, string> = {
  Queued: "Waiting for a free renderer…",
  "Finding the beat": "Finding the beat…",
  "Scouting shots": "Scouting shots…",
  Directing: "Directing…",
  "Preparing clips": "Rendering…",
  Rendering: "Rendering…",
  Done: "Done",
};

export default function GeneratePage({ params }: { params: { pid: string; eid: string } }) {
  const { pid, eid } = params;
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const e = await api.edit(pid, eid);
        if (e.status === "done") return router.replace(`/p/${pid}/edit/${eid}`);
        if (!e.job_id) throw new Error("This edit has no job.");
        const j = await waitForJob(e.job_id, (u) => alive && setJob(u));
        if (!alive) return;
        if (j.status === "finished") router.replace(`/p/${pid}/edit/${eid}`);
        else setErr(j.error || "Something went wrong while making the edit.");
      } catch (e) {
        if (alive) setErr((e as Error).message);
      }
    })();
    return () => {
      alive = false;
    };
  }, [pid, eid, router, attempt]);

  const stage = job?.stage === "Preparing clips" ? "Rendering" : job?.stage ?? "Queued";
  const idx = STAGES.indexOf(stage);

  return (
    <>
      <Header pid={pid} step="style" />
      <main className="mx-auto flex min-h-[70dvh] max-w-md flex-col justify-center gap-8 px-4">
        <div className="space-y-3 text-center">
          <h1 className="font-display text-4xl uppercase tracking-wide" aria-live="polite">
            {err ? "Couldn't finish" : NICE[job?.stage ?? "Queued"] ?? `${job?.stage}…`}
          </h1>
          {!err && <p className="text-sm tabular-nums text-ink-400">{Math.round((job?.progress ?? 0) * 100)}%</p>}
        </div>
        {!err && <Bar value={job?.progress ?? 0} />}
        <ol className="grid grid-cols-4 gap-2 text-center text-[10px] font-semibold uppercase tracking-wider">
          {STAGES.map((s, i) => (
            <li key={s} className={i < idx || job?.status === "finished" ? "text-bone" : i === idx ? "text-blood" : "text-ink-600"}>
              {s}
            </li>
          ))}
        </ol>
        {err && (
          <div className="space-y-3">
            <ErrorNote msg={err} />
            <div className="flex gap-3">
              <button className="btn-ghost flex-1" onClick={() => router.push(`/p/${pid}/style`)}>
                Back to styles
              </button>
              <button
                className="btn-primary flex-1"
                onClick={async () => {
                  setErr(null);
                  setJob(null);
                  const e = await api.edit(pid, eid);
                  const again = await api.createEdit(pid, e.settings);
                  router.replace(`/p/${pid}/generate/${again.id}`);
                  setAttempt((a) => a + 1);
                }}
              >
                Try again
              </button>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
