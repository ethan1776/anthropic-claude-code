// Typed client for the MOG API (served under /api, proxied to the Python server).
import type { Timeline } from "./timeline";

export type Project = {
  id: string;
  name: string;
  created_at: number;
  saved: boolean;
};

export type ProjectSummary = Project & { edits: number; thumb: string | null; latest_preview: string | null };

export type ClipInfo = {
  id: string;
  filename: string;
  size: number;
  url: string;
  thumb: string | null;
  analysed: boolean;
  duration: number | null;
  shots: number | null;
  warnings: string[];
};

export type EditSettings = {
  preset: string;
  length: number;
  section_start: number | null;
  intensity: number;
  seed: number;
  text: string | null;
  fps: 30 | 60;
};

export type Edit = {
  id: string;
  created_at: number;
  settings: EditSettings;
  status: "queued" | "running" | "done" | "failed";
  job_id: string | null;
  error: string | null;
  warnings: string[];
  preview_url: string | null;
  export_url: string | null;
  export_status: "none" | "queued" | "running" | "done" | "failed";
  export_job_id: string | null;
};

export type ProjectFull = Project & {
  clips: ClipInfo[];
  audio: null | { filename: string; size: number; analysed: boolean; playback: string | null };
  edits: Edit[];
};

export type Section = { start: number; end: number; drop: number | null; lead_bars: number; warnings: string[] };

export type SongAnalysis = {
  duration: number;
  bpm: number;
  beats: number[];
  downbeats: number[];
  drop: { t: number; confidence: number } | null;
  sections: Record<string, Section>;
  waveform: number[];
  playback: string;
  warnings: string[];
};

export type Job = { id: string; status: string; stage: string; progress: number; error: string | null };
export type Look = { id: string; name: string; description: string; image: string | null; preview: string };
export type ShotInfo = {
  source: string;
  clip: string;
  shot: number;
  start: number;
  end: number;
  stare: number;
  action: number;
  thumb: string | null;
};
export type Limits = {
  max_clips: number;
  max_upload_bytes: number;
  max_audio_bytes: number;
  video_types: string[];
  audio_types: string[];
  retention_hours: number;
};

export class ApiError extends Error {}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined,
    cache: "no-store",
  });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      msg = j.error || j.detail?.[0]?.msg || msg;
    } catch {
      /* not JSON */
    }
    throw new ApiError(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  limits: () => call<Limits>("/api/limits"),
  createProject: () => call<Project>("/api/projects", { method: "POST", body: JSON.stringify({}) }),
  projects: () => call<ProjectSummary[]>("/api/projects"),
  project: (pid: string) => call<ProjectFull>(`/api/projects/${pid}`),
  saveProject: (pid: string) => call<Project>(`/api/projects/${pid}/save`, { method: "POST" }),
  deleteClip: (pid: string, cid: string) => call(`/api/projects/${pid}/clips/${cid}`, { method: "DELETE" }),
  song: (pid: string) => call<SongAnalysis>(`/api/projects/${pid}/audio/analysis`),
  looks: (pid: string) => call<Look[]>(`/api/projects/${pid}/looks`),
  shots: (pid: string) => call<ShotInfo[]>(`/api/projects/${pid}/shots`),
  job: (id: string) => call<Job>(`/api/jobs/${id}`),
  createEdit: (pid: string, s: Partial<EditSettings>) =>
    call<Edit>(`/api/projects/${pid}/edits`, { method: "POST", body: JSON.stringify(s) }),
  edit: (pid: string, eid: string) => call<Edit & { timeline: Timeline | null }>(`/api/projects/${pid}/edits/${eid}`),
  op: (pid: string, eid: string, body: Record<string, unknown>) =>
    call<{ timeline: Timeline }>(`/api/projects/${pid}/edits/${eid}/ops`, { method: "POST", body: JSON.stringify(body) }),
  render: (pid: string, eid: string) => call<Edit>(`/api/projects/${pid}/edits/${eid}/render`, { method: "POST" }),
  reroll: (pid: string, eid: string) => call<Edit>(`/api/projects/${pid}/edits/${eid}/reroll`, { method: "POST" }),
  exportEdit: (pid: string, eid: string, fps: number, quality: string) =>
    call<Edit>(`/api/projects/${pid}/edits/${eid}/export`, { method: "POST", body: JSON.stringify({ fps, quality }) }),
};

/** Upload with progress (fetch can't report upload progress; XMLHttpRequest can). */
export function upload(
  url: string,
  file: File,
  onProgress: (frac: number) => void,
): Promise<{ job_id: string; id?: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body: { error?: string } & Record<string, unknown> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as { job_id: string; id?: string });
      else reject(new ApiError(body.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError("Network error during upload. Check your connection and try again."));
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

/** Poll a job until it finishes. */
export async function waitForJob(id: string, onUpdate: (j: Job) => void, intervalMs = 900): Promise<Job> {
  for (;;) {
    const j = await api.job(id);
    onUpdate(j);
    if (j.status === "finished" || j.status === "failed" || j.status === "canceled" || j.status === "stopped") return j;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
