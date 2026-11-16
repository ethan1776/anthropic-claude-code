// Forwards /api/* to the Python API. A route handler (not next.config rewrites) so API_URL
// is read at request time: rewrites are baked in at build time in standalone Docker builds.
// Bodies are streamed both ways: big uploads and video range requests pass straight through.
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const HOP = ["connection", "keep-alive", "transfer-encoding", "upgrade", "host", "content-length"];

async function forward(req: NextRequest, { params }: { params: { path: string[] } }) {
  const base = process.env.API_URL || "http://localhost:8000";
  const url = `${base}/api/${params.path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const headers = new Headers(req.headers);
  HOP.forEach((h) => headers.delete(h));
  headers.set("x-forwarded-for", req.ip ?? req.headers.get("x-forwarded-for") ?? "");
  const hasBody = !["GET", "HEAD"].includes(req.method);
  let res: Response;
  try {
    res = await fetch(url, {
      method: req.method,
      headers,
      body: hasBody ? req.body : undefined,
      // @ts-expect-error: Node fetch needs duplex for streamed request bodies
      duplex: hasBody ? "half" : undefined,
      redirect: "manual",
      cache: "no-store",
    });
  } catch {
    return Response.json({ error: "The MOG server isn't reachable. Is the API running?" }, { status: 502 });
  }
  const out = new Headers(res.headers);
  ["connection", "keep-alive", "transfer-encoding", "content-encoding"].forEach((h) => out.delete(h));
  return new Response(res.body, { status: res.status, headers: out });
}

export { forward as GET, forward as POST, forward as PUT, forward as DELETE, forward as PATCH, forward as HEAD };
