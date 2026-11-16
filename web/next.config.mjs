/** @type {import('next').NextConfig} */
// The browser talks to /api on this server; app/api/[...path]/route.ts forwards it to the
// Python API at API_URL (read at request time).
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
};

export default nextConfig;
