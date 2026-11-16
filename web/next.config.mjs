/** @type {import('next').NextConfig} */
const API_URL = process.env.API_URL || "http://localhost:8000";

const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // The browser talks to /api on this server; Next forwards it to the Python API.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
  experimental: { proxyTimeout: 10 * 60 * 1000 },
};

export default nextConfig;
