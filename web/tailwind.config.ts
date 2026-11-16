import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#070708", 900: "#0d0d10", 800: "#16161b", 700: "#212128", 600: "#2e2e37", 400: "#7a7a88" },
        bone: "#ecebe6",
        blood: { DEFAULT: "#ff2d55", dim: "#b3203d" },
        frost: "#7fd6e0",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-anton)", "Impact", "sans-serif"],
      },
    },
  },
  plugins: [],
};
export default config;
