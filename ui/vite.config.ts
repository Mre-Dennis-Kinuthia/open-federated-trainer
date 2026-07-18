import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev UI proxies to the HA TLS edge by default (production compose).
// Override with VITE_API_PROXY=http://127.0.0.1:8000 for the JSON single-node stack.
const apiTarget =
  process.env.VITE_API_PROXY?.trim() || "https://127.0.0.1:8443";

// Default "/" for Vercel/static hosts. Use VITE_BASE=/ui/ when serving under
// the coordinator's /ui path.
const base = process.env.VITE_BASE?.trim() || "/";

export default defineConfig({
  plugins: [react()],
  base,
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
