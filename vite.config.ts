import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The bot's engine runs on the box (cfr/, `hu serve`). In development the dev
// server forwards /api to it, so the app always calls a relative /api/act —
// the same path it uses when the box serves the built app itself.
const ENGINE = process.env.HU_ENGINE ?? "http://192.168.0.185:8790";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": ENGINE },
  },
});
