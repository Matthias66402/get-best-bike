import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5181,
    strictPort: true,
    proxy: {
      // Backend läuft im Docker-Container auf Port 8011
      "/api": { target: "http://localhost:8011", changeOrigin: true },
    },
  },
});
