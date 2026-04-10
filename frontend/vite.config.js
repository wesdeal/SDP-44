import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Vite configuration.
 *
 * DEV PROXY
 * ─────────
 * When VITE_API_BASE is set, the dev server proxies two paths to the backend:
 *
 *   /api/*   → VITE_API_BASE/api/*    (artifact JSON endpoints)
 *   /plots/* → VITE_API_BASE/plots/*  (evaluation plot images)
 *
 * This lets `fetch('/api/runs/...')` work during development without CORS
 * issues, even when the backend runs on a different port.
 *
 * If VITE_API_BASE is not set, no proxy is configured (mock mode has no
 * network requests to forward).
 *
 * See frontend/INTEGRATION.md for setup instructions.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiBase = env.VITE_API_BASE;

  const proxy = apiBase
    ? {
        '/api': {
          target: apiBase,
          changeOrigin: true,
          secure: false,
        },
        '/plots': {
          target: apiBase,
          changeOrigin: true,
          secure: false,
        },
      }
    : undefined;

  return {
    plugins: [react()],
    server: {
      port: 5173,
      open: false,
      ...(proxy ? { proxy } : {}),
    },
  };
});
