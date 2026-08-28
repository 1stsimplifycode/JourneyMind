import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build lands directly in the backend's static directory, so one FastAPI
// process serves both the API and the UI and deployment stays a single service.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../backend/app/static',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8011',
      '/health': 'http://127.0.0.1:8011',
    },
  },
})
