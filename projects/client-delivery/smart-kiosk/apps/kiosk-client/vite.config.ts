import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    // The development VM is accessed through a forwarded host name rather than
    // directly as localhost. Keep this limited to Vite's development server.
    allowedHosts: true,
    proxy: {
      '/v1': 'http://127.0.0.1:3001',
      '/health': 'http://127.0.0.1:3001',
      '/ready': 'http://127.0.0.1:3001'
    }
  }
});
