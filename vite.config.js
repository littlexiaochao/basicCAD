import { defineConfig } from 'vite';

export default defineConfig({
  // 使用相对路径，便于后续部署到任意子路径
  base: './',
  server: {
    host: true,
    port: 5173,
  },
});
