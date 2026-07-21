import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { cpSync, mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'

const APP_VERSION = '0.1.0-debug.1'
const APP_BUILD_TIME = new Date().toISOString()
const APP_BUILD_LABEL = APP_BUILD_TIME.slice(0, 16).replace('T', ' ') + 'Z'

/**
 * Model weights and parity fixtures live under public/ for local development,
 * but they must never be copied into an application deployment. Besides being
 * hosted separately in production, the export workspace contains several large
 * diagnostic graphs that intentionally share hard-linked sidecars. Vite's
 * ordinary public-directory copy expands those links into many gigabytes.
 *
 * Only the ORT WASM fallback assets and the social-preview image are
 * application-owned static files.
 */
function copyRuntimeAssets(): Plugin {
  return {
    name: 'copy-runtime-assets',
    transformIndexHtml(html) {
      return html
        .replaceAll('{{APP_VERSION}}', APP_VERSION)
        .replaceAll('{{APP_BUILD_TIME}}', APP_BUILD_TIME)
        .replaceAll('{{APP_BUILD_LABEL}}', APP_BUILD_LABEL)
    },
    configureServer(server) {
      server.middlewares.use((request, _response, next) => {
        if (request.url?.startsWith('/e2e-web.html')) {
          request.url = request.url.replace('/e2e-web.html', '/e2e-web-debug.html')
        }
        next()
      })
    },
    closeBundle() {
      const outputDirectory = resolve('dist')
      const ortOutputDirectory = resolve(outputDirectory, 'ort')
      rmSync(ortOutputDirectory, { recursive: true, force: true })
      mkdirSync(ortOutputDirectory, { recursive: true })
      for (const file of [
        'ort-wasm-simd-threaded.asyncify.mjs',
        'ort-wasm-simd-threaded.asyncify.wasm',
      ]) {
        cpSync(resolve('public/ort', file), resolve(ortOutputDirectory, file))
      }
      cpSync(resolve('public/vite.svg'), resolve(outputDirectory, 'vite.svg'))
      cpSync(
        resolve('public/corgi.ceo_image_header.social.jpg'),
        resolve(outputDirectory, 'corgi.ceo_image_header.social.jpg'),
      )

      // Keep the established production route as the default while exposing
      // the diagnostic build at a second path on the same origin. Both pages
      // therefore share the browser's verified model cache.
      cpSync(
        resolve(outputDirectory, 'e2e-web-debug.html'),
        resolve(outputDirectory, 'e2e-web.html'),
      )
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), copyRuntimeAssets()],
  build: {
    // Do not deploy local model artifacts or deterministic parity fixtures.
    copyPublicDir: false,
    rollupOptions: {
      input: {
        app: 'index.html',
        learn: 'learn.html',
        e2eWebDebug: 'e2e-web-debug.html',
        e2eWebChecks: 'e2e-web-checks.html',
        e2eWebAnim: 'e2e-web-anim.html',
        sharpLab: 'sharp-lab.html',
        encoderLab: 'encoder-lab.html',
        dinoLab: 'dino-lab.html',
        ditLab: 'dit-lab.html',
        ditProfileLab: 'dit-profile-lab.html',
        flowLab: 'flow-lab.html',
        octreeLab: 'octree-lab.html',
        gaussianLab: 'gaussian-lab.html',
        e2eLab: 'e2e-lab.html',
      },
    },
  },
  // dialkit and motion each import react; without deduping, Vite can hand them a
  // separate React instance than the app, triggering "Invalid hook call".
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
})
