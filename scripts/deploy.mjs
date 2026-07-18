#!/usr/bin/env node

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { spawnSync } from 'node:child_process'

const root = resolve(import.meta.dirname, '..')
const dist = resolve(root, 'dist')
const spaceRepo = process.env.HF_SPACE_REPO ?? 'Yosun/TripoSplat-WebGPU-Demo'
const spaceCard = process.env.HF_SPACE_CARD ?? 'huggingface-space/README.md'
const hfClientVersion = process.env.HF_HUB_VERSION ?? '1.23.0'
const vercelScope = process.env.VERCEL_SCOPE
const skipBuild = process.argv.includes('--skip-build')

function run(command, args, options = {}) {
  console.log(`\n$ ${command} ${args.join(' ')}`)
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: options.input === undefined ? 'inherit' : ['pipe', 'inherit', 'inherit'],
    input: options.input,
    ...options,
  })
  if (result.error) throw result.error
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status ?? 'unknown'}.`)
  }
}

function requirePath(path, description) {
  if (!existsSync(path)) throw new Error(`${description} not found: ${path}`)
}

if (!skipBuild) run('pnpm', ['build'])
requirePath(dist, 'Vite output directory')
requirePath(resolve(dist, 'index.html'), 'Vite output')

const python = `
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = os.environ['DEPLOY_HF_SPACE_REPO']
folder = Path(os.environ['DEPLOY_DIST']).resolve()
card = Path(os.environ['DEPLOY_SPACE_CARD']).resolve()
api = HfApi()

app_commit = api.upload_folder(
    repo_id=repo_id,
    repo_type='space',
    folder_path=str(folder),
    path_in_repo='',
    commit_message=os.environ['DEPLOY_APP_MESSAGE'],
)
print(f'Hugging Face app commit: {app_commit}')

card_commit = api.upload_file(
    repo_id=repo_id,
    repo_type='space',
    path_or_fileobj=str(card),
    path_in_repo='README.md',
    commit_message=os.environ['DEPLOY_CARD_MESSAGE'],
)
print(f'Hugging Face card commit: {card_commit}')
`

run('uv', ['run', '--with', `huggingface_hub==${hfClientVersion}`, 'python', '-'], {
  input: python,
  env: {
    ...process.env,
    DEPLOY_HF_SPACE_REPO: spaceRepo,
    DEPLOY_DIST: dist,
    DEPLOY_SPACE_CARD: resolve(root, spaceCard),
    DEPLOY_APP_MESSAGE: process.env.HF_APP_COMMIT_MESSAGE ?? 'Deploy TripoSplat WebGPU app',
    DEPLOY_CARD_MESSAGE: process.env.HF_CARD_COMMIT_MESSAGE ?? 'Update TripoSplat WebGPU Space card',
  },
})

const vercelArgs = ['--prod', '--yes']
if (vercelScope) vercelArgs.push('--scope', vercelScope)
run('vercel', vercelArgs)

console.log('\nDeployment completed for Hugging Face and Vercel.')
