import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

const matrix = path.resolve(import.meta.dirname, '../scripts/sandbox/generate-e2e-matrix.mjs')
const picker = path.resolve(import.meta.dirname, '../scripts/sandbox/pick-release-tags.sh')

describe('parseTagsJson', () => {
  it('treats empty --tags as no annotations so a skipped pick-releases job cannot crash the chart', async () => {
    const { parseTagsJson } = await import(/* @vite-ignore */ new URL('../scripts/sandbox/generate-e2e-matrix.mjs', import.meta.url).href)
    expect(parseTagsJson('')).toEqual([])
    expect(parseTagsJson('   ')).toEqual([])
    expect(parseTagsJson(undefined)).toEqual([])
    expect(parseTagsJson('[]')).toEqual([])
    expect(parseTagsJson('[{"ref":"v2026.9.11","desktop":true}]')).toEqual([
      { ref: 'v2026.9.11', desktop: true },
    ])
  })

  it('CLI --format results survives the empty-string --tags the report job passes on a failed pick', () => {
    const r = spawnSync(process.execPath, [matrix, '--format', 'results', '--tags', ''], {
      encoding: 'utf8',
      input: '',
    })
    expect(r.status, r.stderr).toBe(0)
    expect(r.stdout).toContain('### Install & Update E2E results')
  })
})

describe('pick-release-tags', () => {
  const env = { ...process.env }
  delete env.GITHUB_ACTIONS

  function git(cwd: string, ...args: string[]) {
    const r = spawnSync('git', ['-c', 'init.defaultBranch=main', ...args], { cwd, encoding: 'utf8', env })
    if (r.status !== 0) throw new Error(r.stderr || r.stdout)
    return r
  }

  it('fails closed on a checkout with no release tags (and does not fetch unless GITHUB_ACTIONS=true)', () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), 'pick-tags-empty-'))
    try {
      git(dir, 'init')
      git(dir, 'config', 'user.email', 't@example.com')
      git(dir, 'config', 'user.name', 't')
      writeFileSync(path.join(dir, 'f'), 'x')
      git(dir, 'add', 'f')
      git(dir, 'commit', '-m', 'init')
      const r = spawnSync('bash', [picker, '--repo', dir, '--count', '2'], { encoding: 'utf8', env })
      expect(r.status).toBe(1)
      expect(r.stderr).toMatch(/no release tags found/)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('emits oldest and newest when asked for two', () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), 'pick-tags-span-'))
    try {
      git(dir, 'init')
      git(dir, 'config', 'user.email', 't@example.com')
      git(dir, 'config', 'user.name', 't')
      writeFileSync(path.join(dir, 'f'), 'x')
      git(dir, 'add', 'f')
      git(dir, 'commit', '-m', 'init')
      for (const tag of ['v2026.1.1', 'v2026.6.1', 'v2026.9.11']) git(dir, 'tag', tag)
      const r = spawnSync('bash', [picker, '--repo', dir, '--count', '2'], { encoding: 'utf8', env })
      expect(r.status, r.stderr).toBe(0)
      expect(JSON.parse(r.stdout)).toEqual(['v2026.1.1', 'v2026.9.11'])
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
