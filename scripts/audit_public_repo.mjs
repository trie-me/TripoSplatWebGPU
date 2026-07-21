import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, realpathSync, statSync, writeFileSync } from 'node:fs'
import { basename, sep } from 'node:path'

const fixLocalPaths = process.argv.includes('--fix-local-paths')
const root = realpathSync(process.cwd())
const rootPrefix = `${root}${sep}`
const files = execFileSync(
  'git',
  ['ls-files', '-z', '--cached', '--others', '--exclude-standard'],
  { encoding: 'utf8' },
)
  .split('\0')
  .filter(Boolean)
const issues = []
const blockedNames = new Set(['.DS_Store', '.env', '.env.local'])
const localPathPattern = new RegExp(
  '(?:/' + 'Users/[^/\\s"\']+|/' + 'home/[^/\\s"\']+|[A-Za-z]:\\\\Users\\\\[^\\\\\\s"\']+)[^\\n"\']*',
  'g',
)
const secretPatterns = [
  ['private key', new RegExp('BEGIN ' + '(?:RSA |EC |OPENSSH )?PRIVATE KEY')],
  ['GitHub token', new RegExp('gh' + '[pousr]_[A-Za-z0-9_]{30,}')],
  ['AWS access key', new RegExp('AK' + 'IA[0-9A-Z]{16}')],
]

for (const file of files) {
  if (!existsSync(file)) continue
  const size = statSync(file).size
  if (blockedNames.has(basename(file))) issues.push(`${file}: tracked local/secret file`)
  if (size > 50 * 1024 * 1024) issues.push(`${file}: ${(size / 1024 / 1024).toFixed(1)} MiB tracked file`)
  if (size > 20 * 1024 * 1024) continue

  let content = readFileSync(file, 'utf8')
  if (content.includes('\0')) continue
  if (fixLocalPaths && content.includes(rootPrefix)) {
    content = content.replaceAll(rootPrefix, '$REPO/')
    writeFileSync(file, content)
  }

  if (!file.startsWith('public/ort/')) {
    for (const match of content.matchAll(localPathPattern)) {
      issues.push(`${file}: machine-local path ${match[0].slice(0, 120)}`)
    }
  }
  for (const [label, pattern] of secretPatterns) {
    if (pattern.test(content)) issues.push(`${file}: possible ${label}`)
  }
}

if (issues.length > 0) {
  console.error(`Public repository audit failed with ${issues.length} issue(s):`)
  for (const issue of issues.slice(0, 100)) console.error(`- ${issue}`)
  if (issues.length > 100) console.error(`- ...and ${issues.length - 100} more`)
  process.exitCode = 1
} else {
  console.log(`Public repository audit passed for ${files.length} candidate files.`)
}
