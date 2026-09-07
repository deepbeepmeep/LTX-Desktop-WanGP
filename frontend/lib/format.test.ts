import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { formatBytes } from './format.ts'

describe('formatBytes', () => {
  it('formats zero', () => {
    assert.equal(formatBytes(0), '0 B')
  })

  it('formats megabytes with one decimal', () => {
    assert.equal(formatBytes(996_000_000), '949.9 MB')
  })

  it('formats terabytes instead of overflowing the unit list', () => {
    assert.equal(formatBytes(1.8 * 1024 ** 4), '1.8 TB')
  })
})
