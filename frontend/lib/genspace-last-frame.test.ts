import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { shouldShowLastFrameChip } from './genspace-last-frame.ts'

describe('shouldShowLastFrameChip', () => {
  it('is hidden without a first frame', () => {
    assert.equal(
      shouldShowLastFrameChip({ mode: 'video', hasFirstFrame: false, duration: 5 }),
      false,
    )
  })

  it('is hidden when duration is auto', () => {
    assert.equal(
      shouldShowLastFrameChip({ mode: 'video', hasFirstFrame: true, duration: null }),
      false,
    )
  })

  it('is hidden outside video mode', () => {
    assert.equal(
      shouldShowLastFrameChip({ mode: 'image', hasFirstFrame: true, duration: 5 }),
      false,
    )
  })

  it('is shown for video with a first frame and a concrete duration', () => {
    assert.equal(
      shouldShowLastFrameChip({ mode: 'video', hasFirstFrame: true, duration: 5 }),
      true,
    )
  })
})
