import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { applyTimecode, nudgeKeyframe } from './keyframe-controls.ts'

const keyframes = [
  { id: 'opening', frameIndex: 10 },
  { id: 'middle', frameIndex: 11 },
  { id: 'closing', frameIndex: 20 },
]

describe('nudgeKeyframe', () => {
  it('moves a keyframe one frame in the requested direction', () => {
    assert.equal(nudgeKeyframe(keyframes, 'closing', -1, 30), 19)
  })

  it('never lands on another keyframe', () => {
    const frameIndex = nudgeKeyframe(keyframes, 'opening', 1, 30)

    assert.equal(frameIndex, 12)
    assert.equal(
      keyframes.some((keyframe) => (
        keyframe.id !== 'opening' && keyframe.frameIndex === frameIndex
      )),
      false,
    )
  })

  it('returns null for an unknown keyframe', () => {
    assert.equal(nudgeKeyframe(keyframes, 'missing', 1, 30), null)
  })
})

describe('applyTimecode', () => {
  it('converts a valid timecode to a frame index', () => {
    assert.equal(applyTimecode(keyframes, 'opening', '00:00.13', 25, 30), 13)
  })

  it('moves a colliding timecode to the nearest free frame', () => {
    const frameIndex = applyTimecode(keyframes, 'opening', '00:00.11', 25, 30)

    assert.equal(frameIndex, 12)
    assert.equal(
      keyframes.some((keyframe) => (
        keyframe.id !== 'opening' && keyframe.frameIndex === frameIndex
      )),
      false,
    )
  })

  it('returns null for invalid timecode', () => {
    assert.equal(applyTimecode(keyframes, 'opening', 'invalid', 25, 30), null)
  })
})
