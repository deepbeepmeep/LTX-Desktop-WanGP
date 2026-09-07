import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { DEFAULT_KEYFRAME_STRENGTH, type KeyframeItem } from './multi-keyframe.ts'
import { buildGenerateVideoImageInputs } from './build-generate-video-body.ts'

const keyframes: KeyframeItem[] = [
  { id: 'opening', path: '/frames/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
  { id: 'ending', path: '/frames/ending.png', frameIndex: 121, strength: DEFAULT_KEYFRAME_STRENGTH },
]

describe('buildGenerateVideoImageInputs', () => {
  it('maps multi-keyframes to the backend request shape', () => {
    assert.deepEqual(
      buildGenerateVideoImageInputs({
        mode: 'multi-keyframe',
        imagePath: null,
        lastImagePath: null,
        keyframes,
      }),
      {
        keyframes: [
          { imagePath: '/frames/opening.png', frameIndex: 0, strength: DEFAULT_KEYFRAME_STRENGTH },
          { imagePath: '/frames/ending.png', frameIndex: 121, strength: DEFAULT_KEYFRAME_STRENGTH },
        ],
      },
    )
  })

  it('omits image inputs when multi-keyframes are present', () => {
    const result = buildGenerateVideoImageInputs({
      mode: 'multi-keyframe',
      imagePath: '/frames/first.png',
      lastImagePath: '/frames/last.png',
      keyframes,
    })

    assert.equal('imagePath' in result, false)
    assert.equal('lastImagePath' in result, false)
    assert.equal('keyframes' in result, true)
  })

  it('forwards each still\'s owned strength on generate', () => {
    assert.deepEqual(
      buildGenerateVideoImageInputs({
        mode: 'multi-keyframe',
        imagePath: null,
        lastImagePath: null,
        keyframes: [
          { id: 'opening', path: '/frames/opening.png', frameIndex: 0, strength: 0.7 },
        ],
      }),
      {
        keyframes: [{ imagePath: '/frames/opening.png', frameIndex: 0, strength: 0.7 }],
      },
    )
  })

  it('preserves image inputs outside multi-keyframe mode', () => {
    assert.deepEqual(
      buildGenerateVideoImageInputs({
        mode: 'video',
        imagePath: '/frames/first.png',
        lastImagePath: '/frames/last.png',
        keyframes,
      }),
      {
        imagePath: '/frames/first.png',
        lastImagePath: '/frames/last.png',
      },
    )
  })
})
