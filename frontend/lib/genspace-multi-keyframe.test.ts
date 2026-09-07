import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  autoDurationOptionVisible,
  canUseMultiKeyframeMode,
  fallbackGenSpaceMode,
  genSpaceUsesAudioInput,
  isGenSpaceLibraryMode,
  isEnhanceAvailableForMode,
  modeAfterCompletedGeneration,
  modeOptionValues,
} from './genspace-multi-keyframe.ts'

const allModesAvailable = {
  canUseMultiKeyframe: true,
  canUseRetake: true,
  canUseExtend: true,
  canUseIcLora: true,
}

describe('autoDurationOptionVisible', () => {
  it('hides Auto in multi-keyframe mode even when the model supports it', () => {
    assert.equal(autoDurationOptionVisible('multi-keyframe', true), false)
  })

  it('shows Auto in video mode when the model supports it', () => {
    assert.equal(autoDurationOptionVisible('video', true), true)
  })

  it('hides Auto when the model does not support it', () => {
    assert.equal(autoDurationOptionVisible('video', false), false)
  })
})

describe('canUseMultiKeyframeMode', () => {
  it('is false in API mode even when the local offering supports it', () => {
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: false,
        localCaps: { multi_keyframe: true },
        enableMultipleKeyframesVideos: true,
      }),
      false,
    )
  })

  it('is false in local mode when the capability is missing', () => {
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: {},
        enableMultipleKeyframesVideos: true,
      }),
      false,
    )
  })

  it('is false in local mode when the capability is disabled', () => {
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: { multi_keyframe: false },
        enableMultipleKeyframesVideos: true,
      }),
      false,
    )
  })

  it('is false when the Dev Panel flag is off, even in local mode with the capability', () => {
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: { multi_keyframe: true },
        enableMultipleKeyframesVideos: false,
      }),
      false,
    )
  })

  it('is true only in local mode when the capability and Dev Panel flag are enabled', () => {
    assert.equal(
      canUseMultiKeyframeMode({
        isLocalMode: true,
        localCaps: { multi_keyframe: true },
        enableMultipleKeyframesVideos: true,
      }),
      true,
    )
  })
})

describe('fallbackGenSpaceMode', () => {
  it('falls back from multi-keyframe when unavailable', () => {
    assert.equal(
      fallbackGenSpaceMode('multi-keyframe', {
        ...allModesAvailable,
        canUseMultiKeyframe: false,
      }),
      'video',
    )
  })

  for (const [mode, flag] of [
    ['retake', 'canUseRetake'],
    ['extend', 'canUseExtend'],
    ['ic-lora', 'canUseIcLora'],
  ] as const) {
    it(`falls back from ${mode} when unavailable`, () => {
      assert.equal(
        fallbackGenSpaceMode(mode, {
          ...allModesAvailable,
          [flag]: false,
        }),
        'video',
      )
    })

    it(`keeps ${mode} when available`, () => {
      assert.equal(fallbackGenSpaceMode(mode, allModesAvailable), mode)
    })
  }
})

describe('modeOptionValues', () => {
  it('excludes multi-keyframe when unavailable', () => {
    const values = modeOptionValues({
      ...allModesAvailable,
      canUseMultiKeyframe: false,
    })

    assert.equal(values.includes('multi-keyframe'), false)
  })

  it('includes multi-keyframe immediately after video when available', () => {
    const values = modeOptionValues(allModesAvailable)

    assert.deepEqual(values.slice(0, 3), ['image', 'video', 'multi-keyframe'])
  })
})

describe('isGenSpaceLibraryMode', () => {
  it('shows the asset library for image and video', () => {
    assert.equal(isGenSpaceLibraryMode('image'), true)
    assert.equal(isGenSpaceLibraryMode('video'), true)
  })

  it('hides the asset library in tool modes, matching retake and extend', () => {
    assert.equal(isGenSpaceLibraryMode('multi-keyframe'), false)
    assert.equal(isGenSpaceLibraryMode('retake'), false)
    assert.equal(isGenSpaceLibraryMode('extend'), false)
    assert.equal(isGenSpaceLibraryMode('ic-lora'), false)
  })
})

describe('modeAfterCompletedGeneration', () => {
  it('returns to video gen space after a multi-keyframe job', () => {
    assert.equal(modeAfterCompletedGeneration('multi-keyframe'), 'video')
  })

  it('does not force a mode change for ordinary video jobs', () => {
    assert.equal(modeAfterCompletedGeneration('text-to-video'), null)
    assert.equal(modeAfterCompletedGeneration('image-to-video'), null)
    assert.equal(modeAfterCompletedGeneration('audio-to-video'), null)
  })
})

describe('isEnhanceAvailableForMode', () => {
  it('includes multi-keyframe alongside video, image, and ic-lora', () => {
    assert.equal(isEnhanceAvailableForMode('multi-keyframe'), true)
    assert.equal(isEnhanceAvailableForMode('video'), true)
    assert.equal(isEnhanceAvailableForMode('image'), true)
    assert.equal(isEnhanceAvailableForMode('ic-lora'), true)
  })

  it('hides Enhance in retake and extend', () => {
    assert.equal(isEnhanceAvailableForMode('retake'), false)
    assert.equal(isEnhanceAvailableForMode('extend'), false)
  })
})

describe('genSpaceUsesAudioInput', () => {
  it('is only video mode, so leftover A2V audio cannot ride along with keyframes', () => {
    assert.equal(genSpaceUsesAudioInput('video'), true)
    assert.equal(genSpaceUsesAudioInput('multi-keyframe'), false)
    assert.equal(genSpaceUsesAudioInput('image'), false)
    assert.equal(genSpaceUsesAudioInput('retake'), false)
    assert.equal(genSpaceUsesAudioInput('extend'), false)
    assert.equal(genSpaceUsesAudioInput('ic-lora'), false)
  })
})
