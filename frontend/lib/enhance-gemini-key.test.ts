import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
  isEnhanceBlockedByMissingGeminiKey,
} from './enhance-gemini-key.ts'

const blocked = {
  enhanceAvailableForMode: true,
  enhanceProvider: 'api' as const,
  hasGeminiApiKey: false,
  hasEnhanceInput: true,
  isGenerationInProgressForEnhance: false,
  isOtherGenerationRunning: false,
}

describe('isEnhanceBlockedByMissingGeminiKey', () => {
  it('is true when Enhance (API) is selected and no Gemini key is configured', () => {
    assert.equal(isEnhanceBlockedByMissingGeminiKey(blocked), true)
  })

  it('is false when enhance is not available for the current mode', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, enhanceAvailableForMode: false }),
      false,
    )
  })

  it('is false when a Gemini key is already configured', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, hasGeminiApiKey: true }),
      false,
    )
  })

  it('is false when Enhance is using the local provider, even without a Gemini key', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, enhanceProvider: 'local' }),
      false,
    )
  })

  it('is false when there is no prompt or image to enhance', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, hasEnhanceInput: false }),
      false,
    )
  })

  it('is false while this project is already generating', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, isGenerationInProgressForEnhance: true }),
      false,
    )
  })

  it('is false while another project is generating', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, isOtherGenerationRunning: true }),
      false,
    )
  })
})

describe('GEMINI_KEY_REQUIRED_SETTINGS_DETAIL', () => {
  it('opens Settings on the API Keys tab for the Gemini key banner', () => {
    assert.deepEqual(GEMINI_KEY_REQUIRED_SETTINGS_DETAIL, {
      tab: 'apiKeys',
      reason: 'geminiKeyRequired',
    })
  })
})
