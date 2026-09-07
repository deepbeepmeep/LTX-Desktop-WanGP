export const GEMINI_KEY_REQUIRED_SETTINGS_DETAIL = {
  tab: 'apiKeys' as const,
  reason: 'geminiKeyRequired' as const,
}

export function isEnhanceBlockedByMissingGeminiKey(input: {
  enhanceAvailableForMode: boolean
  enhanceProvider: 'local' | 'api'
  hasGeminiApiKey: boolean
  hasEnhanceInput: boolean
  isGenerationInProgressForEnhance: boolean
  isOtherGenerationRunning: boolean
}): boolean {
  // True when Enhance would run via Gemini but no key is configured — including when local
  // Enhance is available and the user explicitly picked API. Clicking then opens Settings
  // instead of hiding the API option.
  return (
    input.enhanceAvailableForMode
    && input.enhanceProvider === 'api'
    && !input.hasGeminiApiKey
    && input.hasEnhanceInput
    && !input.isGenerationInProgressForEnhance
    && !input.isOtherGenerationRunning
  )
}
