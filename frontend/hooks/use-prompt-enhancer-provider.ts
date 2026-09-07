import { useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { useAppSettings } from '../contexts/AppSettingsContext'

export type EnhanceProvider = 'local' | 'api'

interface UsePromptEnhancerProviderResult {
  // Local requires the Gemma text-encoder checkpoint to be downloaded AND local generation to
  // actually be usable this run (e.g. not memory-constrained into API-only mode); API requires a
  // stored Gemini key to actually run, but the option stays selectable without one so Enhance
  // can send the user to Settings instead of hiding the choice.
  hasLocalTextEncoder: boolean
  hasGeminiApiKey: boolean
  // The provider Enhance will actually use: the persisted preference when it's currently
  // choosable. API remains choosable without a key (clicking Enhance then opens Settings).
  // Local that's temporarily unavailable (e.g. memory-constrained run) falls back silently —
  // it does NOT overwrite the persisted preference, which only an explicit setProviderPreference
  // call changes.
  provider: EnhanceProvider
  // Shown when local Enhance is available, so the user can still pick API (Gemini) before
  // they've added a key. Hidden when local isn't an option — the button is already API-only.
  canToggleProvider: boolean
  setProviderPreference: (provider: EnhanceProvider) => void
}

// Single source of truth for which prompt-enhancer provider (local Gemma text encoder vs.
// Gemini's hosted API) is available and which one Enhance should use. `enabled` gates the local
// checkpoint lookup so it only fires once the enhancer could plausibly be shown for the current
// mode.
export function usePromptEnhancerProvider(enabled: boolean): UsePromptEnhancerProviderResult {
  const {
    settings: { hasGeminiApiKey, promptEnhancerProviderPreference },
    updateSettings,
    forceApiGenerations,
    modelsVersion,
  } = useAppSettings()

  const [isLocalEncoderUsable, setIsLocalEncoderUsable] = useState(false)
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    void ApiClient.getTextEncoderRecommendation().then((result) => {
      // Deliberately not cp_to_download: the encoder that runs generations isn't always the one
      // that can enhance (LTX 2.5's encodes only, and enhances from a separate checkpoint), so
      // the backend reports enhancer availability on its own.
      if (!cancelled) {
        setIsLocalEncoderUsable(result.ok && result.data.local_enhancement_supported)
      }
    })
    return () => { cancelled = true }
    // modelsVersion: the enhancer is a download the user can make mid-session, and Enhance should
    // become available without a restart.
  }, [enabled, modelsVersion])

  // Downloaded isn't enough on its own — forceApiGenerations is the pure "insufficient memory
  // for local models this run" signal (deliberately NOT shouldVideoGenerateWithLtxApi, which
  // also folds in the user's own preference to use the LTX API for VIDEO specifically — that's
  // unrelated to whether the much smaller Gemma text encoder can run locally right now).
  const hasLocalTextEncoder = isLocalEncoderUsable && !forceApiGenerations
  const canToggleProvider = hasLocalTextEncoder

  // Default to local when the user hasn't made an explicit choice, or when they asked for
  // local and it's currently usable. API preference is honored even without a Gemini key so
  // the Enhance (API) option isn't silently replaced by local.
  const provider: EnhanceProvider =
    promptEnhancerProviderPreference === 'api' ? 'api'
    : promptEnhancerProviderPreference === 'local' && hasLocalTextEncoder ? 'local'
    : hasLocalTextEncoder ? 'local'
    : 'api'

  const setProviderPreference = (next: EnhanceProvider) => {
    updateSettings({ promptEnhancerProviderPreference: next })
  }

  return {
    hasLocalTextEncoder,
    hasGeminiApiKey,
    provider,
    canToggleProvider,
    setProviderPreference,
  }
}
