export const GENSPACE_LAYOUT_STORAGE_KEY = 'ltx-genspace-layout'

export const DEFAULT_PROMPT_BAR_HEIGHT = 160

export const PROMPT_BAR_HEIGHT_LIMITS = { min: 140, max: 400 } as const

export type LayoutStorage = Pick<Storage, 'getItem' | 'setItem'>

export function clampPromptBarHeight(height: number): number {
  return Math.max(
    PROMPT_BAR_HEIGHT_LIMITS.min,
    Math.min(PROMPT_BAR_HEIGHT_LIMITS.max, Math.round(height)),
  )
}

export function loadPromptBarHeight(storage: LayoutStorage = globalThis.localStorage): number {
  try {
    const stored = storage?.getItem(GENSPACE_LAYOUT_STORAGE_KEY)
    if (!stored) return DEFAULT_PROMPT_BAR_HEIGHT
    const parsed = JSON.parse(stored) as { promptBarHeight?: unknown }
    if (typeof parsed.promptBarHeight !== 'number' || !Number.isFinite(parsed.promptBarHeight)) {
      return DEFAULT_PROMPT_BAR_HEIGHT
    }
    return clampPromptBarHeight(parsed.promptBarHeight)
  } catch {
    return DEFAULT_PROMPT_BAR_HEIGHT
  }
}

export function savePromptBarHeight(height: number, storage: LayoutStorage = globalThis.localStorage): void {
  try {
    storage?.setItem(
      GENSPACE_LAYOUT_STORAGE_KEY,
      JSON.stringify({ promptBarHeight: clampPromptBarHeight(height) }),
    )
  } catch {
    // Private mode / quota — layout just won't persist.
  }
}
