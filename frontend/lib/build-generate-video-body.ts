import type { GenSpaceMode } from './genspace-multi-keyframe'
import { toKeyframeInputs, type KeyframeItem } from './multi-keyframe.ts'

interface GenerateVideoImageInputs {
  imagePath?: string
  lastImagePath?: string
  keyframes?: Array<{
    imagePath: string
    frameIndex: number
    strength: number
  }>
}

export function buildGenerateVideoImageInputs({
  mode,
  imagePath,
  lastImagePath,
  keyframes,
}: {
  mode: GenSpaceMode
  imagePath: string | null | undefined
  lastImagePath: string | null | undefined
  keyframes: KeyframeItem[]
}): GenerateVideoImageInputs {
  if (mode === 'multi-keyframe' && keyframes.length > 0) {
    return { keyframes: toKeyframeInputs(keyframes) }
  }

  return {
    ...(imagePath ? { imagePath } : {}),
    ...(lastImagePath ? { lastImagePath } : {}),
  }
}
