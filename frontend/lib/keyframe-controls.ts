import {
  findNearestFreeFrameIndex,
  parseKeyframeTimecode,
} from './keyframe-timeline.ts'

type PositionedKeyframe = {
  id: string
  frameIndex: number
}

function placementContext(
  keyframes: readonly PositionedKeyframe[],
  id: string,
): { current: PositionedKeyframe; others: PositionedKeyframe[] } | null {
  const current = keyframes.find((keyframe) => keyframe.id === id)
  if (!current) return null

  return {
    current,
    others: keyframes.filter((keyframe) => keyframe.id !== id),
  }
}

export function nudgeKeyframe(
  keyframes: readonly PositionedKeyframe[],
  id: string,
  delta: -1 | 1,
  lastFrame: number,
): number | null {
  const context = placementContext(keyframes, id)
  if (!context) return null

  return findNearestFreeFrameIndex(
    context.others,
    context.current.frameIndex + delta,
    lastFrame,
  )
}

export function applyTimecode(
  keyframes: readonly PositionedKeyframe[],
  id: string,
  value: string,
  fps: number,
  lastFrame: number,
): number | null {
  const context = placementContext(keyframes, id)
  const parsedFrame = parseKeyframeTimecode(value, fps)
  if (!context || parsedFrame === null) return null

  return findNearestFreeFrameIndex(context.others, parsedFrame, lastFrame)
}
