"""Resolve the vision stills for an image-conditioned enhance call."""

from __future__ import annotations

from collections.abc import Sequence

# path, frame index, lock strength (0–1). Strength is owned by the still; 1.0 is a full lock.
KeyframeStill = tuple[str, int, float]


def _keyframe_label(index: int, frame_idx: int, fps: int | float | None, strength: float) -> str:
    clock = f" ({frame_idx / fps:.2f}s)" if fps is not None and fps > 0 else ""
    return f"Keyframe {index + 1}: frame {frame_idx}{clock}, strength {strength:g}."


def resolve_i2v_frames(
    image_path: str,
    last_image_path: str | None = None,
    keyframes: Sequence[KeyframeStill] | None = None,
    fps: int | float | None = None,
) -> list[tuple[str, str | None]]:
    """Return ``(path, label)`` pairs in vision order.

    A missing label means the still is unlabeled (single-image i2v, or a lone
    multi-keyframe still when fps is unknown). First/last i2v keeps the existing
    ``First frame:`` / ``Last frame:`` copy. Multi-keyframe stills are numbered
    and tagged with their frame index, clock time when fps is known, and lock
    strength so the rewriter can weight identity lock vs interpolation.
    """
    if keyframes:
        ordered = sorted(keyframes, key=lambda item: (item[1], item[0]))
        if len(ordered) == 1 and (fps is None or fps <= 0):
            return [(ordered[0][0], None)]
        return [
            (path, _keyframe_label(index, frame_idx, fps, strength))
            for index, (path, frame_idx, strength) in enumerate(ordered)
        ]
    if last_image_path:
        return [(image_path, "First frame:"), (last_image_path, "Last frame:")]
    return [(image_path, None)]
