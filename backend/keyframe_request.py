"""Validation for multi-keyframe inputs on GenerateVideoRequest."""

from __future__ import annotations

from api_types import GenerateVideoRequest
from frame_math import compute_num_frames
from server_utils.media_validation import normalize_optional_path


def validate_keyframe_inputs(
    req: GenerateVideoRequest,
    *,
    use_api_specs: bool,
    image_path: str | None,
    last_image_path: str | None,
    audio_path: str | None,
) -> str | None:
    entries = req.keyframes
    if not entries:
        return None

    if use_api_specs:
        return "Multi-keyframe generation is only available for local generation"

    if image_path or last_image_path:
        return "Keyframes cannot be combined with a first or last frame"

    if audio_path:
        return "Keyframes cannot be combined with audio-to-video"

    if req.duration is None:
        return "Keyframes cannot be combined with automatic duration"

    if any(normalize_optional_path(keyframe.imagePath) is None for keyframe in entries):
        return "Each keyframe requires an image path"

    indices = [keyframe.frameIndex for keyframe in entries]
    if len(set(indices)) != len(indices):
        return "Keyframe frame indices must be unique"

    last_frame = compute_num_frames(req.duration, req.fps) - 1
    for frame_idx in indices:
        if frame_idx < 0 or frame_idx > last_frame:
            return f"Keyframe frame index {frame_idx} is outside 0..{last_frame}"

    return None
