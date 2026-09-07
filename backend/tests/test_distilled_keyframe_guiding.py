"""CPU-only checks for the Distilled MKF guiding-helper swap."""

from __future__ import annotations

import pytest

import ltx_pipelines.distilled as distilled
from ltx_pipelines.utils.helpers import image_conditionings_by_adding_guiding_latent
from services.fast_video_pipeline.distilled_keyframe_guiding import distilled_keyframe_guiding


def test_guiding_context_swaps_distilled_combined_helper() -> None:
    original = distilled.combined_image_conditionings
    with distilled_keyframe_guiding():
        assert distilled.combined_image_conditionings is image_conditionings_by_adding_guiding_latent
    assert distilled.combined_image_conditionings is original


def test_guiding_context_restores_helper_after_exception() -> None:
    original = distilled.combined_image_conditionings
    with pytest.raises(RuntimeError, match="swap-failed"):
        with distilled_keyframe_guiding():
            assert distilled.combined_image_conditionings is image_conditionings_by_adding_guiding_latent
            raise RuntimeError("swap-failed")
    assert distilled.combined_image_conditionings is original
