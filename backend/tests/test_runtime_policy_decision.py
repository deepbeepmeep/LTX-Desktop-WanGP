"""Tests for runtime policy decision helpers."""

from __future__ import annotations

import pytest

from runtime_config.runtime_policy import (
    decide_local_generation_mode,
    streaming_prefetch_count_for_mode,
)


def test_darwin_without_mps_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=False, ram_gb=64)
        == "unsupported"
    )


def test_darwin_with_low_ram_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=14)
        == "unsupported"
    )


def test_darwin_with_unknown_ram_unsupported() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=None)
        == "unsupported"
    )


def test_darwin_streams_below_full_resident_floor() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=15)
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=48)
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=84)
        == "streaming_models_loading"
    )


def test_darwin_full_resident_at_and_above_floor() -> None:
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=85)
        == "full_models_loading"
    )
    assert (
        decide_local_generation_mode(system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=128)
        == "full_models_loading"
    )


def test_darwin_defaults_to_unsupported_without_mps_kwargs() -> None:
    """Backward-compat: existing call sites that don't pass mps_available/ram_gb stay unsupported."""
    assert decide_local_generation_mode(system="Darwin", cuda_available=True, vram_gb=64) == "unsupported"


def test_windows_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=False, vram_gb=24) == "unsupported"


def test_windows_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=14) == "unsupported"


def test_windows_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=None) == "unsupported"


def test_windows_streaming_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=24) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_windows_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=31) == "full_models_loading"
    assert decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=96) == "full_models_loading"


def test_linux_without_cuda_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=False, vram_gb=24) == "unsupported"


def test_linux_with_low_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=14) == "unsupported"


def test_linux_with_unknown_vram_unsupported() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=None) == "unsupported"


def test_linux_streaming_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=15) == "streaming_models_loading"
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=30) == "streaming_models_loading"


def test_linux_full_loading_range() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=31) == "full_models_loading"


def test_linux_without_fp8_still_unsupported_below_streaming_floor() -> None:
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=14, fp8_capable=False
        )
        == "unsupported"
    )


def test_linux_without_fp8_streams_even_above_full_floor() -> None:
    """ROCm reports as CUDA but has no fp8; the 31 GB floor assumes an fp8 transformer."""
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=31, fp8_capable=False
        )
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=96, fp8_capable=False
        )
        == "streaming_models_loading"
    )


def test_windows_without_fp8_streams_even_above_full_floor() -> None:
    assert (
        decide_local_generation_mode(
            system="Windows", cuda_available=True, vram_gb=31, fp8_capable=False
        )
        == "streaming_models_loading"
    )


def test_fp8_capable_default_preserves_cuda_full_loading() -> None:
    assert decide_local_generation_mode(system="Linux", cuda_available=True, vram_gb=31) == "full_models_loading"
    assert (
        decide_local_generation_mode(
            system="Linux", cuda_available=True, vram_gb=31, fp8_capable=True
        )
        == "full_models_loading"
    )


def test_darwin_ignores_fp8_capable() -> None:
    assert (
        decide_local_generation_mode(
            system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=48, fp8_capable=False
        )
        == "streaming_models_loading"
    )
    assert (
        decide_local_generation_mode(
            system="Darwin", cuda_available=False, vram_gb=None, mps_available=True, ram_gb=85, fp8_capable=False
        )
        == "full_models_loading"
    )


def test_other_systems_fail_closed() -> None:
    assert decide_local_generation_mode(system="FreeBSD", cuda_available=True, vram_gb=48) == "unsupported"


def test_streaming_prefetch_count_for_full_loading_is_none() -> None:
    assert streaming_prefetch_count_for_mode("full_models_loading") is None


def test_streaming_prefetch_count_for_streaming_mode_is_two() -> None:
    assert streaming_prefetch_count_for_mode("streaming_models_loading") == 2


def test_streaming_prefetch_count_for_unsupported_asserts() -> None:
    with pytest.raises(AssertionError):
        streaming_prefetch_count_for_mode("unsupported")
