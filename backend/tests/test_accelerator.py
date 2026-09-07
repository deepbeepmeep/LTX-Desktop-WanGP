"""Tests for ROCm vs CUDA accelerator detection."""

from __future__ import annotations

from types import SimpleNamespace

from runtime_config.accelerator import accelerator_backend
from services.services_utils import device_supports_fp8


def test_accelerator_backend_rocm_when_hip_is_set(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", "6.3.0", raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    assert accelerator_backend() == "rocm"


def test_accelerator_backend_cuda_when_hip_is_none(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: True)
    assert accelerator_backend() == "cuda"


def test_accelerator_backend_mps_when_no_cuda(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: False)
    mps = getattr(accel.torch.backends, "mps", None)
    if mps is None:
        monkeypatch.setattr(
            accel.torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False
        )
    else:
        monkeypatch.setattr(mps, "is_available", lambda: True)
    assert accelerator_backend() == "mps"


def test_accelerator_backend_cpu_fallback(monkeypatch) -> None:
    import runtime_config.accelerator as accel

    monkeypatch.setattr(accel.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(accel.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(accel.torch.backends.mps, "is_available", lambda: False)
    assert accelerator_backend() == "cpu"


def test_device_supports_fp8_false_on_mps() -> None:
    assert device_supports_fp8("mps") is False
    assert device_supports_fp8(SimpleNamespace(type="mps")) is False


def test_device_supports_fp8_false_on_cpu() -> None:
    assert device_supports_fp8("cpu") is False


def test_device_supports_fp8_true_on_cuda_not_rocm(monkeypatch) -> None:
    monkeypatch.setattr("runtime_config.accelerator.accelerator_backend", lambda: "cuda")
    assert device_supports_fp8("cuda") is True
    assert device_supports_fp8(SimpleNamespace(type="cuda")) is True


def test_device_supports_fp8_false_on_rocm_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr("runtime_config.accelerator.accelerator_backend", lambda: "rocm")
    assert device_supports_fp8("cuda") is False
    assert device_supports_fp8(SimpleNamespace(type="cuda")) is False
