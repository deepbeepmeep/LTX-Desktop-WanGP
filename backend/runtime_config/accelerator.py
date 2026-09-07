"""Accelerator backend detection.

ROCm PyTorch reports itself through the same `torch.cuda` interface as NVIDIA
CUDA builds (`torch.cuda.is_available()`, `device.type == "cuda"`, etc.), so
code that checks `device.type == "cuda"` to mean "this is an NVIDIA GPU" is
wrong under ROCm. `torch.version.hip` is the actual discriminator: it is set
on ROCm builds and `None` on CUDA builds.

Originally contributed by boxwrench in
https://github.com/Lightricks/LTX-Desktop/pull/160
"""

from __future__ import annotations

from typing import Literal

import torch

AcceleratorBackend = Literal["rocm", "cuda", "mps", "cpu"]


def accelerator_backend() -> AcceleratorBackend:
    if getattr(torch.version, "hip", None):
        return "rocm"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
