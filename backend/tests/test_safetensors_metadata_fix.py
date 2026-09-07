"""FP8 scale / key scans must not use safetensors.safe_open (Windows commit charge).

safetensors.safe_open maps the whole file with torch.UntypedStorage.from_file(shared=False),
which reserves copy-on-write commit equal to the file size. On a 46 GB distilled-1.1
checkpoint that raises OSError 1455 before generation starts (Lightricks/LTX-Desktop#158).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ltx_core.block_streaming import builder as streaming_builder
from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
from ltx_core.quantization import fp8_cast
from safetensors.torch import save_file

import services.patches.safetensors_metadata_fix as patch

_PAGING_FILE_ERROR = OSError("The paging file is too small for this operation to complete. (os error 1455)")
_WEIGHT_KEY = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
_SCALE_KEY = f"{_WEIGHT_KEY}_scale"
_NON_BLOCK_KEY = "model.diffusion_model.proj_in.weight"


def _boom(*_args: object, **_kwargs: object) -> object:
    raise _PAGING_FILE_ERROR


def _write_checkpoint(path: Path) -> Path:
    save_file(
        {
            _WEIGHT_KEY: torch.zeros(2, 2, dtype=torch.float8_e4m3fn),
            _SCALE_KEY: torch.tensor(0.5, dtype=torch.float32),
            _NON_BLOCK_KEY: torch.zeros(2, 2, dtype=torch.bfloat16),
        },
        str(path),
    )
    return path


def test_read_scales_survives_safe_open_paging_file_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = _write_checkpoint(tmp_path / "model.safetensors")
    monkeypatch.setattr(fp8_cast.safetensors, "safe_open", _boom)

    scales = fp8_cast._read_scales(ckpt)

    assert set(scales) == {"transformer_blocks.0.attn1.to_q.weight"}
    torch.testing.assert_close(scales["transformer_blocks.0.attn1.to_q.weight"], torch.tensor(0.5))


def test_build_fp8_policy_survives_safe_open_paging_file_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = _write_checkpoint(tmp_path / "model.safetensors")
    monkeypatch.setattr(fp8_cast.safetensors, "safe_open", _boom)

    policy = fp8_cast.build_policy(ckpt)

    # Scale was registered: dropping the sibling key succeeds instead of raising.
    assert policy.sd_ops is not None
    assert policy.sd_ops.apply_to_key_value(
        "transformer_blocks.0.attn1.to_q.weight_scale",
        torch.tensor(0.5, dtype=torch.float32),
    ) == []


def test_read_scales_rejects_scale_key_without_diffusion_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bad.safetensors"
    save_file({"other.weight_scale": torch.tensor(1.0, dtype=torch.float32)}, str(path))
    monkeypatch.setattr(fp8_cast.safetensors, "safe_open", _boom)

    with pytest.raises(ValueError, match="does not start with the expected raw prefix"):
        fp8_cast._read_scales(path)


def test_scan_checkpoint_keys_survives_safe_open_paging_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = _write_checkpoint(tmp_path / "model.safetensors")
    monkeypatch.setattr(streaming_builder.safetensors, "safe_open", _boom)

    block_key_map, non_block_keys = streaming_builder._scan_checkpoint_keys(
        [str(ckpt)], LTXV_MODEL_COMFY_RENAMING_MAP, "transformer_blocks"
    )

    assert set(block_key_map[0]) == {
        (_WEIGHT_KEY, "attn1.to_q.weight"),
        (_SCALE_KEY, "attn1.to_q.weight_scale"),
    }
    assert non_block_keys == [(_NON_BLOCK_KEY, "proj_in.weight")]


def test_patch_rebinds_read_scales_and_scan_checkpoint_keys() -> None:
    assert fp8_cast._read_scales is patch._patched_read_scales
    assert streaming_builder._scan_checkpoint_keys is patch._patched_scan_checkpoint_keys
