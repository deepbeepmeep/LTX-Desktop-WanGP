"""Monkey-patch: replace safe_open metadata reads with direct file reads.

safetensors' safe_open uses torch.UntypedStorage.from_file(shared=False) which
reserves copy-on-write commit charge equal to the file size. For a 22GB
checkpoint, this reserves 22GB of commit charge just to read a small JSON
header. Under memory pressure, this causes "paging file too small" errors.

This patch replaces metadata-only safe_open calls — and the FP8 scale / streaming
key-scan reads that still used safe_open on the full checkpoint — with direct
file reads that parse the safetensors header without mmap or commit charge.

Remove this patch once safetensors supports read-only file mapping.

Usage:
    import services.patches.safetensors_metadata_fix  # noqa: F401
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO, cast

import torch
from ltx_core.loader.sd_ops import SDOps
from ltx_core.quantization.fp8_cast import _RAW_DIFFUSION_MODEL_PREFIX

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Parse the JSON header and byte offset of the tensor payload. No mmap."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
    return header, 8 + header_size


def _read_safetensors_metadata(path: str) -> dict[str, str] | None:
    """Read metadata from a safetensors file header without mmap."""
    header, _ = _read_safetensors_header(path)
    meta = header.get("__metadata__")
    if meta is None:
        return None
    return cast(dict[str, str], meta)


def _read_tensor(path: str, info: dict[str, Any], data_offset: int, fh: BinaryIO | None = None) -> torch.Tensor:
    """Read one tensor by seeking to its payload. Does not map the rest of the file."""
    dtype = _DTYPES[info["dtype"]]
    shape = info["shape"]
    start, end = info["data_offsets"]
    if start == end:
        return torch.empty(shape, dtype=dtype)

    def _load(handle: BinaryIO) -> torch.Tensor:
        handle.seek(data_offset + start)
        raw = handle.read(end - start)
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).clone()

    if fh is not None:
        return _load(fh)
    with open(path, "rb") as handle:
        return _load(handle)


# --- Patch 1: SafetensorsModelStateDictLoader.metadata ---

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader


def _patched_model_metadata(self: SafetensorsModelStateDictLoader, path: str) -> dict:
    """Full ``__metadata__`` dict with JSON-encoded values parsed, mirroring upstream.

    Callers index into it themselves (``config``, ``model_version``,
    ``gemma_source_checkpoint``), so returning only ``config`` silently hides the
    sibling keys.
    """
    meta = _read_safetensors_metadata(path)
    if meta is None:
        return {}
    parsed: dict[str, object] = {}
    for key, value in meta.items():
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


assert hasattr(SafetensorsModelStateDictLoader, "metadata") and callable(
    getattr(SafetensorsModelStateDictLoader, "metadata")
), "SafetensorsModelStateDictLoader.metadata not found — patch needs updating."
SafetensorsModelStateDictLoader.metadata = _patched_model_metadata  # type: ignore[assignment]


# --- Patch 2: ltx_pipelines read_lora_reference_downscale_factor ---
# The upstream MPS-support work moved this helper to ltx_pipelines.iclora_utils and renamed it
# (dropped the leading underscore); ic_lora re-imports it, so both bindings are patched.
# The upstream version still uses safetensors safe_open (Windows commit-charge concern), so
# this mmap-free replacement is still worth applying.

import ltx_pipelines.ic_lora as _ic_lora_module
import ltx_pipelines.iclora_utils as _iclora_utils_module

_DOWNSCALE_FN = "read_lora_reference_downscale_factor"


def _patched_read_lora_reference_downscale_factor(lora_path: str) -> int:
    try:
        meta = _read_safetensors_metadata(lora_path) or {}
        return int(meta.get("reference_downscale_factor", 1))
    except Exception:
        import logging
        logging.warning(f"Failed to read metadata from LoRA file '{lora_path}'")
        return 1


assert hasattr(_iclora_utils_module, _DOWNSCALE_FN), (
    f"ltx_pipelines.iclora_utils.{_DOWNSCALE_FN} not found — patch needs updating."
)
setattr(_iclora_utils_module, _DOWNSCALE_FN, _patched_read_lora_reference_downscale_factor)
# ic_lora binds the name via `from ...iclora_utils import ...`, so its module-local reference
# (the actual call site) must be patched too.
if hasattr(_ic_lora_module, _DOWNSCALE_FN):
    setattr(_ic_lora_module, _DOWNSCALE_FN, _patched_read_lora_reference_downscale_factor)


# --- Patch 3: ltx_pipelines.utils.constants.detect_model_version ---
# Only the metadata read is replaced; the version -> params mapping stays upstream's
# (``detect_params`` calls this by module global), so new generations keep their own defaults.

import ltx_pipelines.distilled as _distilled_module
import ltx_pipelines.utils.constants as _constants_module

from ltx_core.loader.helpers import parse_model_version

_DETECT_VERSION_FN = "detect_model_version"


def _patched_detect_model_version(checkpoint_path: str) -> tuple[int, ...]:
    import logging
    logger = logging.getLogger(__name__)

    try:
        meta = _read_safetensors_metadata(checkpoint_path) or {}
        version = meta.get("model_version", "")
    except Exception:
        logger.warning("Could not read checkpoint metadata from %s, treating it as unversioned", checkpoint_path)
        return ()

    # Pre-release tags come both dot- and hyphen-separated ("2.3.rc1", "2.4-rc2").
    parsed = parse_model_version(version.replace("-", "."))
    logger.info("Checkpoint declares model_version=%s (parsed as %s)", version or "unknown", parsed)
    return parsed


assert hasattr(_constants_module, _DETECT_VERSION_FN), (
    f"ltx_pipelines.utils.constants.{_DETECT_VERSION_FN} not found — patch needs updating."
)
setattr(_constants_module, _DETECT_VERSION_FN, _patched_detect_model_version)
# distilled.py binds the name via `from ...constants import ...`, so its module-local
# reference (the sampler-selection call site) must be patched too.
if hasattr(_distilled_module, _DETECT_VERSION_FN):
    setattr(_distilled_module, _DETECT_VERSION_FN, _patched_detect_model_version)


# --- Patch 4: services.text_encoder.ltx_text_encoder.TextHandler.get_model_id_from_checkpoint ---

from services.text_encoder.ltx_text_encoder import LTXTextEncoder


def _patched_get_model_id_from_checkpoint(self: LTXTextEncoder, checkpoint_path: str) -> str | None:
    try:
        meta = _read_safetensors_metadata(checkpoint_path) or {}
        if "encrypted_wandb_properties" in meta:
            return meta["encrypted_wandb_properties"]
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
    return None


assert hasattr(LTXTextEncoder, "get_model_id_from_checkpoint"), (
    "LTXTextEncoder.get_model_id_from_checkpoint not found — patch needs updating."
)
LTXTextEncoder.get_model_id_from_checkpoint = _patched_get_model_id_from_checkpoint  # type: ignore[assignment]


# --- Patch 5: ltx_core.quantization.fp8_cast._read_scales (Lightricks/LTX-Desktop#158) ---
# build_fp8_cast_policy runs this on the 46 GB distilled-1.1 checkpoint. safe_open
# reserves commit charge equal to the file size (os error 1455) before inference.

import ltx_core.quantization.fp8_cast as _fp8_cast_module


def _patched_read_scales(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Same contract as upstream ``_read_scales``, without mapping the whole file."""
    path = str(checkpoint_path)
    header, data_offset = _read_safetensors_header(path)
    out: dict[str, torch.Tensor] = {}
    with open(path, "rb") as fh:
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if not key.endswith("_scale"):
                continue
            if not key.startswith(_RAW_DIFFUSION_MODEL_PREFIX):
                raise ValueError(
                    f"Scale key {key!r} does not start with the expected raw prefix {_RAW_DIFFUSION_MODEL_PREFIX!r}"
                )
            param_key = key.removeprefix(_RAW_DIFFUSION_MODEL_PREFIX).removesuffix("_scale")
            out[param_key] = _read_tensor(path, info, data_offset, fh)
    return out


assert hasattr(_fp8_cast_module, "_read_scales") and callable(_fp8_cast_module._read_scales), (
    "fp8_cast._read_scales not found — patch needs updating."
)
_fp8_cast_module._read_scales = _patched_read_scales  # type: ignore[assignment]


# --- Patch 6: ltx_core.block_streaming.builder._scan_checkpoint_keys ---
# Same 46 GB safe_open during StreamingModelBuilder.build (next 1455 after #158).

import ltx_core.block_streaming.builder as _streaming_builder_module


def _patched_scan_checkpoint_keys(
    checkpoint_paths: list[str],
    sd_ops: SDOps | None,
    blocks_prefix: str,
) -> tuple[dict[int, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Same contract as upstream ``_scan_checkpoint_keys``, header-only."""
    block_key_map: dict[int, list[tuple[str, str]]] = {}
    non_block_keys: list[tuple[str, str]] = []
    prefix_dot = blocks_prefix + "."
    for path in checkpoint_paths:
        header, _ = _read_safetensors_header(path)
        for sft_key in header:
            if sft_key == "__metadata__":
                continue
            model_key = sft_key if sd_ops is None else sd_ops.apply_to_key(sft_key)
            if model_key is None:
                continue
            if model_key.startswith(prefix_dot):
                rest = model_key[len(prefix_dot) :]
                idx_str, _, param_name = rest.partition(".")
                try:
                    block_idx = int(idx_str)
                except ValueError:
                    non_block_keys.append((sft_key, model_key))
                    continue
                block_key_map.setdefault(block_idx, []).append((sft_key, param_name))
            else:
                non_block_keys.append((sft_key, model_key))
    return block_key_map, non_block_keys


assert hasattr(_streaming_builder_module, "_scan_checkpoint_keys") and callable(
    _streaming_builder_module._scan_checkpoint_keys
), "block_streaming.builder._scan_checkpoint_keys not found — patch needs updating."
_streaming_builder_module._scan_checkpoint_keys = _patched_scan_checkpoint_keys  # type: ignore[assignment]
