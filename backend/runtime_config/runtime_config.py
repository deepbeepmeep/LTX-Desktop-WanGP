"""Runtime configuration model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from runtime_config.model_download_specs import ModelFileDownloadSpec
from state.app_state_types import ModelFileType


@dataclass
class RuntimeConfig:
    device: torch.device
    models_dir: Path
    model_download_specs: Mapping[ModelFileType, ModelFileDownloadSpec]
    required_model_types: frozenset[ModelFileType]
    outputs_dir: Path
    ic_lora_dir: Path
    settings_file: Path
    ltx_api_base_url: str
    force_api_generations: bool
    use_sage_attention: bool
    camera_motion_prompts: dict[str, str]
    default_negative_prompt: str
    wangp_enabled: bool
    wangp_root: Path | None
    wangp_python: str | None
    wangp_config_dir: Path
    wangp_video_model_type: str
    wangp_image_model_type: str
    wangp_extra_args: tuple[str, ...]

    def spec_for(self, model_type: ModelFileType) -> ModelFileDownloadSpec:
        return self.model_download_specs[model_type]

    def model_path(self, model_type: ModelFileType) -> Path:
        return self.models_dir / self.spec_for(model_type).relative_path

    @property
    def downloading_dir(self) -> Path:
        return self.models_dir / ".downloading"

    def downloading_path(self, model_type: ModelFileType) -> Path:
        """Return the staging path under downloading_dir for a model type."""
        spec = self.spec_for(model_type)
        if spec.is_folder:
            return self.downloading_dir / spec.relative_path
        return self.downloading_dir
