"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from runtime_config.runtime_policy import LocalGenerationMode


@dataclass
class RuntimeConfig:
    device: torch.device
    app_data_dir: Path
    default_models_dir: Path
    outputs_dir: Path
    settings_file: Path
    ltx_api_base_url: str
    local_generations_mode: LocalGenerationMode
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
    dev_mode: bool
    backend_port: int
    hf_oauth_client_id: str = ""
    lora_catalog_source: str = ""
    # Bundled catalog used as a fallback when lora_catalog_source is a URL that fails to fetch.
    lora_catalog_fallback_path: str = ""

    def spec_for(self, model_type: ModelFileType) -> ModelFileDownloadSpec:
        return self.model_download_specs[model_type]

    def model_path(self, model_type: ModelFileType) -> Path:
        return self.models_dir / self.spec_for(model_type).relative_path

    @property
    def force_api_generations(self) -> bool:
        """Derived: local generation is unavailable for this runtime."""
        return self.local_generations_mode == "unsupported"
