"""Model availability and model status handlers."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import ModelFileStatus, ModelInfo, ModelsStatusResponse, TextEncoderStatus
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_required_model_types
from services.wangp_bridge import WanGPBridge
from state.app_state_types import AppState, AvailableFiles

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        wangp_bridge: WanGPBridge,
    ) -> None:
        super().__init__(state, lock)
        self._config = config
        self._wangp_bridge = wangp_bridge

    @staticmethod
    def _path_size(path: Path, is_folder: bool) -> int:
        if not is_folder:
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _scan_available_files(self) -> AvailableFiles:
        files: AvailableFiles = {}
        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            path = self._config.model_path(model_type)
            if spec.is_folder:
                ready = path.exists() and any(path.iterdir()) if path.exists() else False
                files[model_type] = path if ready else None
            else:
                files[model_type] = path if path.exists() else None
        return files

    @with_state_lock
    def refresh_available_files(self) -> AvailableFiles:
        self.state.available_files = self._scan_available_files()
        return self.state.available_files.copy()

    def get_text_encoder_status(self) -> TextEncoderStatus:
        if self._config.wangp_enabled:
            status = self._wangp_bridge.get_status()
            return TextEncoderStatus(
                downloaded=status.available,
                size_bytes=0,
                size_gb=0.0,
                expected_size_gb=0.0,
            )

        files = self.refresh_available_files()
        text_encoder_path = files["text_encoder"]
        exists = text_encoder_path is not None
        text_spec = self._config.spec_for("text_encoder")
        size_bytes = self._path_size(text_encoder_path, is_folder=True) if exists else 0
        expected = text_spec.expected_size_bytes

        return TextEncoderStatus(
            downloaded=exists,
            size_bytes=size_bytes if exists else expected,
            size_gb=round((size_bytes if exists else expected) / (1024**3), 1),
            expected_size_gb=round(expected / (1024**3), 1),
        )

    def get_models_list(self) -> list[ModelInfo]:
        if self._config.wangp_enabled:
            return [
                ModelInfo(id="fast", name="Fast (WanGP LTX-2.3 Distilled)", description="WanGP bridge, 8 steps"),
                ModelInfo(id="pro", name="Pro (WanGP LTX-2.3 Distilled)", description="WanGP bridge, configurable steps"),
            ]

        pro_steps = self.state.app_settings.pro_model.steps
        pro_upscaler = self.state.app_settings.pro_model.use_upscaler
        return [
            ModelInfo(id="fast", name="Fast (Distilled)", description="8 steps + 2x upscaler"),
            ModelInfo(
                id="pro",
                name="Pro (Full)",
                description=f"{pro_steps} steps" + (" + 2x upscaler" if pro_upscaler else " (native resolution)"),
            ),
        ]

    def get_models_status(self, has_api_key: bool | None = None) -> ModelsStatusResponse:
        if self._config.wangp_enabled:
            status = self._wangp_bridge.get_status()
            root_label = str(status.root) if status.root is not None else str(self._config.wangp_config_dir)
            reason = status.reason if not status.available else "Uses external WanGP installation"
            return ModelsStatusResponse(
                models=[
                    ModelFileStatus(
                        name="WanGP Bridge",
                        description=reason or "WanGP bridge",
                        downloaded=status.available,
                        size=0,
                        expected_size=0,
                        required=True,
                        is_folder=True,
                        optional_reason=None,
                    )
                ],
                all_downloaded=status.available,
                total_size=0,
                downloaded_size=0,
                total_size_gb=0.0,
                downloaded_size_gb=0.0,
                models_path=root_label,
                has_api_key=False,
                text_encoder_status=self.get_text_encoder_status(),
                use_local_text_encoder=False,
            )

        files = self.refresh_available_files()
        settings = self.state.app_settings.model_copy(deep=True)

        if has_api_key is None:
            has_api_key = bool(settings.ltx_api_key)

        models: list[ModelFileStatus] = []
        total_size = 0
        downloaded_size = 0
        required_types = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key,
            settings.use_local_text_encoder,
        )

        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            path = files[model_type]
            exists = path is not None
            actual_size = self._path_size(path, is_folder=spec.is_folder) if exists else 0
            required = model_type in required_types
            if required:
                total_size += spec.expected_size_bytes
                if exists:
                    downloaded_size += actual_size

            description = spec.description
            optional_reason: str | None = None
            if model_type == "text_encoder":
                description += " (optional with API key)" if has_api_key else ""
                optional_reason = "Uses LTX API for text encoding" if has_api_key else None

            models.append(
                ModelFileStatus(
                    name=spec.name,
                    description=description,
                    downloaded=exists,
                    size=actual_size if exists else spec.expected_size_bytes,
                    expected_size=spec.expected_size_bytes,
                    required=required,
                    is_folder=spec.is_folder,
                    optional_reason=optional_reason if model_type == "text_encoder" else None,
                )
            )

        all_downloaded = all(model.downloaded for model in models if model.required)

        return ModelsStatusResponse(
            models=models,
            all_downloaded=all_downloaded,
            total_size=total_size,
            downloaded_size=downloaded_size,
            total_size_gb=round(total_size / (1024**3), 1),
            downloaded_size_gb=round(downloaded_size / (1024**3), 1),
            models_path=str(self._config.models_dir),
            has_api_key=has_api_key,
            text_encoder_status=self.get_text_encoder_status(),
            use_local_text_encoder=settings.use_local_text_encoder,
        )
