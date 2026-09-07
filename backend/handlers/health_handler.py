"""Health and hardware info handlers."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from api_types import GpuInfoResponse, GpuTelemetry, HealthResponse, ModelStatusItem, MpsMemoryResponse
from handlers.base import StateHandlerBase
from handlers.models_handler import ModelsHandler
from services.interfaces import GpuInfo
<<<<<<< HEAD
from services.wangp_bridge import WanGPBridge
from state.app_state_types import AppState, GpuSlot, StartupError, StartupLoading, StartupPending, StartupReady, VideoPipelineState, VideoPipelineWarmth
=======
from state.app_state_types import AppState, GpuSlot, VideoPipelineState
>>>>>>> upstream/main

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

_BYTES_PER_MIB = 1024 * 1024


class HealthHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        models_handler: ModelsHandler,
        gpu_info: GpuInfo,
        config: RuntimeConfig,
        wangp_bridge: WanGPBridge,
    ) -> None:
        super().__init__(state, lock, config)
        self._models = models_handler
        self._gpu_info = gpu_info
        self._wangp_bridge = wangp_bridge

    def get_health(self) -> HealthResponse:
        if self._config.wangp_enabled:
            bridge = self._wangp_bridge.get_status()
            return HealthResponse(
                status="ok",
                models_loaded=bridge.available,
                active_model="wangp" if bridge.available else None,
                gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
                sage_attention=self._use_sage_attention,
                models_status=[
                    ModelStatusItem(
                        id="fast",
                        name="WanGP LTX-2.3 Distilled",
                        loaded=bridge.available,
                        downloaded=bridge.available,
                    ),
                ],
            )

        active_model: str | None = None
        models_loaded = False

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline)):
                    active_model = pipeline.pipeline_kind
                    models_loaded = True
                case _:
                    pass

        downloaded_checkpoints = self._models.get_downloaded_checkpoints()

        return HealthResponse(
            status="ok",
            models_loaded=models_loaded,
            active_model=active_model,
            gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
            sage_attention=self.config.use_sage_attention,
            models_status=[
                ModelStatusItem(
                    id="fast",
                    name="LTX-2 Fast",
                    loaded=models_loaded,
                    downloaded=any(cp_id.startswith("ltx-") for cp_id in downloaded_checkpoints),
                ),
            ],
        )

    def get_gpu_info(self) -> GpuInfoResponse:
        return GpuInfoResponse(
            cuda_available=self._gpu_info.get_cuda_available(),
            mps_available=self._gpu_info.get_mps_available(),
            gpu_available=self._gpu_info.get_gpu_available(),
            gpu_name=self._gpu_info.get_device_name(),
            vram_gb=self._gpu_info.get_vram_total_gb(),
            gpu_info=GpuTelemetry(**self._gpu_info.get_gpu_info()),
        )

    def get_mps_memory(self) -> MpsMemoryResponse:
        """Read-only Apple Silicon MPS memory snapshot (torch-tracked / driver-allocated /
        recommended-max, MiB). ``available`` is False off MPS. No side effects; torch is
        imported lazily so the call is cheap and safe on non-MPS hosts."""
        import sys

        import torch

        if sys.platform != "darwin" or not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return MpsMemoryResponse(available=False)
        try:
            if self._config.wangp_enabled:
                status = self._wangp_bridge.get_status()
                if status.available:
                    self.set_startup_ready()
                else:
                    self.set_startup_error(status.reason or "WanGP bridge is unavailable")
                return

            self.set_startup_loading("Checking models", 5)
            status = self._models.get_models_status()
            if not status.all_downloaded:
                self.set_startup_pending("Models not downloaded. User needs to download via app.")
                return

            if not self.state.app_settings.load_on_startup:
                self.set_startup_ready()
                return

            if self._config.force_api_generations:
                self.set_startup_ready()
                return

            self.set_startup_loading("Loading Fast pipeline", 30)
            self._pipelines.load_gpu_pipeline("fast", should_warm=False)

            self.set_startup_loading("Warming Fast pipeline", 60)
            self._pipelines.warmup_pipeline("fast")
            with self._lock:
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as state):
                        state.warmth = VideoPipelineWarmth.WARM
                    case _:
                        pass

            zit_models_path = self._config.model_path("zit")
            zit_exists = zit_models_path.exists() and any(zit_models_path.iterdir())
            if zit_exists:
                self.set_startup_loading("Preloading Z-Image-Turbo to CPU", 85)
                if self.state.cpu_slot is None:
                    self._pipelines.preload_zit_to_cpu()

            self.set_startup_ready()
        except Exception as exc:
            log_background_exception("health-default-warmup", exc)
            self.set_startup_error(str(exc))

            return MpsMemoryResponse(
                available=True,
                allocated_mib=round(torch.mps.current_allocated_memory() / _BYTES_PER_MIB),
                driver_mib=round(torch.mps.driver_allocated_memory() / _BYTES_PER_MIB),
                recommended_max_mib=round(torch.mps.recommended_max_memory() / _BYTES_PER_MIB),
            )
        except Exception:  # noqa: BLE001
            return MpsMemoryResponse(available=False)
