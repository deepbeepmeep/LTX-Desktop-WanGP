"""Model download session handler."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from threading import RLock
from typing import TYPE_CHECKING

from api_types import DownloadProgressResponse
from handlers.base import StateHandlerBase, with_state_lock
from handlers.models_handler import ModelsHandler
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_required_model_types
from services.interfaces import ModelDownloader, TaskRunner
from state.app_state_types import AppState, DownloadError, FileDownloadCompleted, FileDownloadRunning, ModelFileType

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class DownloadHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        models_handler: ModelsHandler,
        model_downloader: ModelDownloader,
        task_runner: TaskRunner,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock)
        self._models_handler = models_handler
        self._model_downloader = model_downloader
        self._task_runner = task_runner
        self._config = config

    @with_state_lock
    def is_download_running(self) -> bool:
        return self.state.is_downloading

    @with_state_lock
    def start_download(self, files: dict[ModelFileType, tuple[str, int]]) -> None:
        self.state.downloading_session = {
            file_type: FileDownloadRunning(
                target_path=target,
                progress=0.0,
                downloaded_bytes=0,
                total_bytes=size,
                speed_mbps=0.0,
            )
            for file_type, (target, size) in files.items()
        }

    @with_state_lock
    def update_file_progress(self, file_type: ModelFileType, downloaded: int, total: int, speed_mbps: float) -> None:
        match self.state.downloading_session:
            case dict() as files:
                if file_type not in files:
                    return
                match files[file_type]:
                    case FileDownloadRunning() as running:
                        running.downloaded_bytes = downloaded
                        running.total_bytes = total
                        running.progress = 0.0 if total == 0 else min(1.0, max(0.0, downloaded / total))
                        running.speed_mbps = speed_mbps
                    case FileDownloadCompleted():
                        return
            case _:
                return

    @with_state_lock
    def complete_file(self, file_type: ModelFileType) -> None:
        match self.state.downloading_session:
            case dict() as files:
                files[file_type] = FileDownloadCompleted()
            case _:
                return

    @with_state_lock
    def fail_download(self, error: str) -> None:
        logger.error("Model download failed: %s", error)
        self.state.downloading_session = DownloadError(error=error)

    def _make_progress_callback(self, file_type: ModelFileType) -> Callable[[int, int], None]:
        start_time = time.monotonic()

        def on_progress(downloaded: int, total: int) -> None:
            elapsed = time.monotonic() - start_time
            speed_mbps = (downloaded / elapsed / (1024 * 1024)) if elapsed > 0 else 0.0
            self.update_file_progress(file_type, downloaded, total, speed_mbps)

        return on_progress

    def _on_background_download_error(self, exc: Exception) -> None:
        self.fail_download(str(exc))

    @with_state_lock
    def get_download_progress(self) -> DownloadProgressResponse:
        status = "idle"
        current_file = ""
        current_file_progress = 0
        speed_mbps = 0
        downloaded_bytes = 0
        total_bytes = 0
        files_completed = 0
        total_files = 0
        error: str | None = None

        match self.state.downloading_session:
            case DownloadError(error=err):
                status = "error"
                error = err
            case dict() as files:
                status = "downloading" if self.state.is_downloading else "complete"
                total_files = len(files)
                for file_type, file_state in files.items():
                    size = self._config.spec_for(file_type).expected_size_bytes
                    total_bytes += size
                    match file_state:
                        case FileDownloadCompleted():
                            files_completed += 1
                            downloaded_bytes += size
                        case FileDownloadRunning() as running:
                            current_file = file_type
                            current_file_progress = int(running.progress * 100)
                            speed_mbps = int(running.speed_mbps)
                            downloaded_bytes += running.downloaded_bytes
            case _:
                status = "idle"

        total_progress = int((downloaded_bytes / total_bytes) * 100) if total_bytes > 0 else 0

        return DownloadProgressResponse(
            status=status,
            currentFile=current_file,
            currentFileProgress=current_file_progress,
            totalProgress=total_progress,
            downloadedBytes=downloaded_bytes,
            totalBytes=total_bytes,
            filesCompleted=files_completed,
            totalFiles=total_files,
            error=error,
            speedMbps=speed_mbps,
        )

    def _move_to_final(self, file_type: ModelFileType) -> None:
        """Move downloaded file/folder from downloading dir to final location."""
        spec = self._config.spec_for(file_type)

        if spec.is_folder:
            src = self._config.downloading_dir / spec.relative_path
            dst = self._config.model_path(file_type)
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)
        else:
            src = self._config.downloading_dir / spec.relative_path
            dst = self._config.model_path(file_type)
            if dst.exists():
                dst.unlink()
            src.rename(dst)

    def cleanup_downloading_dir(self) -> None:
        """Remove stale .downloading/ dir (leftover from crashed downloads)."""
        downloading = self._config.downloading_dir
        if downloading.exists():
            shutil.rmtree(downloading)

    def _download_models_worker(self, skip_text_encoder: bool) -> None:
        files_to_download: dict[ModelFileType, tuple[str, int]] = {}

        self._models_handler.refresh_available_files()
        available = self.state.available_files.copy()
        with self._lock:
            has_api_key = bool(self.state.app_settings.ltx_api_key.strip())
        required_types = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key=has_api_key,
        )

        for model_type in MODEL_FILE_ORDER:
            if model_type not in required_types:
                continue
            if model_type == "text_encoder" and skip_text_encoder:
                continue
            if available[model_type] is not None:
                continue
            spec = self._config.spec_for(model_type)
            files_to_download[model_type] = (spec.name, spec.expected_size_bytes)

        if not files_to_download:
            with self._lock:
                self.state.downloading_session = {}
            return

        self.start_download(files_to_download)

        for file_type, (target_name, expected_size) in files_to_download.items():
            spec = self._config.spec_for(file_type)
            logger.info("Downloading %s from %s", target_name, spec.repo_id)
            progress_cb = self._make_progress_callback(file_type)

            try:
                self._config.downloading_dir.mkdir(parents=True, exist_ok=True)

                if spec.is_folder:
                    self._model_downloader.download_snapshot(
                        repo_id=spec.repo_id,
                        local_dir=str(self._config.downloading_path(file_type)),
                        on_progress=progress_cb,
                    )
                else:
                    self._model_downloader.download_file(
                        repo_id=spec.repo_id,
                        filename=spec.name,
                        local_dir=str(self._config.downloading_path(file_type)),
                        on_progress=progress_cb,
                    )

                self._move_to_final(file_type)
            except Exception:
                self.cleanup_downloading_dir()
                raise

            self.update_file_progress(file_type, expected_size, expected_size, 0)
            self.complete_file(file_type)

        self._models_handler.refresh_available_files()

    def start_model_download(self, skip_text_encoder: bool = False) -> bool:
        if self._config.wangp_enabled:
            with self._lock:
                self.state.downloading_session = {}
            return True

        with self._lock:
            if self.state.is_downloading:
                return False

        self._task_runner.run_background(
            lambda: self._download_models_worker(skip_text_encoder),
            task_name="model-download",
            on_error=self._on_background_download_error,
            daemon=True,
        )
        return True

    def start_text_encoder_download(self) -> bool:
        if self._config.wangp_enabled:
            with self._lock:
                self.state.downloading_session = {}
            return True

        with self._lock:
            if self.state.is_downloading:
                return False

        def worker() -> None:
            text_spec = self._config.spec_for("text_encoder")
            self.start_download({"text_encoder": (text_spec.name, text_spec.expected_size_bytes)})
            progress_cb = self._make_progress_callback("text_encoder")
            try:
                self._config.downloading_dir.mkdir(parents=True, exist_ok=True)
                self._model_downloader.download_snapshot(
                    repo_id=text_spec.repo_id,
                    local_dir=str(self._config.downloading_path("text_encoder")),
                    on_progress=progress_cb,
                )
                self._move_to_final("text_encoder")
            except Exception:
                self.cleanup_downloading_dir()
                raise
            self.update_file_progress(
                "text_encoder",
                text_spec.expected_size_bytes,
                text_spec.expected_size_bytes,
                0,
            )
            self.complete_file("text_encoder")
            self._models_handler.refresh_available_files()

        self._task_runner.run_background(
            worker,
            task_name="text-encoder-download",
            on_error=self._on_background_download_error,
            daemon=True,
        )
        return True
