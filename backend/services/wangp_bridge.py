"""Bridge LTX Desktop requests to WanGP's in-process session API."""

from __future__ import annotations

import importlib
import logging
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int | None, int | None], None]
CancelledCallback = Callable[[], bool]

_VIDEO_RESOLUTION_MAP: dict[str, dict[str, str]] = {
    "512p": {"16:9": "832x480", "9:16": "480x832"},
    "540p": {"16:9": "960x544", "9:16": "544x960"},
    "720p": {"16:9": "1280x704", "9:16": "704x1280"},
    "1080p": {"16:9": "1920x1088", "9:16": "1088x1920"},
    "1440p": {"16:9": "2560x1440", "9:16": "1440x2560"},
    "2160p": {"16:9": "3840x2176", "9:16": "2176x3840"},
}

_QWEN_IMAGE_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (1328, 1328),
    (1664, 928),
    (928, 1664),
    (1472, 1140),
    (1140, 1472),
)
_TQDM_PROGRESS_RE = re.compile(r"(?:(?P<label>.*?):\s+)?(?P<percent>\d{1,3})%\|[^|]*\|\s*(?P<current>\d+)/(?P<total>\d+)")


@dataclass(frozen=True)
class WanGPBridgeStatus:
    available: bool
    root: Path | None
    python_executable: str | None
    reason: str | None = None


class WanGPBridge:
    def __init__(
        self,
        *,
        enabled: bool,
        root: Path | None,
        python_executable: str | None,
        config_dir: Path,
        output_dir: Path,
        video_model_type: str,
        image_model_type: str,
        camera_motion_prompts: dict[str, str],
        extra_args: Iterable[str] = (),
    ) -> None:
        self._enabled = enabled
        self._root = root
        self._python = python_executable
        self._config_dir = config_dir
        self._output_dir = output_dir
        self._video_model_type = video_model_type
        self._image_model_type = image_model_type
        self._camera_motion_prompts = camera_motion_prompts
        self._extra_args = tuple(extra_args)
        self._session = None
        self._submitted_manifest_once = False
        self._session_lock = threading.Lock()

    def _resolve_session_config_path(self) -> Path:
        if self._root is not None:
            root_config = self._root / "wgp_config.json"
            if root_config.exists():
                return root_config
        return self._config_dir / "wgp_config.json"

    def get_status(self) -> WanGPBridgeStatus:
        if not self._enabled:
            return WanGPBridgeStatus(
                available=False,
                root=self._root,
                python_executable=self._python,
                reason="WanGP bridge is disabled",
            )

        if self._root is None:
            return WanGPBridgeStatus(
                available=False,
                root=None,
                python_executable=self._python,
                reason="WanGP root was not resolved",
            )

        wgp_path = self._root / "wgp.py"
        if not wgp_path.exists():
            return WanGPBridgeStatus(
                available=False,
                root=self._root,
                python_executable=self._python,
                reason=f"Missing {wgp_path}",
            )

        api_path = self._root / "shared" / "api.py"
        if not api_path.exists():
            return WanGPBridgeStatus(
                available=False,
                root=self._root,
                python_executable=self._python,
                reason=f"Missing {api_path}",
            )

        try:
            self._load_api_module()
        except Exception as exc:
            return WanGPBridgeStatus(
                available=False,
                root=self._root,
                python_executable=self._python,
                reason=f"Unable to import WanGP API: {exc}",
            )

        return WanGPBridgeStatus(
            available=True,
            root=self._root,
            python_executable=self._python,
        )

    def generate_video(
        self,
        *,
        prompt: str,
        resolution_label: str,
        aspect_ratio: str,
        duration_seconds: int,
        fps: int,
        steps: int,
        seed: int | None,
        camera_motion: str,
        negative_prompt: str,
        image_path: str | None,
        audio_path: str | None,
        on_progress: ProgressCallback,
        is_cancelled: CancelledCallback,
    ) -> str:
        resolution = self._map_video_resolution(resolution_label, aspect_ratio)
        merged_prompt = prompt + self._camera_motion_prompts.get(camera_motion, "")
        video_length = self.compute_num_frames(duration_seconds, fps)

        settings: dict[str, object] = {
            "model_type": self._video_model_type,
            "prompt": merged_prompt,
            "resolution": resolution,
            "num_inference_steps": max(1, steps),
            "video_length": video_length,
            "duration_seconds": duration_seconds,
            "force_fps": fps,
        }
        if self._video_model_type.startswith("ltx2_"):
            settings["sliding_window_size"] = video_length
        if negative_prompt.strip():
            settings["negative_prompt"] = negative_prompt.strip()
        if seed is not None:
            settings["seed"] = seed
        if image_path:
            settings["image_prompt_type"] = "S"
            settings["image_start"] = str(Path(image_path).resolve())
        if audio_path:
            settings["audio_prompt_type"] = "A"
            settings["audio_guide"] = str(Path(audio_path).resolve())

        outputs = self._run_manifest(
            manifest=[{"id": 1, "params": settings, "plugin_data": {}}],
            media_suffixes={".mp4", ".mov", ".mkv", ".avi", ".webm"},
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )
        if not outputs:
            raise RuntimeError("WanGP completed without producing a video")
        return outputs[0]

    def generate_images(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        num_images: int,
        seed: int | None,
        on_progress: ProgressCallback,
        is_cancelled: CancelledCallback,
    ) -> list[str]:
        mapped_width, mapped_height = self._map_image_resolution(width, height)
        normalized_steps = self._normalize_image_steps(num_steps)
        settings: dict[str, object] = {
            "model_type": self._image_model_type,
            "prompt": prompt,
            "resolution": f"{mapped_width}x{mapped_height}",
            "num_inference_steps": normalized_steps,
            "batch_size": max(1, num_images),
            "image_mode": 1,
        }
        if seed is not None:
            settings["seed"] = seed

        outputs = self._run_manifest(
            manifest=[{"id": 1, "params": settings, "plugin_data": {}}],
            media_suffixes={".png", ".jpg", ".jpeg", ".webp"},
            on_progress=on_progress,
            is_cancelled=is_cancelled,
        )
        if not outputs:
            raise RuntimeError("WanGP completed without producing any images")
        return outputs

    @staticmethod
    def compute_num_frames(duration_seconds: int, fps: int) -> int:
        return max(((duration_seconds * fps) // 8) * 8 + 1, 9)

    def _map_video_resolution(self, resolution_label: str, aspect_ratio: str) -> str:
        per_aspect = _VIDEO_RESOLUTION_MAP.get(resolution_label)
        if per_aspect is None:
            raise RuntimeError(f"Unsupported WanGP video resolution: {resolution_label}")
        mapped = per_aspect.get(aspect_ratio)
        if mapped is None:
            raise RuntimeError(f"Unsupported WanGP aspect ratio: {aspect_ratio}")
        return mapped

    def _map_image_resolution(self, width: int, height: int) -> tuple[int, int]:
        if "qwen_image" not in self._image_model_type:
            return width, height

        requested_ratio = width / max(height, 1)

        def score(candidate: tuple[int, int]) -> tuple[float, float]:
            candidate_ratio = candidate[0] / candidate[1]
            ratio_delta = abs(candidate_ratio - requested_ratio)
            area_delta = abs((candidate[0] * candidate[1]) - (width * height))
            return (ratio_delta, area_delta)

        mapped = min(_QWEN_IMAGE_RESOLUTIONS, key=score)
        if mapped != (width, height):
            logger.info(
                "Adjusted Qwen image resolution from %sx%s to native preset %sx%s",
                width,
                height,
                mapped[0],
                mapped[1],
            )
        return mapped

    def _normalize_image_steps(self, num_steps: int) -> int:
        normalized_steps = max(1, num_steps)
        if not self._image_model_type.startswith("z_image"):
            return normalized_steps

        adjusted_steps = max(8, normalized_steps)
        if adjusted_steps != normalized_steps:
            logger.info(
                "Adjusted %s inference steps from %s to %s",
                self._image_model_type,
                normalized_steps,
                adjusted_steps,
            )
        return adjusted_steps

    def _load_api_module(self):
        if self._root is None:
            raise RuntimeError("WanGP root is not configured")
        root_str = str(self._root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        module = importlib.import_module("shared.api")
        module_path = Path(module.__file__).resolve()
        expected_path = (self._root / "shared" / "api.py").resolve()
        if module_path != expected_path:
            raise RuntimeError(f"shared.api resolved to {module_path}, expected {expected_path}")
        return module

    def _get_session(self):
        status = self.get_status()
        if not status.available or status.root is None:
            raise RuntimeError(status.reason or "WanGP bridge is unavailable")

        with self._session_lock:
            if self._session is None:
                api_module = self._load_api_module()
                self._session = api_module.WanGPSession(
                    root=status.root,
                    config_path=self._resolve_session_config_path(),
                    output_dir=self._output_dir,
                    cli_args=self._extra_args,
                )
            return self._session

    def _run_manifest(
        self,
        *,
        manifest: list[dict[str, object]],
        media_suffixes: set[str],
        on_progress: ProgressCallback,
        is_cancelled: CancelledCallback,
    ) -> list[str]:
        session = self._get_session()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config_dir.mkdir(parents=True, exist_ok=True)

        startup_phase = "starting_wangp" if not self._submitted_manifest_once else "validating_request"
        self._submitted_manifest_once = True
        on_progress(startup_phase, 2, None, None)
        job = session.submit_manifest(manifest)
        error_lines: deque[str] = deque(maxlen=40)
        cancel_requested = False
        console_progress = {"phase": "", "progress": -1, "logged_at": 0.0}

        while True:
            if is_cancelled() and not cancel_requested:
                cancel_requested = True
                job.cancel()

            event = job.events.get(timeout=0.2)
            if event is not None:
                self._handle_event(event, on_progress, error_lines, console_progress)

            if job.done and event is None:
                break

        result = job.result()

        if cancel_requested or is_cancelled():
            raise RuntimeError("Generation was cancelled")
        if not result.success:
            details = " | ".join(error_lines) if error_lines else "WanGP generation failed"
            raise RuntimeError(details)
        outputs = result.generated_files

        filtered_outputs = [
            str(Path(path).resolve())
            for path in outputs
            if Path(path).suffix.lower() in media_suffixes
        ]
        if not filtered_outputs:
            return []

        on_progress("complete", 100, None, None)
        return self._dedupe_preserve_order(filtered_outputs)

    def _handle_event(
        self,
        event: Any,
        on_progress: ProgressCallback,
        error_lines: deque[str],
        console_progress: dict[str, object],
    ) -> None:
        kind = getattr(event, "kind", "")
        data = getattr(event, "data", None)

        if kind == "stream":
            stream_name = getattr(data, "stream", "stdout")
            line = str(getattr(data, "text", "")).strip()
            if not line:
                return
            stream_phase = self._classify_stream_phase(line)
            parsed_progress = self._parse_tqdm_progress(line)
            if parsed_progress is None and stream_phase is not None:
                self._emit_console_progress(
                    console_progress,
                    stream_phase,
                    self._estimate_progress(stream_phase, None, None),
                    None,
                    None,
                    line,
                )
            if stream_phase is not None:
                on_progress(stream_phase, self._estimate_progress(stream_phase, None, None), None, None)
            if self._should_capture_error_line(stream_name, line):
                error_lines.append(line)
            return

        if kind == "progress":
            phase = str(getattr(data, "phase", "inference"))
            progress = int(getattr(data, "progress", 0))
            current_step = getattr(data, "current_step", None)
            total_steps = getattr(data, "total_steps", None)
            self._emit_console_progress(
                console_progress,
                phase,
                progress,
                current_step if isinstance(current_step, int) else None,
                total_steps if isinstance(total_steps, int) else None,
                str(getattr(data, "status", "")).strip(),
            )
            on_progress(phase, progress, current_step, total_steps)
            return

        if kind == "status":
            text = str(data or "").strip()
            if not text:
                return
            logger.info("[WanGP status] %s", text)
            phase = self._classify_phase(text)
            progress = self._estimate_progress(phase, None, None)
            self._emit_console_progress(console_progress, phase, progress, None, None, text, force=True)
            on_progress(phase, progress, None, None)
            return

        if kind == "info":
            text = str(data or "").strip()
            if text:
                logger.info("[WanGP info] %s", text)
            return

        if kind == "error":
            message = str(data)
            if message:
                error_lines.append(message)
                logger.error("[WanGP error] %s", message)
            return

        if kind == "completed":
            if bool(getattr(data, "success", False)):
                self._emit_console_progress(console_progress, "complete", 100, None, None, "Completed", force=True)
                on_progress("complete", 100, None, None)

    @staticmethod
    def _classify_phase(status_text: str) -> str:
        lowered = status_text.lower()
        if "denoising first pass" in lowered or "denoising 1st pass" in lowered:
            return "inference_stage_1"
        if "denoising second pass" in lowered or "denoising 2nd pass" in lowered:
            return "inference_stage_2"
        if "denoising third pass" in lowered or "denoising 3rd pass" in lowered:
            return "inference_stage_3"
        if "loading" in lowered:
            return "loading_model"
        if "enhancing prompt" in lowered or "encoding" in lowered:
            return "encoding_text"
        if "decoding" in lowered:
            return "decoding"
        if "saved" in lowered or "completed" in lowered or "output" in lowered:
            return "downloading_output"
        if "cancel" in lowered or "abort" in lowered:
            return "cancelled"
        return "inference"

    @staticmethod
    def _estimate_progress(phase: str, current_step: int | None, total_steps: int | None) -> int:
        if total_steps is None or total_steps <= 0 or current_step is None:
            if phase == "preparing_model":
                return 4
            if phase == "downloading_model":
                return 5
            if phase == "loading_model":
                return 10
            if phase == "encoding_text":
                return 18
            if phase == "inference_stage_1":
                return 25
            if phase == "inference_stage_2":
                return 70
            if phase == "inference_stage_3":
                return 80
            if phase == "decoding":
                return 90
            if phase == "downloading_output":
                return 95
            if phase == "cancelled":
                return 0
            return 15
        ratio = max(0.0, min(1.0, current_step / total_steps))
        if phase == "preparing_model":
            return min(6, 2 + int(ratio * 4))
        if phase == "downloading_model":
            return min(9, 3 + int(ratio * 6))
        if phase == "loading_model":
            return min(15, 5 + int(ratio * 10))
        if phase == "encoding_text":
            return min(22, 12 + int(ratio * 10))
        if phase == "inference_stage_1":
            return min(68, 20 + int(ratio * 48))
        if phase == "inference_stage_2":
            return min(88, 68 + int(ratio * 20))
        if phase == "inference_stage_3":
            return min(89, 80 + int(ratio * 9))
        if phase == "decoding":
            return min(95, 85 + int(ratio * 10))
        if phase == "downloading_output":
            return min(98, 92 + int(ratio * 6))
        if phase == "cancelled":
            return 0
        return min(90, 20 + int(ratio * 65))

    @staticmethod
    def _classify_stream_phase(line: str) -> str | None:
        lowered = line.lower()
        if "hf_xet" in lowered or "falling back to regular http download" in lowered:
            return "preparing_model"
        if lowered.startswith("downloading ") or "downloading model" in lowered or "snapshot_download" in lowered:
            return "downloading_model"
        return None

    @staticmethod
    def _parse_tqdm_progress(line: str) -> tuple[int, int | None, int | None, str | None] | None:
        match = _TQDM_PROGRESS_RE.search(line)
        if match is None:
            return None
        label = (match.group("label") or "").strip(" :")
        current_step = int(match.group("current"))
        total_steps = int(match.group("total"))
        return int(match.group("percent")), current_step, total_steps, label or None

    @staticmethod
    def _phase_label(phase: str) -> str:
        labels = {
            "starting_wangp": "Starting WanGP",
            "preparing_model": "Preparing model",
            "downloading_model": "Downloading model",
            "loading_model": "Loading model",
            "encoding_text": "Encoding text",
            "inference": "Generating",
            "inference_stage_1": "Generating pass 1",
            "inference_stage_2": "Generating pass 2",
            "inference_stage_3": "Generating pass 3",
            "decoding": "Decoding",
            "downloading_output": "Saving output",
            "complete": "Completed",
            "cancelled": "Cancelled",
        }
        return labels.get(phase, phase.replace("_", " ").title())

    def _emit_console_progress(
        self,
        tracker: dict[str, object],
        phase: str,
        progress: int,
        current_step: int | None,
        total_steps: int | None,
        status_text: str,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        progress = max(0, min(100, int(progress)))
        last_phase = str(tracker.get("phase", ""))
        last_progress = int(tracker.get("progress", -1))
        last_logged_at = float(tracker.get("logged_at", 0.0))
        if not force and phase == last_phase and progress == last_progress and now - last_logged_at < 1.0:
            return

        tracker["phase"] = phase
        tracker["progress"] = progress
        tracker["logged_at"] = now

        bar_width = 24
        filled = min(bar_width, max(0, round(progress * bar_width / 100)))
        bar = "#" * filled + "-" * (bar_width - filled)
        detail = f" {current_step}/{total_steps}" if current_step is not None and total_steps is not None else ""
        suffix = f" - {status_text}" if status_text else ""
        logger.info("[WanGP progress] [%s] %3d%% %s%s%s", bar, progress, self._phase_label(phase), detail, suffix)

    @staticmethod
    def _should_capture_error_line(stream_name: str, line: str) -> bool:
        lowered = line.lower()
        if line.startswith("Traceback") or line.startswith("File \"") or line.startswith("  File "):
            return True
        if stream_name != "stderr":
            return "[error]" in lowered or "exception" in lowered or "failed" in lowered
        if "%|" in line and "steps/" in lowered:
            return False
        if "| 0/" in line or "| 1/" in line or "| 2/" in line or "| 3/" in line or "| 4/" in line:
            return False
        return "[error]" in lowered or "traceback" in lowered or "exception" in lowered or "failed" in lowered

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered
