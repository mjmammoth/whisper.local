from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
from typing import Any, Iterable

import numpy as np
import sounddevice as sd


logger = logging.getLogger(__name__)
DEFAULT_STOP_TIMEOUT_SECONDS = 0.75


@dataclass(frozen=True)
class AudioInputDeviceInfo:
    key: str
    index: int
    name: str
    hostapi: str
    max_input_channels: int
    default_samplerate: float | None
    is_default: bool
    sample_rate_supported: bool | None = None
    sample_rate_reason: str | None = None


@dataclass(frozen=True)
class AudioInputScanResult:
    devices: list[AudioInputDeviceInfo]
    default_device_key: str | None
    error: str | None = None


def scan_audio_input_devices(sample_rate: int | None = None) -> AudioInputScanResult:
    """Return discovered audio input devices with stable keys and optional sample-rate checks."""
    try:
        raw_devices = sd.query_devices()
        raw_hostapis = sd.query_hostapis()
    except Exception as exc:
        logger.warning("Failed to query audio devices: %s", exc)
        return AudioInputScanResult(devices=[], default_device_key=None, error=str(exc))

    default_index = _default_input_device_index()
    devices: list[AudioInputDeviceInfo] = []
    seen_counts: dict[str, int] = {}

    for index, raw_device in enumerate(raw_devices):
        max_input_channels = _to_int(raw_device.get("max_input_channels"), fallback=0)
        if max_input_channels <= 0:
            continue

        hostapi_index = _to_int(raw_device.get("hostapi"), fallback=-1)
        hostapi_name = _hostapi_name(raw_hostapis, hostapi_index)
        device_name = _device_name(raw_device.get("name"), index=index)
        base_key = f"{hostapi_name}:{device_name}"
        next_count = seen_counts.get(base_key, 0) + 1
        seen_counts[base_key] = next_count
        key = base_key if next_count == 1 else f"{base_key}#{next_count}"

        sample_rate_supported: bool | None = None
        sample_rate_reason: str | None = None
        if sample_rate is not None and sample_rate > 0:
            try:
                sd.check_input_settings(
                    device=index,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sample_rate_supported = True
            except Exception as exc:
                sample_rate_supported = False
                sample_rate_reason = str(exc)

        devices.append(
            AudioInputDeviceInfo(
                key=key,
                index=index,
                name=device_name,
                hostapi=hostapi_name,
                max_input_channels=max_input_channels,
                default_samplerate=_to_float(raw_device.get("default_samplerate")),
                is_default=index == default_index,
                sample_rate_supported=sample_rate_supported,
                sample_rate_reason=sample_rate_reason,
            )
        )

    default_key = next((device.key for device in devices if device.is_default), None)
    return AudioInputScanResult(devices=devices, default_device_key=default_key, error=None)


def resolve_audio_input_device_index(
    device_key: str | None,
    devices: Iterable[AudioInputDeviceInfo],
) -> int | None:
    if device_key is None:
        return None
    normalized = str(device_key).strip()
    if not normalized:
        return None
    for device in devices:
        if device.key == normalized:
            return device.index
    return None


def find_audio_input_device(
    device_key: str | None,
    devices: Iterable[AudioInputDeviceInfo],
) -> AudioInputDeviceInfo | None:
    if device_key is None:
        return None
    normalized = str(device_key).strip()
    if not normalized:
        return None
    for device in devices:
        if device.key == normalized:
            return device
    return None


def default_audio_input_device(
    devices: Iterable[AudioInputDeviceInfo],
) -> AudioInputDeviceInfo | None:
    for device in devices:
        if device.is_default:
            return device
    return None


class AudioRecorder:
    def __init__(self, sample_rate: int, channels: int = 1, device: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream: sd.InputStream | None = None
        self._frames: list[np.ndarray] = []
        self._state_lock = threading.Lock()

    def start(self) -> None:
        with self._state_lock:
            if self._stream is not None:
                return
            self._frames = []
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=self._on_audio,
            )
            self._stream = stream
        try:
            stream.start()
        except Exception:
            with self._state_lock:
                if self._stream is stream:
                    self._stream = None
            raise

    def stop(self, timeout_seconds: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> np.ndarray:
        with self._state_lock:
            stream = self._stream
            if stream is None:
                return np.empty(0, dtype=np.float32)
            frames = self._frames
            self._stream = None
            self._frames = []

        stream_error: Exception | None = None

        def _stop_stream() -> None:
            nonlocal stream_error
            try:
                stream.stop()
            except Exception as exc:
                stream_error = exc
            finally:
                try:
                    stream.close()
                except Exception as exc:
                    if stream_error is None:
                        stream_error = exc

        stopper = threading.Thread(
            target=_stop_stream,
            daemon=True,
            name="murmur-audio-stop",
        )
        stopper.start()
        if timeout_seconds <= 0:
            stopper.join()
        else:
            stopper.join(timeout_seconds)

        if stopper.is_alive():
            logger.error(
                "Timed out stopping audio stream; recorder state reset timeout_seconds=%s",
                timeout_seconds,
            )
        elif stream_error is not None:
            raise stream_error

        return _flatten_frames(frames, self.channels)

    def is_recording(self) -> bool:
        with self._state_lock:
            return self._stream is not None

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time
        if status:
            logger.warning("Audio callback status: %s", status)
        with self._state_lock:
            if self._stream is None:
                return
            self._frames.append(indata.copy())



def _flatten_frames(frames: Iterable[np.ndarray], channels: int) -> np.ndarray:
    if not frames:
        return np.empty(0, dtype=np.float32)
    audio = np.concatenate(list(frames), axis=0)
    if audio.ndim > 1:
        if channels > 1:
            audio = audio[:, 0]
        else:
            audio = audio.reshape(-1)
    return audio.astype(np.float32, copy=False)


def _default_input_device_index() -> int | None:
    default_device = getattr(sd.default, "device", None)
    if isinstance(default_device, (list, tuple)):
        if not default_device:
            return None
        value = default_device[0]
    else:
        value = default_device

    index = _to_int(value, fallback=-1)
    if index < 0:
        return None
    return index


def _hostapi_name(raw_hostapis: Iterable[dict[str, object]], hostapi_index: int) -> str:
    hostapis = list(raw_hostapis)
    if hostapi_index < 0 or hostapi_index >= len(hostapis):
        return "Unknown Host API"
    raw_hostapi = hostapis[hostapi_index]
    name = str(raw_hostapi.get("name", "")).strip()
    return name or "Unknown Host API"


def _device_name(value: object, *, index: int) -> str:
    name = str(value or "").strip()
    if name:
        return name
    return f"Input {index}"


def _to_int(value: object, fallback: int) -> int:
    value_any: Any = value
    try:
        return int(value_any)
    except Exception:
        return fallback


def _to_float(value: object) -> float | None:
    value_any: Any = value
    try:
        return float(value_any)
    except Exception:
        return None
