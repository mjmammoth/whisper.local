from __future__ import annotations

import time

from murmur import audio
import numpy as np


class _FakeDefaultDevice:
    def __init__(self, device):
        self.device = device


def test_scan_audio_input_devices_filters_non_input_devices(monkeypatch):
    monkeypatch.setattr(
        audio.sd,
        "query_devices",
        lambda: [
            {"name": "Output Only", "hostapi": 0, "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Built-in Mic", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 48000},
        ],
    )
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: [{"name": "CoreAudio"}])
    monkeypatch.setattr(audio.sd, "default", _FakeDefaultDevice((1, 3)))
    monkeypatch.setattr(audio.sd, "check_input_settings", lambda **_: None)

    result = audio.scan_audio_input_devices(sample_rate=16000)

    assert result.error is None
    assert len(result.devices) == 1
    assert result.devices[0].name == "Built-in Mic"
    assert result.devices[0].is_default is True
    assert result.default_device_key == "CoreAudio:Built-in Mic"


def test_scan_audio_input_devices_disambiguates_duplicate_keys(monkeypatch):
    monkeypatch.setattr(
        audio.sd,
        "query_devices",
        lambda: [
            {"name": "USB Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "USB Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
        ],
    )
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: [{"name": "CoreAudio"}])
    monkeypatch.setattr(audio.sd, "default", _FakeDefaultDevice((0, 0)))
    monkeypatch.setattr(audio.sd, "check_input_settings", lambda **_: None)

    result = audio.scan_audio_input_devices(sample_rate=16000)

    keys = [device.key for device in result.devices]
    assert keys == ["CoreAudio:USB Mic", "CoreAudio:USB Mic#2"]


def test_resolve_audio_input_device_index(monkeypatch):
    monkeypatch.setattr(
        audio.sd,
        "query_devices",
        lambda: [
            {"name": "Mic A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Mic B", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
        ],
    )
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: [{"name": "CoreAudio"}])
    monkeypatch.setattr(audio.sd, "default", _FakeDefaultDevice((0, 0)))
    monkeypatch.setattr(audio.sd, "check_input_settings", lambda **_: None)

    result = audio.scan_audio_input_devices(sample_rate=16000)
    key = result.devices[1].key

    assert audio.resolve_audio_input_device_index(key, result.devices) == 1
    assert audio.resolve_audio_input_device_index("unknown", result.devices) is None


def test_scan_audio_input_devices_marks_unsupported_sample_rate(monkeypatch):
    monkeypatch.setattr(
        audio.sd,
        "query_devices",
        lambda: [
            {"name": "Mic A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Mic B", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000},
        ],
    )
    monkeypatch.setattr(audio.sd, "query_hostapis", lambda: [{"name": "CoreAudio"}])
    monkeypatch.setattr(audio.sd, "default", _FakeDefaultDevice((0, 0)))

    def _check_input_settings(*, device, samplerate, channels, dtype):
        del samplerate, channels, dtype
        if device == 0:
            raise ValueError("unsupported sample rate")

    monkeypatch.setattr(audio.sd, "check_input_settings", _check_input_settings)

    result = audio.scan_audio_input_devices(sample_rate=32000)

    assert result.devices[0].sample_rate_supported is False
    assert result.devices[0].sample_rate_reason == "unsupported sample rate"
    assert result.devices[1].sample_rate_supported is True


class _FakeStream:
    def __init__(self, *, stop_delay_seconds: float = 0.0, stop_error: Exception | None = None):
        self.stop_delay_seconds = stop_delay_seconds
        self.stop_error = stop_error
        self.stop_called = 0
        self.close_called = 0

    def stop(self) -> None:
        self.stop_called += 1
        if self.stop_delay_seconds > 0:
            time.sleep(self.stop_delay_seconds)
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
        self.close_called += 1


def test_audio_recorder_stop_returns_empty_when_not_recording():
    recorder = audio.AudioRecorder(sample_rate=16_000, channels=1)

    result = recorder.stop()

    assert result.size == 0


def test_audio_recorder_stop_flattens_frames_and_closes_stream():
    recorder = audio.AudioRecorder(sample_rate=16_000, channels=1)
    stream = _FakeStream()
    recorder._stream = stream
    recorder._frames = [np.array([[0.5], [-0.25]], dtype=np.float32)]

    result = recorder.stop(timeout_seconds=0.2)

    assert np.allclose(result, np.array([0.5, -0.25], dtype=np.float32))
    assert stream.stop_called == 1
    assert stream.close_called == 1
    assert recorder.is_recording() is False


def test_audio_recorder_stop_timeout_resets_recorder_state_without_blocking():
    recorder = audio.AudioRecorder(sample_rate=16_000, channels=1)
    stream = _FakeStream(stop_delay_seconds=0.2)
    recorder._stream = stream
    recorder._frames = [np.array([[1.0]], dtype=np.float32)]

    started = time.monotonic()
    result = recorder.stop(timeout_seconds=0.01)
    elapsed = time.monotonic() - started

    assert result.size == 1
    assert elapsed < 0.15
    assert recorder.is_recording() is False
