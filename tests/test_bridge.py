from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from time import monotonic
from unittest.mock import AsyncMock, Mock, patch

import pytest

from murmur.audio import AudioInputDeviceInfo
from murmur.bridge import (
    BridgeLogFilter,
    BridgeServer,
    FIRST_RUN_SETUP_FALLBACK_DELAY_SECONDS,
    FIRST_RUN_SETUP_MESSAGE,
    RUNTIME_INITIALIZING_MESSAGE,
    WebSocketLogHandler,
)
from murmur.config import AppConfig


class FakeServeContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


def spawn_task_stub(coro):
    coro.close()
    return Mock()


class FakeClientWebSocket:
    def __init__(self, messages: list[str] | None = None) -> None:
        self.path = "/ws"
        self._messages = list(messages or [])
        self._iter_index = 0
        self.sent_payloads: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent_payloads.append(payload)

    def __aiter__(self):
        self._iter_index = 0
        return self

    async def __anext__(self) -> str:
        if self._iter_index >= len(self._messages):
            raise StopAsyncIteration
        payload = self._messages[self._iter_index]
        self._iter_index += 1
        return payload


@pytest.fixture
def mock_config():
    """
    Provide a Mock AppConfig preconfigured with typical test defaults.

    The returned Mock mimics an AppConfig instance used by tests and includes preset values:
    - top-level flags: auto_copy=False, auto_paste=False, auto_revert_clipboard=True
    - audio: sample_rate=16000, noise_suppression.enabled=False
    - vad: enabled=False, aggressiveness=2
    - model: name='tiny', runtime='faster-whisper', device='cpu', compute_type='int8', path=None, language=None
    - hotkey: key='f3', mode='ptt'
    - ui: theme='default'
    - output: clipboard=False, file.enabled=False, file.path=Path('/tmp/output.txt')

    Returns:
        Mock: A Mock object spec'd as AppConfig with the above attributes.
    """
    config = Mock(spec=AppConfig)
    config.auto_copy = False
    config.auto_paste = False
    config.auto_revert_clipboard = True
    config.audio = Mock()
    config.audio.sample_rate = 16000
    config.audio.input_device = None
    config.audio.noise_suppression = Mock()
    config.audio.noise_suppression.enabled = False
    config.vad = Mock()
    config.vad.enabled = False
    config.vad.aggressiveness = 2
    config.model = Mock()
    config.model.name = 'tiny'
    config.model.runtime = 'faster-whisper'
    config.model.device = 'cpu'
    config.model.compute_type = 'int8'
    config.model.path = None
    config.model.language = None
    config.hotkey = Mock()
    config.hotkey.key = 'f3'
    config.hotkey.mode = 'ptt'
    config.ui = Mock()
    config.ui.theme = 'default'
    config.output = Mock()
    config.output.clipboard = False
    config.output.file = Mock()
    config.output.file.enabled = False
    config.output.file.path = Path('/tmp/output.txt')
    config.history = Mock()
    config.history.max_entries = 5000
    return config


@pytest.fixture(autouse=True)
def isolate_transcript_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use an isolated transcript SQLite file for each test."""
    monkeypatch.setattr(
        "murmur.bridge.transcript_db_path",
        lambda: tmp_path / "transcripts.sqlite3",
    )


def test_bridge_log_filter_blocks_websocket_handshake_errors():
    """Test BridgeLogFilter blocks websockets handshake errors."""
    log_filter = BridgeLogFilter()

    record = logging.LogRecord(
        name='websockets.server',
        level=logging.ERROR,
        pathname='',
        lineno=0,
        msg='opening handshake failed',
        args=(),
        exc_info=None
    )

    assert not log_filter.filter(record)


def test_bridge_log_filter_blocks_websocket_asyncio_handshake_errors():
    """Test BridgeLogFilter blocks websockets.asyncio handshake errors."""
    log_filter = BridgeLogFilter()

    record = logging.LogRecord(
        name='websockets.asyncio.server',
        level=logging.ERROR,
        pathname='',
        lineno=0,
        msg='connection opening handshake failed',
        args=(),
        exc_info=None
    )

    assert not log_filter.filter(record)


def test_bridge_log_filter_allows_murmur_info():
    """Test BridgeLogFilter allows murmur INFO logs."""
    log_filter = BridgeLogFilter()

    record = logging.LogRecord(
        name='murmur.bridge',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Bridge server running',
        args=(),
        exc_info=None
    )

    assert log_filter.filter(record)


def test_bridge_log_filter_blocks_murmur_debug():
    """Test BridgeLogFilter blocks murmur DEBUG logs."""
    log_filter = BridgeLogFilter()

    record = logging.LogRecord(
        name='murmur.bridge',
        level=logging.DEBUG,
        pathname='',
        lineno=0,
        msg='Debug message',
        args=(),
        exc_info=None
    )

    assert not log_filter.filter(record)


def test_bridge_log_filter_allows_other_warnings():
    """Test BridgeLogFilter allows WARNING logs from other modules."""
    log_filter = BridgeLogFilter()

    record = logging.LogRecord(
        name='some.other.module',
        level=logging.WARNING,
        pathname='',
        lineno=0,
        msg='Warning message',
        args=(),
        exc_info=None
    )

    assert log_filter.filter(record)


def test_bridge_log_filter_allows_websocket_non_handshake_errors():
    """Test BridgeLogFilter allows websocket errors that are not handshake-related."""
    log_filter = BridgeLogFilter()

    record = logging.LogRecord(
        name='websockets.server',
        level=logging.ERROR,
        pathname='',
        lineno=0,
        msg='connection closed unexpectedly',
        args=(),
        exc_info=None
    )

    # Should be allowed since it's a non-murmur logger
    assert log_filter.filter(record)


def test_bridge_server_init(mock_config):
    """Test BridgeServer initialization."""
    server = BridgeServer(mock_config)

    assert server.config == mock_config
    assert server.clients == set()
    assert server._recording is False
    assert server._auto_copy is False
    assert server._auto_paste is False
    assert server._model_loaded is False


def test_bridge_server_init_forces_auto_copy_when_auto_paste_enabled(mock_config):
    """Auto paste startup state should always imply auto copy."""
    mock_config.auto_copy = False
    mock_config.auto_paste = True

    with patch('murmur.bridge.save_config') as mock_save:
        with patch('murmur.bridge.logger') as mock_logger:
            server = BridgeServer(mock_config)

    assert server._auto_paste is True
    assert server._auto_copy is True
    assert mock_config.auto_copy is True
    mock_save.assert_called_once_with(mock_config)
    mock_logger.info.assert_called_once_with(
        'Auto paste enabled in config; forcing auto copy on'
    )


def test_toggle_auto_copy_is_blocked_while_auto_paste_enabled(mock_config):
    """Disabling auto copy should be rejected when auto paste is on."""
    server = BridgeServer(mock_config)
    server._auto_copy = True
    server._auto_paste = True
    mock_config.auto_copy = True
    mock_config.auto_paste = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch('murmur.bridge.save_config') as mock_save:
        with patch('murmur.bridge.logger') as mock_logger:
            asyncio.run(
                server._handle_message(
                    Mock(),
                    json.dumps({'type': 'toggle_auto_copy', 'enabled': False}),
                )
            )

    assert server._auto_copy is True
    assert mock_config.auto_copy is True
    mock_save.assert_not_called()
    server._broadcast.assert_awaited_once_with(
        {
            'type': 'toast',
            'message': 'Auto copy remains on while auto paste is enabled',
            'level': 'info',
        }
    )
    server._broadcast_config.assert_awaited_once()
    mock_logger.info.assert_called_once_with(
        'Rejected auto copy disable request because auto paste is enabled'
    )


def test_toggle_auto_paste_enables_auto_copy(mock_config):
    """Enabling auto paste should automatically enable auto copy."""
    server = BridgeServer(mock_config)
    server._auto_copy = False
    server._auto_paste = False
    mock_config.auto_copy = False
    mock_config.auto_paste = False
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch('murmur.bridge.save_config') as mock_save:
        with patch('murmur.bridge.logger') as mock_logger:
            asyncio.run(
                server._handle_message(
                    Mock(),
                    json.dumps({'type': 'toggle_auto_paste', 'enabled': True}),
                )
            )

    assert server._auto_paste is True
    assert server._auto_copy is True
    assert mock_config.auto_paste is True
    assert mock_config.auto_copy is True
    mock_save.assert_called_once_with(mock_config)
    server._broadcast.assert_awaited_once_with(
        {
            'type': 'toast',
            'message': 'Auto paste on; auto copy on',
            'level': 'success',
        }
    )
    server._broadcast_config.assert_awaited_once()
    mock_logger.info.assert_called_once_with('Auto paste enabled; forcing auto copy on')


def test_toggle_auto_revert_clipboard_persists_and_broadcasts(mock_config):
    """Toggling auto revert clipboard should persist and broadcast config."""
    server = BridgeServer(mock_config)
    server._auto_revert_clipboard = True
    mock_config.auto_revert_clipboard = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch('murmur.bridge.save_config') as mock_save:
        asyncio.run(
            server._handle_message(
                Mock(),
                json.dumps({'type': 'toggle_auto_revert_clipboard', 'enabled': False}),
            )
        )

    assert server._auto_revert_clipboard is False
    assert mock_config.auto_revert_clipboard is False
    mock_save.assert_called_once_with(mock_config)
    server._broadcast.assert_awaited_once_with(
        {
            'type': 'toast',
            'message': 'Auto revert clipboard off',
            'level': 'success',
        }
    )
    server._broadcast_config.assert_awaited_once()


def test_config_payload_includes_auto_revert_clipboard(mock_config):
    """Bridge config payload should include auto_revert_clipboard flag."""
    server = BridgeServer(mock_config)
    server._auto_revert_clipboard = False
    mock_config.to_dict.return_value = {"model": {"runtime": "faster-whisper"}}

    payload = server._config_payload()

    assert payload["auto_revert_clipboard"] is False


def test_config_payload_includes_audio_inputs(mock_config):
    """Bridge config payload should include audio input diagnostics."""
    server = BridgeServer(mock_config)
    mock_config.to_dict.return_value = {
        "model": {"runtime": "faster-whisper"},
        "audio": {"sample_rate": 16000, "input_device": None},
    }
    server._audio_inputs = [
        AudioInputDeviceInfo(
            key="CoreAudio:Built-in Mic",
            index=0,
            name="Built-in Mic",
            hostapi="CoreAudio",
            max_input_channels=2,
            default_samplerate=48000.0,
            is_default=True,
            sample_rate_supported=True,
            sample_rate_reason=None,
        )
    ]
    server._audio_inputs_dirty = False
    server._audio_inputs_updated_at = monotonic()

    payload = server._config_payload()

    assert "audio_inputs" in payload
    assert payload["audio_inputs"]["devices"][0]["key"] == "CoreAudio:Built-in Mic"


def test_config_payload_includes_platform_capabilities(mock_config):
    """Bridge config payload should include platform capability flags."""
    server = BridgeServer(mock_config)
    mock_config.to_dict.return_value = {"model": {"runtime": "faster-whisper"}}

    payload = server._config_payload()

    assert "platform_capabilities" in payload
    caps = payload["platform_capabilities"]
    assert "hotkey_capture" in caps
    assert "hotkey_swallow" in caps
    assert "status_indicator" in caps
    assert "auto_paste" in caps


def test_config_payload_surfaces_detected_platform_capabilities(mock_config):
    server = BridgeServer(mock_config)
    mock_config.to_dict.return_value = {"model": {"runtime": "faster-whisper"}}
    server._platform_capabilities = {
        "hotkey_capture": True,
        "hotkey_swallow": True,
        "status_indicator": False,
        "auto_paste": False,
        "hotkey_guidance": None,
    }

    payload = server._config_payload()

    assert payload["platform_capabilities"]["hotkey_capture"] is True
    assert payload["platform_capabilities"]["hotkey_swallow"] is True


def test_config_payload_includes_startup_state_and_does_not_force_audio_scan(mock_config):
    server = BridgeServer(mock_config)
    mock_config.to_dict.return_value = {"model": {"runtime": "faster-whisper"}}

    with patch.object(server, "_refresh_audio_inputs") as mock_refresh:
        payload = server._config_payload()

    assert "startup" in payload
    assert payload["startup"]["phase"] in {"idle", "running", "ready", "error"}
    mock_refresh.assert_not_called()


def test_startup_onboarding_close_ready_accepts_degraded_non_model_tasks(mock_config):
    server = BridgeServer(mock_config)
    server._first_run_setup_required = False
    server._startup_model = "ready"
    server._startup_runtime_probe = "degraded"
    server._startup_audio_scan = "ready"
    server._startup_components = "degraded"

    assert server._startup_onboarding_close_ready() is True


def test_startup_onboarding_close_ready_blocks_when_selected_model_download_pending(mock_config):
    server = BridgeServer(mock_config)
    server._first_run_setup_required = False
    server._startup_model = "ready"
    server._startup_runtime_probe = "ready"
    server._startup_audio_scan = "ready"
    server._startup_components = "ready"
    pending_task = Mock()
    pending_task.done.return_value = False
    runtime = mock_config.model.runtime
    key = server._download_task_key(mock_config.model.name, runtime)
    server._download_queue.enqueue_download(
        key,
        model=mock_config.model.name,
        runtime=runtime,
        task=pending_task,
    )

    assert server._startup_onboarding_close_ready() is False


def test_begin_onboarding_setup_is_idempotent(mock_config):
    server = BridgeServer(mock_config)
    server._broadcast_config = AsyncMock()

    async def _run():
        with patch.object(server, "_spawn_task", side_effect=spawn_task_stub) as mock_spawn_task:
            await server._begin_onboarding_setup()
            await server._begin_onboarding_setup()
            assert mock_spawn_task.call_count == 1

    asyncio.run(_run())

    assert server._startup_runtime_probe == "running"
    assert server._startup_audio_scan == "running"
    assert server._startup_components == "running"
    server._broadcast_config.assert_awaited_once()


def test_first_run_start_defers_runtime_initialization_until_onboarding(mock_config):
    server = BridgeServer(mock_config)

    async def _run():
        with patch.object(server, "_has_installed_models", return_value=False), patch(
            "murmur.bridge.websockets.serve", return_value=FakeServeContext()
        ), patch.object(server, "_spawn_task", side_effect=spawn_task_stub) as mock_spawn_task:
            task = asyncio.create_task(server.start())
            await asyncio.sleep(0.01)
            assert server._first_run_setup_required is True
            assert server._startup_phase == "idle"
            assert mock_spawn_task.call_count == 1
            spawned_coro = mock_spawn_task.call_args.args[0]
            assert spawned_coro.cr_code.co_name == "_ensure_first_run_setup_fallback"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(_run())


def test_first_run_setup_fallback_starts_onboarding_when_idle(mock_config):
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._onboarding_setup_started = False
    server._begin_onboarding_setup = AsyncMock()

    async def _run():
        with patch("murmur.bridge.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await server._ensure_first_run_setup_fallback()
            mock_sleep.assert_awaited_once_with(FIRST_RUN_SETUP_FALLBACK_DELAY_SECONDS)

    asyncio.run(_run())

    server._begin_onboarding_setup.assert_awaited_once()


def test_first_run_setup_fallback_noops_when_onboarding_already_started(mock_config):
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._onboarding_setup_started = True
    server._begin_onboarding_setup = AsyncMock()

    async def _run():
        with patch("murmur.bridge.asyncio.sleep", new_callable=AsyncMock):
            await server._ensure_first_run_setup_fallback()

    asyncio.run(_run())

    server._begin_onboarding_setup.assert_not_called()


def test_non_first_run_start_spawns_runtime_initialization_task(mock_config):
    server = BridgeServer(mock_config)

    async def _run():
        with patch.object(server, "_has_installed_models", return_value=True), patch(
            "murmur.bridge.websockets.serve", return_value=FakeServeContext()
        ), patch.object(server, "_spawn_task", side_effect=spawn_task_stub) as mock_spawn_task:
            task = asyncio.create_task(server.start())
            await asyncio.sleep(0.01)
            assert server._first_run_setup_required is False
            assert server._startup_phase == "running"
            assert mock_spawn_task.call_count == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(_run())


def test_start_recording_blocks_when_first_run_setup_required(mock_config):
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._broadcast = AsyncMock()
    server._set_status = AsyncMock()
    server.recorder = Mock()

    asyncio.run(server._start_recording())

    server.recorder.start.assert_not_called()
    server._broadcast.assert_awaited_once_with(
        {
            "type": "toast",
            "message": "Recording unavailable: complete first-run model setup.",
            "level": "info",
        }
    )
    server._set_status.assert_awaited_once_with("connecting", FIRST_RUN_SETUP_MESSAGE)


def test_start_recording_blocks_when_runtime_or_model_not_ready(mock_config):
    server = BridgeServer(mock_config)
    server._first_run_setup_required = False
    server._model_loaded = False
    server.recorder = Mock()
    server.transcriber = None
    server._broadcast = AsyncMock()
    server._set_status = AsyncMock()

    asyncio.run(server._start_recording())

    server.recorder.start.assert_not_called()
    server._broadcast.assert_awaited_once_with(
        {
            "type": "toast",
            "message": "Recording unavailable: model/runtime is still starting.",
            "level": "info",
        }
    )
    server._set_status.assert_awaited_once_with(
        "connecting",
        RUNTIME_INITIALIZING_MESSAGE,
    )


def test_stop_recording_runs_stop_off_event_loop_and_spawns_transcription(mock_config):
    server = BridgeServer(mock_config)
    server._recording = True
    server.recorder = Mock()
    server.recorder.stop.return_value = [0.1, -0.2]
    server.transcriber = Mock()
    server._set_status = AsyncMock()
    server._spawn_task = Mock(side_effect=spawn_task_stub)
    server._process_audio = AsyncMock()

    asyncio.run(server._stop_recording())

    server.recorder.stop.assert_called_once()
    assert server._recording is False
    server._set_status.assert_awaited_once_with("transcribing", "Transcribing...")
    assert server._spawn_task.call_count == 1


def test_stop_recording_reports_error_when_recorder_stop_fails(mock_config):
    server = BridgeServer(mock_config)
    server._recording = True
    server.recorder = Mock()
    server.recorder.stop.side_effect = RuntimeError("stop failure")
    server._broadcast = AsyncMock()
    server._set_status = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._spawn_task = Mock()
    server._invalidate_audio_inputs = Mock()

    asyncio.run(server._stop_recording())

    assert server._recording is False
    server._invalidate_audio_inputs.assert_called_once()
    server._set_status.assert_awaited_once_with("error", "Recording stop failed: stop failure")
    server._broadcast_config.assert_awaited_once()
    server._broadcast.assert_awaited()
    server._spawn_task.assert_not_called()


def test_handle_client_does_not_block_message_processing_on_transcript_history(mock_config):
    server = BridgeServer(mock_config)
    websocket = FakeClientWebSocket(messages=[json.dumps({"type": "list_models"})])
    history_gate = asyncio.Event()

    async def _slow_history(_ws):
        await history_gate.wait()

    async def _run():
        server._send_config = AsyncMock()
        server._send_transcript_history_safe = AsyncMock(side_effect=_slow_history)
        server._handle_message = AsyncMock()

        task = asyncio.create_task(server._handle_client(websocket))

        deadline = monotonic() + 0.3
        while server._handle_message.await_count == 0 and monotonic() < deadline:
            await asyncio.sleep(0.005)

        assert server._handle_message.await_count == 1
        history_gate.set()
        await task

    asyncio.run(_run())


def test_set_audio_input_device_updates_recorder_and_config(mock_config):
    """Selecting input device should update config, recorder, and broadcasts."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server.recorder = Mock()
    server._audio_inputs = [
        AudioInputDeviceInfo(
            key="CoreAudio:USB Mic",
            index=3,
            name="USB Mic",
            hostapi="CoreAudio",
            max_input_channels=1,
            default_samplerate=48000.0,
            is_default=False,
            sample_rate_supported=True,
            sample_rate_reason=None,
        )
    ]
    with patch.object(server, "_refresh_audio_inputs"), patch.object(
        server, "_persist_config", return_value=None
    ):
        asyncio.run(server._set_audio_input_device("CoreAudio:USB Mic"))

    assert mock_config.audio.input_device == "CoreAudio:USB Mic"
    assert server.recorder.device == 3
    assert server._active_audio_input_key == "CoreAudio:USB Mic"
    server._broadcast_config.assert_awaited_once()
    server._broadcast.assert_awaited_with(
        {"type": "toast", "message": "Input device USB Mic", "level": "success"}
    )


def test_set_audio_input_device_rejects_unknown_key(mock_config):
    """Unknown input keys should be rejected without state changes."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._audio_inputs = []
    with patch.object(server, "_refresh_audio_inputs"):
        asyncio.run(server._set_audio_input_device("Missing Device"))

    assert mock_config.audio.input_device is None
    server._broadcast.assert_awaited_with(
        {"type": "toast", "message": "Selected input device is unavailable", "level": "error"}
    )
    server._broadcast_config.assert_awaited_once()


def test_set_audio_input_device_rejects_while_recording(mock_config):
    """Input device change should be blocked while recording."""
    server = BridgeServer(mock_config)
    server._recording = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    asyncio.run(server._set_audio_input_device("CoreAudio:USB Mic"))

    assert mock_config.audio.input_device is None
    server._broadcast.assert_awaited_with(
        {"type": "toast", "message": "Cannot change input device while recording", "level": "error"}
    )
    server._broadcast_config.assert_awaited_once()


def test_init_components_falls_back_to_default_when_saved_device_missing(mock_config):
    """Startup should fallback to system default when saved input device is unavailable."""
    mock_config.audio.input_device = "CoreAudio:Missing Mic"
    server = BridgeServer(mock_config)

    def fake_refresh(*, force=False):
        del force
        server._audio_inputs = []
        server._audio_inputs_dirty = False
        server._audio_inputs_updated_at = monotonic()

    with patch.object(server, "_refresh_audio_inputs", side_effect=fake_refresh), patch.object(
        server, "_persist_config", return_value=None
    ), patch("murmur.bridge.AudioRecorder") as mock_recorder, patch(
        "murmur.bridge.RNNoiseSuppressor"
    ), patch("murmur.bridge.VadProcessor"), patch(
        "murmur.bridge.create_hotkey_provider"
    ):
        server._init_components()

    assert mock_config.audio.input_device is None
    assert server._active_audio_input_key is None
    assert server._startup_audio_notice == "Saved input device unavailable; using system default"
    mock_recorder.assert_called_once()
    assert mock_recorder.call_args.kwargs["device"] is None


def test_refresh_audio_inputs_message_forces_refresh_and_broadcast(mock_config):
    """refresh_audio_inputs should force capability refresh and config broadcast."""
    server = BridgeServer(mock_config)
    server._broadcast_config = AsyncMock()
    with patch.object(server, "_refresh_audio_inputs") as mock_refresh:
        asyncio.run(
            server._handle_message(
                Mock(),
                json.dumps({"type": "refresh_audio_inputs"}),
            )
        )

    mock_refresh.assert_called_once_with(force=True)
    server._broadcast_config.assert_awaited_once()


def test_process_audio_auto_paste_reverts_clipboard_in_order(mock_config):
    """Auto-paste with auto-revert should snapshot, paste, and restore in order."""
    server = BridgeServer(mock_config)
    server._auto_paste = True
    server._auto_copy = True
    server._auto_revert_clipboard = True
    server._broadcast = AsyncMock()

    mock_config.output.clipboard = False
    mock_config.output.file.enabled = False
    mock_config.vad.enabled = False

    audio = Mock()
    audio.size = 4
    audio.shape = (4,)

    result = Mock()
    result.text = "hello world"
    result.language = "en"

    transcriber = Mock()
    transcriber.transcribe.return_value = result
    transcriber.runtime_info.return_value = {}

    call_order: list[str] = []
    snapshot = object()

    async def fake_to_thread(func, *args, **kwargs):
        """
        Invoke a synchronous callable directly from async code.

        Parameters:
            func (callable): The function or callable to invoke.
            *args: Positional arguments forwarded to `func`.
            **kwargs: Keyword arguments forwarded to `func`.

        Returns:
            The value returned by calling `func(*args, **kwargs)`.
        """
        return func(*args, **kwargs)

    with patch("murmur.bridge.capture_clipboard_snapshot") as mock_capture, patch(
        "murmur.bridge.copy_to_clipboard"
    ) as mock_copy, patch.object(
        server._paste_provider, "paste_from_clipboard"
    ) as mock_paste, patch(
        "murmur.bridge.restore_clipboard_snapshot"
    ) as mock_restore, patch(
        "murmur.bridge.asyncio.to_thread", side_effect=fake_to_thread
    ), patch("murmur.bridge.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_capture.side_effect = lambda: call_order.append("capture") or snapshot
        mock_copy.side_effect = lambda text: call_order.append("copy") or True
        mock_paste.side_effect = lambda: call_order.append("paste") or True
        mock_restore.side_effect = lambda snap: call_order.append("restore") or True

        asyncio.run(
            server._process_audio(
                audio,
                transcriber=transcriber,
                language=None,
                sample_rate=mock_config.audio.sample_rate,
            )
        )

    assert call_order == ["capture", "copy", "paste", "restore"]
    mock_sleep.assert_awaited_once_with(0.12)
    suppress_payloads = [
        call.args[0]
        for call in server._broadcast.await_args_list
        if call.args and call.args[0].get("type") == "suppress_paste_input"
    ]
    assert len(suppress_payloads) == 1


def test_process_audio_emits_transcript_id_and_created_at(mock_config):
    """Transcript payload should include persisted transcript metadata."""
    server = BridgeServer(mock_config)
    server._auto_paste = False
    server._auto_copy = False
    server._broadcast = AsyncMock()

    mock_config.output.clipboard = False
    mock_config.output.file.enabled = False
    mock_config.vad.enabled = False

    audio = Mock()
    audio.size = 4
    audio.shape = (4,)

    result = Mock()
    result.text = "hello world"
    result.language = "en"

    transcriber = Mock()
    transcriber.transcribe.return_value = result
    transcriber.runtime_info.return_value = {}

    asyncio.run(
        server._process_audio(
            audio,
            transcriber=transcriber,
            language=None,
            sample_rate=mock_config.audio.sample_rate,
        )
    )

    transcript_payloads = [
        call.args[0]
        for call in server._broadcast.await_args_list
        if call.args and call.args[0].get("type") == "transcript"
    ]
    assert len(transcript_payloads) == 1
    payload = transcript_payloads[0]
    assert isinstance(payload.get("id"), int)
    assert isinstance(payload.get("created_at"), str)


def test_process_audio_broadcasts_transcript_when_persist_fails(mock_config):
    """Transcript broadcast should continue even when persistence raises."""
    server = BridgeServer(mock_config)
    server._auto_paste = False
    server._auto_copy = False
    server._broadcast = AsyncMock()

    mock_config.output.clipboard = False
    mock_config.output.file.enabled = False
    mock_config.vad.enabled = False

    audio = Mock()
    audio.size = 4
    audio.shape = (4,)

    result = Mock()
    result.text = "hello world"
    result.language = "en"

    transcriber = Mock()
    transcriber.transcribe.return_value = result
    transcriber.runtime_info.return_value = {}

    server._transcript_store.append = Mock(side_effect=RuntimeError("sqlite write failed"))

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("murmur.bridge.asyncio.to_thread", side_effect=fake_to_thread):
        asyncio.run(
            server._process_audio(
                audio,
                transcriber=transcriber,
                language=None,
                sample_rate=mock_config.audio.sample_rate,
            )
        )

    transcript_payloads = [
        call.args[0]
        for call in server._broadcast.await_args_list
        if call.args and call.args[0].get("type") == "transcript"
    ]
    assert len(transcript_payloads) == 1
    payload = transcript_payloads[0]
    assert payload["text"] == "hello world"
    assert "id" not in payload
    assert "created_at" not in payload


def test_process_audio_auto_paste_without_revert_does_not_restore(mock_config):
    """Auto-paste without auto-revert should not capture or restore clipboard."""
    server = BridgeServer(mock_config)
    server._auto_paste = True
    server._auto_copy = True
    server._auto_revert_clipboard = False
    server._broadcast = AsyncMock()

    mock_config.output.clipboard = False
    mock_config.output.file.enabled = False
    mock_config.vad.enabled = False

    audio = Mock()
    audio.size = 4
    audio.shape = (4,)

    result = Mock()
    result.text = "hello world"
    result.language = "en"

    transcriber = Mock()
    transcriber.transcribe.return_value = result
    transcriber.runtime_info.return_value = {}

    async def fake_to_thread(func, *args, **kwargs):
        """
        Invoke a synchronous callable directly from async code.

        Parameters:
            func (callable): The function or callable to invoke.
            *args: Positional arguments forwarded to `func`.
            **kwargs: Keyword arguments forwarded to `func`.

        Returns:
            The value returned by calling `func(*args, **kwargs)`.
        """
        return func(*args, **kwargs)

    with patch("murmur.bridge.capture_clipboard_snapshot") as mock_capture, patch(
        "murmur.bridge.copy_to_clipboard", return_value=True
    ), patch.object(
        server._paste_provider, "paste_from_clipboard", return_value=True
    ), patch(
        "murmur.bridge.restore_clipboard_snapshot"
    ) as mock_restore, patch("murmur.bridge.asyncio.to_thread", side_effect=fake_to_thread):
        asyncio.run(
            server._process_audio(
                audio,
                transcriber=transcriber,
                language=None,
                sample_rate=mock_config.audio.sample_rate,
            )
        )

    mock_capture.assert_not_called()
    mock_restore.assert_not_called()


def test_process_audio_auto_paste_revert_attempted_when_paste_fails(mock_config):
    """Clipboard restore should still be attempted when paste command fails."""
    server = BridgeServer(mock_config)
    server._auto_paste = True
    server._auto_copy = True
    server._auto_revert_clipboard = True
    server._broadcast = AsyncMock()

    mock_config.output.clipboard = False
    mock_config.output.file.enabled = False
    mock_config.vad.enabled = False

    audio = Mock()
    audio.size = 4
    audio.shape = (4,)

    result = Mock()
    result.text = "hello world"
    result.language = "en"

    transcriber = Mock()
    transcriber.transcribe.return_value = result
    transcriber.runtime_info.return_value = {}

    async def fake_to_thread(func, *args, **kwargs):
        """
        Invoke a synchronous callable directly from async code.

        Parameters:
            func (callable): The function or callable to invoke.
            *args: Positional arguments forwarded to `func`.
            **kwargs: Keyword arguments forwarded to `func`.

        Returns:
            The value returned by calling `func(*args, **kwargs)`.
        """
        return func(*args, **kwargs)

    with patch("murmur.bridge.capture_clipboard_snapshot", return_value=object()), patch(
        "murmur.bridge.copy_to_clipboard", return_value=True
    ), patch.object(
        server._paste_provider, "paste_from_clipboard", return_value=False
    ), patch(
        "murmur.bridge.restore_clipboard_snapshot", return_value=True
    ) as mock_restore, patch(
        "murmur.bridge.asyncio.to_thread", side_effect=fake_to_thread
    ), patch("murmur.bridge.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        asyncio.run(
            server._process_audio(
                audio,
                transcriber=transcriber,
                language=None,
                sample_rate=mock_config.audio.sample_rate,
            )
        )

    mock_restore.assert_called_once()
    mock_sleep.assert_not_awaited()


def test_bridge_server_spawn_task():
    """Test _spawn_task creates and tracks tasks."""
    config = Mock()
    server = BridgeServer(config)

    async def dummy_coro():
        return "result"

    with patch('asyncio.create_task') as mock_create_task:
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        result = server._spawn_task(dummy_coro())

        assert result == mock_task
        assert mock_task in server._background_tasks
        mock_task.add_done_callback.assert_called_once()


def test_bridge_server_on_task_done_removes_task():
    """Test _on_task_done removes task from background tasks."""
    config = Mock()
    server = BridgeServer(config)

    mock_task = Mock()
    mock_task.cancelled.return_value = False
    mock_task.exception.return_value = None
    server._background_tasks.add(mock_task)

    server._on_task_done(mock_task)

    assert mock_task not in server._background_tasks


def test_bridge_server_on_task_done_logs_exception():
    """Test _on_task_done logs exception from failed task."""
    config = Mock()
    server = BridgeServer(config)

    mock_task = Mock()
    mock_task.cancelled.return_value = False
    mock_task.exception.return_value = RuntimeError("Task failed")
    server._background_tasks.add(mock_task)

    with patch('murmur.bridge.logger') as mock_logger:
        server._on_task_done(mock_task)

        mock_logger.error.assert_called_once()


def test_bridge_server_spawn_model_task_cancels_existing():
    """Test _spawn_model_task cancels existing task for same model."""
    config = Mock()
    server = BridgeServer(config)

    existing_task = Mock()
    existing_task.done.return_value = False
    server._model_tasks['tiny'] = existing_task

    async def dummy_coro():
        pass

    with patch('asyncio.create_task') as mock_create_task:
        mock_new_task = Mock()
        mock_create_task.return_value = mock_new_task

        server._spawn_model_task('tiny', dummy_coro())

        existing_task.cancel.assert_called_once()
        assert server._model_tasks['tiny'] == mock_new_task


def test_cancel_model_download_cancels_queued_task_without_error(mock_config):
    """Queued downloads should cancel cleanly instead of reporting no-active errors."""
    server = BridgeServer(mock_config)
    queued_task = Mock()
    queued_task.done.return_value = False
    server._download_queue.enqueue_download(
        "faster-whisper:small",
        model="small",
        runtime="faster-whisper",
        task=queued_task,
    )
    server._broadcast = AsyncMock()

    asyncio.run(server._cancel_model_download("small", runtime="faster-whisper"))

    queued_task.cancel.assert_called_once()
    server._broadcast.assert_awaited_once_with(
        {
            "type": "toast",
            "message": "Cancelled queued download small.",
            "model": "small",
            "runtime": "faster-whisper",
            "action": "download_cancelled",
            "level": "info",
        }
    )


def test_cancel_model_download_infers_single_queued_task(mock_config):
    """Blank cancel requests should resolve a single queued download."""
    server = BridgeServer(mock_config)
    queued_task = Mock()
    queued_task.done.return_value = False
    server._download_queue.enqueue_download(
        "whisper.cpp:base",
        model="base",
        runtime="whisper.cpp",
        task=queued_task,
    )
    server._broadcast = AsyncMock()

    asyncio.run(server._cancel_model_download("", runtime=None))

    queued_task.cancel.assert_called_once()
    server._broadcast.assert_awaited_once_with(
        {
            "type": "toast",
            "message": "Cancelled queued download base.",
            "model": "base",
            "runtime": "whisper.cpp",
            "action": "download_cancelled",
            "level": "info",
        }
    )


def test_cancel_model_download_is_idempotent_without_error_toast(mock_config):
    server = BridgeServer(mock_config)
    queued_task = Mock()
    queued_task.done.return_value = False
    server._download_queue.enqueue_download(
        "faster-whisper:small",
        model="small",
        runtime="faster-whisper",
        task=queued_task,
    )
    server._broadcast = AsyncMock()

    asyncio.run(server._cancel_model_download("small", runtime="faster-whisper"))
    asyncio.run(server._cancel_model_download("small", runtime="faster-whisper"))

    messages = [call.args[0]["message"] for call in server._broadcast.await_args_list if call.args]
    assert "No active download matches request" not in messages


def test_cancel_all_model_downloads_cancels_queued_entries(mock_config):
    server = BridgeServer(mock_config)
    first_task = Mock()
    first_task.done.return_value = False
    second_task = Mock()
    second_task.done.return_value = False
    server._download_queue.enqueue_download(
        "faster-whisper:small",
        model="small",
        runtime="faster-whisper",
        task=first_task,
    )
    server._download_queue.enqueue_download(
        "whisper.cpp:base",
        model="base",
        runtime="whisper.cpp",
        task=second_task,
    )
    server._broadcast = AsyncMock()

    asyncio.run(server._cancel_all_model_downloads())

    first_task.cancel.assert_called_once()
    second_task.cancel.assert_called_once()


def test_bridge_server_client_path_legacy_api():
    """Test _client_path returns path from legacy websockets API."""
    config = Mock()
    server = BridgeServer(config)

    mock_websocket = Mock()
    mock_websocket.path = '/ws?client=status-indicator'

    path = server._client_path(mock_websocket)
    assert path == '/ws?client=status-indicator'


def test_bridge_server_client_path_new_api():
    """Test _client_path returns path from new websockets API."""
    config = Mock()
    server = BridgeServer(config)

    mock_websocket = Mock()
    del mock_websocket.path  # Simulate new API without .path
    mock_websocket.request = Mock()
    mock_websocket.request.path = '/ws?client=passive'

    path = server._client_path(mock_websocket)
    assert path == '/ws?client=passive'


def test_bridge_server_client_path_fallback():
    """Test _client_path returns empty string when no path available."""
    config = Mock()
    server = BridgeServer(config)

    mock_websocket = Mock()
    del mock_websocket.path
    mock_websocket.request = None

    path = server._client_path(mock_websocket)
    assert path == ''


def test_bridge_server_is_passive_client_status_indicator():
    """Test _is_passive_client identifies status-indicator clients."""
    config = Mock()
    server = BridgeServer(config)

    mock_websocket = Mock()
    mock_websocket.path = '/ws?client=status-indicator'

    assert server._is_passive_client(mock_websocket) is True


def test_bridge_server_is_passive_client_passive():
    """Test _is_passive_client identifies passive clients."""
    config = Mock()
    server = BridgeServer(config)

    mock_websocket = Mock()
    mock_websocket.path = '/ws?client=passive'

    assert server._is_passive_client(mock_websocket) is True


def test_bridge_server_is_passive_client_normal():
    """Test _is_passive_client returns False for normal clients."""
    config = Mock()
    server = BridgeServer(config)

    mock_websocket = Mock()
    mock_websocket.path = '/ws'

    assert server._is_passive_client(mock_websocket) is False


def test_bridge_server_has_active_clients():
    """Test _has_active_clients correctly identifies active clients."""
    config = Mock()
    server = BridgeServer(config)

    active_client = Mock()
    passive_client = Mock()

    server.clients = {active_client, passive_client}
    server._passive_clients = {passive_client}

    assert server._has_active_clients() is True


def test_bridge_server_has_no_active_clients():
    """Test _has_active_clients returns False when only passive clients."""
    config = Mock()
    server = BridgeServer(config)

    passive_client = Mock()

    server.clients = {passive_client}
    server._passive_clients = {passive_client}

    assert server._has_active_clients() is False


def test_bridge_server_active_client_count():
    """Test _active_client_count returns correct count."""
    config = Mock()
    server = BridgeServer(config)

    active1 = Mock()
    active2 = Mock()
    passive = Mock()

    server.clients = {active1, active2, passive}
    server._passive_clients = {passive}

    assert server._active_client_count() == 2


def test_handle_client_disconnect_does_not_stop_hotkey(mock_config):
    """Disconnecting the last client should not disable service hotkey capture."""
    server = BridgeServer(mock_config)
    server.hotkey = Mock()
    server._hotkey_started = True
    server._model_loaded = True
    server._send_config = AsyncMock()
    server._send_transcript_history = AsyncMock()

    class DummyWebSocket:
        path = "/ws"

        async def send(self, _payload: str) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    asyncio.run(server._handle_client(DummyWebSocket()))

    server.hotkey.stop.assert_not_called()
    assert server._hotkey_started is True


def test_handle_client_disconnect_cleanup_runs_once(mock_config):
    """Disconnect cleanup and disconnect logging should run exactly once."""
    server = BridgeServer(mock_config)
    server._hotkey_blocked = True
    server._send_config = AsyncMock()
    server._send_transcript_history = AsyncMock()
    server._handle_message = AsyncMock()

    class DummyWebSocket:
        path = "/ws"

        def __init__(self) -> None:
            self._sent_message = False

        async def send(self, _payload: str) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent_message:
                self._sent_message = True
                return json.dumps({"type": "noop"})
            raise RuntimeError("connection closed")

    websocket = DummyWebSocket()

    with patch("murmur.bridge.websockets.ConnectionClosed", RuntimeError), patch(
        "murmur.bridge.logger"
    ) as mock_logger:
        asyncio.run(server._handle_client(websocket))

    assert server._handle_message.await_count == 1
    handled_websocket, handled_message = server._handle_message.await_args.args
    assert handled_websocket is websocket
    assert isinstance(handled_message, str)
    assert websocket not in server.clients
    assert websocket not in server._passive_clients
    assert server._hotkey_blocked is False
    disconnect_logs = [
        call
        for call in mock_logger.info.call_args_list
        if call.args and str(call.args[0]).startswith("Client disconnected.")
    ]
    assert len(disconnect_logs) == 1


def test_send_transcript_history_sends_history_payload(mock_config):
    """New clients should receive transcript history payload."""
    server = BridgeServer(mock_config)
    server._transcript_store.append("first", timestamp="12:00:00")
    websocket = AsyncMock()

    asyncio.run(server._send_transcript_history(websocket))

    websocket.send.assert_awaited_once()
    raw_payload = websocket.send.await_args.args[0]
    payload = json.loads(raw_payload)
    assert payload["type"] == "transcript_history"
    assert len(payload["entries"]) >= 1
    assert payload["entries"][-1]["text"] == "first"


def test_send_transcript_history_handles_history_read_failure(mock_config):
    server = BridgeServer(mock_config)
    server._transcript_store.history = Mock(side_effect=RuntimeError("history read failed"))
    websocket = AsyncMock()

    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(server._send_transcript_history(websocket))

    websocket.send.assert_awaited_once()
    raw_payload = websocket.send.await_args.args[0]
    payload = json.loads(raw_payload)
    assert payload["type"] == "transcript_history"
    assert payload["entries"] == []
    mock_logger.exception.assert_called_once()


def test_handle_client_start_hotkey_failure_runs_cleanup(mock_config):
    server = BridgeServer(mock_config)
    server._model_loaded = True
    server._hotkey_blocked = True
    server._start_hotkey = Mock(side_effect=RuntimeError("hotkey start failed"))
    server._send_config = AsyncMock()
    server._send_transcript_history = AsyncMock()

    class DummyWebSocket:
        path = "/ws"

        async def send(self, _payload: str) -> None:
            return

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    websocket = DummyWebSocket()

    with pytest.raises(RuntimeError, match="hotkey start failed"):
        asyncio.run(server._handle_client(websocket))

    assert websocket not in server.clients
    assert websocket not in server._passive_clients
    assert server._hotkey_blocked is False
    server._send_config.assert_not_awaited()
    server._send_transcript_history.assert_not_awaited()


def test_bridge_server_installed_model_names():
    """Test _installed_model_names returns installed model names."""
    config = Mock()
    server = BridgeServer(config)

    with patch('murmur.bridge.list_installed_models') as mock_list:
        mock_model1 = Mock(installed=True)
        mock_model1.name = 'tiny'
        mock_model2 = Mock(installed=False)
        mock_model2.name = 'base'
        mock_model3 = Mock(installed=True)
        mock_model3.name = 'small'
        mock_list.return_value = [mock_model1, mock_model2, mock_model3]

        result = server._installed_model_names()

        assert result == ['tiny', 'small']


def test_bridge_server_installed_model_names_backend_aware():
    """Test _installed_model_names filters by runtime variant install state."""
    config = Mock()
    config.model = Mock()
    config.model.runtime = "faster-whisper"
    server = BridgeServer(config)

    with patch("murmur.bridge.list_installed_models") as mock_list:
        tiny = Mock()
        tiny.name = "tiny"
        tiny.variants = {
            "faster-whisper": Mock(installed=True),
            "whisper.cpp": Mock(installed=False),
        }
        base = Mock()
        base.name = "base"
        base.variants = {
            "faster-whisper": Mock(installed=False),
            "whisper.cpp": Mock(installed=True),
        }
        mock_list.return_value = [tiny, base]

        assert server._installed_model_names("faster-whisper") == ["tiny"]
        assert server._installed_model_names("whisper.cpp") == ["base"]


def test_bridge_server_set_model_runtime_missing_variant_emits_requirement_event(mock_config):
    """Switching runtime without required variant should emit requirement event and keep runtime."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = False
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    capabilities = {
        "model": {
            "runtimes": {
                "whisper.cpp": {"enabled": True, "reason": None},
            },
            "devices": {},
            "compute_types_by_device": {},
        }
    }

    with patch.object(server, "_detect_runtime_capabilities", return_value=capabilities), patch(
        "murmur.bridge.get_installed_model_path",
        return_value=None,
    ):
        asyncio.run(server._set_model_runtime("whisper.cpp"))

    assert mock_config.model.runtime == "faster-whisper"

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    requirement_payload = next(
        (payload for payload in payloads if payload.get("type") == "runtime_switch_requires_model_variant"),
        None,
    )
    assert requirement_payload is not None
    assert requirement_payload["runtime"] == "whisper.cpp"
    assert requirement_payload["model"] == "tiny"
    assert requirement_payload["format"] == "ggml"


def test_bridge_server_set_model_runtime_first_run_missing_variant_applies_without_prompt(mock_config):
    """First-run runtime switch should apply config without model-variant requirement prompt."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._reload_transcriber = AsyncMock()

    capabilities = {
        "model": {
            "runtimes": {
                "whisper.cpp": {"enabled": True, "reason": None},
            },
            "devices": {
                "cpu": {"enabled": True, "reason": None},
                "mps": {"enabled": True, "reason": None},
                "cuda": {"enabled": False, "reason": "No CUDA GPU detected"},
            },
            "compute_types_by_device": {
                "cpu": ["default"],
                "mps": ["default"],
                "cuda": [],
            },
        }
    }

    with patch.object(server, "_detect_runtime_capabilities", return_value=capabilities), patch(
        "murmur.bridge.get_installed_model_path",
        return_value=None,
    ), patch.object(server, "_persist_config", return_value=None):
        asyncio.run(server._set_model_runtime("whisper.cpp"))

    assert mock_config.model.runtime == "whisper.cpp"
    assert mock_config.model.compute_type == "default"
    server._reload_transcriber.assert_not_awaited()

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert not any(
        payload.get("type") == "runtime_switch_requires_model_variant"
        for payload in payloads
    )
    assert any(
        payload.get("message") == "Model runtime whisper.cpp"
        for payload in payloads
    )


def test_bridge_server_set_model_device_first_run_persists_without_reload(mock_config):
    """First-run model device change should persist without transcriber reload."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._reload_transcriber = AsyncMock()
    server._runtime_capabilities = {
        "model": {
            "devices": {
                "cpu": {"enabled": True, "reason": None},
                "mps": {"enabled": True, "reason": None},
                "cuda": {"enabled": False, "reason": "No CUDA GPU detected"},
            }
        }
    }

    with patch.object(server, "_persist_config", return_value=None), patch.object(
        server, "_refresh_runtime_capabilities"
    ):
        asyncio.run(server._set_model_device("mps"))

    assert mock_config.model.device == "mps"
    server._reload_transcriber.assert_not_awaited()
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any(
        payload.get("message") == "Model device mps"
        for payload in payloads
    )


def test_bridge_server_download_progress_payload_includes_backend(mock_config):
    """Download progress payload should include runtime for variant-aware UI updates."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()
    server._set_selected_model = AsyncMock()

    def fake_download_model(name, runtime, progress_callback=None, cancel_check=None):
        """
        Simulates a model download in tests and reports a single fixed progress update.

        Parameters:
            name (str): Expected model name; the function asserts it equals "tiny".
            runtime (str): Expected runtime identifier; the function asserts it equals "whisper.cpp".
            progress_callback (callable | None): Optional callable invoked with an integer percent (35) to report progress.
            cancel_check (callable | None): Optional callable checked for cancellation; accepted but not used by this fake implementation.
        """
        assert name == "tiny"
        assert runtime == "whisper.cpp"
        if progress_callback:
            progress_callback(35)

    with patch("murmur.bridge.download_model", side_effect=fake_download_model):
        asyncio.run(server._download_model("tiny", runtime="whisper.cpp"))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    progress_payloads = [payload for payload in payloads if payload.get("type") == "download_progress"]
    assert any(
        payload.get("runtime") == "whisper.cpp" and payload.get("model") == "tiny"
        for payload in progress_payloads
    )
    assert any(
        payload.get("runtime") == "whisper.cpp" and payload.get("percent") == 100
        for payload in progress_payloads
    )


def test_bridge_server_has_installed_models_true():
    """Test _has_installed_models returns True when models installed."""
    config = Mock()
    server = BridgeServer(config)

    with patch('murmur.bridge.list_installed_models') as mock_list:
        mock_model = Mock(installed=True)
        mock_model.name = 'tiny'
        mock_list.return_value = [mock_model]

        assert server._has_installed_models() is True


def test_bridge_server_has_installed_models_false():
    """Test _has_installed_models returns False when no models installed."""
    config = Mock()
    server = BridgeServer(config)

    with patch('murmur.bridge.list_installed_models') as mock_list:
        mock_model = Mock(installed=False)
        mock_model.name = 'tiny'
        mock_list.return_value = [mock_model]

        assert server._has_installed_models() is False


def test_bridge_server_init_components(mock_config):
    """Test _init_components initializes all components."""
    server = BridgeServer(mock_config)

    with patch('murmur.bridge.AudioRecorder'), \
         patch('murmur.bridge.RNNoiseSuppressor'), \
         patch('murmur.bridge.VadProcessor'), \
         patch('murmur.bridge.create_hotkey_provider'):

        server._init_components()

        assert server.recorder is not None
        assert server.noise is not None
        assert server.vad is not None
        assert server.transcriber is None
        assert server.hotkey is not None


def test_bridge_server_detect_runtime_capabilities(mock_config):
    """Test _detect_runtime_capabilities calls detect_runtime_capabilities."""
    server = BridgeServer(mock_config)

    with patch('murmur.transcribe.detect_runtime_capabilities') as mock_detect:
        mock_detect.return_value = {'runtime': 'test'}

        result = server._detect_runtime_capabilities()

        assert result == {'runtime': 'test'}
        mock_detect.assert_called_once_with('faster-whisper')


def test_bridge_server_refresh_runtime_capabilities_no_force_within_ttl(mock_config):
    """Test _refresh_runtime_capabilities skips refresh within TTL."""
    server = BridgeServer(mock_config)
    server._runtime_capabilities = {'old': 'data'}
    server._runtime_capabilities_updated_at = monotonic()
    server._runtime_capabilities_dirty = False

    with patch('murmur.transcribe.detect_runtime_capabilities') as mock_detect:
        server._refresh_runtime_capabilities(force=False)

        # Should not call detect since within TTL
        mock_detect.assert_not_called()


def test_bridge_server_refresh_runtime_capabilities_force(mock_config):
    """Test _refresh_runtime_capabilities forces refresh when force=True."""
    server = BridgeServer(mock_config)

    with patch('murmur.transcribe.detect_runtime_capabilities') as mock_detect:
        mock_detect.return_value = {'new': 'data'}

        server._refresh_runtime_capabilities(force=True)

        mock_detect.assert_called_once()
        assert server._runtime_capabilities == {'new': 'data'}


def test_bridge_server_invalidate_runtime_capabilities(mock_config):
    """Test _invalidate_runtime_capabilities sets dirty flag."""
    server = BridgeServer(mock_config)
    server._runtime_capabilities_dirty = False

    server._invalidate_runtime_capabilities()

    assert server._runtime_capabilities_dirty is True


def test_bridge_server_set_runtime_capabilities(mock_config):
    """Test _set_runtime_capabilities updates capabilities and clears dirty flag."""
    server = BridgeServer(mock_config)
    server._runtime_capabilities_dirty = True

    new_caps = {'test': 'capabilities'}
    server._set_runtime_capabilities(new_caps)

    assert server._runtime_capabilities == new_caps
    assert server._runtime_capabilities_dirty is False


def test_bridge_server_persist_config_success(mock_config):
    """Test _persist_config saves config successfully."""
    server = BridgeServer(mock_config)

    with patch('murmur.bridge.save_config') as mock_save:
        result = server._persist_config('test context')

        assert result is None
        mock_save.assert_called_once_with(mock_config)


def test_bridge_server_persist_config_failure(mock_config):
    """Test _persist_config returns error message on failure."""
    server = BridgeServer(mock_config)

    with patch('murmur.bridge.save_config') as mock_save:
        mock_save.side_effect = Exception('Save failed')

        result = server._persist_config('test context')

        assert result == 'Save failed'


def test_bridge_server_extract_paths_from_paste_single_path():
    """Test _extract_paths_from_paste handles single file path."""
    config = Mock()
    server = BridgeServer(config)

    text = '/path/to/file.mp3'
    paths = server._extract_paths_from_paste(text)

    assert len(paths) >= 1


def test_bridge_server_extract_paths_from_paste_multiple_paths():
    """Test _extract_paths_from_paste handles multiple file paths."""
    config = Mock()
    server = BridgeServer(config)

    text = '/path/to/file1.mp3\n/path/to/file2.wav'
    paths = server._extract_paths_from_paste(text)

    assert len(paths) >= 2


def test_bridge_server_extract_paths_from_paste_quoted():
    """Test _extract_paths_from_paste handles quoted paths."""
    config = Mock()
    server = BridgeServer(config)

    text = '"/path/with spaces/file.mp3"'
    paths = server._extract_paths_from_paste(text)

    assert len(paths) >= 1


def test_bridge_server_normalize_paste_path_file_uri():
    """Test _normalize_paste_path handles file:// URIs."""
    config = Mock()
    server = BridgeServer(config)

    result = server._normalize_paste_path('file:///tmp/test.mp3')

    assert result is not None
    assert str(result).endswith('test.mp3')


def test_bridge_server_normalize_paste_path_empty():
    """Test _normalize_paste_path returns None for empty string."""
    config = Mock()
    server = BridgeServer(config)

    result = server._normalize_paste_path('')

    assert result is None


def test_bridge_server_normalize_paste_path_with_quotes():
    """Test _normalize_paste_path strips quotes."""
    config = Mock()
    server = BridgeServer(config)

    result = server._normalize_paste_path('"/tmp/test.mp3"')

    assert result is not None
    assert '"' not in str(result)


def test_bridge_server_shutdown_stops_recorder(mock_config):
    """Test shutdown stops recorder if recording."""
    server = BridgeServer(mock_config)
    server.recorder = Mock()
    server._recording = True

    server.shutdown()

    server.recorder.stop.assert_called_once()
    assert server._recording is False


def test_bridge_server_shutdown_stops_hotkey(mock_config):
    """Test shutdown stops hotkey if started."""
    server = BridgeServer(mock_config)
    server.hotkey = Mock()
    server._hotkey_started = True

    server.shutdown()

    server.hotkey.stop.assert_called_once()


def test_bridge_server_shutdown_closes_noise(mock_config):
    """Test shutdown closes noise suppressor."""
    server = BridgeServer(mock_config)
    server.noise = Mock()

    server.shutdown()

    server.noise.close.assert_called_once()


def test_bridge_server_shutdown_cancels_pending_download_tasks(mock_config):
    server = BridgeServer(mock_config)
    queued_task = Mock()
    queued_task.done.return_value = False
    server._download_queue.enqueue_download(
        "faster-whisper:tiny",
        model="tiny",
        runtime="faster-whisper",
        task=queued_task,
    )
    server._model_tasks["faster-whisper:tiny"] = queued_task

    server.shutdown()

    queued_task.cancel.assert_called()


def test_websocket_log_handler_emits_log_to_clients():
    """Test WebSocketLogHandler emits logs to bridge clients."""
    mock_bridge = Mock()
    mock_bridge.clients = [Mock()]
    mock_bridge._loop = Mock()
    mock_bridge._loop.is_closed.return_value = False

    handler = WebSocketLogHandler(mock_bridge)

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Test message',
        args=(),
        exc_info=None
    )

    handler.emit(record)

    # Should attempt to broadcast
    assert mock_bridge._loop.is_closed.called


def test_websocket_log_handler_no_emit_without_clients():
    """Test WebSocketLogHandler doesn't emit when no clients connected."""
    mock_bridge = Mock()
    mock_bridge.clients = []

    handler = WebSocketLogHandler(mock_bridge)

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Test message',
        args=(),
        exc_info=None
    )

    # Should not raise, just return early
    handler.emit(record)


def test_websocket_log_handler_handles_emit_errors():
    """Test WebSocketLogHandler handles errors during emit gracefully."""
    mock_bridge = Mock()
    mock_bridge.clients = [Mock()]
    mock_bridge._loop = Mock()
    mock_bridge._loop.is_closed.side_effect = Exception('Loop error')

    handler = WebSocketLogHandler(mock_bridge)

    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='',
        lineno=0,
        msg='Test message',
        args=(),
        exc_info=None
    )

    # Should not raise, errors are caught
    handler.emit(record)


def test_broadcast_mirrors_info_toast_to_logger(mock_config):
    server = BridgeServer(mock_config)
    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(
            server._broadcast(
                {
                    "type": "toast",
                    "message": "Model download queued",
                    "level": "info",
                    "action": "download_queued",
                    "runtime": "faster-whisper",
                    "model": "tiny",
                }
            )
        )

    mock_logger.info.assert_called_once_with(
        "toast: %s%s",
        "Model download queued",
        " (action=download_queued, runtime=faster-whisper, model=tiny)",
    )
    mock_logger.error.assert_not_called()


def test_broadcast_mirrors_success_toast_to_info_logger(mock_config):
    server = BridgeServer(mock_config)
    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(
            server._broadcast(
                {
                    "type": "toast",
                    "message": "Auto paste on; auto copy on",
                    "level": "success",
                }
            )
        )

    mock_logger.info.assert_called_once_with("toast: %s%s", "Auto paste on; auto copy on", "")
    mock_logger.error.assert_not_called()


def test_broadcast_mirrors_error_toast_to_error_logger(mock_config):
    server = BridgeServer(mock_config)
    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(
            server._broadcast(
                {
                    "type": "toast",
                    "message": "Failed to persist setting",
                    "level": "error",
                }
            )
        )

    mock_logger.error.assert_called_once_with("toast: %s%s", "Failed to persist setting", "")
    mock_logger.info.assert_not_called()


def test_broadcast_does_not_log_non_toast_messages(mock_config):
    server = BridgeServer(mock_config)
    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(server._broadcast({"type": "status", "status": "ready", "message": "Ready"}))

    mock_logger.info.assert_not_called()
    mock_logger.error.assert_not_called()


def test_broadcast_ignores_empty_toast_message(mock_config):
    server = BridgeServer(mock_config)
    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(server._broadcast({"type": "toast", "message": "   ", "level": "info"}))

    mock_logger.info.assert_not_called()
    mock_logger.error.assert_not_called()


def test_constants_defined():
    """Test that bridge constants are defined."""
    from murmur.bridge import (
        AUTO_PASTE_INPUT_SUPPRESS_MS,
        AUTO_REVERT_CLIPBOARD_DELAY_MS,
        MAX_DROP_AUDIO_SECONDS,
        MAX_DROP_FILE_BYTES,
        MAX_DROP_FILES,
    )

    assert isinstance(MAX_DROP_FILES, int)
    assert isinstance(MAX_DROP_FILE_BYTES, int)
    assert isinstance(MAX_DROP_AUDIO_SECONDS, int)
    assert isinstance(AUTO_PASTE_INPUT_SUPPRESS_MS, int)
    assert isinstance(AUTO_REVERT_CLIPBOARD_DELAY_MS, int)


# ---------------------------------------------------------------------------
# TranscriptionMetrics & _log_transcription_metrics
# ---------------------------------------------------------------------------


def test_transcription_metrics_dataclass_is_frozen():
    """TranscriptionMetrics should be immutable."""
    from murmur.bridge import TranscriptionMetrics

    metrics = TranscriptionMetrics(
        pipeline_started=0.0,
        input_samples=16000,
        post_noise_samples=16000,
        post_vad_samples=16000,
        transcribe_ms=200,
        job_sample_rate=16000,
        job_transcriber=Mock(runtime_info=lambda: {}),
        job_language="en",
        output_language="en",
        noise_enabled=False,
        noise_available=False,
        noise_applied=False,
        noise_backend="none",
        vad_enabled=False,
        vad_available=False,
        vad_applied=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.input_samples = 0  # type: ignore[misc]


def test_log_transcription_metrics_calls_logger(mock_config):
    """_log_transcription_metrics should invoke logger.info with bench data."""
    from murmur.bridge import TranscriptionMetrics

    server = BridgeServer(mock_config)
    transcriber_mock = Mock()
    transcriber_mock.runtime_info.return_value = {
        "runtime": "faster-whisper",
        "model_name": "tiny",
        "effective_device": "cpu",
        "effective_compute_type": "int8",
    }
    metrics = TranscriptionMetrics(
        pipeline_started=monotonic() - 0.5,
        input_samples=16000,
        post_noise_samples=16000,
        post_vad_samples=16000,
        transcribe_ms=200,
        job_sample_rate=16000,
        job_transcriber=transcriber_mock,
        job_language="en",
        output_language="en",
        noise_enabled=False,
        noise_available=True,
        noise_applied=False,
        noise_backend="rnnoise",
        vad_enabled=True,
        vad_available=True,
        vad_applied=True,
    )

    with patch("murmur.bridge.logger") as mock_logger:
        server._log_transcription_metrics(metrics)

    mock_logger.info.assert_called_once()
    log_msg = mock_logger.info.call_args.args[0]
    assert "bench" in log_msg
    assert "rtf=" in log_msg


def test_log_transcription_metrics_handles_zero_sample_rate(mock_config):
    """_log_transcription_metrics should handle zero sample rate gracefully."""
    from murmur.bridge import TranscriptionMetrics

    server = BridgeServer(mock_config)
    metrics = TranscriptionMetrics(
        pipeline_started=monotonic(),
        input_samples=0,
        post_noise_samples=0,
        post_vad_samples=0,
        transcribe_ms=0,
        job_sample_rate=0,
        job_transcriber=Mock(runtime_info=lambda: {}),
        job_language=None,
        output_language=None,
        noise_enabled=False,
        noise_available=False,
        noise_applied=False,
        noise_backend="none",
        vad_enabled=False,
        vad_available=False,
        vad_applied=False,
    )

    with patch("murmur.bridge.logger") as mock_logger:
        server._log_transcription_metrics(metrics)

    mock_logger.info.assert_called_once()
    # Ensure "auto" and "unknown" are used for None languages
    call_args = mock_logger.info.call_args.args
    assert "auto" in call_args
    assert "unknown" in call_args


# ---------------------------------------------------------------------------
# _dispatch_settings_message
# ---------------------------------------------------------------------------


def test_dispatch_settings_message_routes_known_type(mock_config):
    """_dispatch_settings_message should route a known message type to the handler."""
    server = BridgeServer(mock_config)
    server._set_theme = AsyncMock()

    result = asyncio.run(
        server._dispatch_settings_message("set_theme", {"theme": "dark"})
    )

    assert result is True
    server._set_theme.assert_awaited_once_with("dark")


def test_dispatch_settings_message_returns_false_for_unknown(mock_config):
    """_dispatch_settings_message should return False for unknown message types."""
    server = BridgeServer(mock_config)

    result = asyncio.run(
        server._dispatch_settings_message("totally_unknown", {})
    )

    assert result is False


def test_dispatch_settings_message_handles_refresh_audio_inputs(mock_config):
    """_dispatch_settings_message should handle refresh_audio_inputs specially."""
    server = BridgeServer(mock_config)
    server._handle_refresh_audio_inputs = AsyncMock()

    result = asyncio.run(
        server._dispatch_settings_message("refresh_audio_inputs", {})
    )

    assert result is True
    server._handle_refresh_audio_inputs.assert_awaited_once()


def test_dispatch_settings_message_uses_default_when_key_missing(mock_config):
    """_dispatch_settings_message should use default value when param key is absent."""
    server = BridgeServer(mock_config)
    server._set_hotkey_mode = AsyncMock()

    asyncio.run(server._dispatch_settings_message("set_hotkey_mode", {}))

    server._set_hotkey_mode.assert_awaited_once_with("")


# ---------------------------------------------------------------------------
# _dispatch_action_message
# ---------------------------------------------------------------------------


def test_dispatch_action_message_routes_download_model(mock_config):
    """_dispatch_action_message should route download_model to handler."""
    server = BridgeServer(mock_config)
    server._handle_download_model_message = Mock()

    asyncio.run(
        server._dispatch_action_message(
            Mock(), "download_model", {"name": "tiny", "runtime": "faster-whisper"}
        )
    )

    server._handle_download_model_message.assert_called_once()


def test_dispatch_action_message_routes_list_models(mock_config):
    """_dispatch_action_message should route list_models to _send_models."""
    server = BridgeServer(mock_config)
    server._send_models = AsyncMock()
    ws = Mock()

    asyncio.run(server._dispatch_action_message(ws, "list_models", {}))

    server._send_models.assert_awaited_once_with(ws)


def test_dispatch_action_message_routes_get_config(mock_config):
    """_dispatch_action_message should route get_config to _send_config."""
    server = BridgeServer(mock_config)
    server._send_config = AsyncMock()
    ws = Mock()

    asyncio.run(server._dispatch_action_message(ws, "get_config", {}))

    server._send_config.assert_awaited_once_with(ws)


def test_dispatch_action_message_routes_get_config_file(mock_config):
    """_dispatch_action_message should route get_config_file to _send_config_file."""
    server = BridgeServer(mock_config)
    server._send_config_file = AsyncMock()
    ws = Mock()

    asyncio.run(server._dispatch_action_message(ws, "get_config_file", {}))

    server._send_config_file.assert_awaited_once_with(ws)


def test_dispatch_action_message_routes_get_capabilities(mock_config):
    """_dispatch_action_message should route get_capabilities to _send_capabilities."""
    server = BridgeServer(mock_config)
    server._send_capabilities = AsyncMock()
    ws = Mock()

    asyncio.run(server._dispatch_action_message(ws, "get_capabilities", {}))

    server._send_capabilities.assert_awaited_once_with(ws)


def test_dispatch_action_message_logs_unknown_type(mock_config):
    """_dispatch_action_message should log a warning for unknown message types."""
    server = BridgeServer(mock_config)

    with patch("murmur.bridge.logger") as mock_logger:
        asyncio.run(
            server._dispatch_action_message(Mock(), "unknown_action_xyz", {})
        )

    mock_logger.warning.assert_called_once()


def test_dispatch_action_message_routes_begin_onboarding(mock_config):
    """_dispatch_action_message should route begin_onboarding_setup."""
    server = BridgeServer(mock_config)
    server._begin_onboarding_setup = AsyncMock()

    asyncio.run(
        server._dispatch_action_message(Mock(), "begin_onboarding_setup", {})
    )

    server._begin_onboarding_setup.assert_awaited_once()


# ---------------------------------------------------------------------------
# _resolve_enabled_device (static)
# ---------------------------------------------------------------------------


def test_resolve_enabled_device_returns_requested_when_enabled():
    """_resolve_enabled_device should return requested device if enabled."""
    devices = {"cuda": {"enabled": True}, "cpu": {"enabled": True}}
    assert BridgeServer._resolve_enabled_device(devices, "cuda") == "cuda"


def test_resolve_enabled_device_falls_back_when_disabled():
    """_resolve_enabled_device should fall back to first available when requested is disabled."""
    devices = {
        "cuda": {"enabled": False},
        "mps": {"enabled": False},
        "cpu": {"enabled": True},
    }
    assert BridgeServer._resolve_enabled_device(devices, "cuda") == "cpu"


def test_resolve_enabled_device_prefers_cuda_over_cpu():
    """_resolve_enabled_device should prefer cuda > mps > cpu in fallback order."""
    devices = {
        "cuda": {"enabled": True},
        "mps": {"enabled": True},
        "cpu": {"enabled": True},
    }
    # requesting a non-existent device should fall back to cuda first
    assert BridgeServer._resolve_enabled_device(devices, "tpu") == "cuda"


def test_resolve_enabled_device_returns_requested_when_none_enabled():
    """_resolve_enabled_device should return requested when no devices are enabled."""
    devices = {"cpu": {"enabled": False}}
    assert BridgeServer._resolve_enabled_device(devices, "cpu") == "cpu"


# ---------------------------------------------------------------------------
# _resolve_compute_type (static)
# ---------------------------------------------------------------------------


def test_resolve_compute_type_returns_requested_when_valid():
    """_resolve_compute_type should return the requested type when it's in the valid set."""
    valid = {"int8", "float32"}
    assert BridgeServer._resolve_compute_type(valid, "float32") == "float32"


def test_resolve_compute_type_falls_back_when_invalid():
    """_resolve_compute_type should fall back to preferred type when requested is invalid."""
    valid = {"default", "float32"}
    assert BridgeServer._resolve_compute_type(valid, "int8") == "default"


def test_resolve_compute_type_returns_requested_when_set_empty():
    """_resolve_compute_type should accept any type when valid set is empty."""
    assert BridgeServer._resolve_compute_type(set(), "int8") == "int8"


def test_resolve_compute_type_falls_back_to_sorted_first():
    """_resolve_compute_type should fall back to sorted first when no preferred match."""
    valid = {"zzz_custom"}
    assert BridgeServer._resolve_compute_type(valid, "int8") == "zzz_custom"


# ---------------------------------------------------------------------------
# _download_task_key (static)
# ---------------------------------------------------------------------------


def test_download_task_key():
    """_download_task_key should produce 'runtime:name' format."""
    assert BridgeServer._download_task_key("tiny", "faster-whisper") == "faster-whisper:tiny"


# ---------------------------------------------------------------------------
# _resolve_download_cancel_key
# ---------------------------------------------------------------------------


def test_resolve_download_cancel_key_explicit_key(mock_config):
    """_resolve_download_cancel_key should return a key containing ':' as-is."""
    server = BridgeServer(mock_config)
    assert server._resolve_download_cancel_key("faster-whisper:tiny", None) == "faster-whisper:tiny"


def test_resolve_download_cancel_key_with_runtime(mock_config):
    """_resolve_download_cancel_key should build key from model+runtime."""
    server = BridgeServer(mock_config)
    result = server._resolve_download_cancel_key("tiny", "faster-whisper")
    assert result == "faster-whisper:tiny"


def test_resolve_download_cancel_key_empty_resolves_single(mock_config):
    """_resolve_download_cancel_key should resolve single queued entry when name is empty."""
    server = BridgeServer(mock_config)
    task = Mock()
    task.done.return_value = False
    server._download_queue.enqueue_download(
        "faster-whisper:small", model="small", runtime="faster-whisper", task=task
    )

    result = server._resolve_download_cancel_key("", None)
    assert result == "faster-whisper:small"


def test_resolve_download_cancel_key_returns_none_when_ambiguous(mock_config):
    """_resolve_download_cancel_key should return None when multiple candidates match."""
    server = BridgeServer(mock_config)
    for key, runtime in [("faster-whisper:tiny", "faster-whisper"), ("whisper.cpp:tiny", "whisper.cpp")]:
        task = Mock()
        task.done.return_value = False
        server._download_queue.enqueue_download(key, model="tiny", runtime=runtime, task=task)

    result = server._resolve_download_cancel_key("tiny", None)
    assert result is None


# ---------------------------------------------------------------------------
# _on_download_success / _on_download_cancelled / _on_download_error
# ---------------------------------------------------------------------------


def test_on_download_success_broadcasts_completion(mock_config):
    """_on_download_success should broadcast progress 100% and success toast."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()
    server._set_selected_model = AsyncMock()

    asyncio.run(server._on_download_success("tiny", "faster-whisper", None))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any(p.get("type") == "download_progress" and p.get("percent") == 100 for p in payloads)
    assert any(p.get("action") == "download_complete" for p in payloads)
    server._broadcast_models.assert_awaited_once()


def test_on_download_success_activates_runtime_when_specified(mock_config):
    """_on_download_success should switch runtime when activate_runtime_name is given."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()
    server._set_model_runtime = AsyncMock()

    with patch.object(server, "_persist_config", return_value=None):
        asyncio.run(server._on_download_success("tiny", "whisper.cpp", "whisper.cpp"))

    assert mock_config.model.name == "tiny"
    assert mock_config.model.path is None
    server._set_model_runtime.assert_awaited_once_with("whisper.cpp", allow_missing_variant_prompt=False)


def test_on_download_success_selects_model_when_same_runtime(mock_config):
    """_on_download_success should auto-select model when runtime matches config."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()
    server._set_selected_model = AsyncMock()

    asyncio.run(server._on_download_success("small", "faster-whisper", None))

    server._set_selected_model.assert_awaited_once_with("small")


def test_on_download_cancelled_broadcasts_when_not_shutting_down(mock_config):
    """_on_download_cancelled should broadcast cancellation toast."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()

    asyncio.run(server._on_download_cancelled("tiny", "faster-whisper"))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any(p.get("action") == "download_cancelled" for p in payloads)
    server._broadcast_models.assert_awaited_once()


def test_on_download_cancelled_silent_during_shutdown(mock_config):
    """_on_download_cancelled should not broadcast during shutdown."""
    server = BridgeServer(mock_config)
    server._shutdown_requested.set()
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()

    asyncio.run(server._on_download_cancelled("tiny", "faster-whisper"))

    server._broadcast.assert_not_awaited()
    server._broadcast_models.assert_not_awaited()


def test_on_download_error_broadcasts_failure(mock_config):
    """_on_download_error should broadcast error toast with exception message."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()

    asyncio.run(server._on_download_error("tiny", "faster-whisper", RuntimeError("network error")))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any(p.get("action") == "download_failed" and "network error" in p.get("message", "") for p in payloads)
    server._broadcast_models.assert_awaited_once()


def test_on_download_error_silent_during_shutdown(mock_config):
    """_on_download_error should not broadcast during shutdown."""
    server = BridgeServer(mock_config)
    server._shutdown_requested.set()
    server._broadcast = AsyncMock()
    server._broadcast_models = AsyncMock()

    asyncio.run(server._on_download_error("tiny", "faster-whisper", RuntimeError("err")))

    server._broadcast.assert_not_awaited()


# ---------------------------------------------------------------------------
# _validate_audio_input_device
# ---------------------------------------------------------------------------


def test_validate_audio_input_device_none_is_valid(mock_config):
    """_validate_audio_input_device should accept None (system default)."""
    server = BridgeServer(mock_config)

    valid, device = asyncio.run(server._validate_audio_input_device(None))

    assert valid is True
    assert device is None


def test_validate_audio_input_device_missing_device(mock_config):
    """_validate_audio_input_device should reject missing devices."""
    server = BridgeServer(mock_config)
    server._audio_inputs = []
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    valid, device = asyncio.run(server._validate_audio_input_device("Missing:Device"))

    assert valid is False
    assert device is None
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("unavailable" in p.get("message", "") for p in payloads)


def test_validate_audio_input_device_unsupported_sample_rate(mock_config):
    """_validate_audio_input_device should reject devices with unsupported sample rate."""
    server = BridgeServer(mock_config)
    server._audio_inputs = [
        AudioInputDeviceInfo(
            key="CoreAudio:Bad Mic",
            index=1,
            name="Bad Mic",
            hostapi="CoreAudio",
            max_input_channels=1,
            default_samplerate=44100.0,
            is_default=False,
            sample_rate_supported=False,
            sample_rate_reason="Device does not support 16000 Hz",
        )
    ]
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    valid, device = asyncio.run(server._validate_audio_input_device("CoreAudio:Bad Mic"))

    assert valid is False
    assert device is None
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("16000" in p.get("message", "") for p in payloads)


def test_validate_audio_input_device_valid_device(mock_config):
    """_validate_audio_input_device should accept a valid device."""
    server = BridgeServer(mock_config)
    good_device = AudioInputDeviceInfo(
        key="CoreAudio:Good Mic",
        index=2,
        name="Good Mic",
        hostapi="CoreAudio",
        max_input_channels=2,
        default_samplerate=48000.0,
        is_default=True,
        sample_rate_supported=True,
        sample_rate_reason=None,
    )
    server._audio_inputs = [good_device]

    valid, device = asyncio.run(server._validate_audio_input_device("CoreAudio:Good Mic"))

    assert valid is True
    assert device is good_device


# ---------------------------------------------------------------------------
# _apply_config_with_reload
# ---------------------------------------------------------------------------


def test_apply_config_with_reload_success_path(mock_config):
    """_apply_config_with_reload should apply, persist, reload, and toast on success."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = False
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._reload_transcriber = AsyncMock()
    apply_fn = Mock()
    rollback_fn = Mock()

    with patch.object(server, "_persist_config", return_value=None):
        asyncio.run(
            server._apply_config_with_reload(
                apply_fn=apply_fn,
                rollback_fn=rollback_fn,
                context="test context",
                success_message="Test success",
            )
        )

    apply_fn.assert_called_once()
    rollback_fn.assert_not_called()
    server._reload_transcriber.assert_awaited_once()
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any(p.get("message") == "Test success" for p in payloads)


def test_apply_config_with_reload_first_run_skips_reload(mock_config):
    """_apply_config_with_reload should skip reload during first run setup."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._reload_transcriber = AsyncMock()

    with patch.object(server, "_persist_config", return_value=None):
        asyncio.run(
            server._apply_config_with_reload(
                apply_fn=Mock(),
                rollback_fn=Mock(),
                context="test",
                success_message="Test OK",
            )
        )

    server._reload_transcriber.assert_not_awaited()
    server._broadcast_config.assert_awaited_once()


def test_apply_config_with_reload_rollback_on_failure(mock_config):
    """_apply_config_with_reload should rollback on reload failure."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = False
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._reload_transcriber = AsyncMock(side_effect=RuntimeError("load failed"))
    rollback_fn = Mock()

    with patch.object(server, "_persist_config", return_value=None):
        asyncio.run(
            server._apply_config_with_reload(
                apply_fn=Mock(),
                rollback_fn=rollback_fn,
                context="test",
                success_message="Should not appear",
            )
        )

    rollback_fn.assert_called_once()
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("Failed to apply test" in p.get("message", "") for p in payloads)


def test_apply_config_with_reload_persist_error_during_first_run(mock_config):
    """_apply_config_with_reload should show persist error toast during first run."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch.object(server, "_persist_config", return_value="disk full"):
        asyncio.run(
            server._apply_config_with_reload(
                apply_fn=Mock(),
                rollback_fn=Mock(),
                context="test",
                success_message="Test OK",
            )
        )

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("disk full" in p.get("message", "") for p in payloads)


# ---------------------------------------------------------------------------
# _send_capabilities
# ---------------------------------------------------------------------------


def test_send_capabilities_cpu_fallback(mock_config):
    """_send_capabilities should recommend cpu/faster-whisper when no GPU available."""
    server = BridgeServer(mock_config)
    server._runtime_capabilities = {
        "model": {
            "devices_by_runtime": {
                "faster-whisper": {"cpu": {"enabled": True}, "cuda": {"enabled": False}},
                "whisper.cpp": {"cpu": {"enabled": True}, "mps": {"enabled": False}},
            }
        }
    }
    ws = AsyncMock()

    asyncio.run(server._send_capabilities(ws))

    ws.send.assert_awaited_once()
    payload = json.loads(ws.send.await_args.args[0])
    assert payload["type"] == "capabilities"
    assert payload["recommended"]["runtime"] == "faster-whisper"
    assert payload["recommended"]["device"] == "cpu"


def test_send_capabilities_cuda_recommendation(mock_config):
    """_send_capabilities should recommend cuda when available."""
    server = BridgeServer(mock_config)
    server._runtime_capabilities = {
        "model": {
            "devices_by_runtime": {
                "faster-whisper": {"cpu": {"enabled": True}, "cuda": {"enabled": True}},
                "whisper.cpp": {"cpu": {"enabled": True}},
            }
        }
    }
    ws = AsyncMock()

    with patch("murmur.bridge.sys") as mock_sys:
        mock_sys.platform = "linux"
        asyncio.run(server._send_capabilities(ws))

    payload = json.loads(ws.send.await_args.args[0])
    assert payload["recommended"]["runtime"] == "faster-whisper"
    assert payload["recommended"]["device"] == "cuda"


def test_send_capabilities_empty_caps(mock_config):
    """_send_capabilities should handle empty capabilities gracefully."""
    server = BridgeServer(mock_config)
    server._runtime_capabilities = {}
    ws = AsyncMock()

    asyncio.run(server._send_capabilities(ws))

    payload = json.loads(ws.send.await_args.args[0])
    assert payload["type"] == "capabilities"
    assert payload["recommended"]["runtime"] == "faster-whisper"
    assert payload["recommended"]["device"] == "cpu"


# ---------------------------------------------------------------------------
# _send_config_file
# ---------------------------------------------------------------------------


def test_send_config_file_reads_existing_file(mock_config, tmp_path):
    """_send_config_file should send the content of an existing config file."""
    server = BridgeServer(mock_config)
    config_file = tmp_path / "murmur.toml"
    config_file.write_text("[model]\nname = 'tiny'\n")
    ws = AsyncMock()

    with patch("murmur.bridge.default_config_path", return_value=config_file):
        asyncio.run(server._send_config_file(ws))

    payload = json.loads(ws.send.await_args.args[0])
    assert payload["type"] == "config_file"
    assert "tiny" in payload["content"]
    assert str(config_file) == payload["path"]


def test_send_config_file_returns_empty_for_missing(mock_config, tmp_path):
    """_send_config_file should return empty content when config file doesn't exist."""
    server = BridgeServer(mock_config)
    ws = AsyncMock()

    with patch("murmur.bridge.default_config_path", return_value=tmp_path / "nonexistent.toml"):
        asyncio.run(server._send_config_file(ws))

    payload = json.loads(ws.send.await_args.args[0])
    assert payload["content"] == ""


def test_send_config_file_returns_empty_on_read_error(mock_config, tmp_path):
    """_send_config_file should return empty content when read fails."""
    server = BridgeServer(mock_config)
    ws = AsyncMock()
    config_path = Mock()
    config_path.exists.return_value = True
    config_path.read_text.side_effect = PermissionError("access denied")

    with patch("murmur.bridge.default_config_path", return_value=config_path):
        asyncio.run(server._send_config_file(ws))

    payload = json.loads(ws.send.await_args.args[0])
    assert payload["content"] == ""


# ---------------------------------------------------------------------------
# _send_config
# ---------------------------------------------------------------------------


def test_send_config_sends_config_payload(mock_config):
    """_send_config should send config payload to a specific client."""
    server = BridgeServer(mock_config)
    mock_config.to_dict.return_value = {"model": {"runtime": "faster-whisper"}}
    ws = AsyncMock()

    asyncio.run(server._send_config(ws))

    ws.send.assert_awaited_once()
    payload = json.loads(ws.send.await_args.args[0])
    assert payload["type"] == "config"
    assert "config" in payload


# ---------------------------------------------------------------------------
# _handle_toggle_auto_copy with persist error
# ---------------------------------------------------------------------------


def test_toggle_auto_copy_on_with_persist_error(mock_config):
    """Enabling auto copy with persist failure should show error toast."""
    server = BridgeServer(mock_config)
    server._auto_copy = False
    server._auto_paste = False
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch.object(server, "_persist_config", return_value="disk error"):
        asyncio.run(
            server._handle_toggle_auto_copy({"enabled": True})
        )

    assert server._auto_copy is True
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("disk error" in p.get("message", "") for p in payloads)
    assert any(p.get("level") == "error" for p in payloads)


def test_toggle_auto_copy_enable_success(mock_config):
    """Enabling auto copy should persist and broadcast success."""
    server = BridgeServer(mock_config)
    server._auto_copy = False
    server._auto_paste = False
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch.object(server, "_persist_config", return_value=None):
        asyncio.run(server._handle_toggle_auto_copy({"enabled": True}))

    assert server._auto_copy is True
    assert mock_config.auto_copy is True
    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any(p.get("message") == "Auto copy on" for p in payloads)


# ---------------------------------------------------------------------------
# _set_model_device
# ---------------------------------------------------------------------------


def test_set_model_device_rejects_invalid_device(mock_config):
    """_set_model_device should reject invalid device names."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch.object(server, "_refresh_runtime_capabilities"):
        asyncio.run(server._set_model_device("tpu"))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("Invalid model device" in p.get("message", "") for p in payloads)


def test_set_model_device_rejects_disabled_device(mock_config):
    """_set_model_device should reject a disabled device with reason."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._runtime_capabilities = {
        "model": {
            "devices": {
                "cuda": {"enabled": False, "reason": "No CUDA GPU detected"},
            }
        }
    }

    with patch.object(server, "_refresh_runtime_capabilities"):
        asyncio.run(server._set_model_device("cuda"))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("No CUDA GPU detected" in p.get("message", "") for p in payloads)


def test_set_model_device_noops_when_same(mock_config):
    """_set_model_device should no-op when device is already set."""
    server = BridgeServer(mock_config)
    mock_config.model.device = "cpu"
    server._broadcast = AsyncMock()
    server._runtime_capabilities = {
        "model": {"devices": {"cpu": {"enabled": True}}}
    }

    with patch.object(server, "_refresh_runtime_capabilities"):
        asyncio.run(server._set_model_device("cpu"))

    server._broadcast.assert_not_awaited()


# ---------------------------------------------------------------------------
# _set_model_compute_type
# ---------------------------------------------------------------------------


def test_set_model_compute_type_rejects_invalid(mock_config):
    """_set_model_compute_type should reject invalid compute types."""
    server = BridgeServer(mock_config)
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()

    with patch.object(server, "_refresh_runtime_capabilities"):
        asyncio.run(server._set_model_compute_type("invalid_type"))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("Invalid compute type" in p.get("message", "") for p in payloads)


def test_set_model_compute_type_rejects_float16_on_cpu(mock_config):
    """_set_model_compute_type should reject float16 on CPU."""
    server = BridgeServer(mock_config)
    mock_config.model.device = "cpu"
    server._broadcast = AsyncMock()
    server._broadcast_config = AsyncMock()
    server._runtime_capabilities = {"model": {"compute_types_by_device": {"cpu": ["int8", "float32"]}}}

    with patch.object(server, "_refresh_runtime_capabilities"):
        asyncio.run(server._set_model_compute_type("float16"))

    payloads = [call.args[0] for call in server._broadcast.await_args_list if call.args]
    assert any("not usable on CPU" in p.get("message", "") for p in payloads)


# ---------------------------------------------------------------------------
# _startup_blockers and _refresh_startup_phase
# ---------------------------------------------------------------------------


def test_startup_blockers_first_run(mock_config):
    """_startup_blockers should include first-run message when setup required."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._startup_model = "pending"

    blockers = server._startup_blockers()
    assert any("Download and select" in b for b in blockers)


def test_startup_blockers_model_loading(mock_config):
    """_startup_blockers should include model loading message."""
    server = BridgeServer(mock_config)
    server._startup_model = "running"

    blockers = server._startup_blockers()
    assert any("Model is still loading" in b for b in blockers)


def test_startup_blockers_model_error(mock_config):
    """_startup_blockers should include model error message."""
    server = BridgeServer(mock_config)
    server._startup_model = "error"

    blockers = server._startup_blockers()
    assert any("Model failed to load" in b for b in blockers)


def test_refresh_startup_phase_error(mock_config):
    """_refresh_startup_phase should set error when model failed."""
    server = BridgeServer(mock_config)
    server._startup_model = "error"

    server._refresh_startup_phase()

    assert server._startup_phase == "error"


def test_refresh_startup_phase_ready(mock_config):
    """_refresh_startup_phase should set ready when all settled."""
    server = BridgeServer(mock_config)
    server._startup_model = "ready"
    server._startup_runtime_probe = "ready"
    server._startup_audio_scan = "ready"
    server._startup_components = "ready"

    server._refresh_startup_phase()

    assert server._startup_phase == "ready"


def test_refresh_startup_phase_idle_during_first_run(mock_config):
    """_refresh_startup_phase should stay idle before onboarding starts."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._onboarding_setup_started = False
    server._startup_model = "pending"

    server._refresh_startup_phase()

    assert server._startup_phase == "idle"


def test_refresh_startup_phase_running(mock_config):
    """_refresh_startup_phase should set running during active setup."""
    server = BridgeServer(mock_config)
    server._first_run_setup_required = True
    server._onboarding_setup_started = True
    server._startup_model = "running"

    server._refresh_startup_phase()

    assert server._startup_phase == "running"


# ---------------------------------------------------------------------------
# _startup_payload
# ---------------------------------------------------------------------------


def test_startup_payload_structure(mock_config):
    """_startup_payload should return expected keys."""
    server = BridgeServer(mock_config)
    server._startup_model = "ready"
    server._startup_runtime_probe = "ready"
    server._startup_audio_scan = "ready"
    server._startup_components = "ready"

    payload = server._startup_payload()

    assert "phase" in payload
    assert "runtime_probe" in payload
    assert "audio_scan" in payload
    assert "components" in payload
    assert "model" in payload
    assert "onboarding_close_ready" in payload
    assert "blockers" in payload
    assert "last_error" in payload


# ---------------------------------------------------------------------------
# shutdown additional coverage
# ---------------------------------------------------------------------------


def test_shutdown_sets_shutdown_event(mock_config):
    """shutdown should set the shutdown requested event."""
    server = BridgeServer(mock_config)

    server.shutdown()

    assert server._shutdown_requested.is_set()


def test_shutdown_handles_recorder_stop_error(mock_config):
    """shutdown should handle recorder stop errors gracefully."""
    server = BridgeServer(mock_config)
    server.recorder = Mock()
    server.recorder.stop.side_effect = RuntimeError("device error")
    server._recording = True

    server.shutdown()

    assert server._recording is False


# ---------------------------------------------------------------------------
# _handle_refresh_audio_inputs
# ---------------------------------------------------------------------------


def test_handle_refresh_audio_inputs(mock_config):
    """_handle_refresh_audio_inputs should force refresh and broadcast."""
    server = BridgeServer(mock_config)
    server._broadcast_config = AsyncMock()

    with patch.object(server, "_refresh_audio_inputs") as mock_refresh:
        asyncio.run(server._handle_refresh_audio_inputs())

    mock_refresh.assert_called_once_with(force=True)
    server._broadcast_config.assert_awaited_once()


# ---------------------------------------------------------------------------
# _install_log_handler
# ---------------------------------------------------------------------------


def test_install_log_handler_configures_root_logger(mock_config):
    """_install_log_handler should install handler and configure log levels."""
    server = BridgeServer(mock_config)

    with patch("murmur.bridge.logging.getLogger") as mock_get_logger:
        root = Mock()
        root.handlers = [Mock()]
        ws_logger = Mock()
        asyncio_logger = Mock()

        def side_effect(name=None):
            if name is None:
                return root
            if name == "websockets":
                return ws_logger
            if name == "asyncio":
                return asyncio_logger
            return Mock()

        mock_get_logger.side_effect = side_effect

        server._install_log_handler()

    root.addHandler.assert_called_once()
    root.setLevel.assert_called_once_with(logging.INFO)
    ws_logger.setLevel.assert_called_once_with(logging.WARNING)
    asyncio_logger.setLevel.assert_called_once_with(logging.WARNING)
