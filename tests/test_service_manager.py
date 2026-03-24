from __future__ import annotations

import errno
import json
import sys
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import pytest

from murmur import service_manager
from murmur.service_manager import ServiceManager
from murmur.service_state import (
    ServiceState,
    service_log_path,
    service_state_path,
    transcript_db_path,
)


def test_is_pid_alive_treats_eperm_as_alive() -> None:
    with patch(
        "murmur.service_manager.os.kill",
        side_effect=OSError(errno.EPERM, "operation not permitted"),
    ):
        assert service_manager._is_pid_alive(1234) is True


def test_terminate_pid_attempts_non_blocking_waitpid_after_final_kill() -> None:
    expected_kill_signal = getattr(service_manager.signal, "SIGKILL", service_manager.signal.SIGTERM)
    expected_wnohang = getattr(service_manager.os, "WNOHANG", 0)

    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager.os.kill"
    ) as mock_kill, patch(
        "murmur.service_manager.os.waitpid",
        create=True,
    ) as mock_waitpid:
        service_manager._terminate_pid(1234, timeout=0.0)

    mock_kill.assert_any_call(1234, service_manager.signal.SIGTERM)
    mock_kill.assert_any_call(1234, expected_kill_signal)
    mock_waitpid.assert_called_once_with(1234, expected_wnohang)


def test_terminate_pid_ignores_waitpid_errors() -> None:
    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager.os.kill"
    ), patch(
        "murmur.service_manager.os.waitpid",
        side_effect=ChildProcessError,
        create=True,
    ):
        service_manager._terminate_pid(1234, timeout=0.0)


def test_terminate_pid_skips_when_expected_pid_check_fails() -> None:
    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager.os.kill"
    ) as mock_kill:
        service_manager._terminate_pid(1234, is_expected_pid=lambda _: False)

    mock_kill.assert_not_called()


def test_pid_matches_bridge_process_checks_expected_arguments() -> None:
    with patch(
        "murmur.service_manager._process_argv",
        return_value=(
            "/usr/bin/python3",
            "-m",
            "murmur.cli",
            "bridge",
            "--host",
            "localhost",
            "--port",
            "7878",
            "--capture-logs",
        ),
    ):
        assert service_manager._pid_matches_bridge_process(2222, host="localhost", port=7878) is True
        assert service_manager._pid_matches_bridge_process(2222, host="127.0.0.1", port=7878) is False
        assert service_manager._pid_matches_bridge_process(2222, host="localhost", port=9000) is False


def test_pid_matches_status_indicator_process_checks_expected_arguments() -> None:
    with patch(
        "murmur.service_manager._process_argv",
        return_value=(
            "/usr/bin/python3",
            "-m",
            "murmur.status_indicator",
            "--host",
            "localhost",
            "--port",
            "7878",
        ),
    ):
        assert (
            service_manager._pid_matches_status_indicator_process(
                3333,
                host="localhost",
                port=7878,
            )
            is True
        )
        assert (
            service_manager._pid_matches_status_indicator_process(
                3333,
                host="127.0.0.1",
                port=7878,
            )
            is False
        )
        assert (
            service_manager._pid_matches_status_indicator_process(
                3333,
                host="localhost",
                port=9000,
            )
            is False
        )


def test_status_returns_stopped_when_state_missing(tmp_path: Path) -> None:
    manager = ServiceManager(state_path=tmp_path / "service.json")

    status = manager.status()

    assert status.running is False
    assert status.pid is None
    assert status.stale is False
    assert status.pid_alive is False
    assert status.stale_reason is None


def test_status_marks_and_cleans_stale_state(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    state = ServiceState.new(pid=99999, host="localhost", port=7878, status_indicator_pid=None)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path)

    with patch("murmur.service_manager._is_pid_alive", return_value=False):
        status = manager.status()

    assert status.running is False
    assert status.stale is True
    assert status.pid == 99999
    assert status.pid_alive is False
    assert status.stale_reason == "pid_not_alive"
    assert state_path.exists() is False


def test_status_marks_alive_but_unreachable_state_stale(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    state = ServiceState.new(
        pid=2222,
        host="localhost",
        port=7878,
        status_indicator_pid=3333,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path)

    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager._is_port_reachable",
        return_value=False,
    ), patch("murmur.service_manager._terminate_pid") as mock_terminate:
        status = manager.status()

    assert status.running is False
    assert status.stale is True
    assert status.reachable is False
    assert status.pid == 2222
    assert status.host == "localhost"
    assert status.port == 7878
    assert status.status_indicator_pid == 3333
    assert status.pid_alive is True
    assert status.stale_reason == "port_unreachable"
    mock_terminate.assert_any_call(2222, is_expected_pid=ANY)
    mock_terminate.assert_any_call(3333, timeout=1.5, is_expected_pid=ANY)
    assert state_path.exists() is False


def test_status_diagnose_mode_does_not_cleanup_unreachable_state(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    state = ServiceState.new(
        pid=2222,
        host="localhost",
        port=7878,
        status_indicator_pid=3333,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path)

    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager._is_port_reachable",
        return_value=False,
    ), patch("murmur.service_manager._terminate_pid") as mock_terminate:
        status = manager.status(cleanup_stale=False, auto_recover_unreachable=False)

    assert status.running is False
    assert status.stale is True
    assert status.pid_alive is True
    assert status.stale_reason == "port_unreachable"
    mock_terminate.assert_not_called()
    assert state_path.exists() is True


def test_status_auto_restarts_unreachable_bridge(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    state = ServiceState.new(
        pid=2222,
        host="localhost",
        port=7878,
        status_indicator_pid=3333,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path)

    recovered = Mock()
    recovered.running = True
    recovered.stale = False

    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager._is_port_reachable",
        return_value=False,
    ), patch.object(
        manager,
        "start_background",
        return_value=recovered,
    ) as mock_start_background:
        status = manager.status(
            cleanup_stale=True,
            auto_recover_unreachable=True,
            status_indicator=True,
        )

    mock_start_background.assert_called_once_with(
        host="localhost",
        port=7878,
        status_indicator=True,
    )
    assert status is recovered


def test_status_returns_stale_when_cleanup_fails(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    state = ServiceState.new(pid=2222, host="localhost", port=7878, status_indicator_pid=None)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path)

    with patch("murmur.service_manager._is_pid_alive", return_value=True), patch(
        "murmur.service_manager._is_port_reachable",
        return_value=False,
    ), patch.object(
        manager,
        "_cleanup_stale_state",
        side_effect=PermissionError("read-only"),
    ):
        status = manager.status()

    assert status.running is False
    assert status.stale is True
    assert status.reachable is False
    assert status.pid == 2222
    assert state_path.exists() is True


def test_save_state_does_not_ensure_global_state_directory(tmp_path: Path) -> None:
    state_path = tmp_path / "custom-state" / "service.json"
    manager = ServiceManager(state_path=state_path)
    state = ServiceState.new(pid=1234, host="localhost", port=7878, status_indicator_pid=None)

    with patch("murmur.service_manager.ensure_state_directory") as mock_ensure_state_dir:
        manager.save_state(state)

    mock_ensure_state_dir.assert_not_called()
    assert state_path.exists()


def test_start_background_persists_service_state(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    manager = ServiceManager(state_path=state_path, python_executable="/usr/bin/python3")

    process = Mock()
    process.pid = 1234
    indicator_provider = Mock()
    indicator_provider.pid = 5678

    with patch("murmur.service_manager.subprocess.Popen", return_value=process), patch(
        "murmur.service_manager._wait_for_port", return_value=True
    ), patch(
        "murmur.service_manager._is_pid_alive", return_value=True
    ), patch(
        "murmur.service_manager._is_port_reachable", return_value=True
    ), patch(
        "murmur.service_manager.create_status_indicator_provider",
        return_value=indicator_provider,
    ):
        status = manager.start_background(host="localhost", port=7878, status_indicator=True)

    assert status.running is True
    assert state_path.exists()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["pid"] == 1234
    assert saved["host"] == "localhost"
    assert saved["port"] == 7878


def test_start_background_uses_indicator_provider_when_requested_on_non_darwin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "service.json"
    manager = ServiceManager(state_path=state_path, python_executable="/usr/bin/python3")

    process = Mock()
    process.pid = 1234
    indicator_provider = Mock()
    indicator_provider.pid = 5678
    monkeypatch.setattr("murmur.service_manager.sys.platform", "linux")

    with patch("murmur.service_manager.subprocess.Popen", return_value=process), patch(
        "murmur.service_manager._wait_for_port", return_value=True
    ), patch(
        "murmur.service_manager._is_pid_alive", return_value=True
    ), patch(
        "murmur.service_manager._is_port_reachable", return_value=True
    ), patch(
        "murmur.service_manager.create_status_indicator_provider",
        return_value=indicator_provider,
    ) as mock_create_indicator_provider:
        status = manager.start_background(host="localhost", port=7878, status_indicator=True)

    mock_create_indicator_provider.assert_called_once_with(
        host="localhost",
        port=7878,
        python_executable="/usr/bin/python3",
    )
    indicator_provider.start.assert_called_once()
    assert status.status_indicator_pid == 5678


def test_start_background_restarts_when_existing_state_target_differs(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    existing_state = ServiceState.new(
        pid=4444,
        host="localhost",
        port=7878,
        status_indicator_pid=5555,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(existing_state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path, python_executable="/usr/bin/python3")
    process = Mock()
    process.pid = 7777

    replacement_status = Mock()
    replacement_status.running = True
    replacement_status.pid = 7777
    replacement_status.host = "127.0.0.1"
    replacement_status.port = 9000

    with patch("murmur.service_manager.subprocess.Popen", return_value=process), patch(
        "murmur.service_manager._wait_for_port", return_value=True
    ), patch(
        "murmur.service_manager._is_pid_alive", return_value=True
    ), patch(
        "murmur.service_manager._terminate_pid"
    ) as mock_terminate, patch.object(
        manager, "status", return_value=replacement_status
    ):
        status = manager.start_background(host="127.0.0.1", port=9000, status_indicator=False)

    mock_terminate.assert_any_call(4444, is_expected_pid=ANY)
    mock_terminate.assert_any_call(5555, is_expected_pid=ANY)
    assert status.running is True
    assert status.pid == 7777
    assert status.host == "127.0.0.1"
    assert status.port == 9000


def test_start_background_recovers_missing_indicator_on_darwin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_path = tmp_path / "service.json"
    existing_state = ServiceState.new(
        pid=4444,
        host="localhost",
        port=7878,
        status_indicator_pid=5555,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(existing_state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path, python_executable="/usr/bin/python3")
    process = Mock()
    process.pid = 7777
    indicator_provider = Mock()
    indicator_provider.pid = 8888

    def _pid_alive(pid: int | None) -> bool:
        if pid == 4444:
            return True
        if pid == 5555:
            return False
        if pid == 7777:
            return True
        return False

    monkeypatch.setattr("murmur.service_manager.sys.platform", "darwin")
    with patch("murmur.service_manager.ensure_state_directory"), patch(
        "murmur.service_manager.subprocess.Popen",
        return_value=process,
    ), patch(
        "murmur.service_manager._wait_for_port",
        return_value=True,
    ), patch(
        "murmur.service_manager._is_pid_alive",
        side_effect=_pid_alive,
    ), patch(
        "murmur.service_manager._is_port_reachable",
        return_value=True,
    ), patch(
        "murmur.service_manager.create_status_indicator_provider",
        return_value=indicator_provider,
    ) as mock_create_indicator_provider, patch(
        "murmur.service_manager._terminate_pid"
    ) as mock_terminate:
        status = manager.start_background(host="localhost", port=7878, status_indicator=True)

    mock_terminate.assert_any_call(4444, is_expected_pid=ANY)
    mock_terminate.assert_any_call(5555, is_expected_pid=ANY)
    mock_create_indicator_provider.assert_called_once_with(
        host="localhost",
        port=7878,
        python_executable="/usr/bin/python3",
    )
    indicator_provider.start.assert_called_once()
    assert status.running is True
    assert status.pid == 7777
    assert status.host == "localhost"
    assert status.port == 7878
    assert status.status_indicator_pid == 8888


def test_start_background_non_darwin_does_not_restart_for_missing_indicator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "service.json"
    existing_state = ServiceState.new(
        pid=4444,
        host="localhost",
        port=7878,
        status_indicator_pid=None,
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(existing_state.to_dict())}\n", encoding="utf-8")
    manager = ServiceManager(state_path=state_path)
    existing_status = Mock(running=True, host="localhost", port=7878, pid=4444, status_indicator_pid=None)
    monkeypatch.setattr("murmur.service_manager.sys.platform", "linux")

    with patch(
        "murmur.service_manager._is_pid_alive",
        return_value=True,
    ), patch(
        "murmur.service_manager._is_port_reachable",
        return_value=True,
    ), patch.object(
        manager,
        "status",
        return_value=existing_status,
    ) as mock_status, patch(
        "murmur.service_manager.subprocess.Popen",
    ) as mock_popen, patch(
        "murmur.service_manager._terminate_pid"
    ) as mock_terminate:
        status = manager.start_background(host="localhost", port=7878, status_indicator=True)

    mock_status.assert_called_once()
    mock_popen.assert_not_called()
    mock_terminate.assert_not_called()
    assert status is existing_status


def test_start_background_indicator_start_failure_continues_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "service.json"
    manager = ServiceManager(state_path=state_path, python_executable="/usr/bin/python3")
    process = Mock()
    process.pid = 1234
    indicator_provider = Mock()
    indicator_provider.pid = 5678
    indicator_provider.start.side_effect = RuntimeError("indicator start failed")
    monkeypatch.setattr("murmur.service_manager.sys.platform", "darwin")

    with patch("murmur.service_manager.subprocess.Popen", return_value=process), patch(
        "murmur.service_manager._wait_for_port", return_value=True
    ), patch(
        "murmur.service_manager._is_pid_alive", return_value=True
    ), patch(
        "murmur.service_manager._is_port_reachable", return_value=True
    ), patch(
        "murmur.service_manager.create_status_indicator_provider",
        return_value=indicator_provider,
    ), patch("murmur.service_manager._terminate_pid") as mock_terminate:
        manager.start_background(host="localhost", port=7878, status_indicator=True)

    indicator_provider.stop.assert_called_once()
    mock_terminate.assert_not_called()
    assert state_path.exists() is True


def test_stop_is_idempotent_when_state_missing(tmp_path: Path) -> None:
    manager = ServiceManager(state_path=tmp_path / "service.json")

    status = manager.stop()

    assert status.running is False
    assert status.pid is None


def test_stop_terminates_service_and_indicator(tmp_path: Path) -> None:
    state_path = tmp_path / "service.json"
    manager = ServiceManager(state_path=state_path)
    state = ServiceState.new(pid=1234, host="localhost", port=7878, status_indicator_pid=5678)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(f"{json.dumps(state.to_dict())}\n", encoding="utf-8")

    with patch("murmur.service_manager._terminate_pid") as mock_terminate:
        manager.stop()

    assert mock_terminate.call_count == 2
    mock_terminate.assert_any_call(1234, is_expected_pid=ANY)
    mock_terminate.assert_any_call(5678, timeout=1.5, is_expected_pid=ANY)
    assert state_path.exists() is False


def test_ensure_running_starts_service_when_not_running(tmp_path: Path) -> None:
    manager = ServiceManager(state_path=tmp_path / "service.json")
    with patch.object(manager, "status") as mock_status, patch.object(
        manager, "start_background"
    ) as mock_start:
        mock_status.return_value = Mock(running=False)
        mock_start.return_value = Mock(running=True)

        status = manager.ensure_running(host="localhost", port=7878, status_indicator=True)

    mock_start.assert_called_once_with(host="localhost", port=7878, status_indicator=True)
    assert status.running is True


def test_ensure_running_returns_current_when_target_matches(tmp_path: Path) -> None:
    manager = ServiceManager(state_path=tmp_path / "service.json")
    current = Mock(running=True, host="localhost", port=7878)

    with patch.object(manager, "status", return_value=current), patch.object(
        manager,
        "start_background",
    ) as mock_start:
        status = manager.ensure_running(host="localhost", port=7878, status_indicator=False)

    mock_start.assert_not_called()
    assert status is current


def test_ensure_running_matching_target_with_status_indicator_uses_start_background(tmp_path: Path) -> None:
    manager = ServiceManager(state_path=tmp_path / "service.json")
    current = Mock(running=True, host="localhost", port=7878)
    replacement = Mock(running=True, host="localhost", port=7878)

    with patch.object(manager, "status", return_value=current), patch.object(
        manager,
        "start_background",
        return_value=replacement,
    ) as mock_start:
        status = manager.ensure_running(host="localhost", port=7878, status_indicator=True)

    mock_start.assert_called_once_with(host="localhost", port=7878, status_indicator=True)
    assert status is replacement


def test_ensure_running_restarts_when_running_target_differs(tmp_path: Path) -> None:
    manager = ServiceManager(state_path=tmp_path / "service.json")
    current = Mock(running=True, host="localhost", port=7878)
    replacement = Mock(running=True, host="127.0.0.1", port=9000)

    with patch.object(manager, "status", return_value=current), patch.object(
        manager,
        "start_background",
        return_value=replacement,
    ) as mock_start:
        status = manager.ensure_running(host="127.0.0.1", port=9000, status_indicator=False)

    mock_start.assert_called_once_with(host="127.0.0.1", port=9000, status_indicator=False)
    assert status is replacement


# ---------------------------------------------------------------------------
# _argv_contains_sequence
# ---------------------------------------------------------------------------

def test_argv_contains_sequence_found():
    assert service_manager._argv_contains_sequence(
        ("python", "-m", "murmur.cli", "bridge"),
        ("-m", "murmur.cli", "bridge"),
    ) is True


def test_argv_contains_sequence_not_found():
    assert service_manager._argv_contains_sequence(
        ("python", "-m", "other.module"),
        ("-m", "murmur.cli", "bridge"),
    ) is False


def test_argv_contains_sequence_empty():
    assert service_manager._argv_contains_sequence(("a", "b"), ()) is True


def test_argv_contains_sequence_too_short():
    assert service_manager._argv_contains_sequence(("a",), ("a", "b")) is False


# ---------------------------------------------------------------------------
# _argv_contains_option_value
# ---------------------------------------------------------------------------

def test_argv_contains_option_value_found():
    assert service_manager._argv_contains_option_value(
        ("--host", "localhost", "--port", "7878"),
        "--port",
        "7878",
    ) is True


def test_argv_contains_option_value_not_found():
    assert service_manager._argv_contains_option_value(
        ("--host", "localhost"),
        "--port",
        "7878",
    ) is False


# ---------------------------------------------------------------------------
# _pid_matches_bridge_process / _pid_matches_status_indicator_process
# ---------------------------------------------------------------------------

def test_pid_matches_bridge_process_true():
    argv = ("python", "-m", "murmur.cli", "bridge", "--host", "localhost", "--port", "7878")
    with patch("murmur.service_manager._process_argv", return_value=argv):
        assert service_manager._pid_matches_bridge_process(1234, host="localhost", port=7878) is True


def test_pid_matches_bridge_process_wrong_port():
    argv = ("python", "-m", "murmur.cli", "bridge", "--host", "localhost", "--port", "9999")
    with patch("murmur.service_manager._process_argv", return_value=argv):
        assert service_manager._pid_matches_bridge_process(1234, host="localhost", port=7878) is False


def test_pid_matches_bridge_process_no_argv():
    with patch("murmur.service_manager._process_argv", return_value=None):
        assert service_manager._pid_matches_bridge_process(1234, host="localhost", port=7878) is True


def test_pid_matches_status_indicator_process_true():
    argv = ("python", "-m", "murmur.status_indicator", "--host", "localhost", "--port", "7878")
    with patch("murmur.service_manager._process_argv", return_value=argv):
        assert service_manager._pid_matches_status_indicator_process(1234, host="localhost", port=7878) is True


def test_pid_matches_status_indicator_process_no_argv():
    with patch("murmur.service_manager._process_argv", return_value=None):
        assert service_manager._pid_matches_status_indicator_process(1234, host="localhost", port=7878) is True


# ---------------------------------------------------------------------------
# _is_pid_alive edge cases
# ---------------------------------------------------------------------------

def test_is_pid_alive_zero():
    assert service_manager._is_pid_alive(0) is False


def test_is_pid_alive_none():
    assert service_manager._is_pid_alive(None) is False


def test_is_pid_alive_no_error():
    with patch("murmur.service_manager.os.kill"):
        assert service_manager._is_pid_alive(1234) is True


def test_is_pid_alive_esrch():
    with patch("murmur.service_manager.os.kill", side_effect=OSError(errno.ESRCH, "no process")):
        assert service_manager._is_pid_alive(1234) is False


# ---------------------------------------------------------------------------
# _is_port_reachable
# ---------------------------------------------------------------------------

def test_is_port_reachable_true():
    mock_conn = Mock()
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=False)
    with patch("murmur.service_manager.socket.create_connection", return_value=mock_conn):
        assert service_manager._is_port_reachable("localhost", 7878) is True


def test_is_port_reachable_false():
    with patch("murmur.service_manager.socket.create_connection", side_effect=OSError):
        assert service_manager._is_port_reachable("localhost", 7878) is False


# ---------------------------------------------------------------------------
# _wait_for_port
# ---------------------------------------------------------------------------

def test_wait_for_port_immediate():
    with patch("murmur.service_manager._is_port_reachable", return_value=True):
        assert service_manager._wait_for_port("localhost", 7878, timeout=0.1) is True


def test_wait_for_port_timeout():
    with patch("murmur.service_manager._is_port_reachable", return_value=False):
        assert service_manager._wait_for_port("localhost", 7878, timeout=0.1) is False


# ---------------------------------------------------------------------------
# _process_argv
# ---------------------------------------------------------------------------

def test_process_argv_from_proc(tmp_path: Path):
    proc_cmdline = tmp_path / "cmdline"
    proc_cmdline.write_bytes(b"python\x00-m\x00murmur.cli\x00bridge\x00")
    with patch("murmur.service_manager.Path") as MockPath:
        mock_path = Mock()
        mock_path.read_bytes.return_value = proc_cmdline.read_bytes()
        MockPath.return_value = mock_path
        result = service_manager._process_argv(1234)
    assert result == ("python", "-m", "murmur.cli", "bridge")


def test_process_argv_from_ps():
    with patch("murmur.service_manager.Path") as MockPath:
        mock_path = Mock()
        mock_path.read_bytes.return_value = b""
        MockPath.return_value = mock_path

        mock_result = Mock(returncode=0, stdout="python -m murmur.cli bridge\n")
        with patch("murmur.service_manager.subprocess.run", return_value=mock_result):
            result = service_manager._process_argv(1234)
    assert result is not None
    assert "python" in result


def test_process_argv_ps_fails():
    with patch("murmur.service_manager.Path") as MockPath:
        mock_path = Mock()
        mock_path.read_bytes.return_value = b""
        MockPath.return_value = mock_path

        with patch("murmur.service_manager.subprocess.run", side_effect=Exception("fail")):
            result = service_manager._process_argv(1234)
    assert result is None


def test_process_argv_ps_nonzero():
    with patch("murmur.service_manager.Path") as MockPath:
        mock_path = Mock()
        mock_path.read_bytes.return_value = b""
        MockPath.return_value = mock_path

        mock_result = Mock(returncode=1, stdout="")
        with patch("murmur.service_manager.subprocess.run", return_value=mock_result):
            result = service_manager._process_argv(1234)
    assert result is None


def test_process_argv_ps_empty_output():
    with patch("murmur.service_manager.Path") as MockPath:
        mock_path = Mock()
        mock_path.read_bytes.return_value = b""
        MockPath.return_value = mock_path

        mock_result = Mock(returncode=0, stdout="")
        with patch("murmur.service_manager.subprocess.run", return_value=mock_result):
            result = service_manager._process_argv(1234)
    assert result is None


def test_process_argv_windows_uses_psutil() -> None:
    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            assert pid == 1234

        def cmdline(self) -> list[str]:
            return ["python", "-m", "murmur.cli", "bridge"]

    class _FakePsutil:
        class AccessDenied(Exception):
            pass

        class NoSuchProcess(Exception):
            pass

        Process = _FakeProcess

    with patch.object(service_manager.sys, "platform", "win32"), patch(
        "murmur.service_manager.Path"
    ) as MockPath, patch.dict(sys.modules, {"psutil": _FakePsutil}):
        mock_path = Mock()
        mock_path.read_bytes.return_value = b""
        MockPath.return_value = mock_path
        result = service_manager._process_argv(1234)

    assert result == ("python", "-m", "murmur.cli", "bridge")


# ---------------------------------------------------------------------------
# _terminate_pid edge cases
# ---------------------------------------------------------------------------

def test_terminate_pid_none():
    service_manager._terminate_pid(None)  # should return immediately


def test_terminate_pid_zero():
    service_manager._terminate_pid(0)  # should return immediately


def test_terminate_pid_unexpected_pid():
    with patch("murmur.service_manager._is_pid_alive", return_value=True):
        service_manager._terminate_pid(1234, is_expected_pid=lambda pid: False)
        # Should skip without sending signal


def test_terminate_pid_dead():
    with patch("murmur.service_manager._is_pid_alive", return_value=False):
        service_manager._terminate_pid(1234)
        # Should return early


def test_terminate_pid_sigterm_oserror():
    call_count = [0]
    def fake_is_alive(pid):
        call_count[0] += 1
        return call_count[0] <= 1  # alive first, then dead
    with patch("murmur.service_manager._is_pid_alive", side_effect=fake_is_alive), \
         patch("murmur.service_manager.os.kill", side_effect=OSError("perm")):
        service_manager._terminate_pid(1234)


# ---------------------------------------------------------------------------
# ServiceManager.load_state / save_state
# ---------------------------------------------------------------------------

def test_load_state_missing(tmp_path: Path):
    mgr = ServiceManager(state_path=tmp_path / "nonexistent.json")
    assert mgr.load_state() is None


def test_load_state_invalid_json(tmp_path: Path):
    state_path = tmp_path / "service.json"
    state_path.write_text("not json", encoding="utf-8")
    mgr = ServiceManager(state_path=state_path)
    assert mgr.load_state() is None


def test_save_and_load_state(tmp_path: Path):
    state_path = tmp_path / "service.json"
    mgr = ServiceManager(state_path=state_path)
    state = ServiceState.new(pid=1234, host="localhost", port=7878)
    mgr.save_state(state)
    loaded = mgr.load_state()
    assert loaded is not None
    assert loaded.host == "localhost"
    assert loaded.port == 7878
    assert loaded.pid == 1234


# ---------------------------------------------------------------------------
# service_state path helpers
# ---------------------------------------------------------------------------

def test_service_state_path():
    p = service_state_path()
    assert str(p).endswith("service.json")


def test_service_log_path():
    p = service_log_path()
    assert str(p).endswith("service.log")


def test_transcript_db_path():
    p = transcript_db_path()
    assert str(p).endswith("transcripts.sqlite3")
