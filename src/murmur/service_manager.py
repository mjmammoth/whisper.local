from __future__ import annotations

import errno
import functools
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from murmur.platform import create_status_indicator_provider
from murmur.service_state import (
    ServiceState,
    ServiceStatus,
    ensure_state_directory,
    service_state_path,
)


SERVICE_READY_TIMEOUT_SECONDS = 8.0
SERVICE_READY_POLL_SECONDS = 0.1

logger = logging.getLogger(__name__)


def _process_argv_from_proc(pid: int) -> tuple[str, ...] | None:
    """Read process argv from /proc filesystem."""
    proc_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = proc_path.read_bytes()
    except Exception:
        raw = b""
    if raw:
        return tuple(part.decode(errors="replace") for part in raw.split(b"\x00") if part)
    return None


def _process_argv_from_psutil(pid: int) -> tuple[str, ...] | None:
    """Read process argv via psutil on Windows."""
    try:
        import psutil
    except Exception as exc:
        logger.warning("Failed to import psutil for pid cmdline lookup (pid=%s): %s", pid, exc)
        return None

    try:
        cmdline = psutil.Process(pid).cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
        logger.warning("Failed to read process argv via psutil (pid=%s): %s", pid, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error reading process argv via psutil (pid=%s): %s", pid, exc)
        return None

    argv = tuple(str(part) for part in cmdline if part)
    if not argv:
        logger.warning("Process argv lookup returned empty command line (pid=%s)", pid)
        return None
    return argv


def _process_argv_from_ps(pid: int) -> tuple[str, ...] | None:
    """Read process argv via the ps command on Unix."""
    if sys.platform == "darwin":
        command = ["/bin/ps", "-o", "args=", "-p", str(pid)]
    else:
        command = ["ps", "-o", "args=", "-p", str(pid)]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    raw_command = result.stdout.strip()
    if not raw_command:
        return None
    try:
        return tuple(shlex.split(raw_command))
    except ValueError:
        return tuple(raw_command.split())


def _process_argv(pid: int) -> tuple[str, ...] | None:
    result = _process_argv_from_proc(pid)
    if result is not None:
        return result
    if sys.platform.startswith("win"):
        return _process_argv_from_psutil(pid)
    return _process_argv_from_ps(pid)


def _argv_contains_sequence(argv: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    if not sequence:
        return True
    if len(argv) < len(sequence):
        return False
    for index in range(0, len(argv) - len(sequence) + 1):
        if argv[index : index + len(sequence)] == sequence:
            return True
    return False


def _argv_contains_option_value(argv: tuple[str, ...], option: str, expected: str) -> bool:
    for index in range(0, len(argv) - 1):
        if argv[index] == option and argv[index + 1] == expected:
            return True
    return False


def _pid_matches_bridge_process(pid: int, *, host: str, port: int) -> bool:
    argv = _process_argv(pid)
    if argv is None:
        logger.warning(
            "Could not inspect bridge argv for pid=%s; proceeding with cleanup signaling fallback",
            pid,
        )
        return True
    return (
        _argv_contains_sequence(argv, ("-m", "murmur.cli", "bridge"))
        and _argv_contains_option_value(argv, "--host", host)
        and _argv_contains_option_value(argv, "--port", str(port))
    )


def _pid_matches_status_indicator_process(pid: int, *, host: str, port: int) -> bool:
    argv = _process_argv(pid)
    if argv is None:
        logger.warning(
            "Could not inspect status indicator argv for pid=%s; proceeding with cleanup signaling fallback",
            pid,
        )
        return True
    return (
        _argv_contains_sequence(argv, ("-m", "murmur.status_indicator"))
        and _argv_contains_option_value(argv, "--host", host)
        and _argv_contains_option_value(argv, "--port", str(port))
    )


def _is_safe_pid(pid: int | None) -> bool:
    """Validate that *pid* is a positive integer safe to signal.

    Rejects ``None``, non-positive values (which ``os.kill`` interprets as
    process-group signals), and PID 1 (the init/launchd process) to prevent
    accidental system-wide impact.
    """
    return isinstance(pid, int) and pid > 1


def _is_pid_alive(pid: int | None) -> bool:
    if not _is_safe_pid(pid):
        return False
    assert pid is not None  # narrowing for type-checker after _is_safe_pid
    try:
        os.kill(pid, 0)  # Signal 0: existence check only, no signal delivered
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _is_port_reachable(host: str, port: int, *, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_reachable(host, port):
            return True
        time.sleep(SERVICE_READY_POLL_SECONDS)
    return _is_port_reachable(host, port)


def _terminate_pid(
    pid: int | None,
    *,
    timeout: float = 4.0,
    is_expected_pid: Callable[[int], bool] | None = None,
) -> None:
    if not _is_safe_pid(pid):
        return
    assert pid is not None  # narrowing for type-checker after _is_safe_pid
    if is_expected_pid is not None:
        try:
            if not is_expected_pid(pid):
                logger.warning("Skipping signal to unexpected pid=%s", pid)
                return
        except Exception:
            logger.warning(
                "Failed to validate expected pid=%s; proceeding with cleanup signaling fallback",
                pid,
                exc_info=True,
            )
    if not _is_pid_alive(pid):
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return
        time.sleep(0.1)

    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        os.kill(pid, kill_signal)
    except OSError:
        pass

    waitpid = getattr(os, "waitpid", None)
    wnohang = getattr(os, "WNOHANG", 0)
    if callable(waitpid):
        try:
            waitpid(pid, wnohang)
        except Exception:
            pass


class ServiceManager:
    def __init__(
        self,
        *,
        state_path: Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.state_path = (state_path or service_state_path()).expanduser()
        self.log_path = self.state_path.with_name("service.log")
        self.python_executable = python_executable or sys.executable

    def load_state(self) -> ServiceState | None:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return ServiceState.from_dict(payload)
        except Exception:
            return None

    def save_state(self, state: ServiceState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            f"{json.dumps(state.to_dict(), indent=2)}\n",
            encoding="utf-8",
        )

    def clear_state(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    def _cleanup_stale_state(self, state: ServiceState) -> None:
        """Terminate stale bridge and indicator processes, then clear persisted state."""
        if _is_pid_alive(state.pid):
            _terminate_pid(
                state.pid,
                is_expected_pid=functools.partial(
                    _pid_matches_bridge_process,
                    host=state.host,
                    port=state.port,
                ),
            )
        _terminate_pid(
            state.status_indicator_pid,
            timeout=1.5,
            is_expected_pid=functools.partial(
                _pid_matches_status_indicator_process,
                host=state.host,
                port=state.port,
            ),
        )
        self.clear_state()

    def _service_status_from_state(
        self,
        state: ServiceState,
        *,
        running: bool,
        stale: bool,
        reachable: bool,
        pid_alive: bool,
        stale_reason: str | None = None,
    ) -> ServiceStatus:
        return ServiceStatus(
            running=running,
            pid=state.pid,
            host=state.host,
            port=state.port,
            started_at=state.started_at,
            status_indicator_pid=state.status_indicator_pid,
            stale=stale,
            reachable=reachable,
            state_path=self.state_path,
            pid_alive=pid_alive,
            stale_reason=stale_reason,
        )

    def status(
        self,
        *,
        cleanup_stale: bool = True,
        auto_recover_unreachable: bool = False,
        status_indicator: bool = True,
    ) -> ServiceStatus:
        state = self.load_state()
        if state is None:
            return ServiceStatus(
                running=False,
                pid=None,
                host=None,
                port=None,
                started_at=None,
                status_indicator_pid=None,
                stale=False,
                reachable=False,
                state_path=self.state_path,
                pid_alive=False,
                stale_reason=None,
            )

        pid_alive = _is_pid_alive(state.pid)
        reachable = pid_alive and _is_port_reachable(state.host, state.port)

        if not reachable:
            stale_reason = "port_unreachable" if pid_alive else "pid_not_alive"

            if auto_recover_unreachable and stale_reason == "port_unreachable":
                logger.warning(
                    "Bridge became unreachable; attempting watchdog restart "
                    "(pid=%s host=%s port=%s)",
                    state.pid,
                    state.host,
                    state.port,
                )
                try:
                    return self.start_background(
                        host=state.host,
                        port=state.port,
                        status_indicator=status_indicator,
                    )
                except Exception:
                    logger.warning(
                        "Watchdog restart failed for unreachable bridge "
                        "(pid=%s host=%s port=%s)",
                        state.pid,
                        state.host,
                        state.port,
                        exc_info=True,
                    )

            if cleanup_stale:
                try:
                    self._cleanup_stale_state(state)
                except Exception:
                    logger.warning(
                        "Failed to clean stale service state (pid=%s host=%s port=%s)",
                        state.pid,
                        state.host,
                        state.port,
                        exc_info=True,
                    )
            return self._service_status_from_state(
                state,
                running=False,
                stale=True,
                reachable=False,
                pid_alive=pid_alive,
                stale_reason=stale_reason,
            )

        return self._service_status_from_state(
            state,
            running=True,
            stale=False,
            reachable=True,
            pid_alive=True,
            stale_reason=None,
        )

    def ensure_running(
        self,
        *,
        host: str,
        port: int,
        status_indicator: bool,
    ) -> ServiceStatus:
        current = self.status(
            cleanup_stale=True,
            auto_recover_unreachable=True,
            status_indicator=status_indicator,
        )
        if current.running and current.host == host and current.port == port and not status_indicator:
            return current
        return self.start_background(host=host, port=port, status_indicator=status_indicator)

    def _stop_existing_if_needed(
        self,
        host: str,
        port: int,
        status_indicator: bool,
    ) -> ServiceStatus | None:
        """Stop existing service if needed, or return current status if reuse is possible."""
        current_state = self.load_state()
        if not current_state:
            return None
        if not _is_pid_alive(current_state.pid):
            self._cleanup_stale_state(current_state)
            return None

        requested_target_matches = current_state.host == host and current_state.port == port
        if requested_target_matches and _is_port_reachable(current_state.host, current_state.port):
            indicator_degraded = (
                status_indicator
                and sys.platform == "darwin"
                and not _is_pid_alive(current_state.status_indicator_pid)
            )
            if not indicator_degraded:
                return self.status()
            logger.info(
                "Restarting service to recover missing/dead status indicator "
                "(bridge_pid=%s indicator_pid=%s host=%s port=%s)",
                current_state.pid,
                current_state.status_indicator_pid,
                current_state.host,
                current_state.port,
            )

        _terminate_pid(
            current_state.pid,
            is_expected_pid=functools.partial(
                _pid_matches_bridge_process,
                host=current_state.host,
                port=current_state.port,
            ),
        )
        _terminate_pid(
            current_state.status_indicator_pid,
            is_expected_pid=functools.partial(
                _pid_matches_status_indicator_process,
                host=current_state.host,
                port=current_state.port,
            ),
        )
        self.clear_state()
        return None

    def start_background(
        self,
        *,
        host: str = "localhost",
        port: int = 7878,
        status_indicator: bool = True,
    ) -> ServiceStatus:
        reuse_status = self._stop_existing_if_needed(host, port, status_indicator)
        if reuse_status is not None:
            return reuse_status

        ensure_state_directory()
        command = [
            self.python_executable,
            "-m",
            "murmur.cli",
            "bridge",
            "--host",
            host,
            "--port",
            str(port),
            "--capture-logs",
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
            )

        if not _wait_for_port(host, port, timeout=SERVICE_READY_TIMEOUT_SECONDS):
            _terminate_pid(
                process.pid,
                timeout=1.0,
                is_expected_pid=functools.partial(
                    _pid_matches_bridge_process,
                    host=host,
                    port=port,
                ),
            )
            raise RuntimeError(f"Service failed to start on {host}:{port}")

        indicator_pid: int | None = None
        if status_indicator:
            indicator_provider = create_status_indicator_provider(
                host=host,
                port=port,
                python_executable=self.python_executable,
            )
            try:
                indicator_provider.start()
                indicator_pid = indicator_provider.pid
            except Exception:
                logger.exception("Failed to start status indicator provider")
                indicator_pid = None
                try:
                    indicator_provider.stop()
                except Exception:
                    logger.exception("Failed to stop status indicator provider after start failure")

        try:
            self.save_state(
                ServiceState.new(
                    pid=process.pid,
                    host=host,
                    port=port,
                    status_indicator_pid=indicator_pid,
                )
            )
        except Exception:
            _terminate_pid(
                process.pid,
                timeout=1.0,
                is_expected_pid=functools.partial(
                    _pid_matches_bridge_process,
                    host=host,
                    port=port,
                ),
            )
            _terminate_pid(
                indicator_pid,
                timeout=1.0,
                is_expected_pid=functools.partial(
                    _pid_matches_status_indicator_process,
                    host=host,
                    port=port,
                ),
            )
            self.clear_state()
            raise
        return self.status()

    def stop(self) -> ServiceStatus:
        state = self.load_state()
        if state is None:
            return self.status()

        _terminate_pid(
            state.pid,
            is_expected_pid=functools.partial(
                _pid_matches_bridge_process,
                host=state.host,
                port=state.port,
            ),
        )
        _terminate_pid(
            state.status_indicator_pid,
            timeout=1.5,
            is_expected_pid=functools.partial(
                _pid_matches_status_indicator_process,
                host=state.host,
                port=state.port,
            ),
        )
        self.clear_state()
        return self.status()
