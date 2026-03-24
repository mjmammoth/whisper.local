from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING, Union, cast

from murmur import __version__
from murmur.config import SUPPORTED_RUNTIMES, load_config
from murmur.console import get_console, init_console
from murmur.platform import create_status_indicator_provider
from murmur.service_manager import ServiceManager
from murmur.tui_runtime import resolve_tui_runtime

if TYPE_CHECKING:
    from murmur.service_state import ServiceStatus
    from websockets.asyncio.client import ClientConnection
    from websockets.legacy.client import WebSocketClientProtocol

    WebSocketClientType = Union[ClientConnection, WebSocketClientProtocol]
else:
    WebSocketClientType = Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BRIDGE_HOST_HELP = "Bridge host"
_BRIDGE_PORT_HELP = "Bridge port"
NO_STATUS_INDICATOR_AUTOSTART_HELP = (
    "Disable macOS menu bar status indicator while auto-starting service"
)
STATUS_SNAPSHOT_TIMEOUT_SECONDS = 1.5
FIRST_RUN_SETUP_MESSAGE = "First run setup required. Download and select a model in Model Manager."
SETUP_GUIDANCE_MODEL = "small"
RUNNING_LOOP_STATUS_MESSAGE = (
    "Runtime snapshot unavailable while an event loop is active; run murmur status from a synchronous shell."
)


def _parser_formatter_class(
    base_formatter_class: type[argparse.HelpFormatter] | None,
) -> type[argparse.HelpFormatter]:
    """Return a formatter class that suppresses the subparser metavar row."""
    base_class = base_formatter_class or argparse.HelpFormatter

    class _ParserHelpFormatter(base_class):  # type: ignore[misc, valid-type]
        def _format_action(self, action: argparse.Action) -> str:
            if isinstance(action, argparse._SubParsersAction):
                return "".join(
                    self._format_action(subaction)
                    for subaction in self._iter_indented_subactions(action)
                )
            return cast(str, super()._format_action(action))

        def _rich_format_action(self, action: argparse.Action) -> Iterator[Any]:
            if isinstance(action, argparse._SubParsersAction):
                for subaction in self._iter_indented_subactions(action):
                    yield from self._rich_format_action(subaction)
                return
            yield from super()._rich_format_action(action)

    return _ParserHelpFormatter


def build_parser(*, formatter_class: type | None = None) -> argparse.ArgumentParser:
    parser_formatter_class = _parser_formatter_class(formatter_class)
    class _SubparserArgumentParser(argparse.ArgumentParser):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("formatter_class", parser_formatter_class)
            super().__init__(*args, **kwargs)

    common = _SubparserArgumentParser(add_help=False)
    common.add_argument(
        "--plain",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Force plain text output (no colors or formatting)",
    )

    kwargs: dict[str, Any] = {
        "prog": Path(sys.argv[0]).name or "murmur",
        "formatter_class": parser_formatter_class,
    }
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Force plain text output (no colors or formatting)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="<command>",
        parser_class=_SubparserArgumentParser,
    )

    bridge_parser = subparsers.add_parser(
        "bridge",
        parents=[common],
        help="Start only the WebSocket bridge server",
    )
    bridge_parser.add_argument("--host", default="localhost", help=_BRIDGE_HOST_HELP)
    bridge_parser.add_argument("--port", type=int, default=7878, help=_BRIDGE_PORT_HELP)
    bridge_parser.add_argument("--capture-logs", action="store_true", help=argparse.SUPPRESS)

    tui_parser = subparsers.add_parser(
        "tui",
        parents=[common],
        help="Attach the TypeScript TUI to the service (auto-starts service if needed)",
    )
    tui_parser.add_argument("--host", default="localhost", help=_BRIDGE_HOST_HELP)
    tui_parser.add_argument("--port", type=int, default=7878, help=_BRIDGE_PORT_HELP)
    tui_parser.add_argument(
        "--no-status-indicator",
        action="store_true",
        help=NO_STATUS_INDICATOR_AUTOSTART_HELP,
    )

    start_parser = subparsers.add_parser("start", parents=[common], help="Start service")
    start_parser.add_argument("--host", default="localhost", help=_BRIDGE_HOST_HELP)
    start_parser.add_argument("--port", type=int, default=7878, help=_BRIDGE_PORT_HELP)
    start_parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run service in foreground (blocks current terminal)",
    )
    start_parser.add_argument(
        "--no-status-indicator",
        action="store_true",
        help="Disable macOS menu bar status indicator",
    )

    subparsers.add_parser("stop", parents=[common], help="Stop service")
    status_parser = subparsers.add_parser("status", parents=[common], help="Show service status")
    status_parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Read-only health snapshot (no cleanup/restart side effects)",
    )

    trigger_parser = subparsers.add_parser(
        "trigger",
        parents=[common],
        help="Control recording without opening TUI",
    )
    trigger_parser.add_argument("action", choices=("start", "stop", "toggle"))
    trigger_parser.add_argument("--host", default="localhost", help=_BRIDGE_HOST_HELP)
    trigger_parser.add_argument("--port", type=int, default=7878, help=_BRIDGE_PORT_HELP)
    trigger_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3.0,
        help="Max time to wait for trigger status acknowledgement",
    )
    trigger_parser.add_argument(
        "--no-status-indicator",
        action="store_true",
        help=NO_STATUS_INDICATOR_AUTOSTART_HELP,
    )

    models_parser = subparsers.add_parser("models", parents=[common], help="Manage models")
    models_sub = models_parser.add_subparsers(
        dest="models_command",
        parser_class=_SubparserArgumentParser,
    )
    models_sub.add_parser("list", parents=[common], help="List available models")

    pull_parser = models_sub.add_parser("pull", parents=[common], help="Download a model")
    pull_parser.add_argument("name")
    pull_parser.add_argument(
        "--runtime",
        choices=SUPPORTED_RUNTIMES,
        default="faster-whisper",
        help="Model runtime variant to download",
    )

    remove_parser = models_sub.add_parser("remove", parents=[common], help="Remove a model")
    remove_parser.add_argument("name")
    remove_parser.add_argument(
        "--runtime",
        choices=SUPPORTED_RUNTIMES,
        default="faster-whisper",
        help="Model runtime variant to remove",
    )

    select_parser = models_sub.add_parser(
        "select",
        parents=[common],
        aliases=["set-default"],
        help="Select model",
    )
    select_parser.add_argument("name")

    config_parser = subparsers.add_parser("config", parents=[common], help="Show config")
    config_parser.add_argument("--path", type=Path)

    upgrade_parser = subparsers.add_parser(
        "upgrade",
        parents=[common],
        help="Upgrade murmur (auto-upgrade only for installer-managed installs)",
    )
    upgrade_parser.add_argument(
        "--version",
        help="Upgrade to a specific release tag (example: v0.2.0). Defaults to latest.",
    )

    uninstall_parser = subparsers.add_parser(
        "uninstall",
        parents=[common],
        help="Uninstall murmur (auto-uninstall only for installer-managed installs)",
    )
    uninstall_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    uninstall_parser.add_argument(
        "--remove-state",
        action="store_true",
        help="Remove ~/.local/state/murmur during uninstall",
    )
    uninstall_parser.add_argument(
        "--remove-config",
        action="store_true",
        help="Remove ~/.config/murmur during uninstall",
    )
    uninstall_parser.add_argument(
        "--remove-model-cache",
        action="store_true",
        help="Remove murmur model caches under ~/.cache/huggingface",
    )
    uninstall_parser.add_argument(
        "--all-data",
        action="store_true",
        help="Equivalent to --remove-state --remove-config --remove-model-cache",
    )

    subparsers.add_parser("version", parents=[common], help="Print installed murmur version")

    return parser


def _run_bridge(host: str, port: int, capture_logs: bool = False) -> None:
    from murmur.bridge import run_bridge

    config = load_config()
    run_bridge(config, host, port, capture_logs=capture_logs)


def _run_tui(host: str, port: int) -> subprocess.Popen[bytes]:
    runtime = resolve_tui_runtime(cli_file=__file__)
    logger.info("Starting TUI runtime mode=%s", runtime.mode)
    cmd = [*runtime.command, "--host", host, "--port", str(port)]
    return subprocess.Popen(cmd, cwd=str(runtime.cwd) if runtime.cwd else None)


def _restore_terminal_state() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return

    try:
        sys.stdout.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l\x1b[?25h\x1b[0m")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        subprocess.run(["stty", "sane"], check=False)
    except Exception:
        pass


def _ensure_service_running(host: str, port: int, *, status_indicator: bool) -> ServiceStatus:
    manager = ServiceManager()
    return manager.ensure_running(host=host, port=port, status_indicator=status_indicator)


def _run_tui_attach(host: str, port: int, *, status_indicator: bool) -> None:
    console = get_console()
    try:
        service_status = _ensure_service_running(host, port, status_indicator=status_indicator)
    except Exception as exc:
        console.print_error(f"failed to start service: {exc}")
        raise SystemExit(1)

    resolved_host = service_status.host or host
    resolved_port = service_status.port if service_status.port is not None else port

    try:
        tui_process = _run_tui(resolved_host, resolved_port)
        tui_process.wait()
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as exc:
        console.print_error(str(exc))
        raise SystemExit(1)
    finally:
        _restore_terminal_state()


def _service_run(host: str, port: int, *, foreground: bool, status_indicator: bool) -> None:
    if foreground:
        indicator_provider = None
        indicator_started = False
        if status_indicator:
            indicator_provider = create_status_indicator_provider(host=host, port=port)
            try:
                indicator_provider.start()
                indicator_started = True
            except Exception:
                logger.warning("Failed to start status indicator", exc_info=True)
        try:
            _run_bridge(host, port, capture_logs=True)
        finally:
            if indicator_provider is not None and indicator_started:
                try:
                    indicator_provider.stop()
                except Exception:
                    pass
        return

    manager = ServiceManager()
    status = manager.start_background(host=host, port=port, status_indicator=status_indicator)
    console = get_console()
    if status.running:
        console.print_success(f"Service running pid={status.pid} host={status.host} port={status.port}")
    elif status.stale:
        console.print_warning("Service state was stale and has been cleaned up")
    else:
        console.print("Service start requested")


def _service_stop() -> None:
    manager = ServiceManager()
    before = manager.load_state()
    manager.stop()
    console = get_console()
    if before is None:
        console.print_warning("Service is not running")
    else:
        console.print_success("Service stopped")


def _service_status(*, diagnose: bool = False) -> None:
    manager = ServiceManager()
    status = manager.status(
        cleanup_stale=not diagnose,
        auto_recover_unreachable=not diagnose,
        status_indicator=True,
    )
    console = get_console()

    if status.running:
        host = status.host or "localhost"
        port = status.port if status.port is not None else 7878

        # In plain mode, print the running line first (tests expect this before snapshot)
        if not console.is_rich:
            indicator = (
                f" indicator_pid={status.status_indicator_pid}"
                if status.status_indicator_pid is not None
                else ""
            )
            print(f"running pid={status.pid} host={status.host} port={status.port}{indicator}")

        snapshot = None
        try:
            loop_running = False
            try:
                asyncio.get_running_loop()
                loop_running = True
            except RuntimeError:
                pass
            if loop_running:
                raise RuntimeError(RUNNING_LOOP_STATUS_MESSAGE)
            snapshot = asyncio.run(
                _runtime_status_snapshot(
                    host,
                    port,
                    kickoff_onboarding=not diagnose,
                    timeout_seconds=STATUS_SNAPSHOT_TIMEOUT_SECONDS,
                )
            )
        except Exception as exc:
            if console.is_rich:
                console.print_service_status(
                    running=True, pid=status.pid, host=status.host,
                    port=status.port, indicator_pid=status.status_indicator_pid,
                )
                console.print_error(f"Unable to query runtime state: {exc}")
            else:
                console.print_runtime_error_plain(exc)
            return

        if console.is_rich:
            console.print_service_status(
                running=True, pid=status.pid, host=status.host,
                port=status.port, indicator_pid=status.status_indicator_pid,
                snapshot=snapshot,
            )
        else:
            _print_runtime_status_snapshot(snapshot)
        return

    if status.stale:
        if diagnose:
            stale_reason = status.stale_reason or "unknown"
            pid_alive_text = str(bool(status.pid_alive)).lower()
            reachable_text = str(bool(status.reachable)).lower()
            if console.is_rich:
                console.print_warning(
                    "Stale service state detected (diagnose mode, no cleanup): "
                    f"pid={status.pid} host={status.host} port={status.port} reason={stale_reason}"
                )
                console.print(
                    f"  pid_alive={pid_alive_text} reachable={reachable_text} "
                    "cleanup=false auto_recover=false"
                )
            else:
                print(
                    f"stale (diagnose) pid={status.pid} host={status.host} "
                    f"port={status.port} reason={stale_reason}"
                )
                print(
                    f"diagnose pid_alive={pid_alive_text} reachable={reachable_text} "
                    "cleanup=false auto_recover=false"
                )
            return
        console.print_stale_status(pid=status.pid, host=status.host, port=status.port)
        return

    console.print_service_status(running=False)


async def _wait_for_status(
    websocket: WebSocketClientType,
    *,
    timeout_seconds: float,
    expected_statuses: set[str] | None = None,
) -> tuple[str | None, str | None]:
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    last_status: str | None = None
    last_message: str | None = None

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return last_status, last_message
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        except TimeoutError:
            return last_status, last_message

        status_update = _extract_status_update(raw)
        if status_update is None:
            continue

        last_status, last_message = status_update
        if expected_statuses is None or last_status in expected_statuses:
            return last_status, last_message


def _extract_status_update(raw: str | bytes) -> tuple[str | None, str | None] | None:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "status":
        return None

    status = str(payload.get("status", ""))
    last_status = status or None
    status_message = payload.get("message")
    last_message = status_message if isinstance(status_message, str) else None
    return last_status, last_message


def _extract_config_update(raw: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "config":
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    return config


async def _collect_runtime_snapshot(
    websocket: WebSocketClientType,
    *,
    timeout_seconds: float,
    initial_status: str | None = None,
    initial_message: str | None = None,
    initial_config: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    status = initial_status
    message = initial_message
    config = initial_config

    while True:
        if status is not None and config is not None:
            return status, message, config
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return status, message, config
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        except TimeoutError:
            return status, message, config

        status_update = _extract_status_update(raw)
        if status_update is not None:
            status, message = status_update
            continue

        config_update = _extract_config_update(raw)
        if config_update is not None:
            config = config_update


def _startup_phase_from_config(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return ""
    startup = config.get("startup")
    if not isinstance(startup, dict):
        return ""
    value = startup.get("phase")
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _first_run_pending(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(config.get("first_run_setup_required"))


async def _maybe_kickoff_onboarding(
    websocket: WebSocketClientType,
    config: dict[str, Any] | None,
    status: str | None,
    message: str | None,
    timeout_seconds: float,
    snapshot_started: float,
) -> tuple[str | None, str | None, dict[str, Any] | None, bool]:
    """Send onboarding kickoff if first-run is pending and phase is idle, then re-collect."""
    if not _first_run_pending(config):
        return status, message, config, False
    phase = _startup_phase_from_config(config)
    if phase not in {"", "idle"}:
        return status, message, config, False

    await websocket.send(json.dumps({"type": "begin_onboarding_setup"}))
    remaining_timeout = max(0.0, timeout_seconds - (time.monotonic() - snapshot_started))
    if remaining_timeout > 0:
        status, message, config = await _collect_runtime_snapshot(
            websocket,
            timeout_seconds=remaining_timeout,
        )
    return status, message, config, True


async def _runtime_status_snapshot(
    host: str,
    port: int,
    *,
    kickoff_onboarding: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    import websockets

    uri = f"ws://{host}:{port}?client=passive"
    async with websockets.connect(uri, ping_interval=10, ping_timeout=10) as websocket:
        snapshot_started = time.monotonic()
        status, message, config = await _collect_runtime_snapshot(
            websocket,
            timeout_seconds=timeout_seconds,
        )

        kickoff_sent = False
        if kickoff_onboarding:
            status, message, config, kickoff_sent = await _maybe_kickoff_onboarding(
                websocket, config, status, message, timeout_seconds, snapshot_started,
            )

        return {
            "status": status,
            "message": message,
            "config": config,
            "kickoff_sent": kickoff_sent,
        }


def _parse_startup_detail(config: dict[str, Any]) -> tuple[dict[str, Any], str, list[str], bool]:
    """Extract startup sub-fields from a config dict."""
    startup = config.get("startup")
    startup_dict = startup if isinstance(startup, dict) else {}
    phase = _startup_phase_from_config(config) or "unknown"
    blockers = startup_dict.get("blockers")
    blocker_list = [str(item) for item in blockers] if isinstance(blockers, list) else []
    close_ready = bool(startup_dict.get("onboarding_close_ready"))
    return startup_dict, phase, blocker_list, close_ready


def _print_startup_summary(startup_dict: dict[str, Any], phase: str) -> None:
    """Print the startup component summary line."""
    runtime_probe = str(startup_dict.get("runtime_probe", "unknown"))
    audio_scan = str(startup_dict.get("audio_scan", "unknown"))
    components = str(startup_dict.get("components", "unknown"))
    model_state = str(startup_dict.get("model", "unknown"))
    print(
        "startup "
        f"phase={phase} runtime_probe={runtime_probe} "
        f"audio_scan={audio_scan} components={components} model={model_state}"
    )


def _print_first_run_guidance(kickoff_sent: bool) -> None:
    """Print first-run setup guidance to stdout."""
    if kickoff_sent:
        print("setup_init=started_via_status")
    print(f"setup_required=true message={json.dumps(FIRST_RUN_SETUP_MESSAGE, ensure_ascii=True)}")
    print("next_steps:")
    print(f"  murmur models pull {SETUP_GUIDANCE_MODEL}")
    print(f"  murmur models select {SETUP_GUIDANCE_MODEL}")
    print("  murmur status")


def _print_hotkey_diagnostics(config: dict[str, Any]) -> None:
    diagnostics = config.get("hotkey_diagnostics")
    if not isinstance(diagnostics, dict):
        return

    backend = str(diagnostics.get("backend", "unknown"))
    started = bool(diagnostics.get("started", False))
    blocked = bool(diagnostics.get("blocked", False))
    thread_alive = bool(diagnostics.get("thread_alive", False))
    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    tap_disable_count = _as_int(diagnostics.get("tap_disable_count", 0))
    tap_reenable_count = _as_int(diagnostics.get("tap_reenable_count", 0))
    press_count = _as_int(diagnostics.get("press_count", 0))
    release_count = _as_int(diagnostics.get("release_count", 0))

    print(
        "hotkey "
        f"backend={backend} started={str(started).lower()} blocked={str(blocked).lower()} "
        f"thread_alive={str(thread_alive).lower()} press_count={press_count} release_count={release_count} "
        f"tap_disables={tap_disable_count} tap_reenables={tap_reenable_count}"
    )


def _print_runtime_status_snapshot(snapshot: dict[str, Any]) -> None:
    status = snapshot.get("status")
    message = snapshot.get("message")
    config = snapshot.get("config")
    kickoff_sent = bool(snapshot.get("kickoff_sent"))

    status_text = status if isinstance(status, str) and status else "unknown"
    message_text = message if isinstance(message, str) and message else "unknown"
    print(f"app_status={status_text} message={json.dumps(message_text, ensure_ascii=True)}")

    if not isinstance(config, dict):
        print("runtime_detail=unavailable")
        return

    _print_hotkey_diagnostics(config)

    first_run = _first_run_pending(config)
    startup_dict, phase, blocker_list, close_ready = _parse_startup_detail(config)

    if not first_run and status_text == "ready" and close_ready and not blocker_list:
        print("app_ready=true")
        return

    _print_startup_summary(startup_dict, phase)

    if blocker_list:
        print("startup_blockers:")
        for blocker in blocker_list:
            print(f"  - {blocker}")

    if first_run:
        _print_first_run_guidance(kickoff_sent)


async def _trigger_async(host: str, port: int, action: str, timeout_seconds: float) -> str:
    import websockets

    uri = f"ws://{host}:{port}"
    async with websockets.connect(uri, ping_interval=10, ping_timeout=10) as websocket:
        initial_status, _ = await _wait_for_status(
            websocket,
            timeout_seconds=min(max(timeout_seconds, 0.1), 0.75),
        )
        current_status = initial_status or ""

        effective = action
        if action == "toggle":
            effective = "stop" if current_status == "recording" else "start"

        if effective == "start":
            expected_statuses = {"recording", "connecting"}
            should_return_without_send = current_status == "recording"
        else:
            expected_statuses = {"transcribing", "ready"}
            should_return_without_send = current_status in {"transcribing", "ready"}

        if should_return_without_send:
            return current_status

        message_type = "start_recording" if effective == "start" else "stop_recording"
        await websocket.send(json.dumps({"type": message_type}))

        ack_status, ack_message = await _wait_for_status(
            websocket,
            timeout_seconds=timeout_seconds,
            expected_statuses=expected_statuses,
        )
        if ack_status in expected_statuses:
            return str(ack_status)

        last_status = ack_status or current_status or "unknown"
        last_message = ack_message or "no status message"
        raise TimeoutError(
            f"Timed out waiting for trigger acknowledgement ({effective}); "
            f"last_status={last_status}; last_message={last_message}"
        )


def _trigger(
    host: str,
    port: int,
    *,
    action: str,
    status_indicator: bool,
    timeout_seconds: float,
) -> None:
    console = get_console()
    try:
        service_status = _ensure_service_running(host, port, status_indicator=status_indicator)
    except Exception as exc:
        console.print_error(f"failed to start service: {exc}")
        raise SystemExit(1)

    resolved_host = service_status.host or host
    resolved_port = service_status.port if service_status.port is not None else port

    try:
        ack_status = asyncio.run(_trigger_async(resolved_host, resolved_port, action, timeout_seconds))
        console.print_success(f"Trigger acknowledged: status={ack_status}")
    except TimeoutError as exc:
        console.print_error(f"trigger command timed out: {exc}")
        raise SystemExit(2)
    except Exception as exc:
        console.print_error(f"trigger command failed: {exc}")
        raise SystemExit(1)


def _upgrade(*, requested_version: str | None) -> None:
    from murmur.upgrade import UpgradeActionRequired, UpgradeError, run_upgrade

    console = get_console()
    try:
        result = run_upgrade(requested_version=requested_version)
    except UpgradeActionRequired as exc:
        console.print(str(exc))
        raise SystemExit(2)
    except UpgradeError as exc:
        console.print_error(str(exc))
        raise SystemExit(1)

    console.print_success(
        f"Upgraded murmur {result.previous_version} -> {result.new_version} "
        f"({result.tag})"
    )
    if result.restarted_service:
        console.print("Service was running and has been restarted.")


def _resolve_uninstall_scope(args: argparse.Namespace) -> tuple[bool, bool, bool, bool]:
    remove_state = bool(getattr(args, "remove_state", False))
    remove_config = bool(getattr(args, "remove_config", False))
    remove_model_cache = bool(getattr(args, "remove_model_cache", False))
    if bool(getattr(args, "all_data", False)):
        remove_state = True
        remove_config = True
        remove_model_cache = True
    explicit_scope = any([remove_state, remove_config, remove_model_cache, bool(getattr(args, "all_data", False))])
    return remove_state, remove_config, remove_model_cache, explicit_scope


def _prompt_uninstall_scope() -> tuple[bool, bool, bool]:
    return get_console().prompt_uninstall_scope()


def _print_uninstall_plan(*, remove_state: bool, remove_config: bool, remove_model_cache: bool) -> None:
    get_console().print_uninstall_plan(
        remove_state=remove_state, remove_config=remove_config, remove_model_cache=remove_model_cache,
    )


def _confirm_uninstall() -> bool:
    return get_console().confirm_uninstall()


def _resolve_interactive_scope(
    remove_state: bool,
    remove_config: bool,
    remove_model_cache: bool,
    explicit_scope: bool,
    assume_yes: bool,
    tty_session: bool,
) -> tuple[bool, bool, bool]:
    """Resolve uninstall scope interactively if needed, raising SystemExit on cancellation."""
    if not assume_yes and not tty_session and not explicit_scope:
        print(
            "Error: non-interactive uninstall requires --yes or explicit scope flags "
            "(--remove-state/--remove-config/--remove-model-cache/--all-data).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if tty_session and not assume_yes:
        if not explicit_scope:
            remove_state, remove_config, remove_model_cache = _prompt_uninstall_scope()
        _print_uninstall_plan(
            remove_state=remove_state,
            remove_config=remove_config,
            remove_model_cache=remove_model_cache,
        )
        if not _confirm_uninstall():
            get_console().print_warning("Uninstall cancelled.")
            raise SystemExit(1)

    return remove_state, remove_config, remove_model_cache


def _print_uninstall_result(result: Any) -> None:
    """Print uninstall result details, raising SystemExit(1) on failures."""
    console = get_console()
    if result.removed_paths:
        console.print("Removed paths:")
        for path in result.removed_paths:
            console.print(f"  - {path}")
    else:
        console.print("No files were removed.")

    if result.warnings:
        console.print("Warnings:")
        for warning in result.warnings:
            console.print(f"  - {warning}")

    if result.failed_paths:
        console.print("Failed to remove:")
        for failure in result.failed_paths:
            console.print(f"  - {failure.path}: {failure.reason}")
        raise SystemExit(1)

    console.print_success("Uninstall complete.")


def _uninstall(args: argparse.Namespace) -> None:
    from murmur.uninstall import (
        UninstallActionRequired,
        UninstallError,
        UninstallOptions,
        run_uninstall,
    )

    remove_state, remove_config, remove_model_cache, explicit_scope = _resolve_uninstall_scope(args)
    tty_session = sys.stdin.isatty() and sys.stdout.isatty()
    assume_yes = bool(getattr(args, "yes", False))

    remove_state, remove_config, remove_model_cache = _resolve_interactive_scope(
        remove_state, remove_config, remove_model_cache,
        explicit_scope, assume_yes, tty_session,
    )

    options = UninstallOptions(
        remove_state=remove_state,
        remove_config=remove_config,
        remove_model_cache=remove_model_cache,
    )

    console = get_console()
    try:
        result = run_uninstall(options=options)
    except UninstallActionRequired as exc:
        console.print(str(exc))
        raise SystemExit(2)
    except UninstallError as exc:
        console.print_error(str(exc))
        raise SystemExit(1)

    _print_uninstall_result(result)


def _print_version() -> None:
    get_console().print_version(__version__)


def _handle_models_list() -> None:
    from murmur.model_manager import list_installed_models

    console = get_console()
    models = list_installed_models()
    # Determine selected model from config
    selected: str | None = None
    try:
        config = load_config()
        selected = config.model.name
    except Exception:
        pass
    console.print_model_list(models, selected=selected)


def _handle_models_pull(name: str, runtime: str) -> None:
    from murmur.model_manager import download_model

    console = get_console()
    if runtime == "faster-whisper":
        download_model(name)
        console.print_success(f"Downloaded {name}")
    else:
        download_model(name, runtime=runtime)
        console.print_success(f"Downloaded {name} ({runtime})")


def _handle_models_remove(name: str, runtime: str) -> None:
    from murmur.model_manager import remove_model

    console = get_console()
    if runtime == "faster-whisper":
        remove_model(name)
        console.print_success(f"Removed {name}")
    else:
        remove_model(name, runtime=runtime)
        console.print_success(f"Removed {name} ({runtime})")


def _handle_models_command(args: argparse.Namespace) -> None:
    from murmur.model_manager import set_selected_model

    if args.models_command is None:
        print(
            "Error: No subcommand provided for 'models'. Use --help for options.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.models_command == "list":
        _handle_models_list()
        return

    if args.models_command == "pull":
        _handle_models_pull(args.name, args.runtime)
        return

    if args.models_command == "remove":
        _handle_models_remove(args.name, args.runtime)
        return

    if args.models_command in ("select", "set-default"):
        set_selected_model(args.name)
        get_console().print_success(f"Selected model set to {args.name}")


def _handle_config_command(args: argparse.Namespace) -> None:
    config = load_config(args.path)
    get_console().print_config(config.to_dict())


def main() -> None:
    console = init_console(force_plain=("--plain" in sys.argv))
    formatter_class = console.get_help_formatter_class() if console.is_rich else None
    parser = build_parser(formatter_class=formatter_class)
    args = parser.parse_args()

    if args.command is None:
        if console.is_rich:
            console.print_logo()
        _service_status()
        print()
        parser.print_help()
        return

    if args.command == "start":
        _service_run(
            args.host,
            args.port,
            foreground=args.foreground,
            status_indicator=not args.no_status_indicator,
        )
        return

    if args.command == "stop":
        _service_stop()
        return

    if args.command == "status":
        _service_status(diagnose=bool(getattr(args, "diagnose", False)))
        return

    if args.command == "bridge":
        _run_bridge(args.host, args.port, capture_logs=bool(getattr(args, "capture_logs", False)))
        return

    if args.command == "tui":
        _run_tui_attach(args.host, args.port, status_indicator=not args.no_status_indicator)
        return

    if args.command == "trigger":
        _trigger(
            args.host,
            args.port,
            action=args.action,
            status_indicator=not args.no_status_indicator,
            timeout_seconds=float(args.timeout_seconds),
        )
        return

    if args.command == "models":
        _handle_models_command(args)
        return

    if args.command == "config":
        _handle_config_command(args)
        return

    if args.command == "upgrade":
        _upgrade(requested_version=getattr(args, "version", None))
        return

    if args.command == "uninstall":
        _uninstall(args)
        return

    if args.command == "version":
        _print_version()
        return

    if console.is_rich:
        console.print_logo()
    parser.print_help()


if __name__ == "__main__":
    main()
