from __future__ import annotations

import argparse
import asyncio
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from murmur import cli
from murmur.uninstall import (
    RemovalFailure,
    UninstallActionRequired,
    UninstallError,
    UninstallOptions,
    UninstallResult,
)
from murmur.upgrade import UpgradeActionRequired, UpgradeError, UpgradeResult


def test_build_parser_includes_start_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["start"])
    assert args.command == "start"


def test_build_parser_includes_trigger_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["trigger", "toggle"])
    assert args.command == "trigger"
    assert args.action == "toggle"
    assert args.timeout_seconds == 3.0


def test_build_parser_includes_upgrade_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["upgrade", "--version", "v1.2.3"])
    assert args.command == "upgrade"
    assert args.version == "v1.2.3"


def test_build_parser_includes_uninstall_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall", "--yes", "--all-data"])
    assert args.command == "uninstall"
    assert args.yes is True
    assert args.all_data is True


def test_build_parser_includes_version_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["version"])
    assert args.command == "version"


def test_build_parser_tui_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["tui"])
    assert args.host == "localhost"
    assert args.port == 7878


def test_build_parser_start_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["start"])
    assert args.host == "localhost"
    assert args.port == 7878
    assert args.foreground is False


def test_build_parser_accepts_plain_after_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["start", "--plain"])
    assert args.command == "start"
    assert args.plain is True


def test_build_parser_preserves_plain_before_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--plain", "start"])
    assert args.command == "start"
    assert args.plain is True


def test_build_parser_accepts_plain_after_nested_subcommand() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "list", "--plain"])
    assert args.command == "models"
    assert args.models_command == "list"
    assert args.plain is True


def test_build_parser_subparsers_use_custom_formatter() -> None:
    parser = cli.build_parser()
    command_subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    start_parser = command_subparsers.choices["start"]
    assert start_parser.formatter_class is parser.formatter_class

    models_parser = command_subparsers.choices["models"]
    model_subparsers = next(
        action for action in models_parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    list_parser = model_subparsers.choices["list"]
    assert list_parser.formatter_class is parser.formatter_class


def test_build_parser_help_omits_subparser_metavar_row() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "<command> ..." in help_text
    assert "\ncommands:\n" in help_text
    assert "\ncommands:\n  <command>\n" not in help_text


def test_build_parser_help_omits_subparser_metavar_row_with_rich_formatter() -> None:
    rich_argparse = pytest.importorskip("rich_argparse")
    parser = cli.build_parser(formatter_class=rich_argparse.RichHelpFormatter)
    help_text = parser.format_help()
    assert "<command> ..." in help_text
    assert "\nCommands:\n" in help_text
    assert "\n  <command>\n" not in help_text


def test_build_parser_rejects_removed_service_command() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["service", "status"])
    assert exc_info.value.code == 2


def test_main_rejects_removed_service_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "service", "status"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


def test_build_parser_models_pull_with_runtime() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "pull", "tiny", "--runtime", "whisper.cpp"])
    assert args.command == "models"
    assert args.models_command == "pull"
    assert args.runtime == "whisper.cpp"


@patch("murmur.cli.load_config")
@patch("murmur.bridge.run_bridge")
def test_run_bridge_calls_bridge_with_config(mock_run_bridge: Mock, mock_load_config: Mock) -> None:
    mock_config = Mock()
    mock_load_config.return_value = mock_config

    cli._run_bridge("localhost", 7878, capture_logs=True)

    mock_load_config.assert_called_once()
    mock_run_bridge.assert_called_once_with(mock_config, "localhost", 7878, capture_logs=True)


@patch("murmur.cli.resolve_tui_runtime")
@patch("murmur.cli.subprocess.Popen")
def test_run_tui_starts_tui_process(mock_popen: Mock, mock_resolve: Mock) -> None:
    runtime = Mock()
    runtime.mode = "packaged"
    runtime.command = ["/usr/bin/tui"]
    runtime.cwd = Path("/usr/bin")
    mock_resolve.return_value = runtime

    process = Mock()
    mock_popen.return_value = process

    result = cli._run_tui("localhost", 7878)

    assert result == process
    mock_popen.assert_called_once_with(
        ["/usr/bin/tui", "--host", "localhost", "--port", "7878"],
        cwd="/usr/bin",
    )


@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
def test_restore_terminal_state_when_tty(mock_stdin: Mock, mock_stdout: Mock) -> None:
    mock_stdin.isatty.return_value = True
    mock_stdout.isatty.return_value = True

    cli._restore_terminal_state()

    mock_stdout.write.assert_called()
    mock_stdout.flush.assert_called()


@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
def test_restore_terminal_state_when_not_tty(mock_stdin: Mock, mock_stdout: Mock) -> None:
    mock_stdin.isatty.return_value = False
    mock_stdout.isatty.return_value = False

    cli._restore_terminal_state()

    mock_stdout.write.assert_not_called()


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli._run_tui")
@patch("murmur.cli._restore_terminal_state")
def test_run_tui_attach_auto_starts_service(
    mock_restore: Mock,
    mock_run_tui: Mock,
    mock_ensure_service: Mock,
) -> None:
    process = Mock()
    mock_run_tui.return_value = process
    service_status = Mock()
    service_status.host = "127.0.0.1"
    service_status.port = 9000
    mock_ensure_service.return_value = service_status

    cli._run_tui_attach("localhost", 7878, status_indicator=True)

    mock_ensure_service.assert_called_once_with("localhost", 7878, status_indicator=True)
    mock_run_tui.assert_called_once_with("127.0.0.1", 9000)
    process.wait.assert_called_once()
    mock_restore.assert_called_once()


@patch("murmur.cli._run_bridge")
@patch("murmur.cli.create_status_indicator_provider")
def test_service_run_foreground_without_status_indicator_skips_indicator_provider(
    mock_create_indicator_provider: Mock,
    mock_run_bridge: Mock,
) -> None:
    cli._service_run("localhost", 7878, foreground=True, status_indicator=False)

    mock_create_indicator_provider.assert_not_called()
    mock_run_bridge.assert_called_once_with("localhost", 7878, capture_logs=True)


@patch("murmur.cli.logger")
@patch("murmur.cli._run_bridge")
@patch("murmur.cli.create_status_indicator_provider")
def test_service_run_foreground_stops_indicator_after_successful_start(
    mock_create_indicator_provider: Mock,
    mock_run_bridge: Mock,
    mock_logger: Mock,
) -> None:
    indicator_provider = Mock()
    mock_create_indicator_provider.return_value = indicator_provider

    cli._service_run("localhost", 7878, foreground=True, status_indicator=True)

    mock_create_indicator_provider.assert_called_once_with(host="localhost", port=7878)
    indicator_provider.start.assert_called_once()
    indicator_provider.stop.assert_called_once()
    mock_logger.warning.assert_not_called()
    mock_run_bridge.assert_called_once_with("localhost", 7878, capture_logs=True)


@patch("murmur.cli.logger")
@patch("murmur.cli._run_bridge")
@patch("murmur.cli.create_status_indicator_provider")
def test_service_run_foreground_does_not_stop_indicator_when_start_fails(
    mock_create_indicator_provider: Mock,
    mock_run_bridge: Mock,
    mock_logger: Mock,
) -> None:
    indicator_provider = Mock()
    indicator_provider.start.side_effect = RuntimeError("indicator start failed")
    mock_create_indicator_provider.return_value = indicator_provider

    cli._service_run("localhost", 7878, foreground=True, status_indicator=True)

    indicator_provider.start.assert_called_once()
    indicator_provider.stop.assert_not_called()
    mock_logger.warning.assert_called_once()
    mock_run_bridge.assert_called_once_with("localhost", 7878, capture_logs=True)


@patch("murmur.cli._ensure_service_running", side_effect=RuntimeError("boom"))
def test_run_tui_attach_exits_when_service_start_fails(mock_ensure_service: Mock, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli._run_tui_attach("localhost", 7878, status_indicator=True)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "failed to start service" in captured.err
    mock_ensure_service.assert_called_once()


@patch("murmur.cli._service_status")
def test_main_no_command_prints_help(
    mock_service_status: Mock,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(sys, "argv", ["cli"])

    cli.main()

    mock_service_status.assert_called_once()
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_main_rejects_removed_run_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "run", "--host", "127.0.0.1", "--port", "9000"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2


@patch("murmur.cli._run_bridge")
def test_main_runs_bridge_command(mock_run_bridge: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "bridge", "--host", "127.0.0.1", "--port", "9000"])

    cli.main()

    mock_run_bridge.assert_called_once_with("127.0.0.1", 9000, capture_logs=False)


@patch("murmur.cli._run_tui_attach")
def test_main_runs_tui_command(mock_attach: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "tui", "--no-status-indicator"])

    cli.main()

    mock_attach.assert_called_once_with("localhost", 7878, status_indicator=False)


@patch("murmur.cli._service_run")
def test_main_start_command_defaults_to_background_service(
    mock_service_run: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "start"])

    cli.main()

    mock_service_run.assert_called_once_with("localhost", 7878, foreground=False, status_indicator=True)


@patch("murmur.cli._service_run")
def test_main_start_command(mock_service_run: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["cli", "start", "--host", "0.0.0.0", "--port", "8123", "--foreground", "--no-status-indicator"],
    )

    cli.main()

    mock_service_run.assert_called_once_with("0.0.0.0", 8123, foreground=True, status_indicator=False)


@patch("murmur.cli._service_stop")
def test_main_stop_command(mock_service_stop: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "stop"])

    cli.main()

    mock_service_stop.assert_called_once()


@patch("murmur.cli._service_status")
def test_main_status_command(mock_service_status: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "status"])

    cli.main()

    mock_service_status.assert_called_once_with(diagnose=False)


@patch("murmur.cli._service_status")
def test_main_status_command_with_diagnose(mock_service_status: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "status", "--diagnose"])

    cli.main()

    mock_service_status.assert_called_once_with(diagnose=True)


@patch("murmur.cli._trigger")
def test_main_trigger_command(mock_trigger: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "trigger", "toggle"])

    cli.main()

    mock_trigger.assert_called_once_with(
        "localhost",
        7878,
        action="toggle",
        status_indicator=True,
        timeout_seconds=3.0,
    )


@patch("murmur.cli._trigger")
def test_main_trigger_command_with_custom_timeout(
    mock_trigger: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "trigger", "start", "--timeout-seconds", "5.5"])

    cli.main()

    mock_trigger.assert_called_once_with(
        "localhost",
        7878,
        action="start",
        status_indicator=True,
        timeout_seconds=5.5,
    )


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli.asyncio.run")
def test_trigger_success_prints_ack(
    mock_asyncio_run: Mock,
    mock_ensure_service: Mock,
    capsys,
) -> None:
    service_status = Mock()
    service_status.host = "127.0.0.1"
    service_status.port = 9000
    mock_ensure_service.return_value = service_status

    def _runner(coro):
        coro.close()
        return "recording"

    mock_asyncio_run.side_effect = _runner

    cli._trigger(
        "localhost",
        7878,
        action="start",
        status_indicator=True,
        timeout_seconds=3.0,
    )

    mock_ensure_service.assert_called_once_with("localhost", 7878, status_indicator=True)
    captured = capsys.readouterr()
    assert "Trigger acknowledged: status=recording" in captured.out


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli._trigger_async", new_callable=AsyncMock)
def test_trigger_uses_resolved_service_endpoint(
    mock_trigger_async: AsyncMock,
    mock_ensure_service: Mock,
    capsys,
) -> None:
    service_status = Mock()
    service_status.host = "127.0.0.1"
    service_status.port = 9000
    mock_ensure_service.return_value = service_status
    mock_trigger_async.return_value = "ready"

    cli._trigger(
        "localhost",
        7878,
        action="start",
        status_indicator=True,
        timeout_seconds=3.0,
    )

    mock_trigger_async.assert_called_once_with("127.0.0.1", 9000, "start", 3.0)
    captured = capsys.readouterr()
    assert "Trigger acknowledged: status=ready" in captured.out


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli.asyncio.run")
def test_trigger_timeout_exits_non_zero(
    mock_asyncio_run: Mock,
    mock_ensure_service: Mock,
    capsys,
) -> None:
    service_status = Mock()
    service_status.host = "127.0.0.1"
    service_status.port = 9000
    mock_ensure_service.return_value = service_status

    def _runner(coro):
        coro.close()
        raise TimeoutError("ack timeout")

    mock_asyncio_run.side_effect = _runner

    with pytest.raises(SystemExit) as exc_info:
        cli._trigger(
            "localhost",
            7878,
            action="stop",
            status_indicator=True,
            timeout_seconds=1.0,
        )

    assert exc_info.value.code == 2
    mock_ensure_service.assert_called_once_with("localhost", 7878, status_indicator=True)
    captured = capsys.readouterr()
    assert "timed out" in captured.err


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli.asyncio.run")
def test_trigger_error_exits_non_zero(
    mock_asyncio_run: Mock,
    mock_ensure_service: Mock,
    capsys,
) -> None:
    service_status = Mock()
    service_status.host = "127.0.0.1"
    service_status.port = 9000
    mock_ensure_service.return_value = service_status

    def _runner(coro):
        coro.close()
        raise RuntimeError("ws error")

    mock_asyncio_run.side_effect = _runner

    with pytest.raises(SystemExit) as exc_info:
        cli._trigger(
            "localhost",
            7878,
            action="start",
            status_indicator=True,
            timeout_seconds=1.0,
        )

    assert exc_info.value.code == 1
    mock_ensure_service.assert_called_once_with("localhost", 7878, status_indicator=True)
    captured = capsys.readouterr()
    assert "trigger command failed" in captured.err


@patch("murmur.cli._upgrade")
def test_main_upgrade_command(mock_upgrade: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "upgrade", "--version", "v2.0.0"])

    cli.main()

    mock_upgrade.assert_called_once_with(requested_version="v2.0.0")


@patch("murmur.cli._uninstall")
def test_main_uninstall_command(mock_uninstall: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "uninstall", "--yes"])

    cli.main()

    mock_uninstall.assert_called_once()


def test_main_version_flag_prints_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "__version__", "9.9.9")
    monkeypatch.setattr(sys, "argv", ["cli", "--version"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "9.9.9" in captured.out


@patch("murmur.cli._print_version")
def test_main_version_command(mock_print_version: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "version"])

    cli.main()

    mock_print_version.assert_called_once()


def test_print_version_outputs_installed_version(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli, "__version__", "1.2.3")
    cli._print_version()
    captured = capsys.readouterr()
    assert captured.out.strip() == "1.2.3"


@patch("murmur.uninstall.run_uninstall")
@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
def test_uninstall_non_interactive_without_flags_requires_yes(
    mock_stdin: Mock,
    mock_stdout: Mock,
    mock_run_uninstall: Mock,
    capsys,
) -> None:
    mock_stdin.isatty.return_value = False
    mock_stdout.isatty.return_value = False
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall"])

    with pytest.raises(SystemExit) as exc_info:
        cli._uninstall(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "non-interactive uninstall requires --yes" in captured.err
    mock_run_uninstall.assert_not_called()


@patch("murmur.uninstall.run_uninstall")
@patch("builtins.input", side_effect=["2", "y"])
@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
def test_uninstall_interactive_prompt_selects_scope(
    mock_stdin: Mock,
    mock_stdout: Mock,
    mock_input: Mock,
    mock_run_uninstall: Mock,
) -> None:
    mock_stdin.isatty.return_value = True
    mock_stdout.isatty.return_value = True
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall"])
    mock_run_uninstall.return_value = UninstallResult(
        channel="installer",
        removed_paths=(),
        failed_paths=(),
        warnings=(),
    )

    cli._uninstall(args)

    mock_run_uninstall.assert_called_once_with(
        options=UninstallOptions(
            remove_state=True,
            remove_config=True,
            remove_model_cache=False,
        )
    )


@patch("murmur.uninstall.run_uninstall")
def test_uninstall_success_outputs_summary(mock_run_uninstall: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall", "--yes"])
    mock_run_uninstall.return_value = UninstallResult(
        channel="installer",
        removed_paths=(Path("/var/lib/murmur/a"), Path("/var/lib/murmur/b")),
        failed_paths=(),
        warnings=("warn",),
    )

    cli._uninstall(args)

    captured = capsys.readouterr()
    assert "Removed paths:" in captured.out
    assert "Warnings:" in captured.out
    assert "Uninstall complete." in captured.out


@patch("murmur.uninstall.run_uninstall")
def test_uninstall_action_required_exits_with_guidance(mock_run_uninstall: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall", "--yes"])
    mock_run_uninstall.side_effect = UninstallActionRequired(
        channel="homebrew",
        command="brew uninstall murmur",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._uninstall(args)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "brew uninstall murmur" in captured.out


@patch("murmur.uninstall.run_uninstall")
def test_uninstall_error_exits_non_zero(mock_run_uninstall: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall", "--yes"])
    mock_run_uninstall.side_effect = UninstallError("failed")

    with pytest.raises(SystemExit) as exc_info:
        cli._uninstall(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: failed" in captured.err


@patch("murmur.uninstall.run_uninstall")
def test_uninstall_reports_failed_paths_as_non_zero(mock_run_uninstall: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall", "--yes"])
    mock_run_uninstall.return_value = UninstallResult(
        channel="installer",
        removed_paths=(Path("/var/lib/murmur/a"),),
        failed_paths=(RemovalFailure(path=Path("/var/lib/murmur/b"), reason="permission denied"),),
        warnings=(),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._uninstall(args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to remove:" in captured.out


@patch("murmur.upgrade.run_upgrade")
def test_upgrade_success_output(mock_run_upgrade: Mock, capsys) -> None:
    mock_run_upgrade.return_value = UpgradeResult(
        channel="installer",
        tag="v1.2.0",
        previous_version="1.1.0",
        new_version="1.2.0",
        restarted_service=True,
    )

    cli._upgrade(requested_version="v1.2.0")

    captured = capsys.readouterr()
    assert "1.1.0 -> 1.2.0" in captured.out
    assert "restarted" in captured.out


@patch("murmur.upgrade.run_upgrade")
def test_upgrade_action_required_exits_with_guidance(mock_run_upgrade: Mock, capsys) -> None:
    mock_run_upgrade.side_effect = UpgradeActionRequired(
        channel="homebrew",
        command="brew update && brew upgrade murmur",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._upgrade(requested_version=None)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "brew upgrade murmur" in captured.out


@patch("murmur.upgrade.run_upgrade", side_effect=UpgradeError("network error"))
def test_upgrade_error_exits_non_zero(mock_run_upgrade: Mock, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli._upgrade(requested_version=None)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "network error" in captured.err


@patch("murmur.model_manager.list_installed_models")
def test_main_models_list_runtime_variants(mock_list_models: Mock, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "models", "list"])

    model = Mock()
    model.name = "small"
    model.variants = {
        "faster-whisper": Mock(installed=True),
        "whisper.cpp": Mock(installed=False),
    }
    mock_list_models.return_value = [model]

    cli.main()

    captured = capsys.readouterr()
    assert "small: faster-whisper=installed, whisper.cpp=available" in captured.out


def test_main_models_without_subcommand_exits_with_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "models"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "No subcommand provided for 'models'" in captured.err


@patch("murmur.cli.load_config")
def test_main_config_command(mock_load_config: Mock, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["cli", "config"])

    mock_config = Mock()
    mock_config.to_dict.return_value = {
        "model": {"name": "tiny", "runtime": "faster-whisper"},
        "history": {"max_entries": 5000},
    }
    mock_load_config.return_value = mock_config

    cli.main()

    captured = capsys.readouterr()
    assert "[model]" in captured.out
    assert "[history]" in captured.out


def test_prog_name_uses_argv() -> None:
    with patch.object(sys, "argv", ["/usr/bin/murmur"]):
        parser = cli.build_parser()
        assert "murmur" in parser.prog


@patch("murmur.cli.ServiceManager")
def test_ensure_service_running_returns_service_status(mock_service_manager: Mock) -> None:
    expected_status = Mock()
    manager_instance = mock_service_manager.return_value
    manager_instance.ensure_running.return_value = expected_status

    result = cli._ensure_service_running("localhost", 7878, status_indicator=True)

    assert result is expected_status
    manager_instance.ensure_running.assert_called_once_with(
        host="localhost",
        port=7878,
        status_indicator=True,
    )


def test_wait_for_status_ignores_non_object_json_payloads() -> None:
    class DummyWebSocket:
        def __init__(self) -> None:
            self._messages = [
                "[]",
                '"string-payload"',
                '{"type":"status","status":"ready","message":"Ready"}',
            ]

        async def recv(self) -> str:
            if not self._messages:
                raise TimeoutError
            return self._messages.pop(0)

    status, message = asyncio.run(
        cli._wait_for_status(
            DummyWebSocket(),
            timeout_seconds=1.0,
        )
    )

    assert status == "ready"
    assert message == "Ready"


@patch("murmur.cli.subprocess.run", side_effect=RuntimeError("stty failed"))
@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
def test_restore_terminal_state_ignores_stty_failures(
    mock_stdin: Mock,
    mock_stdout: Mock,
    mock_run: Mock,
) -> None:
    mock_stdin.isatty.return_value = True
    mock_stdout.isatty.return_value = True

    cli._restore_terminal_state()

    mock_stdout.write.assert_called_once()
    mock_stdout.flush.assert_called_once()
    mock_run.assert_called_once_with(["stty", "sane"], check=False)


@patch("murmur.cli.subprocess.run")
@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
def test_restore_terminal_state_ignores_terminal_write_failures(
    mock_stdin: Mock,
    mock_stdout: Mock,
    mock_run: Mock,
) -> None:
    mock_stdin.isatty.return_value = True
    mock_stdout.isatty.return_value = True
    mock_stdout.write.side_effect = RuntimeError("stdout write failed")

    cli._restore_terminal_state()

    mock_run.assert_called_once_with(["stty", "sane"], check=False)


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli._run_tui")
@patch("murmur.cli._restore_terminal_state")
def test_run_tui_attach_handles_keyboard_interrupt(
    mock_restore: Mock,
    mock_run_tui: Mock,
    mock_ensure_service: Mock,
) -> None:
    process = Mock()
    process.wait.side_effect = KeyboardInterrupt
    mock_run_tui.return_value = process
    mock_ensure_service.return_value = Mock(host=None, port=None)

    cli._run_tui_attach("localhost", 7878, status_indicator=True)

    process.wait.assert_called_once()
    mock_restore.assert_called_once()


@patch("murmur.cli._ensure_service_running")
@patch("murmur.cli._run_tui", side_effect=FileNotFoundError("tui binary not found"))
@patch("murmur.cli._restore_terminal_state")
def test_run_tui_attach_handles_missing_tui_binary(
    mock_restore: Mock,
    mock_run_tui: Mock,
    mock_ensure_service: Mock,
    capsys,
) -> None:
    mock_ensure_service.return_value = Mock(host=None, port=None)

    with pytest.raises(SystemExit) as exc_info:
        cli._run_tui_attach("localhost", 7878, status_indicator=True)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "tui binary not found" in captured.err
    mock_run_tui.assert_called_once_with("localhost", 7878)
    mock_restore.assert_called_once()


@patch("murmur.cli.logger")
@patch("murmur.cli._run_bridge")
@patch("murmur.cli.create_status_indicator_provider")
def test_service_run_foreground_ignores_indicator_stop_failure(
    mock_create_indicator_provider: Mock,
    mock_run_bridge: Mock,
    mock_logger: Mock,
) -> None:
    indicator_provider = Mock()
    indicator_provider.stop.side_effect = RuntimeError("stop failed")
    mock_create_indicator_provider.return_value = indicator_provider

    cli._service_run("localhost", 7878, foreground=True, status_indicator=True)

    mock_create_indicator_provider.assert_called_once_with(host="localhost", port=7878)
    indicator_provider.start.assert_called_once()
    indicator_provider.stop.assert_called_once()
    mock_logger.warning.assert_not_called()
    mock_run_bridge.assert_called_once_with("localhost", 7878, capture_logs=True)


@patch("murmur.cli.ServiceManager")
def test_service_run_background_reports_running_status(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.start_background.return_value = Mock(running=True, stale=False, pid=42, host="h", port=10)

    cli._service_run("h", 10, foreground=False, status_indicator=True)

    manager.start_background.assert_called_once_with(host="h", port=10, status_indicator=True)
    captured = capsys.readouterr()
    assert "Service running pid=42 host=h port=10" in captured.out


@patch("murmur.cli.ServiceManager")
def test_service_run_background_reports_stale_status(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.start_background.return_value = Mock(running=False, stale=True, pid=None, host="h", port=10)

    cli._service_run("h", 10, foreground=False, status_indicator=False)

    captured = capsys.readouterr()
    assert "Service state was stale and has been cleaned up" in captured.out


@patch("murmur.cli.ServiceManager")
def test_service_run_background_reports_requested_status(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.start_background.return_value = Mock(running=False, stale=False, pid=None, host="h", port=10)

    cli._service_run("h", 10, foreground=False, status_indicator=False)

    captured = capsys.readouterr()
    assert "Service start requested" in captured.out


@patch("murmur.cli.ServiceManager")
def test_service_stop_prints_not_running_when_no_state(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.load_state.return_value = None

    cli._service_stop()

    manager.stop.assert_called_once()
    captured = capsys.readouterr()
    assert "Service is not running" in captured.out


@patch("murmur.cli.ServiceManager")
def test_service_stop_prints_stopped_when_state_exists(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.load_state.return_value = Mock()

    cli._service_stop()

    manager.stop.assert_called_once()
    captured = capsys.readouterr()
    assert "Service stopped" in captured.out


@patch("murmur.cli._runtime_status_snapshot")
@patch("murmur.cli.ServiceManager")
def test_service_status_prints_running_with_indicator(
    mock_service_manager: Mock,
    mock_runtime_snapshot: Mock,
    capsys,
) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=True,
        stale=False,
        pid=55,
        host="127.0.0.1",
        port=8787,
        status_indicator_pid=88,
    )
    mock_runtime_snapshot.return_value = {
        "status": "ready",
        "message": "Ready",
        "config": {
            "first_run_setup_required": False,
            "hotkey_diagnostics": {
                "backend": "macos_event_tap",
                "started": True,
                "blocked": False,
                "thread_alive": True,
                "press_count": 3,
                "release_count": 3,
                "tap_disable_count": 1,
                "tap_reenable_count": 1,
            },
            "startup": {
                "phase": "ready",
                "onboarding_close_ready": True,
                "blockers": [],
            },
        },
        "kickoff_sent": False,
    }

    cli._service_status()

    captured = capsys.readouterr()
    assert "running pid=55 host=127.0.0.1 port=8787 indicator_pid=88" in captured.out
    assert 'app_status=ready message="Ready"' in captured.out
    assert "hotkey backend=macos_event_tap started=true blocked=false thread_alive=true" in captured.out
    assert "app_ready=true" in captured.out
    mock_runtime_snapshot.assert_called_once_with(
        "127.0.0.1",
        8787,
        kickoff_onboarding=True,
        timeout_seconds=cli.STATUS_SNAPSHOT_TIMEOUT_SECONDS,
    )


@patch("murmur.cli._runtime_status_snapshot")
@patch("murmur.cli.ServiceManager")
def test_service_status_prints_first_run_guidance_and_blockers(
    mock_service_manager: Mock,
    mock_runtime_snapshot: Mock,
    capsys,
) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=True,
        stale=False,
        pid=55,
        host="localhost",
        port=7878,
        status_indicator_pid=None,
    )
    mock_runtime_snapshot.return_value = {
        "status": "connecting",
        "message": cli.FIRST_RUN_SETUP_MESSAGE,
        "config": {
            "first_run_setup_required": True,
            "startup": {
                "phase": "running",
                "runtime_probe": "running",
                "audio_scan": "running",
                "components": "running",
                "model": "pending",
                "onboarding_close_ready": False,
                "blockers": ["Download and select a model to continue."],
            },
        },
        "kickoff_sent": True,
    }

    cli._service_status()

    captured = capsys.readouterr()
    assert "app_status=connecting" in captured.out
    assert "startup phase=running" in captured.out
    assert "startup_blockers:" in captured.out
    assert "setup_init=started_via_status" in captured.out
    assert "murmur models pull small" in captured.out
    assert "murmur models select small" in captured.out


@patch("murmur.cli._runtime_status_snapshot", side_effect=RuntimeError("boom"))
@patch("murmur.cli.ServiceManager")
def test_service_status_handles_runtime_snapshot_failure(
    mock_service_manager: Mock,
    mock_runtime_snapshot: Mock,
    capsys,
) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=True,
        stale=False,
        pid=11,
        host="localhost",
        port=7878,
        status_indicator_pid=None,
    )

    cli._service_status()

    captured = capsys.readouterr()
    assert "running pid=11 host=localhost port=7878" in captured.out
    assert 'app_status=unknown message="Unable to query runtime state: boom"' in captured.out
    mock_runtime_snapshot.assert_called_once()


@patch("murmur.cli._runtime_status_snapshot")
@patch("murmur.cli.asyncio.get_running_loop", return_value=Mock())
@patch("murmur.cli.ServiceManager")
def test_service_status_handles_active_event_loop(
    mock_service_manager: Mock,
    mock_get_running_loop: Mock,
    mock_runtime_snapshot: Mock,
    capsys,
) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=True,
        stale=False,
        pid=21,
        host="localhost",
        port=7878,
        status_indicator_pid=None,
    )

    cli._service_status()

    captured = capsys.readouterr()
    assert "running pid=21 host=localhost port=7878" in captured.out
    assert cli.RUNNING_LOOP_STATUS_MESSAGE in captured.out
    mock_get_running_loop.assert_called_once()
    mock_runtime_snapshot.assert_not_called()


@patch("murmur.cli.ServiceManager")
def test_service_status_prints_stale_status(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=False,
        stale=True,
        pid=10,
        host="localhost",
        port=7878,
        status_indicator_pid=None,
    )

    cli._service_status()

    captured = capsys.readouterr()
    assert "stale (cleaned) previous_pid=10 host=localhost port=7878" in captured.out


@patch("murmur.cli.ServiceManager")
def test_service_status_diagnose_prints_read_only_stale_snapshot(
    mock_service_manager: Mock,
    capsys,
) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=False,
        stale=True,
        pid=10,
        host="localhost",
        port=7878,
        status_indicator_pid=None,
        pid_alive=True,
        reachable=False,
        stale_reason="port_unreachable",
    )

    cli._service_status(diagnose=True)

    manager.status.assert_called_once_with(
        cleanup_stale=False,
        auto_recover_unreachable=False,
        status_indicator=True,
    )
    captured = capsys.readouterr()
    assert "stale (diagnose) pid=10 host=localhost port=7878 reason=port_unreachable" in captured.out
    assert "diagnose pid_alive=true reachable=false cleanup=false auto_recover=false" in captured.out


@patch("murmur.cli.ServiceManager")
def test_service_status_prints_stopped(mock_service_manager: Mock, capsys) -> None:
    manager = mock_service_manager.return_value
    manager.status.return_value = Mock(
        running=False,
        stale=False,
        pid=None,
        host=None,
        port=None,
        status_indicator_pid=None,
    )

    cli._service_status()

    captured = capsys.readouterr()
    assert captured.out.strip() == "stopped"


def test_wait_for_status_returns_last_when_deadline_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyWebSocket:
        def recv(self) -> str:
            return "[]"

    call_count = {"value": 0}

    def fake_monotonic() -> float:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return 100.0
        if call_count["value"] == 2:
            return 100.05
        return 1000.0

    monkeypatch.setattr(cli.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(cli.asyncio, "wait_for", AsyncMock(return_value="[]"))

    status, message = asyncio.run(
        cli._wait_for_status(
            DummyWebSocket(),
            timeout_seconds=0.1,
        )
    )

    assert status is None
    assert message is None


def test_wait_for_status_returns_last_when_recv_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyWebSocket:
        def recv(self) -> str:
            return "{}"

    wait_for_mock = AsyncMock(side_effect=TimeoutError)
    monkeypatch.setattr(cli.asyncio, "wait_for", wait_for_mock)

    status, message = asyncio.run(
        cli._wait_for_status(
            DummyWebSocket(),
            timeout_seconds=1.0,
        )
    )

    assert status is None
    assert message is None


def test_extract_status_update_rejects_invalid_utf8_bytes() -> None:
    assert cli._extract_status_update(b"\xff") is None


def test_extract_status_update_rejects_invalid_json() -> None:
    assert cli._extract_status_update("{broken-json") is None


def test_extract_status_update_rejects_non_status_payload() -> None:
    assert cli._extract_status_update('{"type":"not-status","status":"ready"}') is None


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent_messages.append(payload)


class _FakeConnectContext:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self._websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self._websocket

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _install_fake_websockets_module(
    monkeypatch: pytest.MonkeyPatch,
    websocket: _FakeWebSocket,
) -> dict[str, object]:
    state: dict[str, object] = {}

    def connect(uri: str, ping_interval: int, ping_timeout: int) -> _FakeConnectContext:
        state["uri"] = uri
        state["ping_interval"] = ping_interval
        state["ping_timeout"] = ping_timeout
        return _FakeConnectContext(websocket)

    monkeypatch.setitem(sys.modules, "websockets", Mock(connect=connect))
    return state


def test_trigger_async_sends_start_command_and_returns_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = _FakeWebSocket()
    connect_state = _install_fake_websockets_module(monkeypatch, websocket)
    wait_for_status = AsyncMock(side_effect=[(None, None), ("recording", "ack")])

    with patch("murmur.cli._wait_for_status", wait_for_status):
        result = asyncio.run(cli._trigger_async("localhost", 7878, "start", 3.0))

    assert result == "recording"
    assert connect_state == {
        "uri": "ws://localhost:7878",
        "ping_interval": 10,
        "ping_timeout": 10,
    }
    assert [json.loads(msg)["type"] for msg in websocket.sent_messages] == ["start_recording"]
    assert wait_for_status.await_count == 2
    assert wait_for_status.await_args_list[0].kwargs["timeout_seconds"] == 0.75
    assert wait_for_status.await_args_list[1].kwargs["expected_statuses"] == {
        "recording",
        "connecting",
    }


def test_trigger_async_toggle_recording_sends_stop_command(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = _FakeWebSocket()
    _install_fake_websockets_module(monkeypatch, websocket)
    wait_for_status = AsyncMock(side_effect=[("recording", None), ("ready", "ack")])

    with patch("murmur.cli._wait_for_status", wait_for_status):
        result = asyncio.run(cli._trigger_async("localhost", 7878, "toggle", 2.0))

    assert result == "ready"
    assert [json.loads(msg)["type"] for msg in websocket.sent_messages] == ["stop_recording"]
    assert wait_for_status.await_args_list[1].kwargs["expected_statuses"] == {
        "transcribing",
        "ready",
    }


def test_trigger_async_returns_immediately_when_status_already_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _FakeWebSocket()
    _install_fake_websockets_module(monkeypatch, websocket)
    wait_for_status = AsyncMock(return_value=("recording", "already"))

    with patch("murmur.cli._wait_for_status", wait_for_status):
        result = asyncio.run(cli._trigger_async("localhost", 7878, "start", 1.0))

    assert result == "recording"
    assert websocket.sent_messages == []
    assert wait_for_status.await_count == 1


def test_trigger_async_stop_from_error_sends_stop_command(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = _FakeWebSocket()
    _install_fake_websockets_module(monkeypatch, websocket)
    wait_for_status = AsyncMock(side_effect=[("error", "runtime not ready"), ("ready", "stopped")])

    with patch("murmur.cli._wait_for_status", wait_for_status):
        result = asyncio.run(cli._trigger_async("localhost", 7878, "stop", 1.0))

    assert result == "ready"
    assert [json.loads(msg)["type"] for msg in websocket.sent_messages] == ["stop_recording"]
    assert wait_for_status.await_count == 2


def test_trigger_async_timeout_raises_with_status_details(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = _FakeWebSocket()
    _install_fake_websockets_module(monkeypatch, websocket)
    wait_for_status = AsyncMock(side_effect=[(None, None), (None, None)])

    with patch("murmur.cli._wait_for_status", wait_for_status):
        with pytest.raises(TimeoutError) as exc_info:
            asyncio.run(cli._trigger_async("localhost", 7878, "start", 1.0))

    assert "Timed out waiting for trigger acknowledgement (start)" in str(exc_info.value)
    assert "last_status=unknown" in str(exc_info.value)


def test_extract_config_update_rejects_invalid_json() -> None:
    assert cli._extract_config_update("{broken-json") is None


def test_extract_config_update_rejects_non_config_payload() -> None:
    assert cli._extract_config_update('{"type":"status","status":"ready"}') is None


def test_extract_config_update_returns_config_dict() -> None:
    payload = '{"type":"config","config":{"first_run_setup_required":true}}'
    assert cli._extract_config_update(payload) == {"first_run_setup_required": True}


def test_runtime_status_snapshot_kicks_off_setup_when_first_run_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _FakeWebSocket()
    _install_fake_websockets_module(monkeypatch, websocket)
    collect = AsyncMock(
        side_effect=[
            (
                "connecting",
                cli.FIRST_RUN_SETUP_MESSAGE,
                {"first_run_setup_required": True, "startup": {"phase": "idle"}},
            ),
            (
                "connecting",
                cli.FIRST_RUN_SETUP_MESSAGE,
                {"first_run_setup_required": True, "startup": {"phase": "running"}},
            ),
        ]
    )

    with patch("murmur.cli._collect_runtime_snapshot", collect):
        result = asyncio.run(
            cli._runtime_status_snapshot(
                "localhost",
                7878,
                kickoff_onboarding=True,
                timeout_seconds=1.0,
            )
        )

    assert result["kickoff_sent"] is True
    assert [json.loads(msg)["type"] for msg in websocket.sent_messages] == ["begin_onboarding_setup"]
    assert collect.await_count == 2
    assert collect.await_args_list[0].kwargs["timeout_seconds"] == 1.0
    assert 0 <= collect.await_args_list[1].kwargs["timeout_seconds"] <= 1.0


def test_runtime_status_snapshot_skips_kickoff_when_not_first_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = _FakeWebSocket()
    _install_fake_websockets_module(monkeypatch, websocket)
    collect = AsyncMock(
        return_value=(
            "ready",
            "Ready",
            {"first_run_setup_required": False, "startup": {"phase": "ready"}},
        )
    )

    with patch("murmur.cli._collect_runtime_snapshot", collect):
        result = asyncio.run(
            cli._runtime_status_snapshot(
                "localhost",
                7878,
                kickoff_onboarding=True,
                timeout_seconds=1.0,
            )
        )

    assert result["kickoff_sent"] is False
    assert websocket.sent_messages == []
    assert collect.await_count == 1


@patch("murmur.cli._ensure_service_running", side_effect=RuntimeError("start failed"))
def test_trigger_exits_when_service_start_fails(mock_ensure_service: Mock, capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli._trigger(
            "localhost",
            7878,
            action="start",
            status_indicator=True,
            timeout_seconds=1.0,
        )

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "failed to start service" in captured.err
    mock_ensure_service.assert_called_once_with("localhost", 7878, status_indicator=True)


def test_resolve_uninstall_scope_enables_all_flags_when_all_data_enabled() -> None:
    args = argparse.Namespace(
        remove_state=False,
        remove_config=False,
        remove_model_cache=False,
        all_data=True,
    )

    remove_state, remove_config, remove_model_cache, explicit_scope = cli._resolve_uninstall_scope(args)

    assert (remove_state, remove_config, remove_model_cache, explicit_scope) == (True, True, True, True)


@patch("builtins.input", side_effect=[""])
def test_prompt_uninstall_scope_default_choice_is_app_only(mock_input: Mock) -> None:
    assert cli._prompt_uninstall_scope() == (False, False, False)
    mock_input.assert_called_once()


@patch("builtins.input", side_effect=["9", "3"])
def test_prompt_uninstall_scope_reprompts_until_valid_choice(mock_input: Mock, capsys) -> None:
    assert cli._prompt_uninstall_scope() == (True, True, True)
    captured = capsys.readouterr()
    assert "Invalid choice" in captured.out
    assert mock_input.call_count == 2


def test_print_uninstall_plan_includes_model_cache_when_requested(capsys) -> None:
    cli._print_uninstall_plan(
        remove_state=False,
        remove_config=False,
        remove_model_cache=True,
    )

    captured = capsys.readouterr()
    assert "model caches under ~/.cache/huggingface/hub" in captured.out


@patch("builtins.input", side_effect=["yes", "no"])
def test_confirm_uninstall_accepts_yes_only(mock_input: Mock) -> None:
    assert cli._confirm_uninstall() is True
    assert cli._confirm_uninstall() is False
    assert mock_input.call_count == 2


@patch("murmur.uninstall.run_uninstall")
@patch("murmur.cli.sys.stdout")
@patch("murmur.cli.sys.stdin")
@patch("murmur.cli._confirm_uninstall", return_value=False)
def test_uninstall_interactive_cancelled_by_user_exits(
    mock_confirm: Mock,
    mock_stdin: Mock,
    mock_stdout: Mock,
    mock_run_uninstall: Mock,
    capsys,
) -> None:
    mock_stdin.isatty.return_value = True
    mock_stdout.isatty.return_value = True
    parser = cli.build_parser()
    args = parser.parse_args(["uninstall", "--remove-state"])

    with pytest.raises(SystemExit) as exc_info:
        cli._uninstall(args)

    assert exc_info.value.code == 1
    mock_confirm.assert_called_once()
    mock_run_uninstall.assert_not_called()


@patch("murmur.model_manager.list_installed_models")
def test_handle_models_command_list_without_runtime_variants(mock_list_models: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "list"])
    model = Mock(name="tiny")
    model.name = "tiny"
    model.variants = None
    model.installed = True
    mock_list_models.return_value = [model]

    cli._handle_models_command(args)

    captured = capsys.readouterr()
    assert "tiny: installed" in captured.out


@patch("murmur.model_manager.download_model")
def test_handle_models_command_pull_default_runtime(download_model: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "pull", "tiny"])

    cli._handle_models_command(args)

    download_model.assert_called_once_with("tiny")
    captured = capsys.readouterr()
    assert "Downloaded tiny" in captured.out


@patch("murmur.model_manager.download_model")
def test_handle_models_command_pull_with_runtime(download_model: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "pull", "tiny", "--runtime", "whisper.cpp"])

    cli._handle_models_command(args)

    download_model.assert_called_once_with("tiny", runtime="whisper.cpp")
    captured = capsys.readouterr()
    assert "Downloaded tiny (whisper.cpp)" in captured.out


@patch("murmur.model_manager.remove_model")
def test_handle_models_command_remove_default_runtime(remove_model: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "remove", "tiny"])

    cli._handle_models_command(args)

    remove_model.assert_called_once_with("tiny")
    captured = capsys.readouterr()
    assert "Removed tiny" in captured.out


@patch("murmur.model_manager.remove_model")
def test_handle_models_command_remove_with_runtime(remove_model: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "remove", "tiny", "--runtime", "whisper.cpp"])

    cli._handle_models_command(args)

    remove_model.assert_called_once_with("tiny", runtime="whisper.cpp")
    captured = capsys.readouterr()
    assert "Removed tiny (whisper.cpp)" in captured.out


@patch("murmur.model_manager.set_selected_model")
def test_handle_models_command_select(set_selected_model: Mock, capsys) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["models", "select", "base"])

    cli._handle_models_command(args)

    set_selected_model.assert_called_once_with("base")
    captured = capsys.readouterr()
    assert "Selected model set to base" in captured.out


@patch("murmur.cli.load_config")
def test_handle_config_command_prints_scalar_and_nested_values(mock_load_config: Mock, capsys) -> None:
    args = argparse.Namespace(path=None)
    config = Mock()
    config.to_dict.return_value = {
        "channel": "stable",
        "model": {"name": "small", "variant": {"runtime": "faster-whisper"}},
    }
    mock_load_config.return_value = config

    cli._handle_config_command(args)

    captured = capsys.readouterr()
    assert "channel = stable" in captured.out
    assert "[model]" in captured.out
    assert "variant.runtime = faster-whisper" in captured.out


def test_main_unknown_command_prints_help() -> None:
    parser = Mock()
    parser.parse_args.return_value = argparse.Namespace(command="unknown")

    with patch("murmur.cli.build_parser", return_value=parser):
        cli.main()

    parser.print_help.assert_called_once()


def test_cli_module_entrypoint_calls_main(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["murmur"])

    runpy.run_path(str(Path(cli.__file__)), run_name="__main__")

    captured = capsys.readouterr()
    assert "usage:" in captured.out
