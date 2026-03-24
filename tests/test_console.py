"""Tests for murmur.console — MurmurConsole output module."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from murmur.console import MurmurConsole, _format_size, init_console, get_console


@pytest.fixture
def plain_console() -> MurmurConsole:
    return MurmurConsole(force_plain=True)



class TestIsRich:
    def test_force_plain_returns_false(self, plain_console: MurmurConsole) -> None:
        assert plain_console.is_rich is False

    def test_non_tty_returns_false(self) -> None:
        # In tests, stdout is not a TTY so is_rich should be False
        console = MurmurConsole(force_plain=False)
        assert console.is_rich is False


class TestPrintVersion:
    def test_plain_prints_version_string(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_version("1.2.3")
        captured = capsys.readouterr()
        assert captured.out.strip() == "1.2.3"


class TestPrintServiceStatus:
    def test_plain_running(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_service_status(
            running=True, pid=123, host="localhost", port=7878,
        )
        captured = capsys.readouterr()
        assert "running pid=123 host=localhost port=7878" in captured.out

    def test_plain_running_with_indicator(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_service_status(
            running=True, pid=123, host="localhost", port=7878, indicator_pid=456,
        )
        captured = capsys.readouterr()
        assert "indicator_pid=456" in captured.out

    def test_plain_stopped(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_service_status(running=False)
        captured = capsys.readouterr()
        assert captured.out.strip() == "stopped"


class TestPrintStaleStatus:
    def test_plain_output(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_stale_status(pid=10, host="localhost", port=7878)
        captured = capsys.readouterr()
        assert "stale (cleaned) previous_pid=10 host=localhost port=7878" in captured.out


class TestPrintModelList:
    def test_plain_with_variants(self, plain_console: MurmurConsole, capsys) -> None:
        model = Mock()
        model.name = "small"
        model.variants = {
            "faster-whisper": Mock(installed=True, size_bytes=500_000_000),
            "whisper.cpp": Mock(installed=False, size_bytes=485_000_000),
        }
        plain_console.print_model_list([model], selected="small")
        captured = capsys.readouterr()
        assert "small: faster-whisper=installed, whisper.cpp=available" in captured.out

    def test_plain_without_variants(self, plain_console: MurmurConsole, capsys) -> None:
        model = Mock()
        model.name = "tiny"
        model.variants = None
        model.installed = True
        plain_console.print_model_list([model])
        captured = capsys.readouterr()
        assert "tiny: installed" in captured.out


class TestPrintConfig:
    def test_plain_prints_sections(self, plain_console: MurmurConsole, capsys) -> None:
        config_dict = {
            "channel": "stable",
            "model": {"name": "small", "variant": {"runtime": "faster-whisper"}},
        }
        plain_console.print_config(config_dict)
        captured = capsys.readouterr()
        assert "channel = stable" in captured.out
        assert "[model]" in captured.out
        assert "variant.runtime = faster-whisper" in captured.out


class TestFeedbackMessages:
    def test_print_success_plain(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_success("All good")
        assert capsys.readouterr().out.strip() == "All good"

    def test_print_warning_plain(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_warning("Be careful")
        assert capsys.readouterr().out.strip() == "Be careful"

    def test_print_error_plain(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_error("Something broke")
        assert "Error: Something broke" in capsys.readouterr().err

    def test_print_error_with_hint_plain(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_error("Failed", hint="Try again")
        captured = capsys.readouterr()
        assert "Error: Failed" in captured.err
        assert "Hint: Try again" in captured.err


class TestUninstallFlow:
    def test_print_uninstall_plan_plain(self, plain_console: MurmurConsole, capsys) -> None:
        plain_console.print_uninstall_plan(
            remove_state=True, remove_config=False, remove_model_cache=True,
        )
        captured = capsys.readouterr()
        assert "Uninstall plan:" in captured.out
        assert "~/.local/state/murmur" in captured.out
        assert "model caches" in captured.out
        assert "~/.config/murmur" not in captured.out

    def test_prompt_uninstall_scope_plain_default(
        self, plain_console: MurmurConsole, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        assert plain_console.prompt_uninstall_scope() == (False, False, False)

    def test_confirm_uninstall_plain_yes(
        self, plain_console: MurmurConsole, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt: "y")
        assert plain_console.confirm_uninstall() is True

    def test_confirm_uninstall_plain_no(
        self, plain_console: MurmurConsole, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt: "n")
        assert plain_console.confirm_uninstall() is False


class TestDownloadProgress:
    def test_plain_progress_context_manager(self, plain_console: MurmurConsole, capsys) -> None:
        with plain_console.download_progress("small", "faster-whisper") as update:
            update(50)
        captured = capsys.readouterr()
        assert "Downloading small" in captured.out


class TestFormatSize:
    def test_bytes(self) -> None:
        assert _format_size(500) == "500 B"

    def test_kilobytes(self) -> None:
        assert _format_size(2048) == "2 KB"

    def test_megabytes(self) -> None:
        assert _format_size(500 * 1024 * 1024) == "500 MB"

    def test_gigabytes(self) -> None:
        assert _format_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


class TestSingleton:
    def test_init_console_returns_instance(self) -> None:
        import murmur.console as console_mod
        original_console = console_mod._console
        console = init_console(force_plain=True)
        try:
            assert isinstance(console, MurmurConsole)
            assert console.is_rich is False
        finally:
            console_mod._console = original_console

    def test_get_console_returns_default_if_not_initialized(self) -> None:
        import murmur.console as console_mod
        original_console = console_mod._console
        console_mod._console = None
        try:
            console = get_console()
            assert isinstance(console, MurmurConsole)
        finally:
            console_mod._console = original_console


class TestGetHelpFormatterClass:
    def test_plain_returns_none(self, plain_console: MurmurConsole) -> None:
        assert plain_console.get_help_formatter_class() is None


class TestPrintImageLogo:
    def test_returns_false_when_not_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from murmur.logo import print_image_logo
        monkeypatch.setattr("sys.stdout", Mock(isatty=lambda: False))
        assert print_image_logo() is False

    def test_falls_back_on_render_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """print_image_logo returns False instead of raising when rendering fails."""
        from murmur import logo as logo_mod
        from murmur.logo import print_image_logo
        monkeypatch.setattr("sys.stdout", Mock(isatty=lambda: True))
        monkeypatch.setattr(logo_mod, "_supports_iterm_images", lambda: True)
        monkeypatch.setattr(logo_mod, "_render_iterm_image", lambda: (_ for _ in ()).throw(Exception("bad data")))
        assert print_image_logo() is False
