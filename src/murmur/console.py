"""Central console output module for murmur CLI with Rich styling and plain-text fallback."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

MURMUR_THEME = Theme(
    {
        "primary": "bold #9d7cd8",
        "secondary": "#5c9cf5",
        "accent": "#fab283",
        "success": "bold green",
        "error": "bold red",
        "warning": "bold yellow",
        "muted": "dim",
    }
)

_INSTALLED_LABEL = "● installed"
_AVAILABLE_LABEL = "○ available"
_WHISPER_CPP_KEY = "whisper.cpp"


class MurmurConsole:
    """Wraps Rich Console with brand-themed output methods and plain-text fallback."""

    def __init__(self, *, force_plain: bool = False) -> None:
        self._force_plain = force_plain
        self._console = Console(theme=MURMUR_THEME, highlight=False)

    @property
    def is_rich(self) -> bool:
        """True when rich output should be used (TTY and not forced plain)."""
        if self._force_plain:
            return False
        return bool(self._console.is_terminal)

    @property
    def rich(self) -> Console:
        """Access the underlying Rich Console."""
        return self._console

    # ── Logo ──────────────────────────────────────────────────────────

    def print_logo(self) -> None:
        from murmur.logo import print_logo
        if self.is_rich:
            print_logo(self._console)
        # No logo in plain mode

    # ── Version ───────────────────────────────────────────────────────

    def print_version(self, version: str) -> None:
        if self.is_rich:
            self.print_logo()
            self._console.print(f"  [muted]v{version}[/muted]")
        else:
            print(version)

    # ── Service Status ────────────────────────────────────────────────

    def print_service_status(
        self,
        *,
        running: bool,
        pid: int | None = None,
        host: str | None = None,
        port: int | None = None,
        indicator_pid: int | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        if not self.is_rich:
            self._print_service_status_plain(
                running=running, pid=pid, host=host, port=port,
                indicator_pid=indicator_pid,
            )
            return

        _STATUS_PANEL_WIDTH = 48

        if running:
            content = Text()
            content.append("  Status   ", style="muted")
            content.append("● Running\n", style="success")
            content.append("  PID      ", style="muted")
            content.append(f"{pid}\n")
            addr = f"{host or 'localhost'}:{port or 7878}"
            content.append("  Address  ", style="muted")
            content.append(addr)

            if snapshot:
                content.append("\n\n")
                self._append_rich_snapshot(content, snapshot)

            panel = Panel(
                content,
                title="[primary]murmur[/primary]",
                title_align="left",
                border_style="primary",
                width=_STATUS_PANEL_WIDTH,
                padding=(1, 2),
            )
            self._console.print(panel)
        else:
            content = Text()
            content.append("  Status   ", style="muted")
            content.append("○ Stopped", style="muted")
            panel = Panel(
                content,
                title="[primary]murmur[/primary]",
                title_align="left",
                border_style="primary",
                width=_STATUS_PANEL_WIDTH,
                padding=(1, 2),
            )
            self._console.print(panel)

    def _append_rich_snapshot(self, content: Text, snapshot: dict[str, Any]) -> None:
        status = snapshot.get("status")
        config = snapshot.get("config")

        status_text = status if isinstance(status, str) and status else "unknown"

        if not isinstance(config, dict):
            content.append("  Runtime  ", style="muted")
            content.append("unavailable", style="warning")
            return

        startup = config.get("startup")
        startup_dict = startup if isinstance(startup, dict) else {}
        phase = startup_dict.get("phase", "unknown")
        if isinstance(phase, str):
            phase = phase.strip().lower() or "unknown"

        first_run = bool(config.get("first_run_setup_required"))
        blockers = startup_dict.get("blockers")
        blocker_list = [str(item) for item in blockers] if isinstance(blockers, list) else []
        close_ready = bool(startup_dict.get("onboarding_close_ready"))

        # If fully ready, compact output
        if not first_run and status_text == "ready" and close_ready and not blocker_list:
            content.append("  Runtime  ", style="muted")
            content.append("✓ ready", style="success")
            return

        self._append_runtime_tree(content, startup_dict, phase)

        if blocker_list:
            for blocker in blocker_list:
                content.append(f"\n  ⚠ {blocker}", style="warning")

        if first_run:
            content.append("\n")
            self._append_first_run_guidance(content)

    def _append_runtime_tree(
        self, content: Text, startup_dict: dict[str, Any], phase: str
    ) -> None:
        """Append the runtime component tree lines to content."""
        content.append("  Runtime\n", style="muted")

        phase_icon, phase_style = _component_state(phase, {"complete", "ready"})
        content.append("  ├─ Phase    ", style="muted")
        content.append(f"{phase_icon} {phase}\n", style=phase_style)

        audio = str(startup_dict.get("audio_scan", "unknown"))
        audio_icon, audio_style = _component_state(audio, {"done", "ready", "complete"})
        content.append("  ├─ Audio    ", style="muted")
        content.append(f"{audio_icon} {audio}\n", style=audio_style)

        model = str(startup_dict.get("model", "unknown"))
        model_icon, model_style = _component_state(model, {"done", "ready", "loaded", "complete"})
        content.append("  ├─ Model    ", style="muted")
        content.append(f"{model_icon} {model}\n", style=model_style)

        components = str(startup_dict.get("components", "unknown"))
        comp_icon, comp_style = _component_state(components, {"done", "ready", "complete"})
        content.append("  └─ App      ", style="muted")
        content.append(f"{comp_icon} {components}", style=comp_style)

    def _append_first_run_guidance(self, content: Text) -> None:
        from murmur.cli import SETUP_GUIDANCE_MODEL
        content.append("  Next steps:\n", style="accent")
        content.append(f"    murmur models pull {SETUP_GUIDANCE_MODEL}\n", style="secondary")
        content.append(f"    murmur models select {SETUP_GUIDANCE_MODEL}\n", style="secondary")
        content.append("    murmur status", style="secondary")

    def _print_service_status_plain(
        self,
        *,
        running: bool,
        pid: int | None,
        host: str | None,
        port: int | None,
        indicator_pid: int | None,
    ) -> None:
        if running:
            indicator = (
                f" indicator_pid={indicator_pid}"
                if indicator_pid is not None
                else ""
            )
            print(f"running pid={pid} host={host} port={port}{indicator}")
        else:
            print("stopped")

    # ── Service Status (stale) ────────────────────────────────────────

    def print_stale_status(self, *, pid: int | None, host: str | None, port: int | None) -> None:
        if self.is_rich:
            self._console.print(
                f"  [warning]⚠[/warning] Stale service state cleaned up "
                f"[muted](previous PID: {pid}, {host}:{port})[/muted]"
            )
        else:
            print(f"stale (cleaned) previous_pid={pid} host={host} port={port}")

    # ── Runtime Status Snapshot (plain fallback) ──────────────────────

    def print_runtime_status_plain(self, snapshot: dict[str, Any]) -> None:
        """Print runtime status snapshot in plain key=value format."""
        from murmur.cli import (
            FIRST_RUN_SETUP_MESSAGE,
            SETUP_GUIDANCE_MODEL,
            _first_run_pending,
            _parse_startup_detail,
        )

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

        first_run = _first_run_pending(config)
        startup_dict, phase, blocker_list, close_ready = _parse_startup_detail(config)

        if not first_run and status_text == "ready" and close_ready and not blocker_list:
            print("app_ready=true")
            return

        runtime_probe = str(startup_dict.get("runtime_probe", "unknown"))
        audio_scan = str(startup_dict.get("audio_scan", "unknown"))
        components = str(startup_dict.get("components", "unknown"))
        model_state = str(startup_dict.get("model", "unknown"))
        print(
            "startup "
            f"phase={phase} runtime_probe={runtime_probe} "
            f"audio_scan={audio_scan} components={components} model={model_state}"
        )

        if blocker_list:
            print("startup_blockers:")
            for blocker in blocker_list:
                print(f"  - {blocker}")

        if first_run:
            if kickoff_sent:
                print("setup_init=started_via_status")
            print(f"setup_required=true message={json.dumps(FIRST_RUN_SETUP_MESSAGE, ensure_ascii=True)}")
            print("next_steps:")
            print(f"  murmur models pull {SETUP_GUIDANCE_MODEL}")
            print(f"  murmur models select {SETUP_GUIDANCE_MODEL}")
            print("  murmur status")

    def print_runtime_error_plain(self, exc: Exception) -> None:
        """Print runtime status error in plain key=value format."""
        error_message = f"Unable to query runtime state: {exc}"
        print(f"app_status=unknown message={json.dumps(error_message, ensure_ascii=True)}")

    # ── Model List ────────────────────────────────────────────────────

    def print_model_list(self, models: list[Any], selected: str | None = None) -> None:
        if not self.is_rich:
            self._print_model_list_plain(models)
            return

        table = Table(
            show_header=True,
            header_style="muted",
            border_style="primary",
            title="[primary]Models[/primary]",
            title_style="primary",
            padding=(0, 1),
        )
        table.add_column("Name", style="bold")
        table.add_column("faster-whisper", justify="center")
        table.add_column(_WHISPER_CPP_KEY, justify="center")
        table.add_column("Size", justify="right", style="muted")

        for model in models:
            self._add_model_table_row(table, model, selected)

        self._console.print()
        self._console.print(table)
        if selected:
            self._console.print("  [accent]★[/accent] [muted]selected model[/muted]")
        self._console.print()

    def _add_model_table_row(self, table: Table, model: Any, selected: str | None) -> None:
        """Add a single model row to the rich models table."""
        name = model.name
        is_selected = name == selected
        name_display = f"★ {name}" if is_selected else f"  {name}"

        variants = getattr(model, "variants", None)
        if isinstance(variants, dict):
            fw = variants.get("faster-whisper")
            cpp = variants.get(_WHISPER_CPP_KEY)
            fw_text = _variant_text(fw)
            cpp_text = _variant_text(cpp)
            size = next((v.size_bytes for v in [fw, cpp] if v and v.size_bytes), None)
            size_text = _format_size(size) if size else ""
        else:
            installed = bool(getattr(model, "installed", False))
            fw_text = _installed_text(installed)
            cpp_text = Text("-", style="muted")
            size_text = ""

        name_style = "accent" if is_selected else ""
        table.add_row(
            Text(name_display, style=name_style),
            fw_text,
            cpp_text,
            size_text,
        )

    def _print_model_list_plain(self, models: list[Any]) -> None:
        for model in models:
            variants = getattr(model, "variants", None)
            if isinstance(variants, dict):
                fw_variant = variants.get("faster-whisper")
                cpp_variant = variants.get(_WHISPER_CPP_KEY)
                fw_state = "installed" if fw_variant and fw_variant.installed else "available"
                wcpp_state = "installed" if cpp_variant and cpp_variant.installed else "available"
                print(f"{model.name}: faster-whisper={fw_state}, {_WHISPER_CPP_KEY}={wcpp_state}")
            else:
                state = "installed" if bool(getattr(model, "installed", False)) else "available"
                print(f"{model.name}: {state}")

    # ── Config ────────────────────────────────────────────────────────

    def print_config(self, config_dict: dict[str, Any]) -> None:
        if not self.is_rich:
            self._print_config_plain(config_dict)
            return

        import tomli_w

        toml_str = tomli_w.dumps(config_dict)
        from rich.syntax import Syntax

        syntax = Syntax(toml_str, "toml", theme="monokai", line_numbers=False, padding=1)
        panel = Panel(
            syntax,
            title="[primary]Config[/primary]",
            title_align="left",
            border_style="primary",
        )
        self._console.print(panel)

    def _print_config_plain(self, config_dict: dict[str, Any]) -> None:
        for section, values in config_dict.items():
            if not isinstance(values, dict):
                print(f"{section} = {values}")
                continue
            print(f"[{section}]")
            for key, value in values.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"{key}.{sub_key} = {sub_value}")
                else:
                    print(f"{key} = {value}")

    # ── Download Progress ─────────────────────────────────────────────

    @contextmanager
    def download_progress(self, name: str, runtime: str) -> Iterator[Any]:
        """Context manager yielding a callback(percent: int) for download progress."""
        if not self.is_rich:
            print(f"Downloading {name} ({runtime})...")
            yield lambda percent: None
            return

        label = f"Downloading {name} ({runtime})"
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        )
        with progress:
            task = progress.add_task(label, total=100)

            def update(percent: int) -> None:
                progress.update(task, completed=percent)

            yield update

    # ── Feedback Messages ─────────────────────────────────────────────

    def print_success(self, message: str) -> None:
        if self.is_rich:
            self._console.print(f"  [success]✓[/success] {message}")
        else:
            print(message)

    def print_warning(self, message: str) -> None:
        if self.is_rich:
            self._console.print(f"  [warning]⚠[/warning] {message}")
        else:
            print(message)

    def print_error(self, message: str, *, hint: str | None = None) -> None:
        if self.is_rich:
            self._console.print(f"  [error]✗[/error] {message}")
            if hint:
                self._console.print(f"    [muted]Hint: {hint}[/muted]")
        else:
            print(f"Error: {message}", file=sys.stderr)
            if hint:
                print(f"Hint: {hint}", file=sys.stderr)

    def print(self, message: str = "") -> None:
        """Print a plain message."""
        print(message)

    # ── Uninstall Flow ────────────────────────────────────────────────

    def print_uninstall_plan(
        self, *, remove_state: bool, remove_config: bool, remove_model_cache: bool,
    ) -> None:
        if not self.is_rich:
            print("Uninstall plan:")
            print("  - Remove installer launchers and runtime under ~/.local/share/murmur")
            if remove_state:
                print("  - Remove ~/.local/state/murmur")
            if remove_config:
                print("  - Remove ~/.config/murmur")
            if remove_model_cache:
                print("  - Remove murmur model caches under ~/.cache/huggingface/hub")
            return

        self._console.print()
        self._console.print("  [primary]Uninstall plan:[/primary]")
        self._console.print("    • Remove installer launchers and runtime under ~/.local/share/murmur")
        if remove_state:
            self._console.print("    • Remove ~/.local/state/murmur")
        if remove_config:
            self._console.print("    • Remove ~/.config/murmur")
        if remove_model_cache:
            self._console.print("    • Remove murmur model caches under ~/.cache/huggingface/hub")
        self._console.print()

    def prompt_uninstall_scope(self) -> tuple[bool, bool, bool]:
        if not self.is_rich:
            return self._prompt_uninstall_scope_plain()
        return self._prompt_uninstall_scope_rich()

    def _prompt_uninstall_scope_plain(self) -> tuple[bool, bool, bool]:
        print("Select uninstall scope:")
        print("  1) App/runtime only")
        print("  2) App/runtime + local state/config")
        print("  3) App/runtime + local state/config + model cache")
        while True:
            choice = input("Choice [1-3] (default: 1): ").strip() or "1"
            result = _scope_from_choice(choice)
            if result is not None:
                return result
            print("Invalid choice. Enter 1, 2, or 3.")

    def _prompt_uninstall_scope_rich(self) -> tuple[bool, bool, bool]:
        from rich.prompt import Prompt

        self._console.print()
        self._console.print("  [primary]Select uninstall scope:[/primary]")
        self._console.print("    [muted]1)[/muted] App/runtime only")
        self._console.print("    [muted]2)[/muted] App/runtime + local state/config")
        self._console.print("    [muted]3)[/muted] App/runtime + local state/config + model cache")
        while True:
            choice = Prompt.ask("  Choice", choices=["1", "2", "3"], default="1", console=self._console)
            result = _scope_from_choice(choice)
            if result is not None:
                return result

    def confirm_uninstall(self) -> bool:
        if not self.is_rich:
            response = input("Proceed with uninstall? [y/N]: ").strip().lower()
            return response in {"y", "yes"}

        from rich.prompt import Confirm

        return bool(Confirm.ask("  Proceed with uninstall?", default=False, console=self._console))

    # ── Help formatter ────────────────────────────────────────────────

    def get_help_formatter_class(self) -> type | None:
        """Return RichHelpFormatter when rich output is active, else None."""
        if self.is_rich:
            from rich_argparse import RichHelpFormatter

            RichHelpFormatter.styles["argparse.prog"] = "bold #9d7cd8"
            RichHelpFormatter.styles["argparse.args"] = "#5c9cf5"
            RichHelpFormatter.styles["argparse.groups"] = "bold #fab283"
            return RichHelpFormatter  # type: ignore[no-any-return]
        return None


def _scope_from_choice(choice: str) -> tuple[bool, bool, bool] | None:
    """Map a numeric scope choice string to (remove_state, remove_config, remove_model_cache)."""
    if choice == "1":
        return False, False, False
    if choice == "2":
        return True, True, False
    if choice == "3":
        return True, True, True
    return None


def _component_state(value: str, done_states: set[str]) -> tuple[str, str]:
    """Return (icon, style) for a startup component based on its state value."""
    if value in done_states:
        return "✓", "success"
    return "…", "accent"


def _installed_text(installed: bool) -> Text:
    """Return a Rich Text for installed/available state."""
    if installed:
        return Text(_INSTALLED_LABEL, style="success")
    return Text(_AVAILABLE_LABEL, style="muted")


def _variant_text(variant: Any) -> Text:
    """Return a Rich Text for a model variant's installed/available state."""
    return _installed_text(bool(variant and variant.installed))


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes} B"


# ── Singleton ─────────────────────────────────────────────────────────

_console: MurmurConsole | None = None


def init_console(*, force_plain: bool = False) -> MurmurConsole:
    """Initialize the global MurmurConsole singleton."""
    global _console
    _console = MurmurConsole(force_plain=force_plain)
    return _console


def get_console() -> MurmurConsole:
    """Get the global MurmurConsole, initializing with defaults if needed."""
    global _console
    if _console is None:
        _console = MurmurConsole()
    return _console
