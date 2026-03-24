"""Murmur logo display with terminal image protocol support and ASCII fallback."""

from __future__ import annotations

import base64
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

# Small 200px-wide PNG of the murmur banner, base64-encoded (~7KB).
_LOGO_PNG_B64 = (
    "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"
    "TOb QmZstoCflKzgRbXaieJkb2qFwFxl6PiWlrmEpygiyLb5mh+aLii+LGLJMjMIjQSsurVaV"
    "qDREuU1tUVkavTgVTzTIOlGTjI9ucEGVR11MQVGVUh/QsUgiceILJtIeWmEkNkKbUgnKRkJpl"
    "giK7tCo7AW2lna3uRvaQJ3yZ/m55SeLKvEjyrcXZD/BR1PJxuhUduI6Zbx4gtsQUh15UhOlfg"
    "wXRnEvwAVAQY+ChU/7OKQmbusCskFkbmVOQMb02oI2ymSSRm+syDod9TAzYk8lVaQYAiXAeiO"
    "y207k0hGkW+nQCq6RmlycZwlG2LMjlrviJF1yqVUa4kDLCYgMwk+sf2qXyujsB5ydibkFIEk"
    "X8mhqQVYLiS12w6SgXPcdAj7fWoiH0WA1Gm2KyIL14Cig+Gj5Utgg8leH7mqqkqSYllR8eAo"
    "tqI3S0jbfJHAMX7iif1DncLNj5kU+L8kOCYm0sBTt8UXSKRDIYsbomwkEsKYEQ/o6j9txpmkK"
    "nkN3pgBzhAiO/x+fHRe9gkyBzAAAAABJRU5ErkJggg=="
)


def _supports_iterm_images() -> bool:
    """Check for iTerm2 or WezTerm inline image protocol support."""
    term_program = os.environ.get("TERM_PROGRAM", "")
    return term_program.lower() in {"iterm.app", "iterm2", "wezterm"}


def _supports_kitty_images() -> bool:
    """Check for Kitty graphics protocol support."""
    term = os.environ.get("TERM", "")
    return "kitty" in term.lower()


def _render_iterm_image() -> str:
    """Render logo using iTerm2/WezTerm inline image protocol (OSC 1337)."""
    png_data = base64.b64decode(_LOGO_PNG_B64)
    b64 = base64.b64encode(png_data).decode("ascii")
    return f"\033]1337;File=inline=1;width=40;preserveAspectRatio=1:{b64}\a\n"


def _render_kitty_image() -> str:
    """Render logo using Kitty graphics protocol."""
    png_data = base64.b64decode(_LOGO_PNG_B64)
    b64 = base64.b64encode(png_data).decode("ascii")
    # Kitty protocol: transmit PNG data and display immediately
    chunks = [b64[i : i + 4096] for i in range(0, len(b64), 4096)]
    result = ""
    for i, chunk in enumerate(chunks):
        m = 1 if i < len(chunks) - 1 else 0
        if i == 0:
            result += f"\033_Ga=T,f=100,t=d,m={m};{chunk}\033\\"
        else:
            result += f"\033_Gm={m};{chunk}\033\\"
    return result + "\n"


def print_image_logo() -> bool:
    """Try to print the logo as a terminal image. Returns True if successful."""
    if not sys.stdout.isatty():
        return False

    try:
        if _supports_iterm_images():
            sys.stdout.write(_render_iterm_image())
            sys.stdout.flush()
            return True

        if _supports_kitty_images():
            sys.stdout.write(_render_kitty_image())
            sys.stdout.flush()
            return True
    except Exception:
        pass

    return False


def print_rich_logo(console: Console) -> None:
    """Print the ASCII art murmur logo with Rich gradient styling."""
    from rich.text import Text

    # Unicode block art spelling "murmur"
    lines = [
        "  ███╗   ███╗██╗   ██╗██████╗ ███╗   ███╗██╗   ██╗██████╗ ",
        "  ████╗ ████║██║   ██║██╔══██╗████╗ ████║██║   ██║██╔══██╗",
        "  ██╔████╔██║██║   ██║██████╔╝██╔████╔██║██║   ██║██████╔╝",
        "  ██║╚██╔╝██║██║   ██║██╔══██╗██║╚██╔╝██║██║   ██║██╔══██╗",
        "  ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║ ╚═╝ ██║╚██████╔╝██║  ██║",
        "  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝",
    ]

    console.print()
    for line in lines:
        text = Text(line)
        text.stylize("bold #9d7cd8")
        console.print(text)


def print_logo(console: Console) -> None:
    """Print the murmur logo, preferring terminal image protocols with ASCII fallback."""
    if not print_image_logo():
        print_rich_logo(console)
