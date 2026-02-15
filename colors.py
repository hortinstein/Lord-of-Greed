"""
Terminal color utilities using ANSI escape codes.

Respects the NO_COLOR environment variable (https://no-color.org/).
Set NO_COLOR=1 to disable all color output.

No external dependencies required.
"""
from __future__ import annotations

import os
import sys


def _color_enabled() -> bool:
    """Return True if color output should be used."""
    if os.environ.get("NO_COLOR", ""):
        return False
    # Disable color if stdout is not a TTY (e.g. piped to a file)
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


COLOR = _color_enabled()

# ── ANSI escape sequences ────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"

# Foreground colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright foreground colors
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"


# ── Helper functions ─────────────────────────────────────────────────

def _wrap(code: str, text: str) -> str:
    if not COLOR:
        return text
    return f"{code}{text}{RESET}"


def bold(text: str) -> str:
    return _wrap(BOLD, text)


def dim(text: str) -> str:
    return _wrap(DIM, text)


def red(text: str) -> str:
    return _wrap(RED, text)


def green(text: str) -> str:
    return _wrap(GREEN, text)


def yellow(text: str) -> str:
    return _wrap(YELLOW, text)


def blue(text: str) -> str:
    return _wrap(BLUE, text)


def magenta(text: str) -> str:
    return _wrap(MAGENTA, text)


def cyan(text: str) -> str:
    return _wrap(CYAN, text)


def bright_red(text: str) -> str:
    return _wrap(BRIGHT_RED, text)


def bright_green(text: str) -> str:
    return _wrap(BRIGHT_GREEN, text)


def bright_yellow(text: str) -> str:
    return _wrap(BRIGHT_YELLOW, text)


def bright_cyan(text: str) -> str:
    return _wrap(BRIGHT_CYAN, text)


# ── Composite styles ─────────────────────────────────────────────────

def header(text: str) -> str:
    """Bold + cyan for section headers."""
    if not COLOR:
        return text
    return f"{BOLD}{CYAN}{text}{RESET}"


def subheader(text: str) -> str:
    """Bold + white for sub-section headers."""
    if not COLOR:
        return text
    return f"{BOLD}{WHITE}{text}{RESET}"


def success(text: str) -> str:
    """Bright green for success messages."""
    return bright_green(text)


def error(text: str) -> str:
    """Bright red for error messages."""
    return bright_red(text)


def price_gain(text: str) -> str:
    """Green for price increases."""
    return green(text)


def price_loss(text: str) -> str:
    """Red for price decreases."""
    return red(text)


def price_change(value: float, text: str) -> str:
    """Color text green if value > 0, red if < 0, dim if zero."""
    if value > 0:
        return price_gain(text)
    elif value < 0:
        return price_loss(text)
    return dim(text)


def rarity_color(rarity: str, text: str) -> str:
    """Color text based on card rarity."""
    r = rarity.strip().lower()
    if r == "unique":
        return _wrap(BRIGHT_YELLOW, text)
    elif r == "elite":
        return _wrap(BRIGHT_MAGENTA, text)
    elif r == "exceptional":
        return _wrap(BRIGHT_CYAN, text)
    elif r == "ordinary":
        return _wrap(WHITE, text)
    return text
