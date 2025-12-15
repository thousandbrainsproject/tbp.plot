# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass

Color = str | tuple[float, float, float] | tuple[float, float, float, float]


def hex_to_rgb(hex_str: str, alpha: float | None = None) -> Color:
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) != 6:
        raise ValueError(f"Expected 6 hex digits, got: {hex_str!r}")

    r = int(hex_clean[0:2], 16) / 255.0
    g = int(hex_clean[2:4], 16) / 255.0
    b = int(hex_clean[4:6], 16) / 255.0

    if alpha is None:
        return (r, g, b)
    return (r, g, b, alpha)


@dataclass(frozen=True)
class Palette:
    """The TBP color palette.

    If you request a color that doesn't exist, a KeyError is raised with a list of
    available names.
    """

    # Primary Colors
    indigo: str = "#2f2b5c"
    numenta_blue: str = "#00a0df"

    # Secondary Colors
    bossanova: str = "#5c315f"
    vivid_violet: str = "#86308b"
    blue_violet: str = "#655eb2"
    amethyst: str = "#915acc"

    # Accent Colors/Shades
    rich_black: str = "#000000"
    charcoal: str = "#3f3f3f"
    link_water: str = "#dfe6f5"

    # ---------- Internal helper ----------
    @classmethod
    def _validate(cls, name: str) -> str:
        if not hasattr(cls, name):
            available = [k for k in cls.__dict__.keys() if not k.startswith("_")]
            msg = (
                f"Color '{name}' is not defined in Palette.\n"
                f"Available colors: {', '.join(available)}"
            )
            raise KeyError(msg)
        return getattr(cls, name)

    # ---------- Public API ----------
    @classmethod
    def as_hex(cls, name: str) -> Color:
        """Return the raw hex string for a color name."""
        return cls._validate(name)

    @classmethod
    def as_rgb(cls, name: str, alpha: float | None = None) -> Color:
        """Return the color as an RGB(A) tuple in [0,1] range."""
        hex_str = cls._validate(name)
        return hex_to_rgb(hex_str, alpha=alpha)
