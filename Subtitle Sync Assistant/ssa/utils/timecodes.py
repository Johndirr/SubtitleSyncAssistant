# SPDX-License-Identifier: GPL-3.0-or-later
# Subtitle Sync Assistant
# This file is part of Subtitle Sync Assistant.
#
# Copyright (C) 2025 Subtitle Sync Assistant contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Time parsing/formatting helpers.

Centralized to remove duplication across widgets, dialogs and services.
Keeps SRT-friendly HH:MM:SS,mmm conversions and permissive parse-any.
"""
from __future__ import annotations

from typing import Optional


def format_seconds(seconds: float) -> str:
    """Format seconds into SRT timecode HH:MM:SS,mmm.

    Guarantees rollover and non-negative times.
    """
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ms = 0
        s += 1
        if s == 60:
            s = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_srt(ts: str) -> Optional[float]:
    """Parse HH:MM:SS,mmm into seconds. Returns None on failure."""
    try:
        hms, ms = ts.strip().split(",")
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except Exception:
        return None


def normalize_srt(ts: str) -> Optional[str]:
    """Normalize an input time string into HH:MM:SS,mmm if possible.

    Accepts variants like HH:MM:SS.mmm or HH:MM:SS and pads milliseconds.
    Returns None if it cannot be normalized.
    """
    t = (ts or "").strip()
    if not t:
        return None
    if "." in t and "," not in t:
        a, b = t.rsplit(".", 1)
        if b.isdigit() and 1 <= len(b) <= 3:
            t = a + "," + b.ljust(3, "0")
    if "," not in t and t.count(":") == 2:
        t += ",000"
    if "," in t:
        a, b = t.split(",", 1)
        if not b.isdigit():
            return None
        t = a + "," + b[:3].ljust(3, "0")
    return t


def parse_any(text: str) -> Optional[float]:
    """Parse either an SRT timecode or a float seconds value.

    Accepts HH:MM:SS,mmm or HH:MM:SS.mmm or numeric seconds with comma/point.
    """
    t = (text or "").strip()
    if not t:
        return None
    cand = t
    if "," not in cand and cand.count(":") >= 2:
        if "." in cand and cand.rsplit(".", 1)[1].isdigit():
            cand = cand.replace(".", ",", 1)
        else:
            cand += ",000"
    if cand.count(":") >= 2 and "," in cand:
        v = parse_srt(cand)
        if v is not None:
            return v
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None
