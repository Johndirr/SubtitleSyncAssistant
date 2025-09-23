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

"""Offset workers extracted from the monolith.

OffsetWorker: correlates new snippet to reference wave.
SlidingOffsetWorker: uses a per-row shifted reference window.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from ..services.audio import resample_linear  # reuse shared helper

try:
    from audio_offset_finder.audio_offset_finder import find_offset_between_buffers
except ImportError:  # keep runtime behavior the same as monolith
    find_offset_between_buffers = None  # type: ignore


class OffsetWorker(QObject):
    """Find offsets by correlating each new snippet against a reference wave."""

    progress = pyqtSignal(int, str)
    result = pyqtSignal(int, object, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        ref_wave: np.ndarray,
        ref_sr: int,
        new_wave: np.ndarray,
        new_sr: int,
        rows: List[Tuple[int, float, float]],
        min_duration: float = 1.0,
        ref_offset_sec: float = 0.0,
    ):
        """Initialize worker with reference/new waves and row time windows.

        rows is a list of (row_index, start_sec, end_sec) in "new" timebase.
        """
        super().__init__()
        self.ref_wave = ref_wave.astype(np.float32)
        self.ref_sr = ref_sr
        self.new_wave = new_wave.astype(np.float32)
        self.new_sr = new_sr
        self.rows = rows
        self.min_duration = min_duration
        self.ref_offset_sec = ref_offset_sec
        self._abort = False

    def abort(self):
        """Request cooperative cancellation."""
        self._abort = True

    def _check_abort(self):
        """Raise sentinel on cancellation to unwind run() quickly."""
        if self._abort:
            raise RuntimeError("__ABORT__")

    def run(self):
        """Compute a time delta per requested row and emit results.

        For each row window in the new wave, we optionally resample to the
        reference sample rate, then call audio-offset-finder to get a global
        time offset for the snippet relative to ref_wave. Finally, we compute
        delta such that start_sec - delta aligns the subtitle to the reference.
        """
        if find_offset_between_buffers is None:
            self.failed.emit("audio-offset-finder not installed (pip install audio-offset-finder)")
            return
        try:
            total = len(self.rows)
            for seq, (idx, start_sec, end_sec) in enumerate(self.rows, start=1):
                self._check_abort()
                dur = end_sec - start_sec
                if dur <= 0 or dur < self.min_duration:
                    self.result.emit(idx, None, "too short")
                    continue
                start_i = int(start_sec * self.new_sr)
                end_i = min(int(end_sec * self.new_sr), self.new_wave.shape[0])
                if end_i <= start_i:
                    self.result.emit(idx, None, "empty")
                    continue
                snippet = self.new_wave[start_i:end_i]
                if self.new_sr != self.ref_sr:
                    snippet = resample_linear(snippet, self.new_sr, self.ref_sr)
                try:
                    res = find_offset_between_buffers(self.ref_wave, snippet, self.ref_sr)
                except Exception as ex:
                    self.result.emit(idx, None, f"err:{ex}")
                    continue
                if not isinstance(res, dict) or "time_offset" not in res:
                    self.result.emit(idx, None, "bad-result")
                    continue
                # time_offset is the absolute time in reference where snippet best matches
                time_offset_global = float(res["time_offset"]) + self.ref_offset_sec
                # We want delta to add to each row's time to align with the ref timeline
                delta = start_sec - time_offset_global
                self.result.emit(idx, delta, "ok")
                self.progress.emit(idx, f"{seq}/{total} (row {idx+1})")
            self.finished.emit()
        except RuntimeError as ex:
            if str(ex) == "__ABORT__":
                self.cancelled.emit()
            else:
                self.failed.emit(str(ex))
        except Exception as e:
            self.failed.emit(str(e))


class SlidingOffsetWorker(QObject):
    """Find per-row offsets using row-specific reference windows.

    Each row is matched against its own ref window created by sliding a
    base window along the reference to keep similar acoustic content.
    """

    progress = pyqtSignal(int, str)
    result = pyqtSignal(int, object, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        new_wave: np.ndarray,
        new_sr: int,
        rows: List[Tuple[int, float, float]],
        ref_windows: List[Tuple[np.ndarray, int, float]],  # (ref_slice, ref_sr, ref_offset_sec)
        min_duration: float = 1.0,
    ):
        """Initialize worker with new wave, rows, and per-row reference windows."""
        super().__init__()
        self.new_wave = new_wave.astype(np.float32)
        self.new_sr = new_sr
        self.rows = rows
        self.ref_windows = ref_windows
        self.min_duration = min_duration
        self._abort = False

    def abort(self):
        """Request cooperative cancellation."""
        self._abort = True

    def _check_abort(self):
        """Raise sentinel on cancellation to unwind run() quickly."""
        if self._abort:
            raise RuntimeError("__ABORT__")

    def run(self):
        """Compute offsets using per-row reference windows (sliding range mode)."""
        if find_offset_between_buffers is None:
            self.failed.emit("audio-offset-finder not installed (pip install audio-offset-finder)")
            return
        try:
            total = len(self.rows)
            for seq, ((idx, start_sec, end_sec), (ref_slice, ref_sr, ref_offset_sec)) in enumerate(zip(self.rows, self.ref_windows), start=1):
                self._check_abort()
                dur = end_sec - start_sec
                if dur <= 0 or dur < self.min_duration:
                    self.result.emit(idx, None, "too short")
                    continue
                start_i = int(start_sec * self.new_sr)
                end_i = min(int(end_sec * self.new_sr), self.new_wave.shape[0])
                if end_i <= start_i:
                    self.result.emit(idx, None, "empty")
                    continue
                snippet = self.new_wave[start_i:end_i]
                if self.new_sr != ref_sr:
                    snippet = resample_linear(snippet, self.new_sr, ref_sr)
                try:
                    res = find_offset_between_buffers(ref_slice.astype(np.float32), snippet, ref_sr)
                except Exception as ex:
                    self.result.emit(idx, None, f"err:{ex}")
                    continue
                if not isinstance(res, dict) or "time_offset" not in res:
                    self.result.emit(idx, None, "bad-result")
                    continue
                time_offset_global = float(res["time_offset"]) + ref_offset_sec
                delta = start_sec - time_offset_global
                self.result.emit(idx, delta, "ok")
                self.progress.emit(idx, f"{seq}/{total} (row {idx+1})")
            self.finished.emit()
        except RuntimeError as ex:
            if str(ex) == "__ABORT__":
                self.cancelled.emit()
            else:
                self.failed.emit(str(ex))
        except Exception as e:
            self.failed.emit(str(e))
