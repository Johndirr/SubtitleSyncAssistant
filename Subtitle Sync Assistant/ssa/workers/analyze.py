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

"""AnalyzeWorker extracted from the monolith.

Loads two media files (pydub), prepares decimated display arrays,
parses SRT with pysrt, and returns data back via signals.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pysrt
from pydub import AudioSegment
from PyQt5.QtCore import QObject, pyqtSignal


class AnalyzeWorker(QObject):
    """Background worker to prepare waveform data and read subtitles."""

    # progress: (percent, message)
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, ref_media_path: str, new_media_path: str, srt_path: str):
        """Construct worker with file paths to process."""
        super().__init__()
        self.ref_media_path = ref_media_path
        self.new_media_path = new_media_path
        self.srt_path = srt_path
        self._abort = False

    def abort(self):
        """Request cooperative cancellation."""
        self._abort = True

    def _check_abort(self):
        """Raise a sentinel to unwind the run() flow if abort was requested."""
        if self._abort:
            self.cancelled.emit()
            raise RuntimeError("__ABORT__")

    def _load_audio(self, path: str):
        """Load file and create a lightweight display array without touching quality.

        Returns a tuple (display_array, display_sr, full_segment).
        """
        audio = AudioSegment.from_file(path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).T  # (channels, n)
        # Normalize to [-1, 1] based on sample width
        samples /= float(2 ** (8 * audio.sample_width - 1))
        original_sr = audio.frame_rate

        # Optional decimation for plotting performance; full quality kept in segment
        if samples.size > 5_000_000:
            factor = 10
            display = samples[::factor] if samples.ndim == 1 else samples[:, ::factor]
            display_sr = max(1, original_sr // factor)
        else:
            display = samples
            display_sr = original_sr
        return display, display_sr, audio

    def run(self):
        """Execute the analysis, emitting progress and final result dict."""
        try:
            self.progress.emit(0, "Starting analysis...")
            self._check_abort()

            self.progress.emit(10, "Loading reference media...")
            ref_display, ref_rate, ref_full = self._load_audio(self.ref_media_path)
            self._check_abort()

            self.progress.emit(30, "Loading new media...")
            new_display, new_rate, new_full = self._load_audio(self.new_media_path)
            self._check_abort()

            self.progress.emit(50, "Parsing subtitles...")
            subs = pysrt.open(self.srt_path, encoding='utf-8')
            self._check_abort()

            self.progress.emit(65, "Extracting subtitle rows...")
            rows = [{"start": s.start, "end": s.end, "text": s.text} for s in subs]
            self._check_abort()

            self.progress.emit(80, "Building intervals...")
            intervals = []
            for r in rows:
                st, et = r["start"], r["end"]
                start_sec = st.hours * 3600 + st.minutes * 60 + st.seconds + st.milliseconds / 1000.0
                end_sec = et.hours * 3600 + et.minutes * 60 + et.seconds + et.milliseconds / 1000.0
                intervals.append((start_sec, end_sec))
            self._check_abort()

            self.progress.emit(95, "Finalizing...")
            result = {
                "ref_display": ref_display,
                "ref_rate": ref_rate,
                "ref_full": ref_full,
                "new_display": new_display,
                "new_rate": new_rate,
                "new_full": new_full,
                "rows": rows,
                "intervals": intervals,
            }
            self.progress.emit(100, "Done")
            self.finished.emit(result)

        except RuntimeError as ex:
            # __ABORT__ is internal; map to cancelled
            if str(ex) != "__ABORT__":
                self.failed.emit(f"Aborted: {ex}")
        except Exception as e:
            self.failed.emit(str(e))
