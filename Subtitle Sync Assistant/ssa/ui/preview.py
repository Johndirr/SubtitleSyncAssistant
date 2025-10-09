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

"""Small window that shows side-by-side video preview images for two media files.

Images are extracted on-demand at a requested timestamp using ffmpeg, with:
- Debounce/caching per 0.5s bucket per file
- Background extraction in QThread to keep UI responsive
- Simple scaling to a target width (default 320px)

This module purposefully avoids adding runtime dependencies and relies on
ffmpeg being available on PATH (already required by pydub elsewhere).
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple, Dict

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy


def _bucket_time(t: Optional[float], bucket: float = 0.5) -> Optional[float]:
    """Quantize a timestamp to a fixed-size bucket.

    This reduces the number of distinct preview frames we fetch/cached and
    improves cache hits when the user moves selection rapidly.
    """
    if t is None:
        return None
    try:
        return round(t / bucket) * bucket
    except Exception:
        return None


class _FrameGrabWorker(QObject):
    """Background worker to extract a single frame using ffmpeg.

    Emits:
    - finished(path, bucket_time, QPixmap) on success
    - failed(path, bucket_time, error_message) on failure

    Notes:
    - Uses blocking subprocess.run to keep implementation simple.
    - The worker runs inside its own QThread provided by the caller.
    """

    finished = pyqtSignal(str, float, QPixmap)  # path, bucket_time, pixmap
    failed = pyqtSignal(str, float, str)        # path, bucket_time, error

    def __init__(self, media_path: str, time_s: float, width: int = 320):
        """Create the worker for a specific media path/time/width."""
        super().__init__()
        self.media_path = media_path
        self.time_s = float(max(0.0, time_s))
        self.width = width

    def run(self):
        """Invoke ffmpeg to grab one frame and emit a QPixmap (or failure)."""
        # Build ffmpeg command to grab one frame as PNG to stdout
        if not os.path.exists(self.media_path):
            self.failed.emit(self.media_path, _bucket_time(self.time_s) or 0.0, "Media not found")
            return
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-nostdin",
            "-ss", f"{self.time_s:.3f}",
            "-i", self.media_path,
            "-frames:v", "1",
            "-vf", f"scale={self.width}:-1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            self.failed.emit(self.media_path, _bucket_time(self.time_s) or 0.0, "ffmpeg not found in PATH")
            return
        if proc.returncode != 0 or not proc.stdout:
            msg = proc.stderr.decode(errors="ignore") if proc.stderr else "Unknown ffmpeg error"
            self.failed.emit(self.media_path, _bucket_time(self.time_s) or 0.0, msg)
            return
        pix = QPixmap()
        if not pix.loadFromData(proc.stdout, "PNG"):
            self.failed.emit(self.media_path, _bucket_time(self.time_s) or 0.0, "Could not decode image")
            return
        self.finished.emit(self.media_path, _bucket_time(self.time_s) or 0.0, pix)


class PreviewImagesWindow(QWidget):
    """Tool window showing side-by-side previews for Reference and New media.

    High level behavior:
    - Exposed API show_for_selection() is called with paths and timestamps.
    - For each side, we check cache, otherwise spawn a worker thread to fetch.
    - A small in-memory cache avoids redundant ffmpeg invocations.
    - Images are rescaled on window resize to keep them fitting nicely.
    """

    def __init__(self, parent=None, thumb_width: int = 320):
        """Create the window, UI elements, and internal caches/state."""
        super().__init__(parent)
        self.setWindowTitle("Preview Images")
        # Make it a separate tool window (floats over the main app)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        # Keep window around on close; let owner show/hide when needed
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.thumb_width = thumb_width

        # Cache: {(path, bucket_time): QPixmap}
        self._cache: Dict[Tuple[str, float], QPixmap] = {}

        # Track which cache entry is currently shown per side (for rescaling)
        self._ref_key: Optional[Tuple[str, float]] = None
        self._new_key: Optional[Tuple[str, float]] = None

        # UI -----------------------------------------------------------------
        root = QVBoxLayout(self)
        row = QHBoxLayout(); root.addLayout(row)

        # Reference panel ----------------------------------------------------
        self.ref_title = QLabel("Reference", self)
        self.ref_title.setAlignment(Qt.AlignCenter)
        self.ref_img = QLabel(self)
        self.ref_img.setAlignment(Qt.AlignCenter)
        self.ref_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ref_img.setMinimumSize(self.thumb_width, int(self.thumb_width * 9 / 16))
        self.ref_time = QLabel("--:--:--,---", self); self.ref_time.setAlignment(Qt.AlignCenter)
        left = QVBoxLayout(); left.addWidget(self.ref_title); left.addWidget(self.ref_img); left.addWidget(self.ref_time)

        # New panel ----------------------------------------------------------
        self.new_title = QLabel("New", self)
        self.new_title.setAlignment(Qt.AlignCenter)
        self.new_img = QLabel(self)
        self.new_img.setAlignment(Qt.AlignCenter)
        self.new_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.new_img.setMinimumSize(self.thumb_width, int(self.thumb_width * 9 / 16))
        self.new_time = QLabel("--:--:--,---", self); self.new_time.setAlignment(Qt.AlignCenter)
        right = QVBoxLayout(); right.addWidget(self.new_title); right.addWidget(self.new_img); right.addWidget(self.new_time)

        row.addLayout(left); row.addLayout(right)

        # Reasonable minimum so it is always visible
        self.setMinimumWidth(self.thumb_width * 2 + 48)
        self.setMinimumHeight(int(self.thumb_width * 9 / 16) + 96)

        # State (paths + active worker threads) ------------------------------
        self._ref_path: Optional[str] = None
        self._new_path: Optional[str] = None
        self._ref_thread: Optional[QThread] = None
        self._new_thread: Optional[QThread] = None
        self._ref_worker: Optional[_FrameGrabWorker] = None
        self._new_worker: Optional[_FrameGrabWorker] = None

    def set_media_paths(self, ref_path: Optional[str], new_path: Optional[str]):
        """Set the file system paths for reference and new media files."""
        self._ref_path = ref_path
        self._new_path = new_path
        self.ref_title.setText(f"Reference\n{os.path.basename(ref_path) if ref_path else '(none)'}")
        self.new_title.setText(f"New\n{os.path.basename(new_path) if new_path else '(none)'}")

    def show_for_selection(
        self,
        ref_path: Optional[str],
        ref_time: Optional[float],
        new_path: Optional[str],
        new_time: Optional[float],
        ref_line: Optional[int] = None,
        new_line: Optional[int] = None,
    ):
        """Show the window (if hidden) and render previews for the selection."""
        self.set_media_paths(ref_path, new_path)
        self.update_images(ref_time, new_time, ref_line=ref_line, new_line=new_line)
        if not self.isVisible():
            self.show()
            self.raise_(); self.activateWindow()

    def update_images(
        self,
        ref_time: Optional[float],
        new_time: Optional[float],
        ref_line: Optional[int] = None,
        new_line: Optional[int] = None,
    ):
        """Refresh both sides using the provided timestamps (seconds)."""
        # Reference side
        self._update_one(
            side="ref",
            media_path=self._ref_path,
            time_s=ref_time,
            line_no=ref_line,
            img_label=self.ref_img,
            time_label=self.ref_time,
        )
        # New side
        self._update_one(
            side="new",
            media_path=self._new_path,
            time_s=new_time,
            line_no=new_line,
            img_label=self.new_img,
            time_label=self.new_time,
        )

    def _update_one(
        self,
        side: str,
        media_path: Optional[str],
        time_s: Optional[float],
        line_no: Optional[int],
        img_label: QLabel,
        time_label: QLabel,
    ):
        """Render one side (ref/new): try cache then fall back to worker fetch."""
        # Update time text under the image (prefix with line number if provided)
        if time_s is None:
            label_text = "(no selection)"
        else:
            t = self._format_time(time_s)
            label_text = f"Line {line_no + 1}: {t}" if line_no is not None else t
        time_label.setText(label_text)

        # Validate inputs early and show a neutral placeholder
        if not media_path or time_s is None or not os.path.exists(media_path):
            img_label.setText("(no image)")
            img_label.setPixmap(QPixmap())
            return

        b = _bucket_time(time_s)
        if b is None:
            img_label.setText("(no image)")
            img_label.setPixmap(QPixmap())
            return
        key = (media_path, b)
        # If we already have this frame, scale to current label size and show it
        if key in self._cache:
            base = self._cache[key]
            img_label.setText("")
            img_label.setPixmap(base.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if side == "ref":
                self._ref_key = key
            else:
                self._new_key = key
            return
        # No cache: cancel any running worker on this side (avoids overlap)
        self._cancel_thread(side)
        # Show a loading hint while the worker fetches the frame
        img_label.setText("(loading...)")
        img_label.setPixmap(QPixmap())
        # Spawn worker thread; request width close to current label width
        req_w = max(160, img_label.width()) or self.thumb_width
        worker = _FrameGrabWorker(media_path, b, req_w)
        thread = QThread(self)
        worker.moveToThread(thread)
        # Wire success/failure to UI handlers
        worker.finished.connect(lambda path, bt, pix, s=side: self._on_frame_ready(s, path, bt, pix))
        worker.failed.connect(lambda path, bt, err, s=side: self._on_frame_failed(s, path, bt, err))
        thread.started.connect(worker.run)
        thread.start()
        if side == "ref":
            self._ref_thread = thread
            self._ref_worker = worker
        else:
            self._new_thread = thread
            self._new_worker = worker

    def _cancel_thread(self, side: str):
        """Stop and clear any worker thread running for the given side."""
        th = self._ref_thread if side == "ref" else self._new_thread
        if th is not None:
            try:
                th.quit(); th.wait(200)
            except Exception:
                pass
        if side == "ref":
            self._ref_thread = None
            self._ref_worker = None
        else:
            self._new_thread = None
            self._new_worker = None

    def _on_frame_ready(self, side: str, path: str, bucket_time: float, pix: QPixmap):
        """Receive a fetched pixmap, cache it, and display it on the proper side."""
        key = (path, bucket_time)
        self._cache[key] = pix
        scaled = pix.scaled(self.ref_img.size() if side == "ref" else self.new_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if side == "ref":
            self._ref_key = key
            self.ref_img.setText("")
            self.ref_img.setPixmap(scaled)
            self._cancel_thread("ref")
        else:
            self._new_key = key
            self.new_img.setText("")
            self.new_img.setPixmap(scaled)
            self._cancel_thread("new")

    def _on_frame_failed(self, side: str, path: str, bucket_time: float, err: str):
        """Display an error placeholder when frame extraction fails."""
        label = self.ref_img if side == "ref" else self.new_img
        label.setText("(preview unavailable)")
        label.setPixmap(QPixmap())
        self._cancel_thread(side)

    def resizeEvent(self, event):
        """Rescale currently displayed images when the window is resized."""
        try:
            if self._ref_key and self._ref_key in self._cache:
                base = self._cache[self._ref_key]
                self.ref_img.setPixmap(base.scaled(self.ref_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if self._new_key and self._new_key in self._cache:
                base = self._cache[self._new_key]
                self.new_img.setPixmap(base.scaled(self.new_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        finally:
            super().resizeEvent(event)

    @staticmethod
    def _format_time(seconds: Optional[float]) -> str:
        """Format seconds to HH:MM:SS,mmm; keeps negative clamped to zero."""
        if seconds is None:
            return "--:--:--,---"
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms == 1000:
            ms = 0; s += 1
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def closeEvent(self, event):
        """Cancel running frame-grab threads before the window closes."""
        try:
            self._cancel_thread("ref")
        except Exception:
            pass
        try:
            self._cancel_thread("new")
        except Exception:
            pass
        super().closeEvent(event)
