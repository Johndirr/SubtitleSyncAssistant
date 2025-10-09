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

"""MatplotlibPlotWidget extracted from the monolith.

Behavior preserved; comments added and tiny helpers reused from utils.

This widget provides:
- A scrollable/zoomed view of a mono waveform for quick visual inspection
- Subtitle interval overlays and selection highlighting
- Optimized audio playback with larger buffers for smooth performance

It purposely keeps playback state independent so two instances can run
side-by-side in the main window without interfering with each other.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QByteArray, QBuffer, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QSlider, QWidget, QSizePolicy, QStyle

from pydub import AudioSegment


class MatplotlibPlotWidget(QFrame):
    """Scrollable waveform viewer with optimized audio playback.

    Signals
    -------
    playingChanged: bool
        Emitted when playback starts/stops so the other plot can stop itself.
    """

    playingChanged = pyqtSignal(bool)

    def __init__(self, title: str = "Plot", window_duration: int = 20):
        """Create the plot widget.

        Parameters
        ----------
        title: str
            Title shown centered above the plot.
        window_duration: int
            Duration (seconds) of the visible horizontal window.
        """
        super().__init__()
        # Frame makes it visually distinct in the main UI
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setMinimumHeight(200)
        self.setMaximumHeight(220)

        # Plot/runtime state ------------------------------------------------
        self.window_duration = window_duration
        self.samples: Optional[np.ndarray] = None          # original array (mono or multi)
        self.samples_mono: Optional[np.ndarray] = None     # mono view used for plotting
        self.sr: Optional[int] = None                      # sample rate (Hz)
        self.total_duration: float = 0.0                   # seconds

        self.subtitle_intervals: List[Tuple[float, float]] = []  # [(start_s, end_s)]
        self.selected_subtitle_indices: set[int] = set()         # indices selected in the table

        # Minimal playback state (we play the original loaded AudioSegment)
        self.audio_segment: Optional[AudioSegment] = None
        self.audio_output: Optional[QAudioOutput] = None
        self.audio_buffer: Optional[QBuffer] = None
        self.audio_data: Optional[QByteArray] = None
        self.playhead_sec: float = 0.0
        self._play_origin_sec: float = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS UI refresh
        self._timer.timeout.connect(self._on_tick)

        # Layout ------------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top bar: left control cluster + centered title + right spacer
        top_bar = QHBoxLayout(); top_bar.setContentsMargins(4, 0, 4, 0); top_bar.setSpacing(6)
        # Left controls
        left_controls = QWidget(self)
        lc_layout = QHBoxLayout(left_controls); lc_layout.setContentsMargins(0, 0, 0, 0); lc_layout.setSpacing(6)

        # Button with native style icons (robust to fonts)
        self.play_btn = QPushButton("", self)
        self.play_btn.setCheckable(True); self.play_btn.setFixedSize(28, 22)
        self.play_btn.setToolTip("Play/Pause audio from playhead. Right-click waveform to set playhead.")
        self.play_btn.toggled.connect(self._on_play_toggled)
        # A tiny top padding keeps most platform themes visually centered
        self.play_btn.setStyleSheet("QPushButton { padding-top: 2px; padding-bottom: 0px; }")
        self._update_play_glyph(False)

        # Current playhead position label
        self.pos_lbl = QLabel("00:00:00,000", self); self.pos_lbl.setToolTip("Current playhead position")

        # Vertical amplitude zoom (just scales the y-limits)
        amp_lbl = QLabel("Amp:", self); amp_lbl.setToolTip("Visible +/- amplitude (vertical zoom of normalized waveform).")
        self.amp_spin = QDoubleSpinBox(self); self.amp_spin.setRange(0.05, 2.00); self.amp_spin.setSingleStep(0.05); self.amp_spin.setDecimals(2); self.amp_spin.setValue(1.05)
        self.amp_spin.setToolTip("Adjust vertical zoom (does not alter data).")
        self.amp_spin.valueChanged.connect(self._on_amp_changed)

        lc_layout.addWidget(self.play_btn)
        lc_layout.addWidget(self.pos_lbl)
        lc_layout.addWidget(amp_lbl)
        lc_layout.addWidget(self.amp_spin)
        top_bar.addWidget(left_controls)

        # Centered title
        top_bar.addStretch()
        self.label = QLabel(title, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar.addWidget(self.label)
        top_bar.addStretch()

        # Right controls: total length
        right_controls = QWidget(self)
        rc_layout = QHBoxLayout(right_controls); rc_layout.setContentsMargins(0, 0, 0, 0); rc_layout.setSpacing(6)
        self.total_lbl = QLabel("Total: --:--:--,---", self)
        self.total_lbl.setToolTip("Total audio length")
        self.total_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rc_layout.addWidget(self.total_lbl)
        top_bar.addWidget(right_controls)

        # Keep title centered by making left/right clusters same min width
        try:
            w = max(left_controls.sizeHint().width(), right_controls.sizeHint().width())
            left_controls.setMinimumWidth(w)
            right_controls.setMinimumWidth(w)
        except Exception:
            pass

        layout.addLayout(top_bar)

        # The actual plot canvas
        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Horizontal scroll/position slider (in seconds)
        self.slider = QSlider(Qt.Horizontal); self.slider.setMinimum(0); self.slider.setMaximum(0); self.slider.setValue(0); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._update_plot_from_slider)
        layout.addWidget(self.slider)

        # Internal drag state used for mouse-driven panning
        self._dragging = False
        self._drag_start_x_pixel = None
        self._drag_start_slider = None
        self._amp_limit = 1.05  # current half-range of y-axis (vertical zoom)

        # Mouse interactions: right-click sets playhead; left-drag pans the view
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # Public API ---------------------------------------------------------
    def plot_waveform(self, samples: np.ndarray, sr: int):
        """Provide waveform data for plotting.

        The input can be mono or multi-channel. If multi-channel, a mono
        representation is derived by averaging channels for visualization.
        """
        self.samples = samples
        self.sr = sr
        self.samples_mono = samples.mean(axis=0) if samples.ndim > 1 else samples
        total_len = len(self.samples_mono)
        self.total_duration = (total_len - 1) / sr if sr > 0 and total_len > 0 else 0.0

        # Enable panning only if the full audio is longer than our window
        if self.total_duration > self.window_duration:
            self.slider.setMaximum(int(max(0, self.total_duration - self.window_duration)))
            self.slider.setEnabled(True)
        else:
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
        self.slider.setValue(0)

        # Update total label (prefer audio_segment if present)
        self._update_total_label()

        self._plot_window(0)

    def set_subtitle_intervals(self, intervals: List[Tuple[float, float]]):
        """Set the list of subtitle intervals for overlay shading."""
        self.subtitle_intervals = intervals
        self._plot_window(self.slider.value())

    def set_selected_subtitle_indices(self, indices: List[int]):
        """Set which subtitle rows are currently selected in the table.

        Selected intervals are drawn with a stronger color/alpha.
        """
        self.selected_subtitle_indices = set(i for i in indices if i is not None)
        self._plot_window(self.slider.value())

    def jump_to_time(self, target_sec: float, center: bool = True):
        """Scroll the view so that target_sec is visible (optionally centered)."""
        if self.samples_mono is None or self.sr is None:
            return
        if self.total_duration <= self.window_duration:
            self._plot_window(0)
            return
        start = target_sec - self.window_duration / 2.0 if center else target_sec
        start = max(0.0, min(start, self.total_duration - self.window_duration))
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(start))
        self.slider.blockSignals(blocked)
        self._plot_window(start)

    def set_audio_segment(self, segment: AudioSegment):
        """Attach the full-quality audio segment used for playback."""
        self.audio_segment = segment
        total = (len(segment) / 1000.0) if segment else 0.0
        if self.playhead_sec > total:
            self.playhead_sec = max(0.0, total - 0.001)
        self.pos_lbl.setText(self._format_hhmmss_mmm(self.playhead_sec))
        # Update total label now that we have the exact media length
        self._update_total_label()

    def stop_playback_external(self):
        """Stop playback when the other plot starts playing.

        Keeps button state and visuals consistent without re-entrancy.
        """
        if self.audio_output is not None:
            blocked = self.play_btn.blockSignals(True)
            try:
                self.play_btn.setChecked(False)
            finally:
                self.play_btn.blockSignals(blocked)
            self._stop_playback(paused=True)
            self._update_play_glyph(False)

    def reset_view(self):
        """Reset the view to the beginning (0..window_duration or full span)."""
        if self.samples_mono is None or self.sr is None:
            return
        if self.total_duration <= self.window_duration:
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
            self._plot_window(0)
        else:
            self.slider.setMaximum(int(self.total_duration - self.window_duration))
            self.slider.setEnabled(True)
            self.slider.setValue(0)
            self._plot_window(0)

    # UI callbacks -------------------------------------------------------
    def _update_plot_from_slider(self, value: int):
        """Re-render the plot using the new slider start position."""
        self._plot_window(float(value))

    def _on_amp_changed(self, val: float):
        """Adjust vertical zoom by changing +/- amplitude limits."""
        self._amp_limit = max(0.01, float(val))
        self._plot_window(float(self.slider.value()))

    def _on_mouse_press(self, event):
        """Handle plot mouse press.

        Right click sets playhead, left click starts panning (drag).
        """
        if event.button == 3 and event.inaxes and event.xdata is not None:
            self._set_playhead(float(event.xdata))
            return
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._drag_start_x_pixel = event.x
            self._drag_start_slider = self.slider.value()

    def _on_mouse_release(self, _event):
        """End dragging when the mouse button is released."""
        self._dragging = False
        self._drag_start_x_pixel = None
        self._drag_start_slider = None

    def _on_mouse_move(self, event):
        """Pan the view while dragging left mouse button.

        We map pixel delta to seconds using the current axis width and
        the fixed window duration.
        """
        if self._dragging and event.inaxes and self._drag_start_x_pixel is not None:
            ax = event.inaxes
            bbox = ax.get_window_extent()
            axis_width = bbox.width
            if axis_width <= 0:
                return
            seconds_per_pixel = self.window_duration / axis_width
            dx_pixels = self._drag_start_x_pixel - event.x
            dx_seconds = dx_pixels * seconds_per_pixel
            new_slider = int(self._drag_start_slider + dx_seconds)
            new_slider = max(self.slider.minimum(), min(self.slider.maximum(), new_slider))
            if new_slider != self.slider.value():
                self.slider.setValue(new_slider)
                # Continue dragging relative to the last applied position
                self._drag_start_x_pixel = event.x
                self._drag_start_slider = new_slider

    def _update_play_glyph(self, playing: bool):
        """Update the button icon/text based on playing state."""
        try:
            icon = self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay)
            self.play_btn.setIcon(icon)
            self.play_btn.setText("")
        except Exception:
            # Fallback to ASCII text if style icon is unavailable
            self.play_btn.setIcon(None)
            self.play_btn.setText("||" if playing else ">")
        self.play_btn.setToolTip("Pause audio" if playing else "Play/Pause audio from playhead. Right-click waveform to set playhead.")

    def _on_play_toggled(self, playing: bool):
        """Start or stop playback when the play button is toggled."""
        self._update_play_glyph(playing)
        if playing:
            self.playingChanged.emit(True)
            ok = self._start_playback()
            if not ok:
                # Failed to start playback -> restore button state and signal
                blocked = self.play_btn.blockSignals(True)
                self.play_btn.setChecked(False)
                self.play_btn.blockSignals(blocked)
                self._update_play_glyph(False)
                self.playingChanged.emit(False)
        else:
            self._stop_playback(paused=True)
            self.playingChanged.emit(False)

    def _start_playback(self) -> bool:
        """Start audio playback from current playhead position with optimized buffering.

        Returns
        -------
        bool
            True if playback started; False if no data or at end.
        """
        if self.audio_segment is None:
            return False
        media_total = len(self.audio_segment) / 1000.0
        start_sec = max(0.0, min(self.playhead_sec, media_total))
        self._play_origin_sec = start_sec

        # Slice from playhead to end (original behavior for unlimited playback)
        # Optimize by using direct slicing without intermediate copies
        start_ms = int(start_sec * 1000)
        part = self.audio_segment[start_ms:]
        
        if len(part) <= 0:
            return False

        # Optimize: Only convert sample width if necessary
        if part.sample_width != 2:
            part = part.set_sample_width(2)
        
        fmt = QAudioFormat()
        fmt.setSampleRate(part.frame_rate)
        fmt.setChannelCount(part.channels)
        fmt.setSampleSize(part.sample_width * 8)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)

        self._dispose_audio()

        # Use raw_data directly without intermediate copies
        raw_bytes = bytes(part.raw_data)
        self.audio_data = QByteArray(raw_bytes)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        self.audio_output = QAudioOutput(fmt, self)
        self.audio_output.stateChanged.connect(self._on_audio_state_changed)
        
        # Optimize buffer size: larger buffer = smoother playback
        try:
            # Use 500ms buffer for very smooth playback
            buffer_size = int(part.frame_rate * part.channels * part.sample_width * 0.5)
            self.audio_output.setBufferSize(buffer_size)
        except Exception:
            pass
        
        self.audio_output.start(self.audio_buffer)

        self._timer.start()
        self._set_playhead(start_sec, center_if_needed=True)
        return True

    def _stop_playback(self, paused: bool):
        """Stop playback and refresh the plot at current slider position."""
        self._dispose_audio()
        self._plot_window(float(self.slider.value()))

    def _dispose_audio(self):
        """Dispose QAudioOutput/QBuffer safely and stop the UI timer."""
        self._timer.stop()
        out = self.audio_output; buf = self.audio_buffer
        self.audio_output = None; self.audio_buffer = None; self.audio_data = None
        if out is not None:
            try:
                try:
                    out.stateChanged.disconnect(self._on_audio_state_changed)
                except Exception:
                    pass
                out.stop()
            except Exception:
                pass
            try:
                out.deleteLater()
            except Exception:
                pass
        if buf is not None:
            try:
                buf.close()
            except Exception:
                pass
            try:
                buf.deleteLater()
            except Exception:
                pass

    def _on_audio_state_changed(self, state):
        """Auto-stop when the audio output enters idle/stopped states."""
        try:
            idle_state = getattr(QAudioOutput, "IdleState"); stopped_state = getattr(QAudioOutput, "StoppedState")
        except Exception:
            # Fallback numeric values used by some bindings
            idle_state = 3; stopped_state = 2
        if state in (idle_state, stopped_state):
            self._dispose_audio()
            blocked = self.play_btn.blockSignals(True)
            self.play_btn.setChecked(False)
            self.play_btn.blockSignals(blocked)
            self._update_play_glyph(False)
            self.playingChanged.emit(False)
            self._plot_window(float(self.slider.value()))

    def _on_tick(self):
        """Periodic UI update while playing (advances playhead and view).

        We prefer real processed microseconds from QAudioOutput, falling back
        to timer steps when not available (keeps UI responsive on platforms
        with limited QAudioOutput metrics).
        """
        if not self.audio_output:
            self._timer.stop(); return
        try:
            processed = self.audio_output.processedUSecs() / 1_000_000.0
        except Exception:
            processed = self._timer.interval() / 1000.0
        new_pos = self._play_origin_sec + max(0.0, processed)
        self.playhead_sec = new_pos
        self.pos_lbl.setText(self._format_hhmmss_mmm(self.playhead_sec))
        xmin = float(self.slider.value()); xmax = xmin + self.window_duration
        # When playhead approaches the right edge, keep it centered by panning
        if self.playhead_sec > xmax - (self.window_duration * 0.25):
            self.jump_to_time(self.playhead_sec, center=True)
        else:
            self._plot_window(xmin)

    # Plotting -----------------------------------------------------------
    def _plot_window(self, start_sec: float):
        """Render a view of [start_sec, start_sec + window_duration]."""
        if self.samples_mono is None or self.sr is None:
            return
        sr = self.sr
        total_len = len(self.samples_mono)
        if total_len == 0:
            return
        _xmin = start_sec
        _xmax = min(start_sec + self.window_duration, self.total_duration)
        idx_min = int(_xmin * sr)
        idx_max = min(int(_xmax * sr) + 1, total_len)
        t = np.arange(total_len, dtype=np.float64) / sr

        # Basic waveform line plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t[idx_min:idx_max], self.samples_mono[idx_min:idx_max], linewidth=1.0)
        ax.set_xlim([_xmin, _xmax])
        ax.set_ylim([-self._amp_limit, self._amp_limit])
        ax.set_ylabel("Amplitude", fontsize=8, labelpad=0)

        # Ticks: for long spans, label every 5s but only on odd multiples
        span = _xmax - _xmin
        if span > 120:
            step = 5
            ax.xaxis.set_major_locator(mticker.MultipleLocator(base=step))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: self._format_hhmmss(int(round(x))) if (int(round(x)) // step) % 2 == 1 and abs(x - round(x)) < 1e-6 else "")
            )
        else:
            ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_odd_second))

        ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)

        # Subtitle overlays (shade visible intersections)
        if self.subtitle_intervals:
            for i, (start, end) in enumerate(self.subtitle_intervals):
                if end >= _xmin and start <= _xmax:
                    color = "#ff9900" if i in self.selected_subtitle_indices else "orange"
                    alpha = 0.38 if i in self.selected_subtitle_indices else 0.18
                    ax.axvspan(max(start, _xmin), min(end, _xmax), color=color, alpha=alpha, zorder=0)

        # Playhead indicator
        if self.playhead_sec is not None and _xmin <= self.playhead_sec <= _xmax:
            ax.axvline(self.playhead_sec, color="deepskyblue", linewidth=1.2, alpha=0.9, zorder=5)

        self.canvas.draw()

    # Helpers ------------------------------------------------------------
    def _format_hhmmss(self, seconds: float, _pos=None) -> str:
        """Format integer seconds as HH:MM:SS for x-axis tick labels."""
        sec = int(seconds)
        h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
        return f"{h:02}:{m:02}:{s:02}"

    def _format_odd_second(self, x, _pos=None):
        """Matplotlib formatter: label only odd whole seconds to reduce clutter."""
        whole = int(round(x))
        if abs(x - whole) < 1e-6 and whole % 2 == 1:
            return self._format_hhmmss(whole)
        return ""

    def _format_hhmmss_mmm(self, seconds: float) -> str:
        """Format seconds with milliseconds as HH:MM:SS,mmm for the UI label."""
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms == 1000:
            ms = 0; s += 1
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _update_total_label(self):
        """Update the total length label on the top-right."""
        total_sec = 0.0
        if self.audio_segment is not None:
            total_sec = len(self.audio_segment) / 1000.0
        elif self.total_duration:
            total_sec = float(self.total_duration)
        self.total_lbl.setText(f"Total: {self._format_hhmmss_mmm(total_sec)}")

    def _set_playhead(self, sec: float, center_if_needed: bool = False):
        """Move the playhead to a specific time and optionally recenter view."""
        self.playhead_sec = max(0.0, min(sec, self.total_duration if self.total_duration else sec))
        self.pos_lbl.setText(self._format_hhmmss_mmm(self.playhead_sec))
        if center_if_needed:
            xmin = float(self.slider.value()); xmax = xmin + self.window_duration
            if not (xmin <= self.playhead_sec <= xmax):
                self.jump_to_time(self.playhead_sec, center=True)
        self._plot_window(float(self.slider.value()))
