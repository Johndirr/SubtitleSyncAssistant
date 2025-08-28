import sys
import os
from typing import List, Tuple, Optional

from PyQt5.QtCore import Qt, QByteArray, QBuffer, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QSlider, QAbstractItemView,
    QMenu, QAction, QInputDialog, QProgressBar, QDialog, QSpinBox, QDialogButtonBox, QDoubleSpinBox, QTextEdit
)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput
from PyQt5.QtGui import QColor

import numpy as np
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pysrt
from pydub import AudioSegment

try:
    from audio_offset_finder.audio_offset_finder import find_offset_between_buffers
except ImportError:
    find_offset_between_buffers = None


# ============================= Matplotlib Plot Widget =============================

class MatplotlibPlotWidget(QFrame):
    def __init__(self, title: str = "Plot", window_duration: int = 20):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setMinimumHeight(180)
        self.setMaximumHeight(180)

        self.window_duration = window_duration
        self.samples: Optional[np.ndarray] = None
        self.samples_mono: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.total_duration: float = 0.0

        self.subtitle_intervals: List[Tuple[float, float]] = []
        self.selected_subtitle_indices: set[int] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top bar: Amp controls (left) + centered title
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(4, 0, 4, 0)
        top_bar.setSpacing(6)

        # Left controls container (fixed width reference)
        left_controls = QWidget(self)
        lc_layout = QHBoxLayout(left_controls)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.setSpacing(4)
        amp_lbl = QLabel("Amp:", self)
        amp_lbl.setToolTip("Visible +/- amplitude (vertical zoom of normalized waveform).")
        self.amp_spin = QDoubleSpinBox(self)
        self.amp_spin.setRange(0.05, 2.00)
        self.amp_spin.setSingleStep(0.05)
        self.amp_spin.setDecimals(2)
        self.amp_spin.setValue(1.05)
        self.amp_spin.setToolTip("Adjust vertical zoom (does not alter data).")
        self.amp_spin.valueChanged.connect(self._on_amp_changed)
        lc_layout.addWidget(amp_lbl)
        lc_layout.addWidget(self.amp_spin)
        top_bar.addWidget(left_controls)

        # Stretch before title
        top_bar.addStretch()

        # Title (centered between symmetric side blocks)
        self.label = QLabel(title, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar.addWidget(self.label)

        # Stretch after title
        top_bar.addStretch()

        # Right phantom spacer with same width as left_controls to balance centering.
        right_phantom = QWidget(self)
        phantom_w = left_controls.sizeHint().width()
        right_phantom.setFixedWidth(phantom_w)
        top_bar.addWidget(right_phantom)

        layout.addLayout(top_bar)

        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._update_plot_from_slider)
        layout.addWidget(self.slider)

        self._dragging = False
        self._drag_start_x_pixel = None
        self._drag_start_slider = None
        self._amp_limit = 1.05  # current half-range of y-axis

        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def plot_waveform(self, samples: np.ndarray, sr: int):
        self.samples = samples
        self.sr = sr
        self.samples_mono = samples.mean(axis=0) if samples.ndim > 1 else samples
        total_len = len(self.samples_mono)
        # Use (N-1)/sr as true last-sample time so time vector aligns exactly with i/sr
        self.total_duration = (total_len - 1) / sr if sr > 0 and total_len > 0 else 0.0

        if self.total_duration > self.window_duration:
            self.slider.setMaximum(int(max(0, self.total_duration - self.window_duration)))
            self.slider.setEnabled(True)
        else:
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
        self.slider.setValue(0)
        self._plot_window(0)

    def set_subtitle_intervals(self, intervals: List[Tuple[float, float]]):
        self.subtitle_intervals = intervals
        self._plot_window(self.slider.value())

    def set_selected_subtitle_indices(self, indices: List[int]):
        self.selected_subtitle_indices = set(i for i in indices if i is not None)
        self._plot_window(self.slider.value())

    def jump_to_time(self, target_sec: float, center: bool = True):
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

    def _update_plot_from_slider(self, value: int):
        self._plot_window(float(value))

    def _format_hhmmss(self, seconds: float, _pos=None) -> str:
        sec = int(seconds)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02}:{m:02}:{s:02}"

    def _plot_window(self, start_sec: float):
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

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t[idx_min:idx_max], self.samples_mono[idx_min:idx_max], linewidth=1.0)
        ax.set_xlim([_xmin, _xmax])
        ax.set_ylim([-self._amp_limit, self._amp_limit])
        ax.set_ylabel("Amplitude", fontsize=8, labelpad=0)

        # ---- NEW: deterministic integer / coarse tick positioning ----
        span = _xmax - _xmin
        if span <= 30:
            step = 1
        elif span <= 120:
            step = 5
        elif span <= 600:
            step = 10
        elif span <= 1800:
            step = 30
        else:
            step = 60
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=step))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_hhmmss))
        # --------------------------------------------------------------

        ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)

        if self.subtitle_intervals:
            for i, (start, end) in enumerate(self.subtitle_intervals):
                # Only draw spans intersecting current window
                if end >= _xmin and start <= _xmax:
                    color = "#ff9900" if i in self.selected_subtitle_indices else "orange"
                    alpha = 0.38 if i in self.selected_subtitle_indices else 0.18
                    ax.axvspan(max(start, _xmin), min(end, _xmax), color=color, alpha=alpha, zorder=0)

        self.canvas.draw()

    def _on_mouse_press(self, event):
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._drag_start_x_pixel = event.x
            self._drag_start_slider = self.slider.value()

    def _on_mouse_release(self, _event):
        self._dragging = False
        self._drag_start_x_pixel = None
        self._drag_start_slider = None

    def _on_mouse_move(self, event):
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
                self._drag_start_x_pixel = event.x
                self._drag_start_slider = new_slider

    def _on_amp_changed(self, val: float):
        """User changed vertical zoom."""
        self._amp_limit = max(0.01, float(val))
        self._plot_window(float(self.slider.value()))

    def set_amplitude_limit(self, max_abs: float):
        """Programmatic vertical zoom setter."""
        if max_abs <= 0:
            return
        blocked = self.amp_spin.blockSignals(True)
        self.amp_spin.setValue(max_abs)
        self.amp_spin.blockSignals(blocked)
        self._amp_limit = max_abs
        self._plot_window(float(self.slider.value()))


# ============================= Analyze Worker =============================

class AnalyzeWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, ref_media_path: str, new_media_path: str, srt_path: str):
        super().__init__()
        self.ref_media_path = ref_media_path
        self.new_media_path = new_media_path
        self.srt_path = srt_path
        self._abort = False

    def abort(self):
        self._abort = True

    def _check_abort(self):
        if self._abort:
            self.cancelled.emit()
            raise RuntimeError("__ABORT__")

    def _load_audio(self, path: str):
        """
        Load an audio file with pydub and prepare a lightweight numpy representation
        for waveform plotting (without altering the full-quality AudioSegment).

        Returns
        -------
        display : np.ndarray
            Float32 array (mono or multi‑channel) possibly decimated for speed.
            Shape:
              - 1D (n,) if the original file is mono.
              - 2D (channels, n) if multi‑channel.
        display_sr : int
            Effective sample rate matching the decimated data (NOT the original
            frame rate when we take every Nth sample). This is critical so
            duration = len(display_mono) / display_sr is still correct.
        audio : AudioSegment
            The untouched full‑quality pydub segment (used later for playback
            / precise export / offset correlation caches).
        """
        # Decode file using pydub (ffmpeg backend) into an AudioSegment.
        audio = AudioSegment.from_file(path)

        # Convert raw PCM samples to a NumPy float32 array.
        # get_array_of_samples() returns interleaved samples for multi‑channel audio.
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # If multi‑channel, reshape to (channels, n_samples).
        # Pydub provides samples interleaved: L R L R ... (frame major).
        # We reshape to (n_frames, channels) then transpose so axis 0 = channels.
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).T  # (channels, n)

        # Normalize integer PCM to [-1.0, 1.0] range.
        # audio.sample_width is in bytes (e.g. 2 for 16‑bit).
        # Signed PCM max magnitude = 2^(bits-1). Example: 16‑bit -> 32768.
        # Using floating range keeps plotting consistent and prevents overflow in later math.
        full_scale = float(2 ** (8 * audio.sample_width - 1))
        samples /= full_scale

        # Record original sample rate (frame_rate in pydub).
        original_sr = audio.frame_rate

        # OPTIONAL DOWNSAMPLING (purely for plotting performance).
        # If the total sample count is huge (>5M values), decimate by a fixed stride
        # to reduce memory + draw time. We also adjust display_sr accordingly so
        # timeline math (seconds = index / display_sr) stays correct.
        if samples.size > 5_000_000:
            factor = 10  # simple stride decimation (no anti‑aliasing – acceptable for visualization)
            if samples.ndim == 1:
                display = samples[::factor]
            else:
                display = samples[:, ::factor]
            display_sr = max(1, original_sr // factor)
        else:
            display = samples
            display_sr = original_sr

        # Return lightweight array + its effective sample rate + full segment.
        return display, display_sr, audio

    def run(self):
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
                "intervals": intervals
            }
            self.progress.emit(100, "Done")
            self.finished.emit(result)

        except RuntimeError as ex:
            if str(ex) != "__ABORT__":  # explicit abort
                self.failed.emit(f"Aborted: {ex}")
        except Exception as e:
            self.failed.emit(str(e))


# ============================= Offset Worker =============================

class OffsetWorker(QObject):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(int, object, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self,
                 ref_wave: np.ndarray,
                 ref_sr: int,
                 new_wave: np.ndarray,
                 new_sr: int,
                 rows: List[Tuple[int, float, float]],
                 min_duration: float = 1.0,
                 ref_offset_sec: float = 0.0):
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
        self._abort = True

    def _check_abort(self):
        if self._abort:
            raise RuntimeError("__ABORT__")

    @staticmethod
    def _resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or samples.size == 0:
            return samples
        ratio = dst_sr / src_sr
        new_len = max(1, int(round(samples.shape[0] * ratio)))
        x_old = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        return np.interp(x_new, x_old, samples).astype(np.float32)

    def run(self):
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
                    snippet = self._resample_linear(snippet, self.new_sr, self.ref_sr)
                try:
                    res = find_offset_between_buffers(self.ref_wave, snippet, self.ref_sr)
                except Exception as ex:
                    self.result.emit(idx, None, f"err:{ex}")
                    continue
                if not isinstance(res, dict) or "time_offset" not in res:
                    self.result.emit(idx, None, "bad-result")
                    continue
                time_offset_global = float(res["time_offset"]) + self.ref_offset_sec
                delta = start_sec - time_offset_global
                self.result.emit(idx, delta, "ok")
                # Emit sequential progress + 1-based UI row number (idx+1)
                self.progress.emit(idx, f"{seq}/{total} (row {idx+1})")
            self.finished.emit()
        except RuntimeError as ex:
            if str(ex) == "__ABORT__":
                self.cancelled.emit()
            else:
                self.failed.emit(str(ex))
        except Exception as e:
            self.failed.emit(str(e))


# ============================= Busy Dialog =============================

class BusyDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None, title="Working", message="Please wait...", cancellable: bool = False):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(360)
        self._cancellable = cancellable
        self._force_close = False
        self._cancel_sent = False
        layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)
        if cancellable:
            btns = QDialogButtonBox(QDialogButtonBox.Cancel)
            btns.rejected.connect(self._on_cancel)
            layout.addWidget(btns)

    def _on_cancel(self):
        if not self._cancel_sent:
            self._cancel_sent = True
            self.cancel_requested.emit()
        self.set_message("Cancelling …")

    def set_message(self, text: str):
        self.label.setText(text)

    def finish(self):
        """Allow dialog to really close now (worker done)."""
        self._force_close = True
        self.close()

    def closeEvent(self, event):
        # Allow normal close if force_close set or not cancellable
        if self._force_close or not self._cancellable:
            return super().closeEvent(event)
        # Treat as user cancellation; keep dialog until worker ends
        if not self._cancel_sent:
            self._cancel_sent = True
            self.cancel_requested.emit()
        self.set_message("Cancelling …")
        event.ignore()


# ============================= Range Select Dialog =============================

class RangeSelectDialog(QDialog):
    def __init__(self, parent, default_start_sec: float, default_end_sec: float):
        super().__init__(parent)
        self.setWindowTitle("Reference Range")
        self.setModal(True)
        self.setMinimumWidth(360)

        # Always initialize to zero (ignore provided defaults)
        sh = sm = ss = 0
        eh = em = es = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Start time row
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start (HH:MM:SS):"))
        self.start_h = QSpinBox(); self.start_h.setRange(0, 999); self.start_h.setValue(sh)
        self.start_m = QSpinBox(); self.start_m.setRange(0, 59);  self.start_m.setValue(sm)
        self.start_s = QSpinBox(); self.start_s.setRange(0, 59);  self.start_s.setValue(ss)
        for w in (self.start_h, self.start_m, self.start_s):
            w.setFixedWidth(60)
        start_row.addWidget(self.start_h)
        start_row.addWidget(QLabel(":"))
        start_row.addWidget(self.start_m)
        start_row.addWidget(QLabel(":"))
        start_row.addWidget(self.start_s)
        start_row.addStretch()
        layout.addLayout(start_row)

        # End time row
        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End (HH:MM:SS):"))
        self.end_h = QSpinBox(); self.end_h.setRange(0, 999); self.end_h.setValue(eh)
        self.end_m = QSpinBox(); self.end_m.setRange(0, 59);  self.end_m.setValue(em)
        self.end_s = QSpinBox(); self.end_s.setRange(0, 59);  self.end_s.setValue(es)
        for w in (self.end_h, self.end_m, self.end_s):
            w.setFixedWidth(60)
        end_row.addWidget(self.end_h)
        end_row.addWidget(QLabel(":"))
        end_row.addWidget(self.end_m)
        end_row.addWidget(QLabel(":"))
        end_row.addWidget(self.end_s)
        end_row.addStretch()
        layout.addLayout(end_row)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._start_seconds = None
        self._end_seconds = None

    def _on_accept(self):
        start_sec = self.start_h.value()*3600 + self.start_m.value()*60 + self.start_s.value()
        end_sec = self.end_h.value()*3600 + self.end_m.value()*60 + self.end_s.value()
        if end_sec <= start_sec:
            QMessageBox.warning(self, "Invalid Range", "End must be greater than Start.")
            return
        self._start_seconds = start_sec
        self._end_seconds = end_sec
        self.accept()

    @staticmethod
    def get_range(parent, default_start_sec: float, default_end_sec: float) -> Tuple[Optional[int], Optional[int], bool]:
        dlg = RangeSelectDialog(parent, default_start_sec, default_end_sec)
        ok = dlg.exec_() == QDialog.Accepted
        return dlg._start_seconds, dlg._end_seconds, ok


# ============================= Edit Subtitle Dialog =============================

class EditSubtitleDialog(QDialog):
    def __init__(self, parent, start_text: str, end_text: str, subtitle_text: str):
        super().__init__(parent)
        self.setWindowTitle("Edit Subtitle Line")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Start time
        row_start = QHBoxLayout()
        row_start.addWidget(QLabel("Start (HH:MM:SS,mmm):"))
        self.le_start = QLineEdit()
        self.le_start.setText(start_text)
        row_start.addWidget(self.le_start)
        layout.addLayout(row_start)

        # End time
        row_end = QHBoxLayout()
        row_end.addWidget(QLabel("End (HH:MM:SS,mmm):"))
        self.le_end = QLineEdit()
        self.le_end.setText(end_text)
        row_end.addWidget(self.le_end)
        layout.addLayout(row_end)

        # Text
        layout.addWidget(QLabel("Text:"))
        self.te_text = QTextEdit()
        self.te_text.setPlainText(subtitle_text)
        layout.addWidget(self.te_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.result_start = None
        self.result_end = None
        self.result_text = None

    def _normalize_time(self, t: str) -> Optional[str]:
        t = t.strip()
        if not t:
            return None
        # Accept variants like HH:MM:SS.mmm or HH:MM:SS,mmm
        if '.' in t and ',' not in t:
            parts = t.rsplit('.', 1)
            if len(parts[-1]) in (1, 2, 3) and parts[-1].isdigit():
                t = parts[0] + ',' + parts[1].ljust(3, '0')
        # Ensure milliseconds
        if ',' not in t:
            if t.count(':') == 2:
                t += ",000"
        # Pad milliseconds to 3
        if ',' in t:
            a, b = t.split(',', 1)
            if not b.isdigit():
                return None
            b = b[:3].ljust(3, '0')
            t = a + ',' + b
        return t

    def _parse_to_seconds(self, t: str) -> Optional[float]:
        try:
            h, m, rest = t.split(':')
            s, ms = rest.split(',')
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
        except Exception:
            return None

    def _on_accept(self):
        start_raw = self._normalize_time(self.le_start.text())
        end_raw = self._normalize_time(self.le_end.text())
        if not start_raw or not end_raw:
            QMessageBox.warning(self, "Invalid", "Invalid start or end time format.")
            return
        s = self._parse_to_seconds(start_raw)
        e = self._parse_to_seconds(end_raw)
        if s is None or e is None:
            QMessageBox.warning(self, "Invalid", "Could not parse start/end times.")
            return
        if e <= s:
            QMessageBox.warning(self, "Invalid", "End must be greater than Start.")
            return
        self.result_start = start_raw
        self.result_end = end_raw
        self.result_text = self.te_text.toPlainText()
        self.accept()

    @staticmethod
    def edit(parent, start_text: str, end_text: str, subtitle_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
        dlg = EditSubtitleDialog(parent, start_text, end_text, subtitle_text)
        ok = dlg.exec_() == QDialog.Accepted
        return dlg.result_start, dlg.result_end, dlg.result_text, ok


# ============================= Main Window =============================

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle Sync Assistant")
        self.resize(1024, 600)
        self._build_ui()

        self.ref_audio_segment: Optional[AudioSegment] = None
        self.new_audio_segment: Optional[AudioSegment] = None
        self.audio_output: Optional[QAudioOutput] = None
        self.audio_buffer: Optional[QBuffer] = None
        self.audio_data: Optional[QByteArray] = None

        self._analyze_thread: Optional[QThread] = None
        self._analyze_worker: Optional[AnalyzeWorker] = None
        self._busy_dialog: Optional[BusyDialog] = None

        self._offset_thread: Optional[QThread] = None
        self._offset_worker: Optional[OffsetWorker] = None
        self._ref_mono_cache: Optional[np.ndarray] = None
        self._ref_sr_cache: Optional[int] = None
        self._new_mono_cache: Optional[np.ndarray] = None
        self._new_sr_cache: Optional[int] = None
        self._busy_offset: Optional[BusyDialog] = None

    # ---------- UI ----------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(4)

        def add_row(btn_text, line_edit_attr, slot):
            h = QHBoxLayout()
            btn = QPushButton(btn_text); btn.setFixedWidth(220)
            le = QLineEdit(); le.setReadOnly(True); le.setPlaceholderText("No file selected")
            h.addWidget(btn); h.addWidget(le); main_layout.addLayout(h)
            btn.clicked.connect(slot)
            return btn, le

        self.btn1, self.le1 = add_row("Load reference media file", "le1", self.select_media_file_btn1)
        self.btn2, self.le2 = add_row("Load new media file", "le2", self.select_media_file_btn2)
        self.btn3, self.le3 = add_row("Load reference subtitle", "le3", self.select_subtitle_file_btn3)
        self.btn4, self.le4 = add_row("Save subtitle under...", "le4", self.save_subtitle_file_btn4)

        row5 = QHBoxLayout()
        self.btn5 = QPushButton("Analyze...")
        row5.addWidget(self.btn5)
        main_layout.addLayout(row5)
        self.btn5.clicked.connect(self.on_analyze)

        main_layout.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setLineWidth(1)
        main_layout.addWidget(sep)
        main_layout.addSpacing(8)

        self.plot1 = MatplotlibPlotWidget("Reference Audio Waveform")
        self.plot2 = MatplotlibPlotWidget("New Audio Waveform")
        main_layout.addWidget(self.plot1)
        main_layout.addWidget(self.plot2)
        main_layout.addSpacing(8)

        tables_row = QHBoxLayout()
        self.referencetable = QTableWidget(0, 3)
        self.referencetable.setHorizontalHeaderLabels(["Start time", "End time", "Text"])
        self._init_table_column_sizing(self.referencetable, [0, 1], [2])
        self.referencetable.setMinimumHeight(300)
        self.referencetable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.referencetable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.referencetable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.referencetable.customContextMenuRequested.connect(self.show_referencetable_context_menu)

        self.synctable = QTableWidget(0, 4)
        self.synctable.setHorizontalHeaderLabels(["Start time", "End time", "Text", "Found offset"])
        self._init_table_column_sizing(self.synctable, [0, 1, 3], [2])
        self.synctable.setMinimumHeight(300)
        self.synctable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.synctable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.synctable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.synctable.customContextMenuRequested.connect(self.show_synctable_context_menu)

        tables_row.addWidget(self.referencetable)
        tables_row.addWidget(self.synctable)
        tables_row.setContentsMargins(0, 0, 0, 0)

        tables_container = QWidget()
        tables_container.setLayout(tables_row)
        tables_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(tables_container)

        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

        self.referencetable.selectionModel().selectionChanged.connect(self.on_reference_table_selection)
        self.synctable.selectionModel().selectionChanged.connect(self.on_sync_table_selection)

    def _init_table_column_sizing(self, table: QTableWidget, fixed_cols: List[int], stretch_cols: List[int]):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for c in fixed_cols:
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)
            table.setColumnWidth(c, 100)
        for c in stretch_cols:
            table.horizontalHeader().setSectionResizeMode(c, QHeaderView.Stretch)

    def align_table_columns_left(self, table: QTableWidget):
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for row in range(table.rowCount()):
                cell = table.item(row, col)
                if cell:
                    cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    # ---------- Selection -> plot highlight ----------
    def on_reference_table_selection(self):
        indices = [idx.row() for idx in self.referencetable.selectionModel().selectedRows()]
        self.plot1.set_selected_subtitle_indices(indices)

    def on_sync_table_selection(self):
        indices = [idx.row() for idx in self.synctable.selectionModel().selectedRows()]
        self.plot2.set_selected_subtitle_indices(indices)

    # ---------- Context Menus ----------
    def show_synctable_context_menu(self, pos):
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)
        act_edit = QAction("Edit...", self)
        act_delete = QAction("Delete line(s)", self)
        act_shift_sel = QAction("Shift times for selected line(s)", self)
        act_shift_all = QAction("Shift all times", self)
        act_find_offsets = QAction("Find Offset(s) (BBC-offset-finder)", self)
        act_find_offsets_range = QAction("Find Offset(s) in range (BBC-offset-finder)", self)

        menu.addAction(act_play)
        menu.addAction(act_jump)
        menu.addAction(act_edit)
        menu.addSeparator()
        menu.addAction(act_delete)
        menu.addSeparator()
        menu.addAction(act_shift_sel)
        menu.addAction(act_shift_all)
        menu.addSeparator()
        menu.addAction(act_find_offsets)
        menu.addAction(act_find_offsets_range)

        act_play.triggered.connect(self.synctable_play_selected)
        act_jump.triggered.connect(self.synctable_jump_to_selected)
        act_edit.triggered.connect(self.edit_selected_subtitle)
        act_delete.triggered.connect(self.synctable_delete_selected)
        act_shift_sel.triggered.connect(lambda: self.shift_times(selected_only=True))
        act_shift_all.triggered.connect(lambda: self.shift_times(selected_only=False))
        act_find_offsets.triggered.connect(self.find_offsets_for_selected)
        act_find_offsets_range.triggered.connect(self.find_offsets_for_selected_in_range)

        menu.exec_(self.synctable.viewport().mapToGlobal(pos))

    def show_referencetable_context_menu(self, pos):
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)
        menu.addAction(act_play)
        menu.addAction(act_jump)
        act_play.triggered.connect(self.referencetable_play_selected)
        act_jump.triggered.connect(self.referencetable_jump_to_selected)
        menu.exec_(self.referencetable.viewport().mapToGlobal(pos))

    def edit_selected_subtitle(self):
        """Open edit dialog for the first selected sync table row."""
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        s_item = self.synctable.item(row, 0)
        e_item = self.synctable.item(row, 1)
        t_item = self.synctable.item(row, 2)
        if not (s_item and e_item and t_item):
            return

        start_orig = s_item.text()
        end_orig = e_item.text()
        text_orig = t_item.text()

        new_start, new_end, new_text, ok = EditSubtitleDialog.edit(self, start_orig, end_orig, text_orig)
        if not ok:
            return

        s_item.setText(new_start)
        e_item.setText(new_end)
        t_item.setText(new_text)

        # Refresh plot intervals
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())

    # ---------- Playback ----------
    def referencetable_play_selected(self):
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if self.ref_audio_segment is None:
            path = self.le1.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Play", "Reference media file not loaded.")
                return
            try:
                self.ref_audio_segment = AudioSegment.from_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Play", str(e))
                return
        s_item = self.referencetable.item(row, 0)
        e_item = self.referencetable.item(row, 1)
        if not s_item or not e_item:
            return
        s = self._parse_time_to_seconds(s_item.text())
        e = self._parse_time_to_seconds(e_item.text())
        if e > s:
            self._play_audio_segment(self.ref_audio_segment, s, e)

    def synctable_play_selected(self):
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        if self.new_audio_segment is None:
            path = self.le2.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Play", "New media file not loaded.")
                return
            try:
                self.new_audio_segment = AudioSegment.from_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Play", str(e))
                return
        s_item = self.synctable.item(row, 0)
        e_item = self.synctable.item(row, 1)
        if not s_item or not e_item:
            return
        s = self._parse_time_to_seconds(s_item.text())
        e = self._parse_time_to_seconds(e_item.text())
        if e > s:
            self._play_audio_segment(self.new_audio_segment, s, e)

    def _play_audio_segment(self, segment: AudioSegment, start_sec: float, end_sec: float):
        if not segment:
            return
        total_dur = len(segment) / 1000.0
        start_sec = max(0.0, start_sec)
        end_sec = min(total_dur, end_sec)
        if end_sec <= start_sec:
            return
        part = segment[int(start_sec * 1000):int(end_sec * 1000)]
        if part.sample_width != 2:
            part = part.set_sample_width(2)
        fmt = QAudioFormat()
        fmt.setSampleRate(part.frame_rate)
        fmt.setChannelCount(part.channels)
        fmt.setSampleSize(part.sample_width * 8)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)
        if self.audio_output:
            self.audio_output.stop()
            self.audio_output.deleteLater()
            self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer.deleteLater()
            self.audio_buffer = None
        self.audio_data = QByteArray(part.raw_data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        self.audio_output = QAudioOutput(fmt, self)
        self.audio_output.start(self.audio_buffer)

    # ---------- Jump ----------
    def referencetable_jump_to_selected(self):
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        for idx in selected:
            itm = self.referencetable.item(idx.row(), 0)
            if itm:
                starts.append(self._parse_time_to_seconds(itm.text()))
        if starts:
            self.plot1.jump_to_time(min(starts), center=True)

    def synctable_jump_to_selected(self):
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        for idx in selected:
            itm = self.synctable.item(idx.row(), 0)
            if itm:
                starts.append(self._parse_time_to_seconds(itm.text()))
        if starts:
            self.plot2.jump_to_time(min(starts), center=True)

    # ---------- Table Editing ----------
    def _collect_synctable_intervals(self) -> List[Tuple[float, float]]:
        intervals = []
        for r in range(self.synctable.rowCount()):
            s_item = self.synctable.item(r, 0)
            e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text())
            e = self._parse_time_to_seconds(e_item.text())
            intervals.append((s, e))
        return intervals

    def synctable_delete_selected(self):
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        for idx in sorted(selected, key=lambda x: x.row(), reverse=True):
            self.synctable.removeRow(idx.row())
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        self.plot2.set_selected_subtitle_indices([])

    def shift_times(self, selected_only: bool):
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Shift Times", "No rows to shift.")
            return
        if selected_only:
            sel = self.synctable.selectionModel().selectedRows()
            target = [i.row() for i in sel]
            if not target:
                QMessageBox.information(self, "Shift Times", "No rows selected.")
                return
            prompt = "Shift selected line(s) by seconds (e.g. -1.250 or 2.5):"
            prefill = "0.000"
            for r in target:
                cell = self.synctable.item(r, 3)
                if not cell:
                    continue
                raw = cell.text().strip()
                try:
                    if raw and (raw[0].isdigit() or raw[0] in "+-"):
                        val = float(raw)
                        prefill = f"{val:+.3f}"
                        break
                except ValueError:
                    continue
        else:
            target = list(range(self.synctable.rowCount()))
            prompt = "Shift ALL lines by seconds (e.g. -1.250 or 2.5):"
            prefill = "0.000"

        val_str, ok = QInputDialog.getText(self, "Shift Times", prompt, text=prefill)
        if not ok or not val_str.strip():
            return
        try:
            delta = float(val_str.replace(",", "."))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Could not parse shift value.")
            return

        for r in target:
            s_item = self.synctable.item(r, 0)
            e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text()) + delta
            e = self._parse_time_to_seconds(e_item.text()) + delta
            s = max(0.0, s)
            e = max(s, e)
            s_item.setText(self._format_seconds_to_time(s))
            e_item.setText(self._format_seconds_to_time(e))

        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())

    # ---------- File Selection ----------
    def select_media_file_btn1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Media", "",
                                              "Media files (*.avi *.mkv *.mp4 *.mov *.mpg *.mpeg *.wmv *.flv *.webm);;All files (*.*)")
        if path:
            self.le1.setText(path)

    def select_media_file_btn2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select New Media", "",
                                              "Media files (*.avi *.mkv *.mp4 *.mov *.mpg *.mpeg *.wmv *.flv *.webm);;All files (*.*)")
        if path:
            self.le2.setText(path)

    def select_subtitle_file_btn3(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Subtitle", "",
                                              "Subtitle files (*.srt);;All files (*.*)")
        if path:
            self.le3.setText(path)
            base, ext = os.path.splitext(path)
            self.le4.setText(f"{base}_resync{ext}")

    def save_subtitle_file_btn4(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Subtitle As", "",
                                              "Subtitle files (*.srt);;All files (*.*)")
        if path:
            if not os.path.splitext(path)[1]:
                path += ".srt"
            self.le4.setText(path)

    # ---------- Analysis ----------
    def sanity_check_files(self) -> bool:
        missing = []
        if not self.le1.text() or not os.path.exists(self.le1.text()):
            missing.append("Reference media file")
        if not self.le2.text() or not os.path.exists(self.le2.text()):
            missing.append("New media file")
        if not self.le3.text() or not os.path.exists(self.le3.text()):
            missing.append("Reference subtitle file")
        if not self.le4.text():
            missing.append("Subtitle save path")
        if missing:
            QMessageBox.warning(self, "Missing Files", "Missing or invalid:\n\n" + "\n".join(missing))
            return False
        return True

    def on_analyze(self):
        if self._analyze_thread is not None:
            QMessageBox.information(self, "Analyze", "Analysis already running.")
            return
        if not self.sanity_check_files():
            return
        self._busy_dialog = BusyDialog(self, title="Analyzing", message="Starting analysis …", cancellable=True)
        self._busy_dialog.cancel_requested.connect(lambda: self._analyze_worker and self._analyze_worker.abort())
        self._busy_dialog.show()

        thread = QThread()
        worker = AnalyzeWorker(self.le1.text().strip(), self.le2.text().strip(), self.le3.text().strip())
        self._analyze_worker = worker
        self._analyze_thread = thread
        worker.moveToThread(thread)

        worker.progress.connect(lambda _v, msg: self._busy_dialog and self._busy_dialog.set_message(msg))

        def finished(result):
            self._teardown_analysis()
            try:
                self._apply_analysis_result(result)
            except Exception as e:
                QMessageBox.critical(self, "Result Error", str(e))

        def failed(msg):
            self._teardown_analysis()
            QMessageBox.critical(self, "Analyze Failed", msg)

        def cancelled():
            self._teardown_analysis()
            QMessageBox.information(self, "Analyze", "Analysis cancelled.")

        def cleanup():
            thread.quit()
            thread.wait()
            self._analyze_worker = None
            self._analyze_thread = None

        worker.finished.connect(finished)
        worker.failed.connect(failed)
        worker.cancelled.connect(cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)

        self.btn5.setEnabled(False)
        thread.start()

    def _teardown_analysis(self):
        self.btn5.setEnabled(True)
        if self._busy_dialog:
            try:
                self._busy_dialog.finish()
            except Exception:
                pass
            self._busy_dialog = None

    def _apply_analysis_result(self, result: dict):
        self.ref_audio_segment = result["ref_full"]
        self.new_audio_segment = result["new_full"]

        self.plot1.plot_waveform(result["ref_display"], result["ref_rate"])
        self.plot2.plot_waveform(result["new_display"], result["new_rate"])

        rows = result["rows"]
        fmt = lambda t: f"{t.hours:02}:{t.minutes:02}:{t.seconds:02},{t.milliseconds:03}"
        self.referencetable.setRowCount(len(rows))
        self.synctable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.referencetable.setItem(i, 0, QTableWidgetItem(fmt(r["start"])))
            self.referencetable.setItem(i, 1, QTableWidgetItem(fmt(r["end"])))
            self.referencetable.setItem(i, 2, QTableWidgetItem(r["text"]))

            self.synctable.setItem(i, 0, QTableWidgetItem(fmt(r["start"])))
            self.synctable.setItem(i, 1, QTableWidgetItem(fmt(r["end"])))
            self.synctable.setItem(i, 2, QTableWidgetItem(r["text"]))
            self.synctable.setItem(i, 3, QTableWidgetItem(""))


            bg = QColor(245, 245, 245) if i % 2 == 0 else QColor(230, 230, 230)
            for c in range(3):
                self.referencetable.item(i, c).setBackground(bg)
            for c in range(4):
                self.synctable.item(i, c).setBackground(bg)

        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)
        self.plot1.set_subtitle_intervals(result["intervals"])
        self.plot2.set_subtitle_intervals(result["intervals"])

    # ---------- Offset Support ----------
    def _ensure_audio_caches_for_offsets(self) -> bool:
        if find_offset_between_buffers is None:
            QMessageBox.warning(self, "Offset Finder", "audio-offset-finder not installed.")
            return False

        def load(seg_attr, path_line_edit):
            if getattr(self, seg_attr) is None:
                p = path_line_edit.text().strip()
                if not p or not os.path.exists(p):
                    return False
                try:
                    setattr(self, seg_attr, AudioSegment.from_file(p))
                except Exception as e:
                    QMessageBox.critical(self, "Offset Finder", str(e))
                    return False
            return True

        if not load("ref_audio_segment", self.le1):
            return False
        if not load("new_audio_segment", self.le2):
            return False

        def to_mono(seg: AudioSegment) -> Tuple[np.ndarray, int]:
            arr = np.array(seg.get_array_of_samples()).astype(np.float32)
            if seg.channels > 1:
                arr = arr.reshape((-1, seg.channels)).mean(axis=1)
            arr /= (2 ** (8 * seg.sample_width - 1))
            return arr, seg.frame_rate

        if self._ref_mono_cache is None:
            self._ref_mono_cache, self._ref_sr_cache = to_mono(self.ref_audio_segment)
        if self._new_mono_cache is None:
            self._new_mono_cache, self._new_sr_cache = to_mono(self.new_audio_segment)
        return True

    def cancel_offset_worker(self):
        if self._offset_worker:
            self._offset_worker.abort()
            if self._busy_offset:
                self._busy_offset.set_message("Cancelling …")

    def find_offsets_for_selected(self):
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Offset Finder", "No rows available.")
            return
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Offset Finder", "No rows selected.")
            return
        if not self._ensure_audio_caches_for_offsets():
            return
        if self._offset_thread is not None:
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return
        rows = []
        for idx in sel:
            r = idx.row()
            s_item = self.synctable.item(r, 0)
            e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text())
            e = self._parse_time_to_seconds(e_item.text())
            rows.append((r, s, e))
        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return

        self._busy_offset = BusyDialog(self, title="Finding Offsets", message="Searching …", cancellable=True)
        self._busy_offset.cancel_requested.connect(self.cancel_offset_worker)
        self._busy_offset.show()

        thread = QThread()
        worker = OffsetWorker(self._ref_mono_cache, self._ref_sr_cache,
                              self._new_mono_cache, self._new_sr_cache,
                              rows, 1.0, ref_offset_sec=0.0)
        self._offset_thread = thread
        self._offset_worker = worker
        worker.moveToThread(thread)

        def on_result(row_index: int, delta_val, status: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            if delta_val is None:
                cell.setText(status)
                cell.setBackground(QColor(240, 240, 200) if not status.startswith("err") else QColor(255, 210, 210))
            else:
                sign = "+" if delta_val >= 0 else "-"
                cell.setText(f"{sign}{abs(delta_val):.3f}")
                cell.setBackground(QColor(210, 245, 210) if status == "ok" else QColor(255, 210, 210))

        def on_progress(row_index: int, msg: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            if self._busy_offset:
                self._busy_offset.set_message(f"Processed {msg}")

        def on_finished(_worker=worker):
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def on_failed(err: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            if err != "Aborted":
                QMessageBox.critical(self, "Offset Finder", err)
            self._finish_offset_worker()

        def on_cancelled(_worker=worker):
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def cleanup(_worker=worker):
            if _worker is self._offset_worker:
                thread.quit()
                thread.wait()
                self._offset_thread = None
                self._offset_worker = None

        worker.result.connect(on_result)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)
        thread.start()

    def find_offsets_for_selected_in_range(self):
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Offset Finder", "No rows available.")
            return
        sel = self.synctable.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Offset Finder", "No rows selected.")
            return
        if not self._ensure_audio_caches_for_offsets():
            return
        if self._offset_thread is not None:
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return
        rows = []
        earliest = float('inf')
        latest = 0.0
        for idx in sel:
            r = idx.row()
            s_item = self.synctable.item(r, 0)
            e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text())
            e = self._parse_time_to_seconds(e_item.text())
            rows.append((r, s, e))
            earliest = min(earliest, s)
            latest = max(latest, e)
        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return

        if self.ref_audio_segment is None:
            QMessageBox.warning(self, "Offset Finder", "Reference audio not loaded.")
            return
        ref_total_sec = len(self.ref_audio_segment) / 1000.0
        pad = 5.0
        default_start = max(0.0, earliest - pad)
        default_end = min(ref_total_sec, latest + pad)

        ref_start, ref_end, ok = RangeSelectDialog.get_range(self, default_start, default_end)
        if not ok:
            return
        if ref_start < 0 or ref_end > ref_total_sec:
            QMessageBox.warning(self, "Reference Range", "Range outside reference duration.")
            return

        start_i = int(ref_start * self._ref_sr_cache)
        end_i = min(int(ref_end * self._ref_sr_cache), self._ref_mono_cache.shape[0])
        if end_i - start_i < 100:
            QMessageBox.warning(self, "Reference Range", "Range too short.")
            return

        ref_slice = self._ref_mono_cache[start_i:end_i]

        self._busy_offset = BusyDialog(self, title="Finding Offsets (Range)",
                                       message=f"Searching {ref_start:.3f}s–{ref_end:.3f}s …",
                                       cancellable=True)
        self._busy_offset.cancel_requested.connect(self.cancel_offset_worker)
        self._busy_offset.show()

        thread = QThread()
        worker = OffsetWorker(ref_slice, self._ref_sr_cache,
                              self._new_mono_cache, self._new_sr_cache,
                              rows, 1.0, ref_offset_sec=ref_start)
        self._offset_thread = thread
        self._offset_worker = worker
        worker.moveToThread(thread)

        def on_result(row_index: int, delta_val, status: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            if delta_val is None:
                cell.setText(status)
                cell.setBackground(QColor(240, 240, 200) if not status.startswith("err") else QColor(255, 210, 210))
            else:
                sign = "+" if delta_val >= 0 else "-"
                cell.setText(f"{sign}{abs(delta_val):.3f}")
                cell.setBackground(QColor(210, 245, 210) if status == "ok" else QColor(255, 210, 210))

        def on_progress(row_index: int, msg: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            if self._busy_offset:
                self._busy_offset.set_message(f"Processed {msg}")

        def on_finished(_worker=worker):
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def on_failed(err: str, _worker=worker):
            if _worker is not self._offset_worker:
                return
            if err != "Aborted":
                QMessageBox.critical(self, "Offset Finder", err)
            self._finish_offset_worker()

        def on_cancelled(_worker=worker):
            if _worker is not self._offset_worker:
                return
            self._finish_offset_worker()

        def cleanup(_worker=worker):
            if _worker is self._offset_worker:
                thread.quit()
                thread.wait()
                self._offset_thread = None
                self._offset_worker = None

        worker.result.connect(on_result)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.cancelled.connect(on_cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)
        thread.start()

    def _finish_offset_worker(self):
        if self._busy_offset:
            try:
                self._busy_offset.finish()
            except Exception:
                pass
            self._busy_offset = None

    # ---------- Time Helpers ----------
    def _parse_time_to_seconds(self, text: str) -> float:
        try:
            h, m, rest = text.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except Exception:
            return 0.0

    def _format_seconds_to_time(self, seconds: float) -> str:
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

    def _parse_user_time_any(self, text: str) -> Optional[float]:
        if not text:
            return None
        t = text.strip()
        try:
            cand = t
            if ',' not in cand and cand.count(':') >= 2:
                if '.' in cand and cand.rsplit('.', 1)[1].isdigit():
                    cand = cand.replace('.', ',', 1)
                else:
                    cand += ",000"
            if cand.count(':') >= 2 and ',' in cand:
                return self._parse_time_to_seconds(cand)
        except Exception:
            pass
        try:
            return float(t.replace(',', '.'))
        except ValueError:
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())