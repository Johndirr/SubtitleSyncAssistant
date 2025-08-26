import sys
import os
from typing import List, Tuple, Optional

from PyQt5.QtCore import Qt, QByteArray, QBuffer, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QSlider, QAbstractItemView,
    QMenu, QAction, QInputDialog, QProgressBar, QDialog
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
    """
    Lightweight widget embedding a matplotlib plot + horizontal slider for windowed
    waveform navigation. Also renders shaded subtitle interval bands and highlights
    selections.
    """
    def __init__(self, title: str = "Plot", window_duration: int = 20):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setMinimumHeight(180)
        self.setMaximumHeight(180)

        self.window_duration = window_duration  # seconds width of view window
        self.samples: Optional[np.ndarray] = None
        self.samples_mono: Optional[np.ndarray] = None
        self.sr: Optional[int] = None
        self.total_duration: float = 0.0

        self.subtitle_intervals: List[Tuple[float, float]] = []   # (start_sec, end_sec)
        self.selected_subtitle_indices: set[int] = set()          # current highlighted indices

        # --- Layout / UI ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(title, self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Horizontal slider for window navigation
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._update_plot_from_slider)
        layout.addWidget(self.slider)

        # Mouse-drag state for click+drag horizontal scrolling
        self._dragging = False
        self._drag_start_x_pixel = None
        self._drag_start_slider = None

        # matplotlib event hooks
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # ---------------- Public API ----------------

    def plot_waveform(self, samples: np.ndarray, sr: int):
        """
        Accept raw (possibly multi-channel) samples and sample-rate.
        Converts to mono (mean) only for plotting; retains slider navigation.
        """
        self.samples = samples
        self.sr = sr
        self.samples_mono = samples.mean(axis=0) if samples.ndim > 1 else samples
        self.total_duration = len(self.samples_mono) / sr if sr > 0 else 0.0

        if self.total_duration > self.window_duration:
            self.slider.setMaximum(int(self.total_duration - self.window_duration))
            self.slider.setEnabled(True)
        else:
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
        self.slider.setValue(0)
        self._plot_window(0)

    def set_subtitle_intervals(self, intervals: List[Tuple[float, float]]):
        """Store interval list and re-render."""
        self.subtitle_intervals = intervals
        self._plot_window(self.slider.value())

    def set_selected_subtitle_indices(self, indices: List[int]):
        """Highlight multiple intervals by index."""
        self.selected_subtitle_indices = set(i for i in indices if i is not None)
        self._plot_window(self.slider.value())

    def jump_to_time(self, target_sec: float, center: bool = True):
        """
        Move viewing window so that target_sec is visible (center if possible).
        Supports sub-second placement (not limited by slider granularity).
        """
        if self.samples_mono is None or self.sr is None:
            return
        if self.total_duration <= self.window_duration:
            self._plot_window(0)
            return

        start = target_sec - self.window_duration / 2.0 if center else target_sec
        start = max(0.0, min(start, self.total_duration - self.window_duration))

        # Slider is integer-second; block signals to avoid double redraw
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(start))
        self.slider.blockSignals(blocked)

        self._plot_window(start)

    # ---------------- Internal plotting helpers ----------------

    def _update_plot_from_slider(self, value: int):
        """Triggered by slider movement (integer seconds)."""
        self._plot_window(float(value))

    def _format_hhmmss(self, seconds: float, _pos=None) -> str:
        """Formatter for x-axis tick labels."""
        sec = int(seconds)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02}:{m:02}:{s:02}"

    def _plot_window(self, start_sec: float):
        """Render currently visible window [start_sec, start_sec + window_duration]."""
        if self.samples_mono is None or self.sr is None:
            return

        sr = self.sr
        total_len = len(self.samples_mono)
        _xmin = start_sec
        _xmax = min(start_sec + self.window_duration, self.total_duration)

        idx_min = int(_xmin * sr)
        idx_max = min(int(_xmax * sr), total_len)

        t = np.linspace(0, self.total_duration, num=total_len)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t[idx_min:idx_max], self.samples_mono[idx_min:idx_max], linewidth=1.0)
        ax.set_xlim([_xmin, _xmax])
        ax.set_ylim([-1.05, 1.05])
        ax.set_ylabel("Amplitude", fontsize=8, labelpad=0)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_hhmmss))
        ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)

        # Draw interval bands (selected intervals get stronger color/alpha)
        if self.subtitle_intervals:
            for i, (start, end) in enumerate(self.subtitle_intervals):
                if end >= _xmin and start <= _xmax:
                    color = "#ff9900" if i in self.selected_subtitle_indices else "orange"
                    alpha = 0.38 if i in self.selected_subtitle_indices else 0.18
                    ax.axvspan(max(start, _xmin), min(end, _xmax), color=color, alpha=alpha, zorder=0)

        self.canvas.draw()

    # ---------------- Mouse drag (scroll) logic ----------------

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
        """Translate horizontal drag into slider movement."""
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
                # Reset baseline to allow continuous drag without jumpiness
                self._drag_start_x_pixel = event.x
                self._drag_start_slider = new_slider


# ============================= Worker for Background Analysis =============================

class AnalyzeWorker(QObject):
    """
    Offloads media + subtitle loading & preprocessing to a background thread
    to keep the GUI responsive. Emits progress messages (value,message) and a
    single result object when complete.
    """
    progress = pyqtSignal(int, str)          # (percent, message)
    finished = pyqtSignal(object)            # result dict
    failed = pyqtSignal(str)                 # error detail
    cancelled = pyqtSignal()                 # if aborted mid-run

    def __init__(self, ref_media_path: str, new_media_path: str, srt_path: str):
        super().__init__()
        self.ref_media_path = ref_media_path
        self.new_media_path = new_media_path
        self.srt_path = srt_path
        self._abort = False

    def abort(self):
        """Request cooperative cancellation."""
        self._abort = True

    def _check_abort(self):
        if self._abort:
            self.cancelled.emit()
            raise RuntimeError("__ABORT__")

    def _load_audio(self, path: str):
        """
        Load audio via pydub, return (display_samples, effective_sample_rate, full_segment).

        IMPORTANT FIX:
        When we downsample the samples array for plotting (to keep it lightweight),
        we must ALSO reduce the reported sample rate by the same factor. Previously
        we always returned the original samplerate even after decimating every 10th
        sample, which shrank the computed total_duration (len / sr) by 10x. That is
        why you could only scroll to about 4:43 instead of the true full length.

        This method now returns the adjusted (display) sample rate when downsampling.
        """
        audio = AudioSegment.from_file(path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        if audio.channels > 1:
            # reshape to (channels, n_samples)
            samples = samples.reshape((-1, audio.channels)).T

        # Normalize to [-1,1]
        samples /= (2 ** (8 * audio.sample_width - 1))
        original_sr = audio.frame_rate

        # Downsample very large signals for plotting speed
        if samples.size > 5_000_000:
            factor = 10
            # Decimate by fixed stride
            display = samples[::factor] if samples.ndim == 1 else samples[:, ::factor]
            display_sr = max(1, original_sr // factor)
        else:
            display = samples
            display_sr = original_sr

        return display, display_sr, audio

    def run(self):
        """
        Main execution routine. Emits progress updates.
        Packs all data needed for UI consumption in one dict and emits finished.
        """
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
            if str(ex) != "__ABORT__":
                self.failed.emit(f"Aborted: {ex}")
        except Exception as e:
            self.failed.emit(str(e))


# ============================= Worker for Background Analysis =============================
class OffsetWorker(QObject):
    """
    Background worker to compute offsets for selected subtitle rows using
    audio_offset_finder.find_offset_between_buffers(buffer1, buffer2, fs, ...)

    buffer1: full reference mono waveform
    buffer2: snippet (new media) mono waveform for each row (resampled to reference rate if needed)
    Returned dict['time_offset'] is where buffer2 aligns relative to buffer1 (seconds).

    We output delta = time_offset - subtitle_start_time  (difference between detected location in reference
    and the nominal start time from the subtitle). Adjust sign logic if you prefer storing raw time_offset.
    """
    progress = pyqtSignal(int, str)              # row_index, message
    result = pyqtSignal(int, object, str)        # row_index, delta(float)|None, status ("ok"/reason)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self,
                 ref_wave: np.ndarray,
                 ref_sr: int,
                 new_wave: np.ndarray,
                 new_sr: int,
                 rows: List[Tuple[int, float, float]],   # (row_index, start_sec, end_sec)
                 min_duration: float = 1.0):
        super().__init__()
        self.ref_wave = ref_wave.astype(np.float32)
        self.ref_sr = ref_sr
        self.new_wave = new_wave.astype(np.float32)
        self.new_sr = new_sr
        self.rows = rows
        self.min_duration = min_duration
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
            for idx, start_sec, end_sec in self.rows:
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

                time_offset = float(res["time_offset"])
                # CHANGED: invert sign so what formerly showed e.g. -19.123 now shows +19.123
                # Previous: delta = time_offset - start_sec
                delta = start_sec - time_offset

                self.result.emit(idx, delta, "ok")
                self.progress.emit(idx, f"row {idx+1} done")
            self.finished.emit()
        except RuntimeError as ex:
            if str(ex) == "__ABORT__":
                self.failed.emit("Aborted")
            else:
                self.failed.emit(str(ex))
        except Exception as e:
            self.failed.emit(str(e))

# ============================= Busy Dialog (Indeterminate Progress) =============================

class BusyDialog(QDialog):
    """
    Simple modal dialog with a label + animated (pulsing) QProgressBar.
    Removes the Windows "?" help button for a cleaner look.
    """
    def __init__(self, parent=None, title="Working", message="Please wait..."):
        super().__init__(parent)
        # Remove context help button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)  # determinate range (we animate ourselves)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self._pulse_value = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.start(60)  # ~16 FPS (1000/60ms) -> smooth enough

    def _pulse(self):
        """Simple wrap-around pulsing animation (0..100)."""
        self._pulse_value = (self._pulse_value + 3) % 101
        self.bar.setValue(self._pulse_value)

    def set_message(self, text: str):
        """Update message (safe to call from queued signal in GUI thread)."""
        self.label.setText(text)


# ============================= Main Application Window =============================

class MainWindow(QWidget):
    """
    Main GUI:
      - File selection for reference/new media and subtitle paths.
      - Two waveform plots (reference / new).
      - Two tables (reference subtitles / working copy for sync adjustments).
      - Context menus: play, jump, delete, shift times.
      - Background analysis using QThread + AnalyzeWorker.
      - Waveform interval visualization and selection highlighting.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle Sync Assistant")
        self.resize(1024, 600)

        # UI build
        self.init_ui()

        # Audio playback state
        self.ref_audio_segment: Optional[AudioSegment] = None
        self.new_audio_segment: Optional[AudioSegment] = None
        self.audio_output: Optional[QAudioOutput] = None
        self.audio_buffer: Optional[QBuffer] = None
        self.audio_data: Optional[QByteArray] = None

        # Analysis thread state
        self._analyze_thread: Optional[QThread] = None
        self._analyze_worker: Optional[AnalyzeWorker] = None
        self._busy_dialog: Optional[BusyDialog] = None

        # Offset finder state
        self._offset_thread: Optional[QThread] = None
        self._offset_worker: Optional[OffsetWorker] = None
        self._ref_mono_cache: Optional[np.ndarray] = None
        self._ref_sr_cache: Optional[int] = None
        self._new_mono_cache: Optional[np.ndarray] = None
        self._new_sr_cache: Optional[int] = None

    # ---------- Utility: Table alignment ----------

    def align_table_columns_left(self, table: QTableWidget):
        """Ensure all headers & cells are left-aligned for readability."""
        if not table:
            return
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for row in range(table.rowCount()):
                cell = table.item(row, col)
                if cell:
                    cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    # ---------- UI Construction ----------

    def init_ui(self):
        """Construct all widgets / layouts and connect signals."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(4)

        # Row 1: Reference media
        row1 = QHBoxLayout()
        self.btn1 = QPushButton("Load reference media file")
        self.btn1.setFixedWidth(220)
        self.le1 = QLineEdit()
        self.le1.setReadOnly(True)
        self.le1.setPlaceholderText("No file selected")
        row1.addWidget(self.btn1)
        row1.addWidget(self.le1)
        self.btn1.clicked.connect(self.select_media_file_btn1)
        main_layout.addLayout(row1)

        # Row 2: New media
        row2 = QHBoxLayout()
        self.btn2 = QPushButton("Load new media file")
        self.btn2.setFixedWidth(220)
        self.le2 = QLineEdit()
        self.le2.setReadOnly(True)
        self.le2.setPlaceholderText("No file selected")
        row2.addWidget(self.btn2)
        row2.addWidget(self.le2)
        self.btn2.clicked.connect(self.select_media_file_btn2)
        main_layout.addLayout(row2)

        # Row 3: Subtitles
        row3 = QHBoxLayout()
        self.btn3 = QPushButton("Load reference subtitle")
        self.btn3.setFixedWidth(220)
        self.le3 = QLineEdit()
        self.le3.setReadOnly(True)
        self.le3.setPlaceholderText("No file selected")
        row3.addWidget(self.btn3)
        row3.addWidget(self.le3)
        self.btn3.clicked.connect(self.select_subtitle_file_btn3)
        main_layout.addLayout(row3)

        # Row 4: Save path
        row4 = QHBoxLayout()
        self.btn4 = QPushButton("Save subtitle under...")
        self.btn4.setFixedWidth(220)
        self.le4 = QLineEdit()
        self.le4.setReadOnly(True)
        self.le4.setPlaceholderText("No file selected")
        row4.addWidget(self.btn4)
        row4.addWidget(self.le4)
        self.btn4.clicked.connect(self.save_subtitle_file_btn4)
        main_layout.addLayout(row4)

        # Row 5: Analyze
        row5 = QHBoxLayout()
        self.btn5 = QPushButton("Analyze...")
        row5.addWidget(self.btn5)
        self.btn5.clicked.connect(self.on_analyze)
        main_layout.addLayout(row5)

        # Separator
        main_layout.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setLineWidth(1)
        main_layout.addWidget(sep)
        main_layout.addSpacing(8)

        # Waveform plots
        self.plot1 = MatplotlibPlotWidget("Reference Audio Waveform")
        self.plot2 = MatplotlibPlotWidget("New Audio Waveform")
        main_layout.addWidget(self.plot1)
        main_layout.addWidget(self.plot2)

        main_layout.addSpacing(8)

        # Tables container row
        tables_row = QHBoxLayout()

        # Reference table
        self.referencetable = QTableWidget(0, 3)
        self.referencetable.setHorizontalHeaderLabels(["Start time", "End time", "Text"])
        self._init_table_column_sizing(self.referencetable, fixed_cols=[0, 1], stretch_cols=[2])
        self.referencetable.setMinimumHeight(300)
        self.referencetable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.referencetable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.referencetable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.referencetable.customContextMenuRequested.connect(self.show_referencetable_context_menu)

        # Sync table
        self.synctable = QTableWidget(0, 4)
        self.synctable.setHorizontalHeaderLabels(["Start time", "End time", "Text", "Found offset"])
        self._init_table_column_sizing(self.synctable, fixed_cols=[0, 1, 3], stretch_cols=[2])
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

        # Initial alignment
        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

        # Selection -> highlight callbacks
        self.referencetable.selectionModel().selectionChanged.connect(self.on_reference_table_selection)
        self.synctable.selectionModel().selectionChanged.connect(self.on_sync_table_selection)

    def _init_table_column_sizing(self, table: QTableWidget, fixed_cols: List[int], stretch_cols: List[int]):
        """Configure table column resize behavior: fixed width for some, stretch for others."""
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for col in fixed_cols:
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Fixed)
        for col in stretch_cols:
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
        # Provide typical widths
        for col in fixed_cols:
            table.setColumnWidth(col, 100)

    # ---------- Selections -> highlight in plots ----------

    def on_reference_table_selection(self):
        """Update highlighted subtitle bands for reference plot based on selection."""
        if not hasattr(self, "referencetable"):
            return
        indices = [idx.row() for idx in self.referencetable.selectionModel().selectedRows()]
        self.plot1.set_selected_subtitle_indices(indices)

    def on_sync_table_selection(self):
        """Update highlighted subtitle bands for new media plot based on selection."""
        if not hasattr(self, "synctable"):
            return
        indices = [idx.row() for idx in self.synctable.selectionModel().selectedRows()]
        self.plot2.set_selected_subtitle_indices(indices)

    # ---------- Context Menus ----------

    def show_synctable_context_menu(self, pos):
        """Right-click menu for sync table rows."""
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)
        act_delete = QAction("Delete line(s)", self)
        act_shift_sel = QAction("Shift times for selected line(s)", self)
        act_shift_all = QAction("Shift all times", self)
        act_find_offsets = QAction("Find Offset(s) (BBC-offset-finder)", self)  # NEW

        menu.addAction(act_play)
        menu.addAction(act_jump)
        menu.addSeparator()
        menu.addAction(act_delete)
        menu.addSeparator()
        menu.addAction(act_shift_sel)
        menu.addAction(act_shift_all)
        menu.addSeparator()
        menu.addAction(act_find_offsets)  # NEW last

        act_play.triggered.connect(self.synctable_play_selected)
        act_jump.triggered.connect(self.synctable_jump_to_selected)
        act_delete.triggered.connect(self.synctable_delete_selected)
        act_shift_sel.triggered.connect(lambda: self.shift_times(selected_only=True))
        act_shift_all.triggered.connect(lambda: self.shift_times(selected_only=False))
        act_find_offsets.triggered.connect(self.find_offsets_for_selected)  # NEW hook

        menu.exec_(self.synctable.viewport().mapToGlobal(pos))

    def show_referencetable_context_menu(self, pos):
        """Right-click menu for reference table rows."""
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)
        menu.addAction(act_play)
        menu.addAction(act_jump)
        act_play.triggered.connect(self.referencetable_play_selected)
        act_jump.triggered.connect(self.referencetable_jump_to_selected)
        menu.exec_(self.referencetable.viewport().mapToGlobal(pos))

    # ---------- Playback & Navigation ----------

    def referencetable_play_selected(self):
        """Play audio snippet from reference media for first selected reference row."""
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "Play", "No row selected.")
            return
        row = selected[0].row()

        # Lazy load reference audio
        if self.ref_audio_segment is None:
            path = self.le1.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Play", "Reference media file not loaded.")
                return
            try:
                self.ref_audio_segment = AudioSegment.from_file(path)
            except Exception as e:
                QMessageBox.critical(self, "Play", f"Failed to load reference media:\n{e}")
                return

        start_item = self.referencetable.item(row, 0)
        end_item = self.referencetable.item(row, 1)
        if not start_item or not end_item:
            QMessageBox.warning(self, "Play", "Missing start / end time.")
            return

        try:
            start_sec = self._parse_time_to_seconds(start_item.text())
            end_sec = self._parse_time_to_seconds(end_item.text())
        except Exception:
            QMessageBox.warning(self, "Play", "Time parse error.")
            return

        if end_sec <= start_sec:
            QMessageBox.warning(self, "Play", "End time must be greater than start.")
            return

        self._play_audio_segment(self.ref_audio_segment, start_sec, end_sec)

    def synctable_play_selected(self):
        """Play audio snippet from NEW media for first selected sync row."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "Play", "No row selected.")
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
                QMessageBox.critical(self, "Play", f"Failed to load new media:\n{e}")
                return

        start_item = self.synctable.item(row, 0)
        end_item = self.synctable.item(row, 1)
        if not start_item or not end_item:
            QMessageBox.warning(self, "Play", "Missing start / end time.")
            return
        try:
            start_sec = self._parse_time_to_seconds(start_item.text())
            end_sec = self._parse_time_to_seconds(end_item.text())
        except Exception:
            QMessageBox.warning(self, "Play", "Time parse error.")
            return
        if end_sec <= start_sec:
            QMessageBox.warning(self, "Play", "End time must be greater than start.")
            return
        self._play_audio_segment(self.new_audio_segment, start_sec, end_sec)

    def _play_audio_segment(self, segment: AudioSegment, start_sec: float, end_sec: float):
        """
        Play a clipped audio segment (start_sec..end_sec) using QAudioOutput.
        Recreates audio output each time (simple approach; efficient enough for short clips).
        """
        if not segment:
            QMessageBox.warning(self, "Playback", "Audio segment not available.")
            return

        total_dur = len(segment) / 1000.0
        start_sec = max(0.0, start_sec)
        end_sec = min(total_dur, end_sec)
        if end_sec <= start_sec:
            QMessageBox.warning(self, "Playback", "Invalid playback interval.")
            return

        # Extract slice
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        part = segment[start_ms:end_ms]

        # Ensure format is 16-bit PCM for Qt
        if part.sample_width != 2:
            part = part.set_sample_width(2)

        # Prepare Qt audio format
        fmt = QAudioFormat()
        fmt.setSampleRate(part.frame_rate)
        fmt.setChannelCount(part.channels)
        fmt.setSampleSize(part.sample_width * 8)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)

        # Stop previous playback if any
        if self.audio_output:
            self.audio_output.stop()
            self.audio_output.deleteLater()
            self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer.deleteLater()
            self.audio_buffer = None

        # Create buffer over raw data & start playback
        self.audio_data = QByteArray(part.raw_data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)
        self.audio_output = QAudioOutput(fmt, self)
        self.audio_output.start(self.audio_buffer)

    # ---------- Jump (scroll plot to selection) ----------

    def referencetable_jump_to_selected(self):
        """Center reference plot on earliest selected start time."""
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        indices = []
        for idx in selected:
            row = idx.row()
            itm = self.referencetable.item(row, 0)
            if itm:
                starts.append(self._parse_time_to_seconds(itm.text()))
                indices.append(row)
        if not starts:
            return
        self.plot1.jump_to_time(min(starts), center=True)
        self.plot1.set_selected_subtitle_indices(indices)

    def synctable_jump_to_selected(self):
        """Center new-media plot on earliest selected sync start time."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        indices = []
        for idx in selected:
            row = idx.row()
            itm = self.synctable.item(row, 0)
            if itm:
                starts.append(self._parse_time_to_seconds(itm.text()))
                indices.append(row)
        if not starts:
            return
        self.plot2.jump_to_time(min(starts), center=True)
        self.plot2.set_selected_subtitle_indices(indices)

    # ---------- Sync Table Editing (delete / shift) ----------

    def _collect_synctable_intervals(self) -> List[Tuple[float, float]]:
        """Internal: recompute all intervals from sync table current values."""
        intervals: List[Tuple[float, float]] = []
        for row in range(self.synctable.rowCount()):
            s_item = self.synctable.item(row, 0)
            e_item = self.synctable.item(row, 1)
            if not s_item or not e_item:
                continue
            start_sec = self._parse_time_to_seconds(s_item.text())
            end_sec = self._parse_time_to_seconds(e_item.text())
            intervals.append((start_sec, end_sec))
        return intervals

    def synctable_delete_selected(self):
        """Remove all selected rows from sync table and refresh plot bands."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "Delete", "No rows selected.")
            return
        for idx in sorted(selected, key=lambda i: i.row(), reverse=True):
            self.synctable.removeRow(idx.row())
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        self.plot2.set_selected_subtitle_indices([])

    def shift_times(self, selected_only: bool):
        """
        Shift start/end times for selected or all rows by a user-specified offset (seconds).
        Negative values shift backward in time.
        """
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Shift Times", "No rows to shift.")
            return

        if selected_only:
            target = [idx.row() for idx in self.synctable.selectionModel().selectedRows()]
            if not target:
                QMessageBox.information(self, "Shift Times", "No rows selected.")
                return
            prompt = "Shift selected line(s) by seconds (e.g. -1.250 or 2.5):"
        else:
            target = list(range(self.synctable.rowCount()))
            prompt = "Shift ALL lines by seconds (e.g. -1.250 or 2.5):"

        val_str, ok = QInputDialog.getText(self, "Shift Times", prompt, text="0.000")
        if not ok or not val_str.strip():
            return
        try:
            delta = float(val_str.replace(",", "."))
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Could not parse shift value.")
            return

        for row in target:
            s_item = self.synctable.item(row, 0)
            e_item = self.synctable.item(row, 1)
            if not s_item or not e_item:
                continue
            s = self._parse_time_to_seconds(s_item.text()) + delta
            e = self._parse_time_to_seconds(e_item.text()) + delta
            s = max(0.0, s)
            e = max(s, e)
            s_item.setText(self._format_seconds_to_time(s))
            e_item.setText(self._format_seconds_to_time(e))

        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())

    # ---------- File Selection Helpers ----------

    def select_media_file_btn1(self):
        filters = "Media files (*.avi *.mkv *.mpg *.mpeg *.mp4 *.mov *.wmv *.flv *.webm);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Media", "", filters)
        if path:
            self.le1.setText(path)

    def select_media_file_btn2(self):
        filters = "Media files (*.avi *.mkv *.mpg *.mpeg *.mp4 *.mov *.wmv *.flv *.webm);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select New Media", "", filters)
        if path:
            self.le2.setText(path)

    def select_subtitle_file_btn3(self):
        filters = "Subtitle files (*.srt);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Subtitle", "", filters)
        if path:
            self.le3.setText(path)
            base, ext = os.path.splitext(path)
            # Auto-suggest output file (resync suffix)
            self.le4.setText(f"{base}_resync{ext}")

    def save_subtitle_file_btn4(self):
        filters = "Subtitle files (*.srt);;All files (*.*)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Subtitle As", "", filters)
        if path:
            if not os.path.splitext(path)[1]:
                path += ".srt"
            self.le4.setText(path)

    # ---------- Analysis Lifecycle ----------

    def sanity_check_files(self) -> bool:
        """Return True if all required paths look valid, else warn user."""
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
            QMessageBox.warning(self, "Missing Files",
                                "The following are missing or invalid:\n\n" + "\n".join(missing))
            return False
        return True

    def on_analyze(self):
        """
        Kick off asynchronous loading + parsing of media & subtitles.
        Shows BusyDialog while work proceeds. Disables Analyze button to
        prevent concurrent runs.
        """
        if self._analyze_thread is not None:
            QMessageBox.information(self, "Analyze", "Analysis already running.")
            return
        if not self.sanity_check_files():
            return

        ref_media = self.le1.text().strip()
        new_media = self.le2.text().strip()
        srt_path = self.le3.text().strip()

        # Show busy dialog immediately
        self._busy_dialog = BusyDialog(self, title="Analyzing", message="Starting analysis …")
        self._busy_dialog.show()

        # Worker + Thread
        thread = QThread()
        worker = AnalyzeWorker(ref_media, new_media, srt_path)
        self._analyze_worker = worker
        worker.moveToThread(thread)

        # Connect signals
        worker.progress.connect(lambda _v, msg: self._busy_dialog and self._busy_dialog.set_message(msg))

        def finished(result):
            self._teardown_analysis()
            try:
                self._apply_analysis_result(result)
            except Exception as e:
                QMessageBox.critical(self, "Result Error", f"{e}")

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

        worker.finished.connect(finished)
        worker.failed.connect(failed)
        worker.cancelled.connect(cancelled)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        worker.cancelled.connect(cleanup)
        thread.started.connect(worker.run)

        # Launch
        self.btn5.setEnabled(False)
        self._analyze_thread = thread
        thread.start()

    def _teardown_analysis(self):
        """Internal helper: re-enable controls & close dialog after worker ends."""
        self.btn5.setEnabled(True)
        if self._busy_dialog:
            try:
                self._busy_dialog.close()
            except Exception:
                pass
            self._busy_dialog = None
        self._analyze_thread = None

    def _apply_analysis_result(self, result: dict):
        """
        Populate tables + plots from analysis outcome:
          - Waveforms
          - Subtitle rows
          - Interval bands
        """
        # Store full-quality audio segments for playback
        self.ref_audio_segment = result["ref_full"]
        self.new_audio_segment = result["new_full"]

        # Waveforms
        self.plot1.plot_waveform(result["ref_display"], result["ref_rate"])
        self.plot2.plot_waveform(result["new_display"], result["new_rate"])

        # Tables
        rows = result["rows"]
        fmt_time = lambda t: f"{t.hours:02}:{t.minutes:02}:{t.seconds:02},{t.milliseconds:03}"

        self.referencetable.setRowCount(len(rows))
        self.synctable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            # Reference view
            self.referencetable.setItem(i, 0, QTableWidgetItem(fmt_time(r["start"])))
            self.referencetable.setItem(i, 1, QTableWidgetItem(fmt_time(r["end"])))
            self.referencetable.setItem(i, 2, QTableWidgetItem(r["text"]))


            # Sync view (copy + empty offset col)
            self.synctable.setItem(i, 0, QTableWidgetItem(fmt_time(r["start"])))
            self.synctable.setItem(i, 1, QTableWidgetItem(fmt_time(r["end"])))
            self.synctable.setItem(i, 2, QTableWidgetItem(r["text"]))
            self.synctable.setItem(i, 3, QTableWidgetItem(""))

            # Row striping
            bg = QColor(245, 245, 245) if i % 2 == 0 else QColor(230, 230, 230)
            for c in range(3):
                self.referencetable.item(i, c).setBackground(bg)
            for c in range(4):
                self.synctable.item(i, c).setBackground(bg)

        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

        # Interval shading
        self.plot1.set_subtitle_intervals(result["intervals"])
        self.plot2.set_subtitle_intervals(result["intervals"])

    def _ensure_audio_caches_for_offsets(self) -> bool:
        """
        Ensure mono float32 numpy arrays + sample rates for reference and new media
        are ready in self._ref_mono_cache / self._new_mono_cache.
        Returns True on success, False on failure (after showing a message box).
        """
        if find_offset_between_buffers is None:
            QMessageBox.warning(self, "Offset Finder",
                                "audio-offset-finder not installed.\nRun:\n  pip install audio-offset-finder")
            return False

        # Load full audio segments if not already loaded
        if self.ref_audio_segment is None:
            ref_path = self.le1.text().strip()
            if not ref_path or not os.path.exists(ref_path):
                QMessageBox.warning(self, "Offset Finder", "Reference media file not available.")
                return False
            try:
                self.ref_audio_segment = AudioSegment.from_file(ref_path)
            except Exception as e:
                QMessageBox.critical(self, "Offset Finder", f"Could not load reference media:\n{e}")
                return False

        if self.new_audio_segment is None:
            new_path = self.le2.text().strip()
            if not new_path or not os.path.exists(new_path):
                QMessageBox.warning(self, "Offset Finder", "New media file not available.")
                return False
            try:
                self.new_audio_segment = AudioSegment.from_file(new_path)
            except Exception as e:
                QMessageBox.critical(self, "Offset Finder", f"Could not load new media:\n{e}")
                return False

        def to_mono_array(seg: AudioSegment) -> Tuple[np.ndarray, int]:
            arr = np.array(seg.get_array_of_samples()).astype(np.float32)
            if seg.channels > 1:
                arr = arr.reshape((-1, seg.channels)).mean(axis=1)
            arr /= (2 ** (8 * seg.sample_width - 1))
            return arr, seg.frame_rate

        if self._ref_mono_cache is None or self._ref_sr_cache is None:
            self._ref_mono_cache, self._ref_sr_cache = to_mono_array(self.ref_audio_segment)
        if self._new_mono_cache is None or self._new_sr_cache is None:
            self._new_mono_cache, self._new_sr_cache = to_mono_array(self.new_audio_segment)

        return True
    
    def find_offsets_for_selected(self):
        """Compute offsets for currently selected sync table rows using BBC audio-offset-finder."""
        if self.synctable.rowCount() == 0:
            QMessageBox.information(self, "Offset Finder", "No rows available.")
            return

        selection = self.synctable.selectionModel().selectedRows()
        if not selection:
            QMessageBox.information(self, "Offset Finder", "No rows selected.")
            return

        if not self._ensure_audio_caches_for_offsets():
            return

        if self._offset_thread is not None:
            QMessageBox.information(self, "Offset Finder", "Offset computation already running.")
            return

        # Collect (row_index, start_sec, end_sec)
        rows: List[Tuple[int, float, float]] = []
        for idx in selection:
            r = idx.row()
            s_item = self.synctable.item(r, 0)
            e_item = self.synctable.item(r, 1)
            if not s_item or not e_item:
                continue
            start_sec = self._parse_time_to_seconds(s_item.text())
            end_sec = self._parse_time_to_seconds(e_item.text())
            rows.append((r, start_sec, end_sec))

        if not rows:
            QMessageBox.information(self, "Offset Finder", "Selected rows have no valid times.")
            return

        # Progress dialog (reuse BusyDialog pattern)
        self._busy_offset = BusyDialog(self, title="Finding Offsets", message="Searching …")
        self._busy_offset.show()

        thread = QThread()
        worker = OffsetWorker(
            ref_wave=self._ref_mono_cache,
            ref_sr=self._ref_sr_cache,
            new_wave=self._new_mono_cache,
            new_sr=self._new_sr_cache,
            rows=rows,
            min_duration=1.0
        )
        self._offset_thread = thread
        self._offset_worker = worker
        worker.moveToThread(thread)

        def on_result(row_index: int, delta_val, status: str):
            # Column 3 = "Found offset"
            cell = self.synctable.item(row_index, 3)
            if cell is None:
                self.synctable.setItem(row_index, 3, QTableWidgetItem(""))
                cell = self.synctable.item(row_index, 3)
            if delta_val is None:
                cell.setText(status)
                cell.setBackground(QColor(240, 240, 200) if status not in ("err",) else QColor(255, 210, 210))
            else:
                sign = "+" if delta_val >= 0 else "-"
                cell.setText(f"{sign}{abs(delta_val):.3f}")
                cell.setBackground(QColor(210, 245, 210) if status == "ok" else QColor(255, 210, 210))

        def on_progress(row_index: int, msg: str):
            if self._busy_offset:
                self._busy_offset.set_message(f"Processing {msg}")

        def on_finished():
            self._finish_offset_worker()

        def on_failed(err: str):
            QMessageBox.critical(self, "Offset Finder", err)
            self._finish_offset_worker()

        def cleanup():
            thread.quit()
            thread.wait()
            self._offset_thread = None
            self._offset_worker = None

        worker.result.connect(on_result)
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.finished.connect(cleanup)
        worker.failed.connect(cleanup)
        thread.started.connect(worker.run)
        thread.start()

    def _finish_offset_worker(self):
        if hasattr(self, "_busy_offset") and self._busy_offset:
            try:
                self._busy_offset.close()
            except Exception:
                pass
            self._busy_offset = None

    # ---------- Time helpers (parsing / formatting) ----------

    def _parse_time_to_seconds(self, text: str) -> float:
        """
        Parse SRT-style timestamp "HH:MM:SS,mmm" into floating seconds.
        Returns 0.0 on failure (caller may decide how to handle).
        """
        try:
            h, m, rest = text.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        except Exception:
            return 0.0

    def _format_seconds_to_time(self, seconds: float) -> str:
        """Format floating seconds into SRT timestamp "HH:MM:SS,mmm"."""
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        # Normalize rounding overspill
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


# ============================= Entrypoint =============================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())