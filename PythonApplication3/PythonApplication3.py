import sys
import os
from PyQt5.QtCore import Qt, QUrl, QByteArray, QBuffer, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QSlider, QAbstractItemView,
    QMenu, QAction, QInputDialog, QProgressBar, QDialog
)
from PyQt5.QtMultimedia import QAudioFormat, QAudioOutput   # <-- NEW (Qt based audio playback)
from pydub import AudioSegment
import numpy as np
import matplotlib.ticker as mticker

import pysrt
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import List, Tuple

class MatplotlibPlotWidget(QFrame):
    def __init__(self, title="Plot", window_duration=20):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setMinimumHeight(180)
        self.setMaximumHeight(180)
        self.window_duration = window_duration  # seconds
        self.samples = None
        self.sr = None
        self.subtitle_intervals: List[Tuple[float, float]] = []
        self.selected_subtitle_indices: set[int] = set()  # support multi-selection

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Optionally comment out the next two lines to remove the label above the plot:
        self.label = QLabel(title, self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.figure = Figure(figsize=(5, 2))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add slider for scrolling
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_plot_from_slider)
        layout.addWidget(self.slider)

        # Mouse drag state
        self._dragging = False
        self._drag_start_x = None
        self._drag_start_slider = None

        # Connect matplotlib mouse events
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # --- NEW: programmatic jump helper ---
    def jump_to_time(self, target_sec: float, center: bool = True):
        """
        Scroll the window so that target_sec is visible (optionally centered) and refresh plot.
        """
        if self.samples is None or self.sr is None:
            return
        if self.total_duration <= self.window_duration:
            # Whole signal already visible
            self._plot_window(0)
            return

        if center:
            start = target_sec - self.window_duration / 2.0
        else:
            start = target_sec

        start = max(0.0, min(start, self.total_duration - self.window_duration))

        # Update slider (integer seconds resolution)
        slider_value = int(start)
        signals_were_blocked = self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(signals_were_blocked)

        # Plot with the (possibly fractional) start for better centering
        self._plot_window(start)

    def plot_waveform(self, samples, sr):
        self.samples = samples
        self.sr = sr
        if samples.ndim > 1:
            samples_mono = samples.mean(axis=0)
        else:
            samples_mono = samples
        self.samples_mono = samples_mono
        self.total_duration = len(samples_mono) / sr if sr > 0 else 1

        if self.total_duration > self.window_duration:
            max_slider = int(self.total_duration - self.window_duration)
            self.slider.setMaximum(max_slider)
            self.slider.setEnabled(True)
        else:
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
        self.slider.setValue(0)
        self._plot_window(0)

    def update_plot_from_slider(self, value):
        self._plot_window(value)

    def _format_hhmmss(self, seconds, pos=None):
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02}:{m:02}:{s:02}"

    def set_selected_subtitle_indices(self, indices: List[int]):
        """Set (possibly multiple) selected subtitle indices to highlight."""
        self.selected_subtitle_indices = set(i for i in indices if i is not None)
        self._plot_window(self.slider.value())

    def set_selected_subtitle_index(self, index: int):
        """Backward compatible single selection API."""
        if index is None:
            self.selected_subtitle_indices = set()
        else:
            self.selected_subtitle_indices = {index}
        self._plot_window(self.slider.value())

    def _plot_window(self, start_sec):
        if self.samples is None or self.sr is None:
            return
        samples_mono = self.samples_mono
        sr = self.sr
        total_len = len(samples_mono)
        t = np.linspace(0, self.total_duration, num=total_len)
        _xmin = start_sec
        _xmax = min(start_sec + self.window_duration, self.total_duration)
        idx_min = int(_xmin * sr)
        idx_max = int(_xmax * sr)
        idx_max = min(idx_max, total_len)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t[idx_min:idx_max], samples_mono[idx_min:idx_max], linewidth=1.0)
        ax.set_xlim([_xmin, _xmax])
        ax.set_ylim([-1.05, 1.05])
        ax.set_ylabel('Amplitude', fontsize=8, labelpad=0)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_hhmmss))
        ax.tick_params(axis='both', which='major', labelsize=7, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)
        selected_indices = getattr(self, "selected_subtitle_indices", set())
        if self.subtitle_intervals:
            for i, (start, end) in enumerate(self.subtitle_intervals):
                if end >= _xmin and start <= _xmax:
                    # default band
                    color = 'orange'
                    alpha = 0.18
                    # highlight if selected
                    if i in selected_indices:
                        color = '#ff9900'
                        alpha = 0.38
                    ax.axvspan(max(start, _xmin), min(end, _xmax), color=color, alpha=alpha, zorder=0)
        self.canvas.draw()

    def _on_mouse_press(self, event):
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._drag_start_x_pixel = event.x
            self._drag_start_slider = self.slider.value()

    def _on_mouse_release(self, event):
        self._dragging = False
        self._drag_start_x_pixel = None
        self._drag_start_slider = None

    def _on_mouse_move(self, event):
        if self._dragging and event.inaxes and self._drag_start_x_pixel is not None:
            # Calculate how many seconds per pixel
            ax = event.inaxes
            bbox = ax.get_window_extent()
            axis_width_pixels = bbox.width
            if axis_width_pixels == 0:
                return
            seconds_per_pixel = self.window_duration / axis_width_pixels
            dx_pixels = self._drag_start_x_pixel - event.x
            dx_seconds = dx_pixels * seconds_per_pixel
            new_slider = int(self._drag_start_slider + dx_seconds)
            new_slider = max(self.slider.minimum(), min(self.slider.maximum(), new_slider))
            if new_slider != self.slider.value():
                self.slider.setValue(new_slider)
                # Update drag reference to avoid toggling/jumping
                self._drag_start_x_pixel = event.x
                self._drag_start_slider = new_slider

    def set_subtitle_intervals(self, intervals: List[Tuple[float, float]]):
        self.subtitle_intervals = intervals
        self._plot_window(self.slider.value())

    def plot_subtitle_bands(self, intervals: List[Tuple[float, float]], color='orange', alpha=0.18):
        """Draw translucent bands for subtitle intervals (list of (start, end) in seconds)."""
        ax = self.figure.gca()
        for start, end in intervals:
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)
        self.canvas.draw()

# --- Worker infrastructure for non-blocking Analyze ---
class AnalyzeWorker(QObject):
    progress = pyqtSignal(int, str)          # value 0..100, message
    finished = pyqtSignal(object)            # result dict
    failed = pyqtSignal(str)                 # error message
    cancelled = pyqtSignal()
    def __init__(self, ref_media_path:str, new_media_path:str, srt_path:str):
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

    def _load_audio(self, path:str):
        audio = AudioSegment.from_file(path)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).T
        samples = samples / (2 ** (8 * audio.sample_width - 1))
        samplerate = audio.frame_rate
        # Downsample for display if huge
        if samples.size > 5_000_000:
            display = samples[::10] if samples.ndim == 1 else samples[:, ::10]
            display_rate = samplerate // 10
        else:
            display = samples
            display_rate = samplerate
        return display, samplerate, audio  # keep original full-quality segment

    def run(self):
        try:
            # Step 0 sanity / start
            self.progress.emit(0, "Starting analysis...")
            self._check_abort()

            # Step 1 load reference media
            self.progress.emit(10, "Loading reference media...")
            ref_display, ref_rate, ref_full = self._load_audio(self.ref_media_path)
            self._check_abort()

            # Step 2 load new media
            self.progress.emit(30, "Loading new media...")
            new_display, new_rate, new_full = self._load_audio(self.new_media_path)
            self._check_abort()

            # Step 3 load subtitles
            self.progress.emit(50, "Parsing subtitles...")
            subs = pysrt.open(self.srt_path, encoding='utf-8')
            self._check_abort()

            # Step 4 prepare table data
            table_rows = []
            for sub in subs:
                table_rows.append({
                    "start": sub.start,
                    "end": sub.end,
                    "text": sub.text
                })
            self._check_abort()

            # Step 5 prepare intervals
            self.progress.emit(70, "Preparing intervals...")
            intervals = []
            for r in table_rows:
                st = r["start"]
                et = r["end"]
                start_sec = st.hours*3600 + st.minutes*60 + st.seconds + st.milliseconds/1000.0
                end_sec   = et.hours*3600 + et.minutes*60 + et.seconds + et.milliseconds/1000.0
                intervals.append((start_sec, end_sec))
            self._check_abort()

            # Step 6 final packaging
            self.progress.emit(90, "Finalizing...")
            result = {
                "ref_display": ref_display,
                "ref_rate": ref_rate,
                "ref_full": ref_full,
                "new_display": new_display,
                "new_rate": new_rate,
                "new_full": new_full,
                "rows": table_rows,
                "intervals": intervals
            }
            self.progress.emit(100, "Done")
            self.finished.emit(result)
        except RuntimeError as ex:
            if str(ex) != "__ABORT__":
                self.failed.emit(f"Aborted: {ex}")
        except Exception as e:
            self.failed.emit(str(e))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Button-LineEdit Pairs Example")
        self.resize(1024, 600)
        self.init_ui()
        self.ref_audio_segment: AudioSegment | None = None
        self.new_audio_segment: AudioSegment | None = None
        self.audio_output: QAudioOutput | None = None
        self.audio_buffer: QBuffer | None = None
        self.audio_data: QByteArray | None = None
        self._analyze_thread: QThread | None = None
        self._analyze_worker: AnalyzeWorker | None = None
        self._busy_dialog: 'BusyDialog | None' = None

    # ================== ADD / RESTORE MISSING CONTEXT MENU HELPERS ==================

    def _collect_synctable_intervals(self) -> List[Tuple[float, float]]:
        """Return list of (start_sec, end_sec) for every row currently in the sync table."""
        intervals: List[Tuple[float, float]] = []
        if not hasattr(self, "synctable"):
            return intervals
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
        """Delete selected row(s) from sync table and refresh plot bands."""
        if not hasattr(self, "synctable"):
            return
        sel = self.synctable.selectionModel().selectedRows() if self.synctable.selectionModel() else []
        if not sel:
            QMessageBox.information(self, "Delete", "No rows selected.")
            return
        # Remove from bottom to top
        for model_index in sorted(sel, key=lambda x: x.row(), reverse=True):
            self.synctable.removeRow(model_index.row())
        # Update intervals & clear highlight
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())
        self.plot2.set_selected_subtitle_indices([])

    def shift_times(self, selected_only: bool):
        """
        Shift start/end times in sync table by a user-entered delta (seconds, may be negative).
        If selected_only is True, only selected rows shift; else all rows.
        """
        if not hasattr(self, "synctable"):
            return
        table = self.synctable
        if selected_only:
            target_rows = [idx.row() for idx in (table.selectionModel().selectedRows()
                                                 if table.selectionModel() else [])]
            if not target_rows:
                QMessageBox.information(self, "Shift Times", "No rows selected.")
                return
            caption = "Shift selected line(s) (seconds, e.g. -1.250 or 2.5):"
        else:
            if table.rowCount() == 0:
                QMessageBox.information(self, "Shift Times", "No rows to shift.")
                return
            target_rows = list(range(table.rowCount()))
            caption = "Shift ALL lines (seconds, e.g. -1.250 or 2.5):"

        val_str, ok = QInputDialog.getText(self, "Shift Times", caption, text="0.000")
        if not ok or not val_str.strip():
            return
        val_str = val_str.replace(",", ".")
        try:
            delta = float(val_str)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Could not parse the shift value.")
            return

        for row in target_rows:
            s_item = table.item(row, 0)
            e_item = table.item(row, 1)
            if not s_item or not e_item:
                continue
            s_sec = self._parse_time_to_seconds(s_item.text())
            e_sec = self._parse_time_to_seconds(e_item.text())
            s_sec = max(0.0, s_sec + delta)
            e_sec = max(s_sec, e_sec + delta)  # prevent negative / inverted
            s_item.setText(self._format_seconds_to_time(s_sec))
            e_item.setText(self._format_seconds_to_time(e_sec))

        # Refresh interval shading
        self.plot2.set_subtitle_intervals(self._collect_synctable_intervals())

    def sanity_check_files(self) -> bool:
        """Validate that required file paths are set and exist."""
        missing = []
        if not self.le1.text() or not os.path.exists(self.le1.text()): missing.append("Reference media file")
        if not self.le2.text() or not os.path.exists(self.le2.text()): missing.append("New media file")
        if not self.le3.text() or not os.path.exists(self.le3.text()): missing.append("Reference subtitle file")
        if not self.le4.text(): missing.append("Subtitle save path")
        if missing:
            QMessageBox.warning(self, "Missing Files",
                                "The following are missing or invalid:\n\n" + "\n".join(missing))
            return False
        return True

    def on_analyze(self):
        """Asynchronous analyze entry point (audio + subtitles) with busy dialog."""
        if self._analyze_thread is not None:
            QMessageBox.information(self, "Analyze", "Analysis already running.")
            return
        if not self.sanity_check_files():
            return

        ref_media = self.le1.text().strip()
        new_media = self.le2.text().strip()
        srt_path  = self.le3.text().strip()

        # Busy dialog
        self._busy_dialog = BusyDialog(self, title="Analyzing", message="Starting analysis …")
        self._busy_dialog.show()

        # Thread + worker
        thread = QThread()
        worker = AnalyzeWorker(ref_media, new_media, srt_path)
        self._analyze_worker = worker
        worker.moveToThread(thread)

        # Signal handlers
        worker.progress.connect(lambda _val, msg: self._busy_dialog and self._busy_dialog.set_message(msg))

        def finished(result):
            self._teardown_analysis()
            try:
                self._apply_analysis_result(result)
            except Exception as e:
                QMessageBox.critical(self, "Apply Result Error", f"{e}")

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

        self.btn5.setEnabled(False)
        self._analyze_thread = thread
        thread.start()

    def _teardown_analysis(self):
        self.btn5.setEnabled(True)
        if self._busy_dialog:
            try: self._busy_dialog.close()
            except Exception: pass
            self._busy_dialog = None
        self._analyze_thread = None

    def _apply_analysis_result(self, result: dict):
        """Populate UI with analysis results."""
        # Assign full quality segments
        self.ref_audio_segment = result["ref_full"]
        self.new_audio_segment = result["new_full"]

        # Plot waveforms
        self.plot1.plot_waveform(result["ref_display"], result["ref_rate"])
        self.plot2.plot_waveform(result["new_display"], result["new_rate"])

        fmt_time = lambda t: f"{t.hours:02}:{t.minutes:02}:{t.seconds:02},{t.milliseconds:03}"
        rows = result["rows"]

        # Reference table
        self.referencetable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.referencetable.setItem(i, 0, QTableWidgetItem(fmt_time(r["start"])))
            self.referencetable.setItem(i, 1, QTableWidgetItem(fmt_time(r["end"])))
            self.referencetable.setItem(i, 2, QTableWidgetItem(r["text"]))
            bg = QColor(245,245,245) if i % 2 == 0 else QColor(230,230,230)
            for c in range(3):
                self.referencetable.item(i, c).setBackground(bg)

        # Sync table
        self.synctable.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.synctable.setItem(i, 0, QTableWidgetItem(fmt_time(r["start"])))
            self.synctable.setItem(i, 1, QTableWidgetItem(fmt_time(r["end"])))
            self.synctable.setItem(i, 2, QTableWidgetItem(r["text"]))
            self.synctable.setItem(i, 3, QTableWidgetItem(""))
            bg = QColor(245,245,245) if i % 2 == 0 else QColor(230,230,230)
            for c in range(4):
                self.synctable.item(i, c).setBackground(bg)

        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

        # Intervals to plots
        self.plot1.set_subtitle_intervals(result["intervals"])
        self.plot2.set_subtitle_intervals(result["intervals"])

    # Basic time helpers (if lost during refactors)
    def _parse_time_to_seconds(self, text: str) -> float:
        try:
            h, m, rest = text.split(":")
            s, ms = rest.split(",")
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
        except Exception:
            return 0.0

    def _format_seconds_to_time(self, seconds: float) -> str:
        if seconds < 0: seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds))*1000))
        if ms == 1000:
            ms = 0; s += 1
            if s == 60:
                s = 0; m += 1
                if m == 60:
                    m = 0; h += 1
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def align_table_columns_left(self, table: QTableWidget):
        """Left-align header text and cell contents for all columns of the given table."""
        if table is None:
            return
        cols = table.columnCount()
        rows = table.rowCount()
        for col in range(cols):
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for row in range(rows):
                cell = table.item(row, col)
                if cell:
                    cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(2)

        row1 = QHBoxLayout()
        self.btn1 = QPushButton("Load reference media file")
        self.btn1.setFixedWidth(220)
        self.le1 = QLineEdit()
        self.le1.setPlaceholderText("No file selected")
        self.le1.setReadOnly(True)
        row1.addWidget(self.btn1)
        row1.addWidget(self.le1)
        main_layout.addLayout(row1)
        self.btn1.clicked.connect(self.select_media_file_btn1)

        row2 = QHBoxLayout()
        self.btn2 = QPushButton("Load new media file")
        self.btn2.setFixedWidth(220)
        self.le2 = QLineEdit()
        self.le2.setPlaceholderText("No file selected")
        self.le2.setReadOnly(True)
        row2.addWidget(self.btn2)
        row2.addWidget(self.le2)
        main_layout.addLayout(row2)
        self.btn2.clicked.connect(self.select_media_file_btn2)

        row3 = QHBoxLayout()
        self.btn3 = QPushButton("Load reference subtitle")
        self.btn3.setFixedWidth(220)
        self.le3 = QLineEdit()
        self.le3.setPlaceholderText("No file selected")
        self.le3.setReadOnly(True)
        row3.addWidget(self.btn3)
        row3.addWidget(self.le3)
        main_layout.addLayout(row3)
        self.btn3.clicked.connect(self.select_subtitle_file_btn3)

        row4 = QHBoxLayout()
        self.btn4 = QPushButton("Save subtitle under...")
        self.btn4.setFixedWidth(220)
        self.le4 = QLineEdit()
        self.le4.setPlaceholderText("No file selected")
        self.le4.setReadOnly(True)
        row4.addWidget(self.btn4)
        row4.addWidget(self.le4)
        main_layout.addLayout(row4)
        self.btn4.clicked.connect(self.save_subtitle_file_btn4)

        row5 = QHBoxLayout()
        self.btn5 = QPushButton("Analyze...")
        row5.addWidget(self.btn5)
        main_layout.addLayout(row5)
        self.btn5.clicked.connect(self.on_analyze)

        # Add a horizontal line as a separator as well as some spacing
        main_layout.addSpacing(10)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(1)
        main_layout.addWidget(separator)
        main_layout.addSpacing(10)

        # --- Use MatplotlibPlotWidget for plots ---
        self.plot1 = MatplotlibPlotWidget("Reference Audio Waveform")
        self.plot2 = MatplotlibPlotWidget("New Audio Waveform")
        main_layout.addWidget(self.plot1)
        main_layout.addWidget(self.plot2)

        main_layout.addSpacing(10)

        tables_row = QHBoxLayout()

        self.referencetable = QTableWidget(0, 3)
        self.referencetable.setHorizontalHeaderLabels(["Start time", "End time", "Text"])
        self.referencetable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.referencetable.setColumnWidth(0, 100)
        self.referencetable.setColumnWidth(1, 100)
        self.referencetable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.referencetable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.referencetable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.referencetable.setMinimumHeight(300)
        self.referencetable.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Allow multi-row selection (Ctrl / Shift click)
        self.referencetable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Add context menu (Play only) for reference table
        self.referencetable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.referencetable.customContextMenuRequested.connect(self.show_referencetable_context_menu)

        self.synctable = QTableWidget(0, 4)
        self.synctable.setHorizontalHeaderLabels(["Start time", "End time", "Text", "Found offset"])
        self.synctable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.synctable.setColumnWidth(0, 100)
        self.synctable.setColumnWidth(1, 100)
        self.synctable.setColumnWidth(3, 100)
        self.synctable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.synctable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.synctable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.synctable.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.synctable.setMinimumHeight(300)
        self.synctable.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Allow multi-row selection (Ctrl / Shift click)
        self.synctable.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Add context menu for sync table
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

    def on_reference_table_selection(self):
        """
        Highlight selected subtitle bands in plot1 (reference) when the reference table selection changes.
        Connected to selectionModel().selectionChanged.
        """
        if not hasattr(self, "referencetable"):
            return
        selected_rows = self.referencetable.selectionModel().selectedRows()
        indices = [idx.row() for idx in selected_rows]
        self.plot1.set_selected_subtitle_indices(indices)

    def on_sync_table_selection(self):
        """
        Highlight selected subtitle bands in plot2 (new media) when the sync table selection changes.
        """
        if not hasattr(self, "synctable"):
            return
        selected_rows = self.synctable.selectionModel().selectedRows()
        indices = [idx.row() for idx in selected_rows]
        self.plot2.set_selected_subtitle_indices(indices)

    # --- Context menu handlers ---
    def show_synctable_context_menu(self, pos):
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)   # NEW
        act_delete = QAction("Delete line(s)", self)
        act_shift_sel = QAction("Shift times for selected line(s)", self)
        act_shift_all = QAction("Shift all times", self)

        menu.addAction(act_play)
        menu.addAction(act_jump)              # NEW
        menu.addSeparator()
        menu.addAction(act_delete)
        menu.addSeparator()
        menu.addAction(act_shift_sel)
        menu.addAction(act_shift_all)

        act_play.triggered.connect(self.synctable_play_selected)
        act_jump.triggered.connect(self.synctable_jump_to_selected)   # NEW
        act_delete.triggered.connect(self.synctable_delete_selected)
        act_shift_sel.triggered.connect(lambda: self.shift_times(selected_only=True))
        act_shift_all.triggered.connect(lambda: self.shift_times(selected_only=False))

        menu.exec_(self.synctable.viewport().mapToGlobal(pos))

    def show_referencetable_context_menu(self, pos):
        """Context menu for reference table (Play only + Jump)."""
        menu = QMenu(self)
        act_play = QAction("Play", self)
        act_jump = QAction("Jump to", self)    # NEW
        menu.addAction(act_play)
        menu.addAction(act_jump)
        act_play.triggered.connect(self.referencetable_play_selected)
        act_jump.triggered.connect(self.referencetable_jump_to_selected)  # NEW
        menu.exec_(self.referencetable.viewport().mapToGlobal(pos))

    def referencetable_play_selected(self):
        """Play the REFERENCE media audio (self.ref_audio_segment) for the first selected row in referencetable."""
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "Play", "No row selected.")
            return

        row = selected[0].row()

        # Ensure reference audio segment is loaded (lazy load if Analyze not yet run)
        if self.ref_audio_segment is None:
            ref_path = self.le1.text().strip()
            if not ref_path or not os.path.exists(ref_path):
                QMessageBox.warning(self, "Play", "Reference media file not available (load it first).")
                return
            try:
                self.ref_audio_segment = AudioSegment.from_file(ref_path)
            except Exception as e:
                QMessageBox.critical(self, "Play", f"Failed to load reference media file:\n{e}")
                return

        start_item = self.referencetable.item(row, 0)
        end_item = self.referencetable.item(row, 1)
        if not start_item or not end_item:
            QMessageBox.warning(self, "Play", "Missing start or end time in the selected row.")
            return

        try:
            start_sec = self._parse_time_to_seconds(start_item.text())
            end_sec = self._parse_time_to_seconds(end_item.text())
        except Exception:
            QMessageBox.warning(self, "Play", "Could not parse times.")
            return

        if end_sec <= start_sec:
            QMessageBox.warning(self, "Play", "End time must be greater than start time.")
            return

        self._play_audio_segment(self.ref_audio_segment, start_sec, end_sec)

    def synctable_play_selected(self):
        """Play NEW media audio for first selected sync table row."""
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "Play", "No row selected.")
            return
        row = selected[0].row()

        # Lazy load new audio if needed
        if self.new_audio_segment is None:
            path_new = self.le2.text().strip()
            if not path_new or not os.path.exists(path_new):
                QMessageBox.warning(self, "Play", "New media file not available.")
                return
            try:
                self.new_audio_segment = AudioSegment.from_file(path_new)
            except Exception as e:
                QMessageBox.critical(self, "Play", f"Failed to load new media:\n{e}")
                return

        start_item = self.synctable.item(row, 0)
        end_item = self.synctable.item(row, 1)
        if not start_item or not end_item:
            QMessageBox.warning(self, "Play", "Missing start/end time.")
            return
        try:
            start_sec = self._parse_time_to_seconds(start_item.text())
            end_sec = self._parse_time_to_seconds(end_item.text())
        except Exception:
            QMessageBox.warning(self, "Play", "Could not parse times.")
            return
        if end_sec <= start_sec:
            QMessageBox.warning(self, "Play", "End must be greater than start.")
            return
        self._play_audio_segment(self.new_audio_segment, start_sec, end_sec)

    def _play_audio_segment(self, segment: AudioSegment, start_sec: float, end_sec: float):
        """
        Play a portion of an AudioSegment (start_sec <= t < end_sec) via QAudioOutput.
        Replaces the previously missing helper required by context menu Play actions.
        """
        if segment is None:
            QMessageBox.warning(self, "Playback", "Audio not loaded.")
            return

        total = len(segment) / 1000.0
        if start_sec < 0:
            start_sec = 0
        if end_sec > total:
            end_sec = total
        if end_sec <= start_sec:
            QMessageBox.warning(self, "Playback", "Invalid time range.")
            return

        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        part = segment[start_ms:end_ms]

        # Force 16‑bit signed PCM for Qt
        if part.sample_width != 2:
            part = part.set_sample_width(2)

        channels = part.channels
        rate = part.frame_rate
        sample_width_bytes = part.sample_width  # 2 after adjustment

        fmt = QAudioFormat()
        fmt.setSampleRate(rate)
        fmt.setChannelCount(channels)
        fmt.setSampleSize(sample_width_bytes * 8)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)

        # Stop any previous playback
        if self.audio_output:
            self.audio_output.stop()
            self.audio_output.deleteLater()
            self.audio_output = None
        if self.audio_buffer:
            self.audio_buffer.close()
            self.audio_buffer.deleteLater()
            self.audio_buffer = None

        # Prepare buffer
        self.audio_data = QByteArray(part.raw_data)
        self.audio_buffer = QBuffer()
        self.audio_buffer.setData(self.audio_data)
        self.audio_buffer.open(QBuffer.ReadOnly)

        self.audio_output = QAudioOutput(fmt, self)
        self.audio_output.start(self.audio_buffer)

    # --- Jump handlers (ADD these inside MainWindow, e.g. after synctable_play_selected) ---
    def referencetable_jump_to_selected(self):
        """
        Jump first plot (reference) to earliest start time of the selected reference rows
        and highlight those subtitle bands.
        """
        if not hasattr(self, "referencetable") or self.referencetable.selectionModel() is None:
            return
        selected = self.referencetable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        sel_indices = []
        for idx in selected:
            row = idx.row()
            start_item = self.referencetable.item(row, 0)
            if start_item:
                starts.append(self._parse_time_to_seconds(start_item.text()))
                sel_indices.append(row)
        if not starts:
            return
        target = min(starts)
        if hasattr(self, "plot1") and hasattr(self.plot1, "jump_to_time"):
            self.plot1.jump_to_time(target, center=True)
            self.plot1.set_selected_subtitle_indices(sel_indices)

    def synctable_jump_to_selected(self):
        """
        Jump second plot (new media) to earliest start time of the selected sync rows
        and highlight those subtitle bands.
        """
        if not hasattr(self, "synctable") or self.synctable.selectionModel() is None:
            return
        selected = self.synctable.selectionModel().selectedRows()
        if not selected:
            return
        starts = []
        sel_indices = []
        for idx in selected:
            row = idx.row()
            start_item = self.synctable.item(row, 0)
            if start_item:
                starts.append(self._parse_time_to_seconds(start_item.text()))
                sel_indices.append(row)
        if not starts:
            return
        target = min(starts)
        if hasattr(self, "plot2") and hasattr(self.plot2, "jump_to_time"):
            self.plot2.jump_to_time(target, center=True)
            self.plot2.set_selected_subtitle_indices(sel_indices)
        

    def select_media_file_btn1(self):
        """Pick reference media file and put its path into le1."""
        filters = "Media files (*.avi *.mkv *.mpg *.mpeg *.mp4 *.mov *.wmv *.flv *.webm);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Media", "", filters)
        if path:
            self.le1.setText(path)

    def select_media_file_btn2(self):
        """Pick new media file and put its path into le2."""
        filters = "Media files (*.avi *.mkv *.mpg *.mpeg *.mp4 *.mov *.wmv *.flv *.webm);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select New Media", "", filters)
        if path:
            self.le2.setText(path)

    def select_subtitle_file_btn3(self):
        """Pick reference subtitle (.srt) and fill le3 + auto-generate save path in le4."""
        filters = "Subtitle files (*.srt);;All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference Subtitle", "", filters)
        if path:
            self.le3.setText(path)
            base, ext = os.path.splitext(path)
            self.le4.setText(f"{base}_resync{ext}")

    def save_subtitle_file_btn4(self):
        """Pick destination for saving adjusted subtitle file."""
        filters = "Subtitle files (*.srt);;All files (*.*)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Subtitle As", "", filters)
        if path:
            # Ensure .srt extension if user omitted (optional)
            if not os.path.splitext(path)[1]:
                path += ".srt"
            self.le4.setText(path)

# --- Simple non-blocking busy dialog (no cancel, custom animation) ---
class BusyDialog(QDialog):
    def __init__(self, parent=None, title="Working", message="Please wait..."):
        super().__init__(parent)
        # Remove the "?" (context help) button from the title bar
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.setWindowTitle(title)
        self.setModal(True)  # Blocks input only; event loop still runs
        self.setMinimumWidth(320)
        lay = QVBoxLayout(self)
        self.label = QLabel(message, self)
        self.label.setWordWrap(True)
        lay.addWidget(self.label)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        lay.addWidget(self.bar)

        self._pulse_value = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.start(60)

    def _pulse(self):
        self._pulse_value = (self._pulse_value + 3) % 101
        self.bar.setValue(self._pulse_value)

    def set_message(self, text: str):
        self.label.setText(text)
        # Force repaint without starving events
        QApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())