import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt
from pydub import AudioSegment
import numpy as np
import matplotlib.ticker as mticker

import pysrt
from PyQt5.QtGui import QColor

# --- Add matplotlib imports ---
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
        self.subtitle_intervals = []  # <-- Add this line

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

    def set_selected_subtitle_index(self, index: int):
        self.selected_subtitle_index = index
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
        # Draw subtitle bands if present
        selected_idx = getattr(self, "selected_subtitle_index", None)
        if hasattr(self, "subtitle_intervals") and self.subtitle_intervals:
            for i, (start, end) in enumerate(self.subtitle_intervals):
                if end >= _xmin and start <= _xmax:
                    color = 'orange'
                    alpha = 0.18
                    if selected_idx is not None and i == selected_idx:
                        color = '#ff9900'  # darker orange
                        alpha = 0.38
                    ax.axvspan(max(start, _xmin), min(end, _xmax), color=color, alpha=alpha, zorder=0)
        self.canvas.draw()

    # --- Mouse drag handlers for panning ---
    def _on_mouse_press(self, event):
        if event.button == 1 and event.inaxes:
            self._dragging = True
            self._drag_start_x_pixel = event.x  # pixel position
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

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Button-LineEdit Pairs Example")
        self.resize(1024, 600)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(2)

        # Pair 1: Load reference media file
        row1 = QHBoxLayout()
        self.btn1 = QPushButton("Load reference media file")
        self.btn1.setFixedWidth(220)
        self.btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.le1 = QLineEdit()
        self.le1.setPlaceholderText("No file selected")
        self.le1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.le1.setReadOnly(True)  # Make not editable
        row1.addWidget(self.btn1)
        row1.addWidget(self.le1)
        main_layout.addLayout(row1)

        # Connect btn1 to file dialog
        self.btn1.clicked.connect(self.select_media_file_btn1)

        # Pair 2: Load new media file
        row2 = QHBoxLayout()
        self.btn2 = QPushButton("Load new media file")
        self.btn2.setFixedWidth(220)
        self.btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.le2 = QLineEdit()
        self.le2.setPlaceholderText("No file selected")
        self.le2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.le2.setReadOnly(True)  # Make not editable
        row2.addWidget(self.btn2)
        row2.addWidget(self.le2)
        main_layout.addLayout(row2)

        # Connect btn2 to file dialog
        self.btn2.clicked.connect(self.select_media_file_btn2)

        # Pair 3: Load reference subtitle
        row3 = QHBoxLayout()
        self.btn3 = QPushButton("Load reference subtitle")
        self.btn3.setFixedWidth(220)
        self.btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.le3 = QLineEdit()
        self.le3.setPlaceholderText("No file selected")
        self.le3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.le3.setReadOnly(True)  # Make not editable
        row3.addWidget(self.btn3)
        row3.addWidget(self.le3)
        main_layout.addLayout(row3)

        # Connect btn3 to file dialog for .srt files
        self.btn3.clicked.connect(self.select_subtitle_file_btn3)

        # Pair 4: Save subtitle under...
        row4 = QHBoxLayout()
        self.btn4 = QPushButton("Save subtitle under...")
        self.btn4.setFixedWidth(220)
        self.btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.le4 = QLineEdit()
        self.le4.setPlaceholderText("No file selected")
        self.le4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.le4.setReadOnly(True)  # Make not editable
        row4.addWidget(self.btn4)
        row4.addWidget(self.le4)
        main_layout.addLayout(row4)

        # Connect btn4 to file save dialog
        self.btn4.clicked.connect(self.save_subtitle_file_btn4)

        # Analyze button
        row5 = QHBoxLayout()
        self.btn5 = QPushButton("Analyze...")
        self.btn5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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

        # Add some spacing before the tables
        main_layout.addSpacing(10)

        # Add two tables side by side below the plots, and make them expand to fill the remaining space
        tables_row = QHBoxLayout()

        # Left table: referencetable
        self.referencetable = QTableWidget(0, 3)
        self.referencetable.setHorizontalHeaderLabels(["Start time", "End time", "Text"])
        self.referencetable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.referencetable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.referencetable.setColumnWidth(0, 100)
        self.referencetable.setColumnWidth(1, 100)
        self.referencetable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)    # "Start time"
        self.referencetable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)    # "End time"
        self.referencetable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # "Text"
        self.referencetable.setMinimumHeight(300)

        # Right table: synctable
        self.synctable = QTableWidget(0, 4)
        self.synctable.setHorizontalHeaderLabels(["Start time", "End time", "Text", "Found offset"])
        self.synctable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.synctable.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.synctable.setColumnWidth(0, 100)
        self.synctable.setColumnWidth(1, 100)
        self.synctable.setColumnWidth(3, 100)
        self.synctable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)    # "Start time"
        self.synctable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)    # "End time"
        self.synctable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # "Text"
        self.synctable.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)    # "Found offset"
        self.synctable.setMinimumHeight(300)

        tables_row.addWidget(self.referencetable)
        tables_row.addWidget(self.synctable)
        tables_row.setContentsMargins(0, 0, 0, 0) # Remove margins between tables so that they have the same width as the plots

        # Use a container widget for the tables and set its size policy to expanding
        tables_container = QWidget()
        tables_container.setLayout(tables_row)
        tables_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(tables_container)

        # Align all columns' text to the left for both tables
        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

        # Connect selection changes
        self.referencetable.selectionModel().selectionChanged.connect(self.on_reference_table_selection)
        self.synctable.selectionModel().selectionChanged.connect(self.on_sync_table_selection)

    def select_media_file_btn1(self):
        filters = "Media files (*.avi *.mkv *.mpg *.mpeg *.mp4 *.mov *.wmv *.flv *.webm);;All files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Media File", "", filters)
        if file_path:
            self.le1.setText(file_path)

    def select_media_file_btn2(self):
        filters = "Media files (*.avi *.mkv *.mpg *.mpeg *.mp4 *.mov *.wmv *.flv *.webm);;All files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Media File", "", filters)
        if file_path:
            self.le2.setText(file_path)

    def select_subtitle_file_btn3(self):
        filters = "Subtitle files (*.srt);;All files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Subtitle File", "", filters)
        if file_path:
            self.le3.setText(file_path)

    def save_subtitle_file_btn4(self):
        filters = "Subtitle files (*.srt);;All files (*.*)"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Subtitle As", "", filters)
        if file_path:
            self.le4.setText(file_path)

    def sanity_check_files(self):
        missing_files = []
        if not self.le1.text() or not os.path.exists(self.le1.text()):
            missing_files.append("Reference media file")
        if not self.le2.text() or not os.path.exists(self.le2.text()):
            missing_files.append("New media file")
        if not self.le3.text() or not os.path.exists(self.le3.text()):
            missing_files.append("Reference subtitle file")
        if not self.le4.text():
            missing_files.append("Subtitle save path")

        if missing_files:
            QMessageBox.warning(
                self,
                "File(s) not found",
                "The following file(s) do not exist or are not selected:\n\n" + "\n".join(missing_files)
            )
            return False
        return True

    def load_audio_samples(self, filepath):
        """Load an audio file and return normalized samples (amplitude [-1, 1] as numpy array."""
        audio = AudioSegment.from_file(filepath)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).T
        # Normalize samples to range [-1, 1]
        samples = samples / (2 ** (8 * audio.sample_width - 1))
        # Get the sample rate (Hz)
        samplerate = audio.frame_rate

        # Downsample if necessary for display (when more than 5 million samples))
        if samples.size > 5000000:
            # Downsample to every 10th sample
            display_samples = samples[::10] if samples.ndim == 1 else samples[:, ::10]
            display_samplerate = samplerate // 10
            return display_samples, audio, display_samplerate
        else:
            return samples, audio, samplerate

    def on_analyze(self):
        if not self.sanity_check_files():
            return
        Refsamples, Refaudio, Refsamplerate = self.load_audio_samples(self.le1.text())
        Newsamples, Newaudio, Newsamplerate = self.load_audio_samples(self.le2.text())

        # --- Plot the waveforms ---
        self.plot1.plot_waveform(Refsamples, Refsamplerate)
        self.plot2.plot_waveform(Newsamples, Newsamplerate)

        # --- Read subtitle into tables ---
        # Load the reference subtitle file
        srt_path = self.le3.text()
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
        except Exception as e:
            QMessageBox.critical(self, "Subtitle Error", f"Failed to load subtitle file:\n{e}")
            return

        # Helper to format time as hh:mm:ss,ms
        def fmt_time(t):
            return f"{t.hours:02}:{t.minutes:02}:{t.seconds:02},{t.milliseconds:03}"

        # Fill referencetable
        self.referencetable.setRowCount(len(subs))
        for i, sub in enumerate(subs):
            self.referencetable.setItem(i, 0, QTableWidgetItem(fmt_time(sub.start)))
            self.referencetable.setItem(i, 1, QTableWidgetItem(fmt_time(sub.end)))
            self.referencetable.setItem(i, 2, QTableWidgetItem(sub.text))
            # Alternate row color
            if i % 2 == 0:
                for col in range(3):
                    self.referencetable.item(i, col).setBackground(QColor(245, 245, 245))
            else:
                for col in range(3):
                    self.referencetable.item(i, col).setBackground(QColor(230, 230, 230))

        # Fill synctable (copy original, leave "Found offset" empty)
        self.synctable.setRowCount(len(subs))
        for i, sub in enumerate(subs):
            self.synctable.setItem(i, 0, QTableWidgetItem(fmt_time(sub.start)))
            self.synctable.setItem(i, 1, QTableWidgetItem(fmt_time(sub.end)))
            self.synctable.setItem(i, 2, QTableWidgetItem(sub.text))
            self.synctable.setItem(i, 3, QTableWidgetItem(""))  # Found offset empty
            # Alternate row color
            if i % 2 == 0:
                for col in range(4):
                    self.synctable.item(i, col).setBackground(QColor(245, 245, 245))
            else:
                for col in range(4):
                    self.synctable.item(i, col).setBackground(QColor(230, 230, 230))

        # Align text left for all columns
        self.align_table_columns_left(self.referencetable)
        self.align_table_columns_left(self.synctable)

        # --- Draw subtitle bands in both plots ---
        # Extract intervals in seconds from pysrt subs
        intervals = []
        for sub in subs:
            start_sec = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
            end_sec = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
            intervals.append((start_sec, end_sec))
        self.plot1.set_subtitle_intervals(intervals)
        self.plot2.set_subtitle_intervals(intervals)

    def on_reference_table_selection(self):
        selected = self.referencetable.selectionModel().selectedRows()
        if selected:
            idx = selected[0].row()
        else:
            idx = None
        self.plot1.set_selected_subtitle_index(idx)

    def on_sync_table_selection(self):
        selected = self.synctable.selectionModel().selectedRows()
        if selected:
            idx = selected[0].row()
        else:
            idx = None
        self.plot2.set_selected_subtitle_index(idx)

    def align_table_columns_left(self, table):
        header = table.horizontalHeader()
        for col in range(table.columnCount()):
            # Set alignment for header
            item = table.horizontalHeaderItem(col)
            if item is not None:
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            # Set alignment for all existing cells in this column
            for row in range(table.rowCount()):
                cell = table.item(row, col)
                if cell is not None:
                    cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
