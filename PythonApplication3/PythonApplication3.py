import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel,
    QTableWidget, QHeaderView, QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt
from pydub import AudioSegment
import numpy as np
import matplotlib.ticker as mticker

# --- Add matplotlib imports ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MatplotlibPlotWidget(QFrame):
    def __init__(self, title="Plot", window_duration=20):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setMinimumHeight(160)
        self.window_duration = window_duration  # seconds
        self.samples = None
        self.sr = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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

    def plot_waveform(self, samples, sr):
        self.samples = samples
        self.sr = sr
        # Calculate total duration
        if samples.ndim > 1:
            samples_mono = samples.mean(axis=0)
        else:
            samples_mono = samples
        self.samples_mono = samples_mono
        self.total_duration = len(samples_mono) / sr if sr > 0 else 1

        # Configure slider
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
        """Format seconds to hh:mm:ss for axis."""
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02}:{m:02}:{s:02}"

    def _plot_window(self, start_sec):
        if self.samples is None or self.sr is None:
            return
        samples_mono = self.samples_mono
        sr = self.sr
        total_len = len(samples_mono)
        t = np.linspace(0, self.total_duration, num=total_len)
        # Calculate window indices
        _xmin = start_sec
        _xmax = min(start_sec + self.window_duration, self.total_duration)
        idx_min = int(_xmin * sr)
        idx_max = int(_xmax * sr)
        idx_max = min(idx_max, total_len)
        # Plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(t[idx_min:idx_max], samples_mono[idx_min:idx_max], linewidth=0.8)
        ax.set_xlim([_xmin, _xmax])
        ax.set_ylim([-1.05, 1.05])
        ax.set_xlabel('Time (hh:mm:ss)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=8)
        # Set x-axis formatter to hh:mm:ss
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_hhmmss))
        self.figure.tight_layout()
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
