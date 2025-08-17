import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel, QTableWidget, QHeaderView, QFileDialog
)
from PyQt5.QtCore import Qt

class PlotWidget(QFrame):
    def __init__(self, title="Plot"):
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(1)
        self.setMinimumHeight(120)
        label = QLabel(title, self)
        label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(label)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Button-LineEdit Pairs Example")
        self.resize(1024, 600)  # Increased height for tables
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
        #self.btn5.setFixedWidth(220)
        self.btn5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row5.addWidget(self.btn5)
        main_layout.addLayout(row5)

        # Add a horizontal line as a separator as well as some spacing
        main_layout.addSpacing(10)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(1)
        main_layout.addWidget(separator)
        main_layout.addSpacing(10)

        # Two PlotWidgets below the pairs, stacked vertically
        plot1 = PlotWidget("Plot 1")
        plot2 = PlotWidget("Plot 2")
        main_layout.addWidget(plot1)
        main_layout.addWidget(plot2)

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
