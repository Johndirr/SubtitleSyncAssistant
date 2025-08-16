import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel, QTableWidget, QHeaderView
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
        btn1 = QPushButton("Load reference media file")
        btn1.setFixedWidth(220)
        btn1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        le1 = QLineEdit()
        le1.setPlaceholderText("No file selected")
        le1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row1.addWidget(btn1)
        row1.addWidget(le1)
        main_layout.addLayout(row1)

        # Pair 2: Load new media file
        row2 = QHBoxLayout()
        btn2 = QPushButton("Load new media file")
        btn2.setFixedWidth(220)
        btn2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        le2 = QLineEdit()
        le2.setPlaceholderText("No file selected")
        le2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row2.addWidget(btn2)
        row2.addWidget(le2)
        main_layout.addLayout(row2)

        # Pair 3: Load reference subtitle
        row3 = QHBoxLayout()
        btn3 = QPushButton("Load reference subtitle")
        btn3.setFixedWidth(220)
        btn3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        le3 = QLineEdit()
        le3.setPlaceholderText("No file selected")
        le3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row3.addWidget(btn3)
        row3.addWidget(le3)
        main_layout.addLayout(row3)

        # Pair 4: Save subtitle under...
        row4 = QHBoxLayout()
        btn4 = QPushButton("Save subtitle under...")
        btn4.setFixedWidth(220)
        btn4.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        le4 = QLineEdit()
        le4.setPlaceholderText("No file selected")
        le4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row4.addWidget(btn4)
        row4.addWidget(le4)
        main_layout.addLayout(row4)

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
