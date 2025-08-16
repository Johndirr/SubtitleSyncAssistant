import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QSizePolicy, QFrame, QLabel
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
        self.resize(1024, 400)  # Increased height for plots
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

        main_layout.addStretch(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
