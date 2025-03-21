from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QLabel, QSpinBox, QHBoxLayout
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import sys

class CustomCompoundWidget(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        # Create your layout here
        self.main_window = main_window  # explicitly store main_window reference
        layout = QVBoxLayout(self)

        # Add widgets (figures, spinboxes, etc.)
        label = QLabel("Component Index:")
        spinbox = QSpinBox()
        spinbox.setRange(0, 10)
        # Connect the spinbox value change to a custom slot
        spinbox.valueChanged.connect(self.on_spinbox_value_changed)

        self.spinbox=spinbox
        
        plot_widget = pg.PlotWidget()
        plot_widget.plot([1, 2, 3, 4], [10, 20, 30, 40])

        # Adding widgets to layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(label)
        control_layout.addWidget(spinbox)

        layout.addLayout(control_layout)
        layout.addWidget(plot_widget)

    def on_spinbox_value_changed(self, value):
        # Change effect logic, e.g., update plot or change appearance
        print(f"Spinbox value changed to: {value}")
        # Example effect: modify plot line thickness based on spinbox value
        variable=self.parent().parent().variable
        print(variable)
        
        print(self.main_window.variable)
        
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QSplitter with Compound Layout")

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self.variable=22
        # Container widget with layout inside splitter
        compound_widget = CustomCompoundWidget(self, self)

        # Another simple widget for comparison
        simple_widget = QLabel("Right Widget")

        splitter.addWidget(compound_widget)
        splitter.addWidget(simple_widget)
        print(compound_widget.spinbox.value())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 500)
    window.show()
    sys.exit(app.exec())
