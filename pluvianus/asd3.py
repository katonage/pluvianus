import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
import pyqtgraph as pg

app = QApplication(sys.argv)

w = QWidget()
layout = QVBoxLayout(w)
plot_widget = pg.PlotWidget()  # Should be recognized as a QWidget
layout.addWidget(plot_widget)   # Should not throw an error
w.show()

app.exec()
