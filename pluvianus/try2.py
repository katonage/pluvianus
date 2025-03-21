import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
import pyqtgraph as pg

class HeatmapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sinus Wavelet Heatmap Test")
        self.resize(800, 800)

        # Central Widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # PyQtGraph GraphicsLayoutWidget integration
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)

        # Plotting Heatmap
        self.plot_heatmap()

    def plot_heatmap(self):
        # Generate sample sinusoidal wavelet data
        x = np.linspace(0, 4 * np.pi, 400)
        y = np.linspace(0, 2 * np.pi, 400)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.sin(Y)

        # Create an ImageItem for the heatmap
        img = pg.ImageItem(Z)

        # Set colormap (using standard grayscale as an example)
        colormap = pg.colormap.get("CET-L17")  # CET perceptually uniform colormap
        lut = colormap.getLookupTable()
        img.setLookupTable(lut)
        img.setLevels([Z.min(), Z.max()])

        # Add image to plot area
        plot = self.graph_widget.addPlot()
        plot.addItem(img)
        plot.setAspectLocked(True)  # Set aspect ratio to match image aspect ratio
        plot.enableAutoRange()
        #plot.showAxis('left', False)
        #plot.showAxis('bottom', False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeatmapWindow()
    window.show()
    sys.exit(app.exec())

