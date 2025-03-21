import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stretched Layout + PlotWidget Tightly Fit to Data")

        # 1) Create a central widget with a layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # 2) Use PlotWidget, which is a proper QWidget subclass
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # 3) Get the underlying PlotItem
        self.plot = self.plot_widget.getPlotItem()

        # Show all four axes => "box" look
        for side in ("left", "bottom", "top", "right"):
            self.plot.showAxis(side, True)

        # Style axes in black, with tick marks drawn inward (positive length)
        for side in ("left", "bottom", "top", "right"):
            ax = self.plot.getAxis(side)
            ax.setPen("k")
            ax.setTextPen("k")
            ax.setStyle(tickLength=10)
        
        # Only show numeric labels on bottom & left
        self.plot.getAxis("top").setStyle(showValues=False)
        self.plot.getAxis("right").setStyle(showValues=False)

        # Disable context menu (no right-click)
        self.plot.setMenuEnabled(False)

        # White background
        self.plot_widget.setBackground("w")

        # 4) Generate a 2D sinusoidal array (400×500)
        x = np.linspace(0, 4*np.pi, 500)   # 500 columns (width)
        y = np.linspace(0, 2*np.pi, 400)   # 400 rows (height)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.sin(Y)  # shape: (400, 500)

        # 5) Create an ImageItem with this data
        self.img_item = pg.ImageItem(Z)
        self.plot.addItem(self.img_item)

        # Apply a viridis colormap from matplotlib
        cmap = pg.colormap.get("viridis", source="matplotlib")
        self.img_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        # 6) Set the plot range tightly around [0..500] in x and [0..400] in y
        self.plot.setRange(xRange=(0, 500), yRange=(0, 400))

        # By default, PyQtGraph inverts Y for images; set invertY(False) for normal Cartesian orientation
        self.plot.invertY(False)

        # Fix aspect ratio to 1:1
        self.plot.getViewBox().setAspectLocked(True, ratio=1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
