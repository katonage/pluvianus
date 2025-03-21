import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    
    # Create a GraphicsLayoutWidget window with a white background
    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle("Sinusoidal (400x500), Viridis, Box Axes, Aspect=1, No Context Menu")
    win.setBackground('w')
    
    # Add a PlotItem
    plot = win.addPlot()
    
    # Show four axes => "box" look
    for side in ('left', 'bottom', 'top', 'right'):
        plot.showAxis(side, True)
    
    # Style the axes in black, with tick marks inward (positive value).
    for side in ('left', 'bottom', 'top', 'right'):
        ax = plot.getAxis(side)
        ax.setPen('k')
        ax.setTextPen('k')
        ax.setStyle(tickLength=10)  # positive => ticks drawn inside the plot
    
    # Show numeric labels only on bottom, left
    plot.getAxis('top').setStyle(showValues=False)
    plot.getAxis('right').setStyle(showValues=False)
    
    # Disable context menu
    plot.setMenuEnabled(False)
    
    # Generate a sinusoidal 2D array:
    #   - 500 "columns" in X, 400 "rows" in Y
    x = np.linspace(0, 4 * np.pi, 500)
    y = np.linspace(0, 2 * np.pi, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.sin(Y)  # shape: (400, 500)
    
    # Create the ImageItem
    img_item = pg.ImageItem(Z)
    plot.addItem(img_item)
    
    # For color mapping, retrieve the 'viridis' colormap from matplotlib
    cm = pg.colormap.get('viridis', source='matplotlib')
    # Apply it to the image
    img_item.setLookupTable(cm.getLookupTable(0.0, 1.0, 256))
    
    # Range from x=[0..500], y=[0..400] => tight to the image
    plot.setRange(xRange=(0, Z.shape[0]), yRange=(0, Z.shape[1]))
    # Ensure standard Cartesian orientation (Y up)
    plot.invertY(False)
    
    # Fix the aspect ratio to 1:1
    plot.getViewBox().setAspectLocked(True, ratio=1)
    
    # Show the window
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
