import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout

def main():
    app = QApplication(sys.argv)
    
    # Create a GraphicsLayoutWidget window with a white background
    central_widget = QWidget()
    #self.setCentralWidget(central_widget)
    layout = QHBoxLayout(central_widget)
    
    win = pg.GraphicsLayoutWidget()
    layout.addWidget(win)
    layout.addWidget(QLabel("S my label here nu"), stretch=1)
        
    central_widget.setWindowTitle("Sinusoida Context Menu")
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
    plot.setDefaultPadding(padding=0.0)
    #plot.setLimits(xMin=0, xMax=400, yMin=0, yMax=500)
    plot.setRange(xRange=(0,400), yRange=None)
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
    if cm is None:
        print("Error: Could not retrieve colormap 'viridis' from matplotlib")
        sys.exit(1)
    # Apply it to the image
    img_item.setLookupTable(cm.getLookupTable(0.0, 1.0, 256))
    
    # Range from x=[0..500], y=[0..400] => tight to the image
    plot.autoRange( padding=0.0 )
    # Ensure standard Cartesian orientation (Y up)
    plot.invertY(False)
    
    # Fix the aspect ratio to 1:1
    plot.getViewBox().setAspectLocked(True, ratio=1)
    
    # Show the window
    central_widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

