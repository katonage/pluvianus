import sys
import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog
import pynapple as nap

# Set pyqtgraph global configuration
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("Pynapple NPZ viewer")

# Create a central widget with a vertical layout
central_widget = QWidget()
layout = QVBoxLayout(central_widget)

# Create a PlotWidget to display curves
win = pg.PlotWidget()
layout.addWidget(win)

# Add a legend to the plot widget
legend = win.addLegend(offset=(10, 10))

# Define a palette with 10 different colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Open a file dialog to choose one or more NPZ files
data_files, _ = QFileDialog.getOpenFileNames(
    central_widget,
    'Open Pynapple NPZ containing Tsd or TsdFrame',
    '',
    'NPZ files (*.npz)'
)
if not data_files:
    sys.exit("No files selected.")

color_index = 0
for data_file in data_files:
    # Load the curve using pynapple
    curve = nap.load_file(data_file)
    
    # If the file contains a Tsd (time-series data) object
    if isinstance(curve, nap.Tsd):
        name = os.path.basename(data_file)
        win.plot(curve.times(), curve.data(),
                 pen=pg.mkPen(color=colors[color_index % len(colors)]),
                 name=name)
        color_index += 1
    # If the file contains a TsdFrame (a multi-column time-series) object
    elif isinstance(curve, nap.TsdFrame):
        for col in curve.columns:
            name = os.path.basename(data_file) + '/' + str(col)
            win.plot(curve.times(), curve.loc[col].data(),
                     pen=pg.mkPen(color=colors[color_index % len(colors)]),
                     name=name)
            color_index += 1
    else:
        raise Exception("Unsupported format: {}".format(type(curve)))
    
   

win.setLabel('bottom', 'Time (s)')

central_widget.show()
sys.exit(app.exec())
