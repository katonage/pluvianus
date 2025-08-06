#!/usr/bin/env python3

import json
import os
import sys
import tempfile

import numpy as np
import pynapple as nap
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFileDialog

# Set pyqtgraph global configuration
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

def save_state(mypath=None):
    filename = 'pynapple_npz_viewer_state.json'
    filename = os.path.join(tempfile.gettempdir(), filename)
    state = {'path': mypath}
    with open(filename, 'w') as f:
        json.dump(state, f)

def load_state():
    filename = 'pynapple_npz_viewer_state.json'
    filename = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(filename):
        return ""
    with open(filename, 'r') as f:
        state = json.load(f)
        if state['path'] is not None and os.path.exists(state['path']):
            mypath = state['path']
            return mypath
        else:
            return ""

app = QApplication(sys.argv)
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

# --- Argument handling
data_files = []
# Ignore the 0-th arg (script name)
if len(sys.argv) > 1:
    # All arguments after script name are assumed to be filenames
    # (Supports more than one file)
    data_files = sys.argv[1:]
    # Check if all files exist
    for f in data_files:
        if not os.path.exists(f):
            sys.exit(f"File not found: {f}")
else:
    # No CLI file given: use file dialog
    data_files, _ = QFileDialog.getOpenFileNames(
        central_widget,
        'Open Pynapple NPZ containing Tsd or TsdFrame',
        load_state(),
        'NPZ files (*.npz)'
    )
    if not data_files:
        sys.exit("No files selected.")

color_index = 0
file_titles = []
for data_file in data_files:
    # Load the curve using pynapple
    curve = nap.load_file(data_file)
    file_titles.append(os.path.basename(data_file))

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

# Save last used directory
save_state(os.path.dirname(data_files[0]))

win.setLabel('bottom', 'Time (s)')

# Set window title to show opened file(s)
if len(file_titles) == 1:
    title = f"Pynapple NPZ viewer – {file_titles[0]}"
else:
    # Show all, or the first and count
    title = f"Pynapple NPZ viewer – {', '.join(file_titles[:2])}"
    if len(file_titles) > 2:
        title += f" (+{len(file_titles)-2} more)"
central_widget.setWindowTitle(title)

central_widget.show()
sys.exit(app.exec())
