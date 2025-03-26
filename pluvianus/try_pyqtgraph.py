import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
import numpy as np
from PyQt6.QtWidgets import QSpinBox, QVBoxLayout, QWidget

app = QtWidgets.QApplication(sys.argv)

spinbox = QSpinBox()
spinbox.setRange(0, 100)
spinbox.setValue(np.random.randint(0, 100))

central_widget = QWidget()
layout = QVBoxLayout(central_widget)
layout.addWidget(spinbox)

win = pg.PlotWidget()
layout.addWidget(win)

win.plot([1, 2, 3], [4, 5, 6])

# add clearly visible red vertical InfiniteLine at x=2
line = pg.InfiniteLine(pos=2, angle=90, pen=pg.mkPen('r', width=2), hoverPen=pg.mkPen('m', width=4), movable=True)
win.addItem(line)

def sigDragged(line):
    print('dragged to', line.value())
    line.setPen(pg.mkPen('g', width=4))
    line.setValue(line.value())
    

def sigPositionChangeFinished(line):
    print('position finished', line.value())
    line.setPen(pg.mkPen('r', width=2))


def sigPositionChanged(line):
    print('position changed', line.value())
    line.setValue(spinbox.value())

def spinPositionChanged(value):
    print('spin position changed', value)
    line.setValue(spinbox.value())
    
def sigClicked(line, ev):
    print('clicked', line.value())

line.sigDragged.connect(sigDragged)
line.sigPositionChangeFinished.connect(sigPositionChangeFinished)
#line.sigPositionChanged.connect(sigPositionChanged)
line.sigClicked.connect(sigClicked)
spinbox.valueChanged.connect(spinPositionChanged)

central_widget.show()
sys.exit(app.exec())
