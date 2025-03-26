import sys
from PySide6 import QtWidgets

# Ensure QApplication is initialized first
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

# Now it's safe to import pyqtgraph and use its functions
import pyqtgraph as pg
from PySide6 import QtCore

# Set global PyQtGraph styles
pg.setConfigOption('background', 'w')  # white background
pg.setConfigOption('foreground', 'k')  # black foreground

class PlotWidgetWithRightAxis(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        """
        Initialize a PlotWidgetWithRightAxis instance, a child of pg.PlotWidget.

        This constructor sets up the plot widget with an additional right axis.
        It creates a new ViewBox for the right axis, links it to the main plot,
        and applies styling to the right axis. The right axis color is set to 
        dark blue by default. The view geometry is updated to synchronize with 
        the main plot's ViewBox when resized.
        
        You can acces the right axis by calling self.RightViewBox, e.g.:
            self.RightViewBox.addItem(pg.PlotCurveItem(...))   
        You can set the right axis color by calling self.setRightColor(...) e.g.:
            self.setRightColor('#ff008b') 

        Parameters (passed to the PlotWidget parent class constructor):
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super(PlotWidgetWithRightAxis, self).__init__(*args, **kwargs)

        self.showAxis('right')
        self.RightViewBox = pg.ViewBox()
        self.plotItem.scene().addItem(self.RightViewBox)

        right_axis = self.getAxis('right')
        right_axis.linkToView(self.RightViewBox)
        self.RightViewBox.setXLink(self)

        self.RightColor = '#00008b'
        # Styling the right axis properly
        right_axis.setStyle(showValues=True)
        self.setRightColor(self.RightColor)

        self._updateViews()
        self.plotItem.vb.sigResized.connect(self._updateViews)

    def _updateViews(self):
        self.RightViewBox.setGeometry(self.plotItem.vb.sceneBoundingRect())
        self.RightViewBox.linkedViewChanged(self.plotItem.vb, self.RightViewBox.XAxis)
        
    def setRightColor(self, color):
        self.RightColor = color
        right_axis = self.getAxis('right')
        right_axis.setPen(pg.mkPen(self.RightColor))       # dark blue ticks/line
        right_axis.setLabel('axis2', color=self.RightColor)
        right_axis.setTextPen(pg.mkPen(self.RightColor))


def add_plots(plot_widget):
    # Left axis plot (red)
    plot_widget.plot([1, 2, 4, 5], pen='r')

    # Right axis plot (dark blue)
    right_curve = pg.PlotCurveItem([1, 1, 2, 3], pen=pg.mkPen('#00008b', width=2))
    plot_widget.RightViewBox.addItem(right_curve)
    plot_widget.setRightColor('#ff008b')

if __name__ == '__main__':
    plot = PlotWidgetWithRightAxis()
    add_plots(plot)
    plot.show()
    sys.exit(app.exec())
