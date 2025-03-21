import pyqtgraph.examples
#pyqtgraph.examples.run()


"""
This example demonstrates the use of ColorBarItem, which displays a simple interactive color bar.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, mkQApp


class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Set the global background color for pyqtgraph
        
         
        data = np.fromfunction(lambda i, j: (1+0.3*np.sin(i)) * (i)**2 + (j)**2, (100, 100))
        
        
        pg.setConfigOption('background', 'w')  # Set to white, you can change 'w' to any color you prefer
        pg.setConfigOption('foreground', 'k')  # Set to white, you can change 'w' to any color you prefer

        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('pyqtgraph example: Interactive color bar')
        self.resize(800,700)
        self.show()
        
        p1 = gr_wid.addPlot(title="interactive")
        # Basic steps to create a false color image with color bar:
        i1 = pg.ImageItem(image=data)
        p1.addItem( i1 )
        p1.addColorBar( i1, colorMap='viridis',values=(0, 30_000)) # , interactive=False)
        
        for side in ( 'top', 'right'):
            ax = p1.getAxis(side)
            ax.setStyle(tickLength=0) 
        for side in ('left', 'bottom'):
            ax = p1.getAxis(side)
            ax.setStyle(tickLength=10) 
        
        p1.setMenuEnabled(False)
        p1.setRange(xRange=(0,100), yRange=(0,100), padding=0)
        p1.showAxes(True, showValues=(True,False,False,True) )
        p1.setAspectLocked(True)
        p1.setDefaultPadding( 0.0 )
        


mkQApp("ColorBarItem Example")
main_window = MainWindow()

## Start Qt event loop
if __name__ == '__main__':
    pg.exec()
