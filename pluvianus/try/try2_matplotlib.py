import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        self.plot_3d_scatter()

    def plot_3d_scatter(self):
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.x = np.random.rand(100)
        self.y = np.random.rand(100)
        self.z = np.log10(np.random.rand(100)*300)

        self.scatter = self.ax.scatter(self.x, self.y, self.z, c=self.z, cmap='viridis', picker=True)
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.set_title('Interactive 3D Scatter Plot')

        self.figure.colorbar(self.scatter, ax=self.ax, shrink=0.5)

        self.annot = self.figure.text(0.00, 1.0, "", va='top', ha='left')
        self.annot.set_visible(False)

        self.selection_marker = self.ax.scatter([], [], [], s=100, edgecolor='purple', facecolor='none', linewidth=2)
        
        self.hover_marker = self.ax.plot([1], [1], [1], color='k', linewidth=0.5)[0]
        self.hover_marker.set_visible(False)

        def log_tick_formatter(val, pos=None):
            if val >=1:
                return "{:.0f}".format(round(10**val))
            else:
                return "{:.1g}".format(10**val)


        self.ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos=None: "{:.0f}".format(round(10**val)) if val >= 1 else "{:.1g}".format(10**val)))
        #self.ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos=None: "{:.1g}".format(10**val)))
        
        magnitudes=np.arange(-5.0, 6.0, dtype=float)
        stubs = np.array([1,2,3,4,5])
        allstubs = np.concatenate([stubs * 10 ** m for m in magnitudes])
        self.ax.zaxis.set_major_locator(plt.FixedLocator(np.log10(allstubs)))
        
        stubs = np.array([1,2,3,4,5,6,7,8,9])
        allstubs = np.concatenate([stubs * 10 ** m for m in magnitudes])
        self.ax.zaxis.set_minor_locator(plt.FixedLocator(np.log10(allstubs)))
        
        self.ax.set_box_aspect((1,1,1),  zoom=1.1)

        
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        self.canvas.mpl_connect("pick_event", self.on_pick)

        self.canvas.draw()



    def hover(self, event):
        def update_annot(self, ind):
            x, y, z = self.x[ind["ind"][0]], self.y[ind["ind"][0]], self.z[ind["ind"][0]]
            self.annot.set_text(f"multiple\nlines\n({x:.2f}, {y:.2f}, {z:.2f})")
            self.annot.set_visible(True)
            self.hover_marker.set_visible(False)
            
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                update_annot(self, ind)
                
                xlim=self.ax.get_xlim()
                ylim=self.ax.get_ylim()
                zlim=self.ax.get_zlim()
                x, y, z = self.x[ind["ind"][0]], self.y[ind["ind"][0]], self.z[ind["ind"][0]]
                
                x=[xlim[0], xlim[1],np.nan, x, x, np.nan,x,x]
                y=[y,y, np.nan,ylim[0], ylim[1], np.nan,y,y]
                z=[z, z, np.nan,z,z,np.nan,zlim[0], zlim[1]]
                
                self.hover_marker._verts3d = (x, y, z)
                self.hover_marker.set_visible(True)
                
                self.canvas.draw_idle()
                
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.hover_marker.set_visible(False)
                    self.canvas.draw_idle()

    def on_pick(self, event):
        ind = event.ind[0]
        x, y, z = self.x[ind], self.y[ind], self.z[ind]
        print(f"Point clicked: ({x:.2f}, {y:.2f}, {z:.2f})")
        self.selection_marker._offsets3d = ([x], [y], [z])
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 - Matplotlib 3D Scatter with Hover and Click")
        self.resize(800, 600)

        plot_widget = MatplotlibWidget(self)
        self.setCentralWidget(plot_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())