import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


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
        self.z = np.random.rand(100)

        self.scatter = self.ax.scatter(self.x, self.y, self.z, c=self.z, cmap='viridis', picker=True)
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')
        self.ax.set_title('Interactive 3D Scatter Plot')

        self.figure.colorbar(self.scatter, ax=self.ax, shrink=0.5)

        self.annot = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)
        self.annot.set_visible(False)

        self.selection_marker = self.ax.scatter([], [], [], s=100, edgecolor='purple', facecolor='none', linewidth=2)

        self.canvas.mpl_connect("motion_notify_event", self.hover)
        self.canvas.mpl_connect("pick_event", self.on_pick)

        self.canvas.draw()

    def update_annot(self, ind):
        x, y, z = self.x[ind["ind"][0]], self.y[ind["ind"][0]], self.z[ind["ind"][0]]
        self.annot.set_text(f"({x:.2f}, {y:.2f}, {z:.2f})")
        self.annot.set_visible(True)

    def hover(self, event):
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                self.update_annot(ind)
                self.canvas.draw_idle()
            else:
                if self.annot.get_visible():
                    self.annot.set_visible(False)
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