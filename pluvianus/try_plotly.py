import sys
import os
import tempfile
import numpy as np

# Check PySide6 availability
try:
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QMainWindow, QWidget
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtCore import Qt, QUrl
except ModuleNotFoundError as e:
    print("Error: PySide6 module is not installed. Please install it using 'pip install PySide6'.")
    sys.exit(1)

import plotly.graph_objs as go
import plotly.io as pio

class PlotlyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sinusoidal (400x500), Viridis, Box Axes, Aspect=1, No Context Menu")
        self.resize(800, 800)

        # Create a central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Add WebEngineView
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

        # Generate sinusoidal data
        x = np.linspace(0, 4 * np.pi, 400)
        y = np.linspace(0, 2 * np.pi, 300)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.sin(Y)

        # Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=Z, colorscale='Viridis'))

        # Update layout properties to set axes tightly around the image
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, range=[0, Z.shape[1]], constrain="range", zeroline=True, linecolor='black'),
            yaxis=dict(showgrid=False, scaleanchor="x", scaleratio=1, range=[0, Z.shape[0]], constrain="range", zeroline=True, linecolor='black'),
            showlegend=False
        )

        # Render Plotly figure as HTML to temporary file
        html = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        temp_file.write(html.encode('utf-8'))
        temp_file.close()
        self.web_view.load(QUrl.fromLocalFile(temp_file.name))

        # Disable context menu
        self.web_view.setContextMenuPolicy(Qt.NoContextMenu)

    def closeEvent(self, event):
        # Clean up temporary HTML file when closing
        temp_html = self.web_view.url().toLocalFile()
        if os.path.exists(temp_html):
            os.remove(temp_html)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotlyWindow()
    window.show()
    sys.exit(app.exec())
