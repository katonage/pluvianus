
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout,
    QPushButton, QScrollArea, QFrame
)
import sys

class ScrollWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QScrollArea with 20 Buttons")
        self.resize(400, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout for central widget
        layout = QVBoxLayout(central_widget)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Makes the scroll area resize with the window
        layout.addWidget(scroll_area)

        # Create a container widget inside scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Add 20 buttons to the scroll layout
        for i in range(1, 21):
            btn = QPushButton(f"Button {i}")
            scroll_layout.addWidget(btn)

        scroll_layout.addStretch()  # Push buttons to top

        # Set scroll content as scroll area widget
        scroll_area.setWidget(scroll_content)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScrollWindow()
    window.show()
    sys.exit(app.exec())
