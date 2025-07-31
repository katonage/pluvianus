from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QColor, QPolygon
from PySide6.QtWidgets import QSplitter, QSplitterHandle

"""
GripSplitter

A QSplitter that adds three dots in the middle of the handle
to make it more visible. This is a drop-in replacement.

"""
class GripSplitterHandle(QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.bar_width = 6      # Light gray bar thickness
        self.dot_radius = 2     # Small circle (dot) radius
        self.dot_spacing = 8   # Spacing between dot centers
        self.setMinimumSize(10, 10)  # Ensure enough space

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        handle_rect = self.rect()
        
        # 1. Fill handle background with very light gray
        painter.fillRect(handle_rect, QColor('#F0F0F0'))

        # 2. Draw the light gray bar
        bar_color = QColor('#E0E0E0')
        painter.setPen(Qt.NoPen)
        painter.setBrush(bar_color)

        if self.orientation() == Qt.Horizontal:
            x = handle_rect.width() // 2 - self.bar_width // 2
            painter.drawRect(x, 0, self.bar_width, handle_rect.height())
        else:
            y = handle_rect.height() // 2 - self.bar_width // 2
            painter.drawRect(0, y, handle_rect.width(), self.bar_width)

        # 3. Draw three small grip circles (classic look)
        grip_color = QColor('#B0B0B0')
        painter.setBrush(grip_color)
        painter.setPen(Qt.NoPen)

        if self.orientation() == Qt.Horizontal:
            x = handle_rect.width() // 2
            y_center = handle_rect.height() // 2
            for i in range(-1, 2):
                y = y_center + i * self.dot_spacing
                painter.drawEllipse(QPoint(x, y), self.dot_radius, self.dot_radius)
        else:
            y = handle_rect.height() // 2
            x_center = handle_rect.width() // 2
            for i in range(-1, 2):
                x = x_center + i * self.dot_spacing
                painter.drawEllipse(QPoint(x, y), self.dot_radius, self.dot_radius)

class GripSplitter(QSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHandleWidth(14)  # Set handle width by default

    def createHandle(self):
        return GripSplitterHandle(self.orientation(), self)