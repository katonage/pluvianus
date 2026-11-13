"""High-state epochs and lightweight full-height annotation shading."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor


def high_state_epochs(times, values, time_support):
    """Return sample-and-hold high epochs, or None for a non-binary curve.

    The greater of two finite levels is high. Constant zero/one signals are
    valid binary signals too. Missing samples and time_support gaps end epochs.
    """
    if (not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_)
            or np.iscomplexobj(values) or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        return None
    # Continuous traces usually reveal a third level immediately. Avoid sorting
    # or copying an entire long trace merely to decide whether to offer the mode.
    levels = np.empty(0, dtype=values.dtype)
    for offset in range(0, len(values), 4096):
        chunk = values[offset:offset + 4096]
        levels = np.union1d(levels, np.unique(chunk[np.isfinite(chunk)]))
        if len(levels) > 2:
            return None
    if len(levels) == 0 or (len(levels) == 1 and levels[0] not in (0, 1)):
        return None
    high_level = levels[-1] if len(levels) == 2 else 1
    high = np.isfinite(values) & (values == high_level)
    epochs = []
    for start, end in zip(time_support.start, time_support.end):
        first = np.searchsorted(times, start, side='left')
        last = np.searchsorted(times, end, side='right')
        state = high[first:last]
        if len(state) == 0:
            continue
        rises = np.flatnonzero(state & ~np.r_[False, state[:-1]]) + first
        falls = np.flatnonzero(state & ~np.r_[state[1:], False]) + first + 1
        for rise, fall in zip(rises, falls):
            stop = times[fall] if fall < last else end
            if stop > times[rise]:
                epochs.append((float(times[rise]), float(stop)))
    return np.asarray(epochs, dtype=float).reshape(-1, 2)


class IntervalShadeItem(pg.UIGraphicsItem):
    """Paint all visible epochs in one noninteractive, viewport-height item."""

    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.brush = pg.mkBrush(None)
        self.setZValue(10)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    def set_color(self, color):
        color = QColor(color)
        color.setAlphaF(color.alphaF() * .14)
        self.brush = pg.mkBrush(color)
        self.update()

    def viewTransformChanged(self):
        super().viewTransformChanged()
        self.setNewBounds()
        self.update()

    def paint(self, painter, option, widget=None):
        bounds = self.boundingRect()
        first = np.searchsorted(self.epochs[:, 1], bounds.left(), side='right')
        last = np.searchsorted(self.epochs[:, 0], bounds.right(), side='left')
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.brush)
        for start, end in self.epochs[first:last]:
            left, right = max(start, bounds.left()), min(end, bounds.right())
            if right > left:
                painter.drawRect(QRectF(left, bounds.top(), right - left, bounds.height()))


class IntervalShading:
    """Remember a curve's representation and manage one shade item per panel."""

    def __init__(self, epochs):
        self.epochs = epochs
        self.axis = 'Top left'
        self.enabled = False
        self.items = []

    def attach(self, views, color, visible):
        if not self.items:
            self.items = [IntervalShadeItem(self.epochs) for _ in views]
        for view, item in zip(views, self.items):
            item.set_color(color)
            view.addItem(item, ignoreBounds=True)
            item.setVisible(visible)

    def remove(self):
        for item in self.items:
            view = item.getViewBox()
            if view is not None:
                view.removeItem(item)

    def set_visible(self, visible):
        for item in self.items:
            item.setVisible(visible)
