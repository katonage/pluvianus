"""Linear spectrogram geometry and rendering for ordinary Pynapple TsdFrames."""

import numpy as np
import pynapple as nap
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGraphicsRectItem,
    QPushButton, QVBoxLayout,
)

SPECTROGRAM_COLORMAPS = ('hot', 'CET-L1', 'viridis', 'inferno', 'plasma', 'magma', 'cividis', 'turbo')


def regular_step(coordinates):
    """Return a positive grid step, allowing timestamp rounding, or None."""
    if len(coordinates) < 2 or not np.isfinite(coordinates).all():
        return None
    differences = np.diff(coordinates)
    step = (coordinates[-1] - coordinates[0]) / (len(coordinates) - 1)
    tolerance = max(1e-9, 4 * abs(np.spacing(np.abs(coordinates).max())))
    if step <= 0 or not np.all(differences > 0):
        return None
    if not np.allclose(differences, step, rtol=1e-5, atol=tolerance):
        return None
    return step


def spectrogram_geometry(data):
    """Return (column order, image rectangles) when a frame has a linear grid.

    Columns are frequency centers in Hz. Time centers must be regularly spaced
    within each nonempty time_support interval; intervals may have different
    steps. Rectangles retain pixel centers and are clipped to time support.
    """
    if not isinstance(data, nap.TsdFrame) or data.shape[1] < 2:
        return None
    values = data.values
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        return None
    try:
        frequencies = np.asarray(data.columns, dtype=float)
    except (TypeError, ValueError):
        return None
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    frequency_step = regular_step(frequencies)
    if frequency_step is None or frequencies[0] < 0:
        return None
    times = data.times()
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        return None
    rectangles = []
    count = 0
    for start, end in zip(data.time_support.start, data.time_support.end):
        if not np.isfinite([start, end]).all() or end <= start:
            return None
        first = np.searchsorted(times, start, side='left')
        last = np.searchsorted(times, end, side='right')
        if first == last:
            continue
        step = regular_step(times[first:last])
        if step is None:
            return None
        rect = QRectF(
            times[first] - step / 2, frequencies[0] - frequency_step / 2,
            step * (last - first), frequency_step * len(frequencies),
        )
        clip = QRectF(max(start, rect.left()), rect.top(),
                      min(end, rect.right()) - max(start, rect.left()), rect.height())
        rectangles.append((slice(first, last), rect, clip))
        count += last - first
    if not rectangles or count != len(times):
        return None
    return order, rectangles


def hot_colormap():
    return pg.ColorMap([0, 1 / 3, 2 / 3, 1],
                       [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)])


class SpectrogramItem(pg.GraphicsObject):
    """One movable/visible plot item containing all the ImageItem pieces."""

    def __init__(self, data, geometry):
        super().__init__()
        self.images = []
        self._bounds = QRectF()
        self.setZValue(-10)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        order, rectangles = geometry
        for rows, rect, clip in rectangles:
            # Clip edge pixels without rescaling their centers away from t/f.
            clipping_item = QGraphicsRectItem(clip, self)
            clipping_item.setPen(pg.mkPen(None))
            clipping_item.setBrush(pg.mkBrush(None))
            clipping_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            clipping_item.setFlag(clipping_item.GraphicsItemFlag.ItemClipsChildrenToShape)
            values = np.array(data.values[rows][:, order].T,
                              dtype=np.result_type(data.values.dtype, np.float32), order='C')
            values[~np.isfinite(values)] = np.nan
            image = pg.ImageItem(values, axisOrder='row-major', autoLevels=False)
            image.setRect(rect)
            image.setParentItem(clipping_item)
            self.images.append(image)
            self._bounds = self._bounds.united(clip)
        finite = data.values[np.isfinite(data.values)]
        self.default_levels = tuple(np.percentile(finite, [1, 99])) if finite.size else (0., 1.)
        low, high = self.default_levels
        if low == high:
            padding = max(abs(low) * 1e-6, 1e-12)
            self.default_levels = (low - padding, high + padding)
        self.set_colors('hot', self.default_levels)

    def boundingRect(self):
        return QRectF(self._bounds)

    def dataBounds(self, ax, frac=1., orthoRange=None):
        if ax == 0:
            return self._bounds.left(), self._bounds.right()
        return self._bounds.top(), self._bounds.bottom()

    def paint(self, painter, option, widget=None):
        pass  # The child ImageItems paint the spectrogram.

    def set_colors(self, name, levels):
        self.color_name = name
        self.levels = tuple(levels)
        color_map = hot_colormap() if name == 'hot' else pg.colormap.get(name)
        for image in self.images:
            image.setColorMap(color_map)
            image.setLevels(self.levels)


class SpectrogramColorsDialog(QDialog):
    def __init__(self, spectrogram, parent=None):
        super().__init__(parent)
        self.spectrogram = spectrogram
        self.setWindowTitle('Spectrogram colors')
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.colormap = QComboBox()
        self.colormap.addItems(SPECTROGRAM_COLORMAPS)
        self.colormap.setCurrentText(spectrogram.color_name)
        form.addRow('Colormap', self.colormap)
        self.low = pg.SpinBox(value=spectrogram.levels[0], decimals=8)
        self.high = pg.SpinBox(value=spectrogram.levels[1], decimals=8)
        form.addRow('Minimum', self.low)
        form.addRow('Maximum', self.high)
        reset = QPushButton('Reset hot / 1–99 percentiles')
        reset.clicked.connect(self.reset_colors)
        layout.addWidget(reset)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.colormap.currentTextChanged.connect(self.apply_colors)
        self.low.sigValueChanged.connect(self.apply_colors)
        self.high.sigValueChanged.connect(self.apply_colors)

    def apply_colors(self, *args):
        low, high = self.low.value(), self.high.value()
        valid = low < high
        self.high.setToolTip('Maximum must exceed minimum.' if not valid else '')
        if valid:
            self.spectrogram.set_colors(self.colormap.currentText(), (low, high))

    def reset_colors(self):
        # Intermediate edits must not apply an invalid or partially reset range.
        widgets = (self.colormap, self.low, self.high)
        for widget in widgets:
            widget.blockSignals(True)
        self.colormap.setCurrentText('hot')
        self.low.setValue(self.spectrogram.default_levels[0])
        self.high.setValue(self.spectrogram.default_levels[1])
        for widget in widgets:
            widget.blockSignals(False)
        self.apply_colors()
