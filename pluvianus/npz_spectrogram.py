"""Linear spectrogram geometry and rendering for ordinary Pynapple TsdFrames."""

import math
import re

import numpy as np
import pynapple as nap
import pyqtgraph as pg
from PySide6.QtCore import QLocale, QPointF, QRectF, Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGraphicsRectItem, QLabel,
    QPushButton, QVBoxLayout,
)
from pluvianus.npz_data import column_info, common_metadata, frequency_coordinates

SPECTROGRAM_COLORMAPS = ('hot', 'CET-L1', 'viridis', 'inferno', 'plasma', 'magma', 'cividis', 'turbo')
COLOR_LIMIT_REGEX = re.compile(
    r'^(?P<number>[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)'
    # SpinBox's SI parser appends X internally when its suffix is empty.
    r'\s*(?P<siPrefix>)(?P<suffix>X?)$'
)


def color_limit_step(low, high):
    """A convenient 1/2/5 step close to one percent of the current range."""
    target = high / 100 - low / 100
    smallest = float(np.nextafter(0., 1.))
    if not math.isfinite(target) or target <= 0:
        return smallest
    magnitude = 10. ** math.floor(math.log10(target))
    if magnitude == 0:
        return smallest
    return min((factor * magnitude for factor in (1, 2, 5, 10)), key=lambda step: abs(step - target))


class ColorLimitSpinBox(pg.SpinBox):
    def __init__(self, value, step_provider):
        self.step_provider = step_provider
        super().__init__(value=value, decimals=5, dec=False, step=step_provider(),
                         compactHeight=False, locale=QLocale.c(), regex=COLOR_LIMIT_REGEX)

    def stepBy(self, steps):
        # Commit typed text before stepping, and calculate the step at the time
        # of the click so editing a field never resets its unfinished text.
        self.editingFinishedEvent()
        self.setOpts(step=self.step_provider())
        super().stepBy(steps)


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
    """Return image pieces at explicit frequency_hz coordinates and real times.

    Regular frequency grids use one image per recording interval. Nonuniform
    grids use one image row per frequency bin, with midpoint bin boundaries.
    Time centers must be regular within each nonempty time_support interval.
    """
    if not isinstance(data, nap.TsdFrame) or data.shape[1] < 2:
        return None
    values = data.values
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        return None
    frequencies = frequency_coordinates(data)
    if frequencies is None:
        return None
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    if not np.all(np.diff(frequencies) > 0):
        return None
    frequency_step = regular_step(frequencies)
    if frequency_step is not None:
        bands = [(slice(None), frequencies[0] - frequency_step / 2,
                  frequencies[-1] + frequency_step / 2)]
    else:
        edges = np.r_[frequencies[0] - (frequencies[1] - frequencies[0]) / 2,
                      frequencies[:-1] + np.diff(frequencies) / 2,
                      frequencies[-1] + (frequencies[-1] - frequencies[-2]) / 2]
        bands = [(slice(index, index + 1), edges[index], edges[index + 1])
                 for index in range(len(frequencies))]
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
        for columns, lower, upper in bands:
            rect = QRectF(times[first] - step / 2, lower, step * (last - first), upper - lower)
            clip = QRectF(max(start, rect.left()), rect.top(),
                          min(end, rect.right()) - max(start, rect.left()), rect.height())
            rectangles.append((slice(first, last), columns, rect, clip))
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
        self.linear_images = []
        self.data_scale = 'linear'
        self.levels_by_scale = {}
        infos = column_info(data)
        self.data_unit = common_metadata(infos, 'data_unit') or ''
        self.data_label = common_metadata(infos, 'data_label') or 'Value'
        self.source_data_unit = common_metadata(infos, 'source_data_unit')
        self._bounds = QRectF()
        self.setZValue(-10)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        order, rectangles = geometry
        for rows, columns, rect, clip in rectangles:
            # Clip edge pixels without rescaling their centers away from t/f.
            clipping_item = QGraphicsRectItem(clip, self)
            clipping_item.setPen(pg.mkPen(None))
            clipping_item.setBrush(pg.mkBrush(None))
            clipping_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            clipping_item.setFlag(clipping_item.GraphicsItemFlag.ItemClipsChildrenToShape)
            values = np.array(data.values[rows][:, order[columns]].T,
                              dtype=np.result_type(data.values.dtype, np.float32), order='C')
            values[~np.isfinite(values)] = np.nan
            image = pg.ImageItem(values, axisOrder='row-major', autoLevels=False)
            image.setRect(rect)
            image.setParentItem(clipping_item)
            self.images.append(image)
            self.linear_images.append(values)
            self._bounds = self._bounds.united(clip)
        self.default_levels = self.default_contrast('linear')
        self.set_colors('hot', self.default_levels)

    @property
    def display_unit(self):
        return ('dB re 1 ' + self.data_unit).strip() if self.data_scale == 'db' else self.data_unit

    def default_contrast(self, scale):
        finite = [values[np.isfinite(values) & (values > 0 if scale == 'db' else True)]
                  for values in self.linear_images]
        values = np.concatenate(finite)
        if scale == 'db':
            values = 10 * np.log10(values)
        low, high = map(float, np.percentile(values, [1, 99])) if values.size else (0., 1.)
        if low == high:
            padding = max(abs(low) * 1e-6, 1e-12)
            low, high = low - padding, high + padding
        return low, high

    def set_scale(self, scale):
        if scale not in ('linear', 'db'):
            raise ValueError(f'Unknown spectrogram scale: {scale}')
        if scale == self.data_scale:
            return
        for image, values in zip(self.images, self.linear_images):
            display = values
            if scale == 'db':
                display = np.full_like(values, np.nan)
                positive = np.isfinite(values) & (values > 0)
                display[positive] = 10 * np.log10(values[positive])
            image.setImage(display, autoLevels=False)
        self.data_scale = scale
        self.default_levels = self.default_contrast(scale)
        self.set_colors(self.color_name, self.levels_by_scale.get(scale, self.default_levels))

    def boundingRect(self):
        return QRectF(self._bounds)

    def dataBounds(self, ax, frac=1., orthoRange=None):
        if ax == 0:
            return self._bounds.left(), self._bounds.right()
        return self._bounds.top(), self._bounds.bottom()

    def paint(self, painter, option, widget=None):
        pass  # The child ImageItems paint the spectrogram.

    def value_at(self, time, frequency):
        """Return the raw pixel value, or None outside valid, finite data."""
        point = QPointF(time, frequency)
        for image, values in zip(self.images, self.linear_images):
            clip = image.parentItem().rect()
            if not (clip.left() <= time < clip.right() and clip.top() <= frequency < clip.bottom()):
                continue
            pixel = image.mapFromItem(self, point)
            column, row = math.floor(pixel.x()), math.floor(pixel.y())
            if 0 <= row < image.image.shape[0] and 0 <= column < image.image.shape[1]:
                value = float(values[row, column])
                return value if math.isfinite(value) else None
        return None

    def set_colors(self, name, levels):
        self.color_name = name
        self.levels = tuple(levels)
        self.levels_by_scale[self.data_scale] = self.levels
        color_map = hot_colormap() if name == 'hot' else pg.colormap.get(name)
        for image in self.images:
            image.setColorMap(color_map)
            image.setLevels(self.levels)


class SpectrogramColorsDialog(QDialog):
    def __init__(self, spectrogram, parent=None):
        super().__init__(parent)
        self.spectrogram = spectrogram
        self._resetting = False
        self.setWindowTitle('Spectrogram colors')
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.value_label = QLabel()
        self.value_label.setWordWrap(True)
        layout.insertWidget(0, self.value_label)
        self.scale = QComboBox()
        self.scale.addItem('Linear PSD', 'linear')
        self.scale.addItem('dB', 'db')
        self.scale.setCurrentIndex(self.scale.findData(spectrogram.data_scale))
        form.addRow('Display', self.scale)
        self.colormap = QComboBox()
        self.colormap.addItems(SPECTROGRAM_COLORMAPS)
        self.colormap.setCurrentText(spectrogram.color_name)
        form.addRow('Colormap', self.colormap)
        self.low = self.create_limit_spinbox(spectrogram.levels[0])
        self.high = self.create_limit_spinbox(spectrogram.levels[1])
        form.addRow('Minimum', self.low)
        form.addRow('Maximum', self.high)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet('color: #b00020;')
        self.error_label.hide()
        layout.addWidget(self.error_label)
        reset = QPushButton('Reset hot / 1–99 percentiles')
        reset.clicked.connect(self.reset_colors)
        layout.addWidget(reset)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.colormap.currentTextChanged.connect(self.apply_colors)
        self.low.sigValueChanging.connect(self.apply_colors)
        self.high.sigValueChanging.connect(self.apply_colors)
        self.low.lineEdit().textChanged.connect(self.apply_colors)
        self.high.lineEdit().textChanged.connect(self.apply_colors)
        self.scale.currentIndexChanged.connect(self.change_scale)
        self.update_value_label()

    def update_value_label(self):
        unit = self.spectrogram.display_unit
        self.value_label.setText(self.spectrogram.data_label + (f' ({unit})' if unit else ''))

    def change_scale(self):
        self._resetting = True
        self.spectrogram.set_scale(self.scale.currentData())
        self.low.setValue(self.spectrogram.levels[0])
        self.high.setValue(self.spectrogram.levels[1])
        self.update_value_label()
        self._resetting = False
        self.apply_colors()

    def create_limit_spinbox(self, value):
        spin = ColorLimitSpinBox(value, lambda: color_limit_step(*self.spectrogram.levels))
        spin.setToolTip('Enter a number such as 1e-10. Decimal points and commas are accepted.')
        return spin

    def apply_colors(self, *args):
        if self._resetting:
            return
        # Read the entered text rather than the last committed spinbox value.
        # This also validates a range while either field is still being edited.
        try:
            values = [spin.value() if spin.lineEdit().text() == spin.lastText else spin.interpret()
                      for spin in (self.low, self.high)]
        except (ArithmeticError, ValueError):
            values = [False]
        if any(value is False or not math.isfinite(float(value)) for value in values):
            message = 'Enter finite numbers, for example 1e-10 and 1e-7.'
        else:
            low, high = map(float, values)
            message = '' if low < high else 'Maximum must be greater than minimum.'
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
        if not message:
            self.spectrogram.set_colors(self.colormap.currentText(), (low, high))

    def reset_colors(self):
        # Intermediate edits must not apply an invalid or partially reset range.
        self._resetting = True
        widgets = (self.colormap, self.low, self.high)
        for widget in widgets:
            widget.blockSignals(True)
        self.colormap.setCurrentText('hot')
        self.low.setValue(self.spectrogram.default_levels[0])
        self.high.setValue(self.spectrogram.default_levels[1])
        for widget in widgets:
            widget.blockSignals(False)
        self._resetting = False
        self.apply_colors()
