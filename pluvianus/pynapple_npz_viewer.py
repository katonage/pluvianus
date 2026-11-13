#!/usr/bin/env python3

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pynapple as nap
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from PySide6.QtCore import QEvent, QLineF, QPointF, QSettings, QSignalBlocker, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGraphicsLineItem,
    QHeaderView, QLabel, QMenuBar, QMessageBox, QPushButton, QSizePolicy,
    QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from pluvianus.GripSplitter import GripSplitter
from pluvianus.npz_intervals import IntervalShading, high_state_epochs
from pluvianus.npz_spectrogram import (
    SPECTROGRAM_COLORMAPS, SpectrogramColorsDialog, SpectrogramItem, spectrogram_geometry,
)
from pluvianus import npz_viewer_session as session

try:
    from pluvianus import __version__
except ImportError:
    __version__ = '0.0.0-dev'

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class CurveLegendSample(ItemSample):
    def mouseClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.sigClicked.emit(self.item)


@dataclass
class SpectrogramState:
    data: nap.TsdFrame
    geometry: tuple
    curves: list
    image: SpectrogramItem | None = None
    image_settings: dict | None = None


class MissingSessionFilesDialog(QDialog):
    def __init__(self, missing, directory, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.replacements = {}
        self.setWindowTitle('Locate missing NPZ files')
        self.resize(700, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Select a missing file to locate it, or load the available files.'))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Referenced file', 'Replacement'])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for index, reference in missing:
            item = QTreeWidgetItem(self.tree, [reference, 'Missing'])
            item.setToolTip(0, reference)
            item.setData(0, Qt.ItemDataRole.UserRole, index)
        self.tree.setCurrentItem(self.tree.topLevelItem(0))
        layout.addWidget(self.tree)
        locate = QPushButton('Locate…')
        locate.clicked.connect(self.locate_selected)
        layout.addWidget(locate)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open
                                  | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Open).setText('Load available files')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def locate_selected(self):
        item = self.tree.currentItem()
        if item is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Locate ' + os.path.basename(item.text(0)),
                                             self.directory, 'NPZ files (*.npz)')
        if path:
            self.replacements[item.data(0, Qt.ItemDataRole.UserRole)] = path
            item.setText(1, path)
            item.setToolTip(1, path)
            self.directory = os.path.dirname(path)


class NpzViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('pluvianus', 'pynapple_npz_viewer')
        self.recent_files = self.settings.value('files/recent', [], type=list)[:20]
        self.axis_preferences = self.settings.value('files/axis_preferences', {})
        self.files = {}
        self.file_data = {}
        self.spectrograms = {}
        self.session_path = None
        self._restoring_session = False
        self.curve_axes = {}
        self.interval_states = {}
        self.color_index = 0
        self.measurement_line = None
        self.measurement_start = None
        self.measurement_plot = None
        self.setWindowTitle('Pynapple NPZ viewer')
        self.resize(1100, 700)

        layout = QVBoxLayout(self)
        menu_bar = QMenuBar(self)
        layout.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu('&File')
        open_action = file_menu.addAction('&Open files…')
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_files)
        self.recent_menu = file_menu.addMenu('Open &recent')
        self.recent_menu.aboutToShow.connect(self.update_recent_menu)
        self.recent_menu.setEnabled(bool(self.recent_files))
        file_menu.addSeparator()
        load_action = file_menu.addAction('&Load settings…')
        load_action.setShortcut(QKeySequence('Ctrl+Shift+O'))
        load_action.triggered.connect(self.load_settings)
        save_action = file_menu.addAction('&Save settings…')
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_settings)
        save_as_action = file_menu.addAction('Save settings &as…')
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_settings_as)
        file_menu.addSeparator()
        exit_action = file_menu.addAction('E&xit')
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setMenuRole(QAction.MenuRole.QuitRole)
        exit_action.triggered.connect(self.close)
        view_menu = menu_bar.addMenu('&View')
        self.legend_action = view_menu.addAction('Show &legends')
        self.legend_action.setCheckable(True)
        self.legend_action.setChecked(self.settings.value('view/legends_visible', True, type=bool))
        self.scale_bar_action = view_menu.addAction('Show time &scale bar')
        self.scale_bar_action.setCheckable(True)
        self.scale_bar_action.setChecked(self.settings.value('view/scale_bar_visible', True, type=bool))
        help_menu = menu_bar.addMenu('&Help')
        about_action = help_menu.addAction('&About NPZ viewer…')
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self.show_about)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Files / data', 'Axis / display', ''])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setMinimumSectionSize(24)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(2, 28)
        self.tree.itemChanged.connect(self.set_curve_visibility)
        sidebar_layout.addWidget(self.tree, 1)
        self.coordinate_status = QLabel('x: — s\ny1 (left): —\ny2 (right): —')
        self.coordinate_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.coordinate_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.coordinate_status.setToolTip('Drag with the middle mouse button to measure distances on both axes.')
        sidebar_layout.addWidget(self.coordinate_status)
        measurement_hint = QLabel('Drag with the middle mouse button to measure Δx, Δy1 and Δy2.')
        measurement_hint.setWordWrap(True)
        sidebar_layout.addWidget(measurement_hint)
        self.splitter.addWidget(sidebar)

        self.plot_splitter = GripSplitter(Qt.Orientation.Vertical)
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot, self.right_view = self.create_plot()
        self.bottom_plot, self.bottom_right_view = self.create_plot()
        self.plot_views = {
            self.plot: self.right_view,
            self.bottom_plot: self.bottom_right_view,
        }
        self.axis_targets = {
            'Top left': (self.plot, self.plot.getViewBox()),
            'Top right': (self.plot, self.right_view),
            'Bottom left': (self.bottom_plot, self.bottom_plot.getViewBox()),
            'Bottom right': (self.bottom_plot, self.bottom_right_view),
        }
        self.bottom_plot.setXLink(self.plot)
        for plot in self.plot_views:
            plot.plotItem.legend.sigSampleClicked.connect(self.select_legend_curve)
            self.plot_splitter.addWidget(plot)
            # Tick labels can move the plot without changing its size.
            plot.getViewBox().geometryChanged.connect(self.update_views)
            plot.scene().sigMouseMoved.connect(
                lambda position, plot=plot: self.update_coordinates(position, plot)
            )
            plot.viewport().installEventFilter(self)
        self.update_views()
        self.plot_splitter.setSizes([350, 350])
        self.splitter.addWidget(self.plot_splitter)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([300, 800])
        self.time_scale_bar = pg.ScaleBar(
            size=1, width=6, brush='k', pen='k', suffix='s', offset=(-20, 35),
        )
        self.time_scale_bar.setParentItem(self.plot.getViewBox())
        self.time_scale_bar.setZValue(100)
        self.time_scale_bar.text.setColor('k')
        self.plot.getViewBox().sigXRangeChanged.connect(self.update_time_scale_bar)
        self.update_time_scale_bar()
        self.legend_action.toggled.connect(self.update_view_decorations)
        self.scale_bar_action.toggled.connect(self.update_view_decorations)
        self.update_view_decorations()
        geometry = self.settings.value('window/geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self.settings.value('window/splitter')
        if splitter_state is not None:
            self.splitter.restoreState(splitter_state)
        plot_splitter_state = self.settings.value('window/plot_splitter')
        if plot_splitter_state is not None:
            self.plot_splitter.restoreState(plot_splitter_state)

    @staticmethod
    def create_plot():
        plot = pg.PlotWidget()
        plot.addLegend(offset=(10, 10), sampleType=CurveLegendSample)
        plot.setLabel('bottom', 'Time (s)')
        plot.setLabel('left', 'Left axis')
        plot.showAxis('right')
        right_view = pg.ViewBox()
        plot.scene().addItem(right_view)
        plot.getAxis('right').linkToView(right_view)
        right_view.setXLink(plot.getViewBox())
        plot.setLabel('right', 'Right axis', color='#00008b')
        plot.getAxis('right').setPen(pg.mkPen('#00008b'))
        plot.getAxis('right').setTextPen(pg.mkPen('#00008b'))
        # Matching gutters keep linked time coordinates aligned on screen.
        for side in ('left', 'right'):
            plot.getAxis(side).setWidth(80)
        return plot, right_view

    def sync_legends(self):
        for panel in self.plot_views:
            legend = panel.plotItem.legend
            visible = [curve for curve, axis in self.curve_axes.items()
                       if isinstance(curve, pg.PlotDataItem) and curve.isVisible()
                       and not self.is_interval_curve(curve) and self.axis_targets[axis][0] is panel]
            for sample, _ in tuple(legend.items):
                if sample.item not in visible:
                    legend.removeItem(sample.item)
            existing = {sample.item for sample, _ in legend.items}
            for curve in visible:
                if curve not in existing:
                    legend.addItem(curve, curve.name())
            for sample, label in legend.items:
                # Both the line sample and its text select the same sidebar row.
                label.mouseClickEvent = sample.mouseClickEvent
                label.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
            legend.setVisible(self.legend_action.isChecked() and bool(legend.items))

    def select_legend_curve(self, curve):
        for parent, _ in self.files.values():
            for index in range(parent.childCount()):
                child = parent.child(index)
                if child.data(0, Qt.ItemDataRole.UserRole) is curve:
                    parent.setExpanded(True)
                    self.tree.setCurrentItem(child)
                    self.tree.scrollToItem(child)
                    self.tree.setFocus(Qt.FocusReason.OtherFocusReason)
                    return

    def update_view_decorations(self, *args):
        self.sync_legends()
        self.time_scale_bar.setVisible(self.scale_bar_action.isChecked())
        self.settings.setValue('view/legends_visible', self.legend_action.isChecked())
        self.settings.setValue('view/scale_bar_visible', self.scale_bar_action.isChecked())

    def update_time_scale_bar(self, *args):
        lower, upper = self.plot.getViewBox().viewRange()[0]
        target = (upper - lower) / 5
        if not math.isfinite(target) or target <= 0:
            return
        magnitude = 10 ** math.floor(math.log10(target))
        size = max((step * magnitude for step in (1, 2, 5)
                    if step * magnitude <= target), default=magnitude)
        self.time_scale_bar.size = size
        if size >= 1:
            label = f'{size:g}s'
        else:
            label = pg.siFormat(size, suffix='s', space='')
        self.time_scale_bar.text.setText(label)
        self.time_scale_bar.updateBar()

    def show_about(self):
        text = f"""
            <p><b>Pynapple NPZ viewer (npzview)</b><br>
            A standalone viewer for Pynapple Tsd and TsdFrame files,
            included with Pluvianus.</p>
            <p>by Gergely Katona</p>
            <p>Version: {__version__}</p>
            <p>If you use Pluvianus in your work, please cite our associated publication:</p>
            <p><b>Pluvianus: Interactive GUI for Exploration and Quality Control of
            CaImAn Calcium Imaging Analysis Results.</b> (2026)
            Katona, G., Dávid, A., Slézia, A., &amp; Kaszás, A.
            <i>Journal of Open Research Software</i>, 14.
            <a href="https://doi.org/10.5334/jors.623">https://doi.org/10.5334/jors.623</a></p>
            """
        QMessageBox.about(self, 'About NPZ viewer', text)

    def update_coordinates(self, position, plot=None):
        if self.measurement_line is not None:
            return
        plot = self.plot if plot is None else plot
        left_view = plot.getViewBox()
        if not left_view.sceneBoundingRect().contains(position):
            self.clear_coordinates()
            return
        left = left_view.mapSceneToView(position)
        right = self.plot_views[plot].mapSceneToView(position)
        panel = 'Top' if plot is self.plot else 'Bottom'
        status = (
            f'{panel} plot\nx: {left.x():.6f} s\ny1 (left): {left.y():.6g}\ny2 (right): {right.y():.6g}'
        )
        for parent, items in self.files.values():
            for item in items:
                if not isinstance(item, SpectrogramItem) or not item.isVisible():
                    continue
                target_plot, target_view = self.axis_targets[self.curve_axes[item]]
                if target_plot is plot:
                    point = target_view.mapSceneToView(position)
                    value = item.value_at(point.x(), point.y())
                    status += f'\n{parent.text(0)} value: ' + (f'{value:.6g}' if value is not None else 'no data')
        self.coordinate_status.setText(status)

    def clear_coordinates(self):
        self.coordinate_status.setText('x: — s\ny1 (left): —\ny2 (right): —')

    def eventFilter(self, watched, event):
        plot = next((plot for plot in self.plot_views if watched is plot.viewport()), None)
        if plot is not None:
            event_type = event.type()
            if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.MiddleButton:
                position = plot.mapToScene(event.position().toPoint())
                if plot.getViewBox().sceneBoundingRect().contains(position):
                    self.measurement_plot = plot
                    self.measurement_start = position
                    self.measurement_line = QGraphicsLineItem()
                    self.measurement_line.setPen(pg.mkPen('#d32f2f', width=2, style=Qt.PenStyle.DashLine))
                    self.measurement_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self.measurement_line.setZValue(1000)
                    plot.scene().addItem(self.measurement_line)
                    self.update_measurement(position)
                    return True
            elif self.measurement_line is not None and self.measurement_plot is plot:
                if event_type == QEvent.Type.MouseMove:
                    self.update_measurement(plot.mapToScene(event.position().toPoint()))
                    return True
                if event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.MiddleButton:
                    self.update_measurement(plot.mapToScene(event.position().toPoint()))
                    self.finish_measurement()
                    return True
                if event_type == QEvent.Type.Wheel:
                    return True
            if event_type == QEvent.Type.Wheel:
                # Let pyqtgraph zoom first, then sample the data now under the pointer.
                point = event.position().toPoint()
                QTimer.singleShot(0, lambda plot=plot, point=point: self.update_coordinates(
                    plot.mapToScene(point), plot))
            if event_type == QEvent.Type.Leave and self.measurement_line is None:
                self.clear_coordinates()
        return super().eventFilter(watched, event)

    def update_measurement(self, position):
        left_view = self.measurement_plot.getViewBox()
        bounds = left_view.sceneBoundingRect()
        end = QPointF(
            max(bounds.left(), min(position.x(), bounds.right())),
            max(bounds.top(), min(position.y(), bounds.bottom())),
        )
        self.measurement_line.setLine(QLineF(self.measurement_start, end))
        left_delta = left_view.mapSceneToView(end) - left_view.mapSceneToView(self.measurement_start)
        right_view = self.plot_views[self.measurement_plot]
        right_delta = right_view.mapSceneToView(end) - right_view.mapSceneToView(self.measurement_start)
        panel = 'Top' if self.measurement_plot is self.plot else 'Bottom'
        self.coordinate_status.setText(
            f'{panel} plot\nΔx: {abs(left_delta.x()):.6f} s\n'
            f'Δy1 (left): {abs(left_delta.y()):.6g}\n'
            f'Δy2 (right): {abs(right_delta.y()):.6g}'
        )

    def finish_measurement(self):
        if self.measurement_line is not None:
            self.measurement_plot.scene().removeItem(self.measurement_line)
            self.measurement_line = None
            self.measurement_start = None
            self.measurement_plot = None

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open Pynapple NPZ containing Tsd or TsdFrame',
            self.settings.value('files/last_directory', '', type=str),
            'NPZ files (*.npz)',
        )
        if paths:
            self.settings.setValue('files/last_directory', os.path.dirname(os.path.abspath(paths[0])))
            self.settings.sync()
        self.add_files(paths)

    @staticmethod
    def column_labels(data):
        if isinstance(data, nap.Tsd):
            return [None]
        return [label.item() if isinstance(label, np.generic) else label for label in data.columns]

    def session_document(self, path):
        """Capture both display representations without constructing hidden images."""
        files = []
        for key, (parent, active) in self.files.items():
            state = self.spectrograms.get(key)
            curves = state.curves if state is not None else active
            labels = self.column_labels(self.file_data[key])
            preferences = self.axis_preferences.get(key, {})
            entry = {'path': session.source_reference(parent.toolTip(0), path),
                     'mode': 'spectrogram' if isinstance(active[0], SpectrogramItem) else 'curves',
                     'curves': []}
            for index, (label, plot) in enumerate(zip(labels, curves)):
                shading = self.interval_states.get(plot)
                axis = self.curve_axes.get(plot, shading.axis if shading is not None
                                           else preferences.get(str(label), 'Top left'))
                entry['curves'].append({
                    'column': {'index': index, 'label': label},
                    'axis': axis.lower().replace(' ', '_'), 'visible': plot.isVisible(),
                    'color': plot.opts['pen'].color().name(QColor.NameFormat.HexArgb),
                    'representation': 'intervals' if shading is not None and shading.enabled else 'curve',
                })
            if state is not None:
                if state.image is not None:
                    image = state.image
                    axis = self.curve_axes.get(image, preferences.get('Spectrogram', 'Top left'))
                    entry['spectrogram'] = {
                        'axis': axis.lower().replace(' ', '_'), 'visible': image.isVisible(),
                        'colormap': image.color_name, 'levels': list(map(float, image.levels)),
                    }
                elif state.image_settings is not None:
                    entry['spectrogram'] = dict(state.image_settings)
            files.append(entry)
        view = {'time_range': list(self.plot.getViewBox().viewRange()[0]),
                'time_auto_range': bool(self.plot.getViewBox().autoRangeEnabled()[0]),
                'legends_visible': self.legend_action.isChecked(),
                'scale_bar_visible': self.scale_bar_action.isChecked(),
                'axes': {}, 'splitters': {}}
        for axis, (_, target) in self.axis_targets.items():
            view['axes'][axis.lower().replace(' ', '_')] = {
                'range': list(target.viewRange()[1]),
                'auto_range': bool(target.autoRangeEnabled()[1]),
            }
        for name, splitter in [('sidebar', self.splitter), ('plots', self.plot_splitter)]:
            sizes = splitter.sizes()
            view['splitters'][name] = [size / sum(sizes) for size in sizes] if sum(sizes) else [.5, .5]
        return {'format': session.FORMAT, 'version': session.VERSION, 'files': files, 'view': view}

    def settings_dialog_path(self):
        return self.session_path or self.settings.value(
            'files/settings_directory', self.settings.value('files/last_directory', '', type=str), type=str,
        )

    def save_settings(self):
        if self.session_path is None:
            self.save_settings_as()
        else:
            self.save_session(self.session_path)

    def save_settings_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save NPZ viewer settings', self.settings_dialog_path(),
            'NPZ viewer settings (*.npzview.json);;JSON files (*.json)',
        )
        if path:
            if not path.lower().endswith('.json'):
                path += session.SUFFIX
            self.save_session(path)

    def save_session(self, path):
        try:
            session.write_session(path, self.session_document(path))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Could not save settings', str(exc))
            return False
        self.remember_session(path)
        return True

    def remember_session(self, path):
        self.session_path = os.path.abspath(path)
        self.settings.setValue('files/settings_directory', os.path.dirname(self.session_path))
        self.remember_recent(self.session_path)
        self.update_title()

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load NPZ viewer settings', self.settings_dialog_path(),
            'NPZ viewer settings (*.npzview.json);;JSON files (*.json)',
        )
        if path:
            self.load_session(path, locate_missing=True)

    def load_session(self, path, *, locate_missing=False):
        try:
            document = session.read_session(path)
            references = [session.resolve_reference(entry['path'], path) for entry in document['files']]
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Could not load settings', str(exc))
            return False
        missing = [(index, entry['path']) for index, entry in enumerate(document['files'])
                   if not os.path.isfile(references[index])]
        if missing and locate_missing:
            dialog = MissingSessionFilesDialog(missing, os.path.dirname(os.path.abspath(path)), self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
            for index, replacement in dialog.replacements.items():
                references[index] = os.path.realpath(replacement)
        # Read all sources first, so malformed settings or entirely unavailable
        # recordings leave the existing session intact.
        staged = {}
        entries = []
        issues = []
        for entry, reference in zip(document['files'], references):
            key = os.path.normcase(reference)
            try:
                if key in staged:
                    raise ValueError('Duplicate file reference; using its first occurrence.')
                data = nap.load_file(reference)
                if not isinstance(data, (nap.Tsd, nap.TsdFrame)):
                    raise ValueError(f'Unsupported format: {type(data).__name__}')
                if isinstance(data, nap.TsdFrame) and data.shape[1] == 0:
                    raise ValueError('The frame has no data columns.')
                if (not (np.issubdtype(data.values.dtype, np.number)
                         or np.issubdtype(data.values.dtype, np.bool_)) or np.iscomplexobj(data.values)):
                    raise ValueError('Display data must be real numeric values.')
                geometry = spectrogram_geometry(data)
                if geometry is None and ('spectrogram' in entry or entry.get('mode') == 'spectrogram'):
                    issues.append(f'{reference}\nThe frequency/time grid is no longer suitable for a spectrogram; using curves.')
                    entry.pop('spectrogram', None)
                    entry['mode'] = 'curves'
                image = entry.get('spectrogram')
                if image is not None and image.get('colormap', 'hot') not in SPECTROGRAM_COLORMAPS:
                    issues.append(f'{reference}\nUnknown colormap {image["colormap"]!r}; using hot.')
                    image['colormap'] = 'hot'
                staged[key] = data
                entries.append((key, reference, entry))
            except Exception as exc:
                issues.append(f'{reference}\n{exc}')
        if document['files'] and not entries:
            QMessageBox.warning(self, 'Could not load settings',
                                'No referenced files could be loaded.\n\n' + '\n\n'.join(issues))
            return False
        self.finish_measurement()
        self._restoring_session = True
        try:
            for key in list(self.files):
                self.close_file(key)
            for key, _, _ in entries:
                self.axis_preferences[key] = {}
            issues.extend(self.add_files([reference for _, reference, _ in entries],
                                         data_by_path=staged, remember=False, report_errors=False))
            for key, reference, entry in entries:
                if key in self.files:
                    self.restore_file_settings(key, entry, issues)
            self.restore_view_settings(document.get('view', {}))
        finally:
            self._restoring_session = False
        for key, reference, _ in entries:
            if key in self.files:
                self.remember_file(reference)
        self.save_axis_preferences()
        self.remember_session(path)
        if issues:
            QMessageBox.warning(self, 'Settings loaded with issues', '\n\n'.join(issues))
        return True

    def restore_file_settings(self, key, entry, issues):
        parent, curves = self.files[key]
        labels = self.column_labels(self.file_data[key])
        if 'curves' in entry and len(entry['curves']) != len(curves):
            issues.append(f'{parent.toolTip(0)}\nColumn count changed; unmatched columns use default settings.')
        restored = set()
        reordered = False
        for saved in entry.get('curves', []):
            column = saved['column']
            index = column['index']
            label = column['label']
            matches = [i for i, current in enumerate(labels)
                       if type(current) is type(label) and current == label]
            if index not in matches:
                if len(matches) != 1:
                    issues.append(f'{parent.toolTip(0)}\nColumn {label!r} changed or is missing; its settings were skipped.')
                    continue
                index = matches[0]
                reordered = True
            if index in restored:
                issues.append(f'{parent.toolTip(0)}\nAmbiguous settings for column {label!r}; its duplicate was skipped.')
                continue
            restored.add(index)
            child, plot = parent.child(index), curves[index]
            self.tree.itemWidget(child, 1).setCurrentText(saved.get('axis', 'top_left').replace('_', ' ').capitalize())
            if 'color' in saved:
                color = QColor(saved['color'])
                pen = pg.mkPen(plot.opts['pen'])
                pen.setColor(color)
                plot.setPen(pen)
                child.setForeground(0, color)
            child.setCheckState(0, Qt.CheckState.Checked if saved.get('visible', True)
                                else Qt.CheckState.Unchecked)
            if saved.get('representation', 'curve') == 'intervals':
                if plot in self.interval_states:
                    self.tree.itemWidget(child, 1).setCurrentText('Intervals')
                else:
                    issues.append(f'{parent.toolTip(0)}\nColumn {label!r} no longer has two levels; using a curve.')
        if reordered:
            issues.append(f'{parent.toolTip(0)}\nColumns were reordered; display settings were matched by label.')
        state = self.spectrograms.get(key)
        if state is not None and 'spectrogram' in entry:
            state.image_settings = dict(entry['spectrogram'])
            self.axis_preferences[key]['Spectrogram'] = entry['spectrogram'].get(
                'axis', 'top_left').replace('_', ' ').capitalize()
        if entry.get('mode') == 'spectrogram':
            self.tree.itemWidget(parent, 1).setCurrentText('Spectrogram')

    def restore_view_settings(self, view):
        # Load view ranges last: adding/moving data and changing layout can autorange.
        if 'legends_visible' in view:
            self.legend_action.setChecked(view['legends_visible'])
        if 'scale_bar_visible' in view:
            self.scale_bar_action.setChecked(view['scale_bar_visible'])
        for _, target in self.axis_targets.values():
            target.enableAutoRange(x=False, y=False)
        for name, splitter in [('sidebar', self.splitter), ('plots', self.plot_splitter)]:
            proportions = view.get('splitters', {}).get(name)
            if proportions is not None:
                total = sum(splitter.sizes()) or 1000
                splitter.setSizes([round(value / sum(proportions) * total) for value in proportions])
        self.update_views()
        for axis, (_, target) in self.axis_targets.items():
            saved = view.get('axes', {}).get(axis.lower().replace(' ', '_'), {})
            if 'range' in saved:
                target.setYRange(*saved['range'], padding=0)
            target.enableAutoRange(y=saved.get('auto_range', 'range' not in saved))
            target.updateAutoRange()
        main = self.plot.getViewBox()
        if 'time_range' in view:
            main.setXRange(*view['time_range'], padding=0)
        main.enableAutoRange(x=view.get('time_auto_range', 'time_range' not in view))
        main.updateAutoRange()

    def update_recent_menu(self):
        self.recent_menu.clear()
        for path in self.recent_files:
            action = self.recent_menu.addAction(path.replace('&', '&&'))
            action.triggered.connect(
                lambda checked=False, path=path: self.open_recent(path)
            )

    def open_recent(self, path):
        if path.lower().endswith('.json'):
            self.load_session(path, locate_missing=True)
        else:
            self.add_files([path])

    def remember_file(self, path):
        self.settings.setValue('files/last_directory', os.path.dirname(path))
        self.remember_recent(path)

    def remember_recent(self, path):
        path = os.path.abspath(path)
        key = os.path.normcase(os.path.realpath(path))
        self.recent_files = [path] + [
            recent for recent in self.recent_files
            if os.path.normcase(os.path.realpath(recent)) != key
        ]
        self.recent_files = self.recent_files[:20]
        self.settings.setValue('files/recent', self.recent_files)
        self.save_axis_preferences()
        self.settings.sync()
        self.recent_menu.setEnabled(True)

    def save_axis_preferences(self):
        for key, (parent, plots) in self.files.items():
            # Keep destinations for both representations when switching modes.
            self.axis_preferences.setdefault(key, {}).update({
                parent.child(index).text(0): ('Intervals' if self.is_interval_curve(plot)
                                              else self.curve_axes[plot])
                for index, plot in enumerate(plots)
            })
        if self._restoring_session:
            return
        recent_keys = {os.path.normcase(path) for path in self.recent_files} | self.files.keys()
        self.axis_preferences = {
            key: axes for key, axes in self.axis_preferences.items() if key in recent_keys
        }
        self.settings.setValue('files/axis_preferences', self.axis_preferences)

    def add_files(self, paths, *, data_by_path=None, remember=True, report_errors=True):
        errors = []
        for path in paths:
            path = os.path.realpath(os.path.abspath(path))
            key = os.path.normcase(path)
            if key in self.files:
                self.tree.setCurrentItem(self.files[key][0])
                if remember:
                    self.remember_file(path)
                continue
            plots = []
            parent = None
            try:
                data = nap.load_file(path) if data_by_path is None else data_by_path[key]
                if isinstance(data, nap.Tsd):
                    curves = [('Tsd', data.values)]
                elif isinstance(data, nap.TsdFrame):
                    if data.shape[1] == 0:
                        raise ValueError('The frame has no data columns.')
                    curves = [(str(col), data.values[:, index])
                              for index, col in enumerate(data.columns)]
                else:
                    raise ValueError(f'Unsupported format: {type(data).__name__}')
                times = data.times()
                geometry = spectrogram_geometry(data)
                interval_ids = np.searchsorted(data.time_support.end, times)
                connections = np.zeros(len(times), dtype=bool)
                connections[:-1] = interval_ids[:-1] == interval_ids[1:]
                filename = os.path.basename(path)
                parent = QTreeWidgetItem([filename, '', ''])
                parent.setToolTip(0, path)
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
                                | Qt.ItemFlag.ItemIsAutoTristate)
                parent.setCheckState(0, Qt.CheckState.Checked)
                for label, values in curves:
                    color = COLORS[self.color_index % len(COLORS)]
                    name = filename if isinstance(data, nap.Tsd) else f'{filename}/{label}'
                    connect = connections.copy()
                    connect[:-1] &= np.isfinite(values[:-1]) & np.isfinite(values[1:])
                    plot = self.plot.plot(times, values, pen=pg.mkPen(color), name=name,
                                          connect=connect)
                    plots.append(plot)
                    self.curve_axes[plot] = 'Top left'
                    plot.visibleChanged.connect(self.sync_legends)
                    epochs = high_state_epochs(times, np.asarray(values), data.time_support)
                    if epochs is not None:
                        self.interval_states[plot] = IntervalShading(epochs)
                    self.color_index += 1
                    child = QTreeWidgetItem(parent, [label, ''])
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked)
                    child.setForeground(0, QColor(color))
                    child.setData(0, Qt.ItemDataRole.UserRole, plot)
                self.tree.addTopLevelItem(parent)
                # Establish the shared time range before curves leave the top left view.
                self.plot.getViewBox().updateAutoRange()
                for index, plot in enumerate(plots):
                    axis = self.axis_preferences.get(key, {}).get(parent.child(index).text(0), 'Top left')
                    self.add_axis_selector(parent.child(index), plot, axis)
                    self.add_color_button(parent.child(index), plot)
                close_button = QToolButton()
                close_button.setText('×')
                close_button.setStyleSheet('QToolButton { color: #d32f2f; font-size: 18px; font-weight: bold; }')
                close_button.setAutoRaise(True)
                close_button.setFixedSize(24, 24)
                close_button.setAccessibleName(f'Close {filename}')
                close_button.setToolTip(f'Close {path}')
                close_button.clicked.connect(lambda checked=False, key=key: self.close_file(key))
                self.tree.setItemWidget(parent, 2, close_button)
                parent.setExpanded(True)
                self.files[key] = (parent, plots)
                self.file_data[key] = data
                if geometry is not None:
                    self.spectrograms[key] = SpectrogramState(data, geometry, plots)
                    display_selector = QComboBox()
                    display_selector.addItems(['Curves', 'Spectrogram'])
                    display_selector.setToolTip('Show frequency columns as curves or a spectrogram')
                    display_selector.currentTextChanged.connect(
                        lambda mode, key=key: self.set_file_display(key, mode)
                    )
                    self.tree.setItemWidget(parent, 1, display_selector)
                if remember:
                    self.remember_file(path)
            except Exception as exc:
                for plot in plots:
                    self.remove_curve(plot)
                    self.interval_states.pop(plot, None)
                self.spectrograms.pop(key, None)
                self.file_data.pop(key, None)
                self.files.pop(key, None)
                if parent is not None:
                    index = self.tree.indexOfTopLevelItem(parent)
                    if index >= 0:
                        self.tree.takeTopLevelItem(index)
                errors.append(f'{path}\n{exc}')
        self.update_title()
        self.sync_legends()
        if errors and report_errors:
            QMessageBox.warning(self, 'Could not open files', '\n\n'.join(errors))
        return errors

    def add_axis_selector(self, item, plot, axis):
        axis = {'Left': 'Top left', 'Right': 'Top right'}.get(axis, axis)
        choices = list(self.axis_targets)
        if plot in self.interval_states:
            choices.append('Intervals')
        if axis not in choices:
            axis = 'Top left'
        selector = QComboBox()
        selector.addItems(choices)
        selector.setToolTip('Choose a plot axis, or Intervals to shade high states in both panels'
                            if plot in self.interval_states else 'Choose the plot and vertical axis for this item')
        self.set_curve_axis(plot, axis, persist=False)
        selector.setCurrentText(axis)
        selector.currentTextChanged.connect(
            lambda axis, plot=plot: self.set_curve_axis(plot, axis)
        )
        self.tree.setItemWidget(item, 1, selector)

    def add_color_button(self, item, plot):
        button = QToolButton()
        button.setText('…')
        if isinstance(plot, SpectrogramItem):
            button.setToolTip('Change spectrogram colormap and color limits')
            button.setAccessibleName('Spectrogram colors')
            button.clicked.connect(
                lambda checked=False, plot=plot: SpectrogramColorsDialog(plot, self).exec()
            )
        else:
            button.setToolTip('Change line or interval color')
            button.setAccessibleName('Line or interval color')
            button.clicked.connect(
                lambda checked=False, item=item, plot=plot: self.edit_curve_color(item, plot)
            )
        self.tree.setItemWidget(item, 2, button)

    def edit_curve_color(self, item, plot):
        color = QColorDialog.getColor(
            plot.opts['pen'].color(), self,
            'Interval color' if self.is_interval_curve(plot) else 'Line color',
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if not color.isValid():
            return
        pen = pg.mkPen(plot.opts['pen'])
        pen.setColor(color)
        plot.setPen(pen)
        item.setForeground(0, color)
        shading = self.interval_states.get(plot)
        if shading is not None:
            for shade in shading.items:
                shade.set_color(color)
        for panel in self.plot_views:
            for sample, _ in panel.plotItem.legend.items:
                if sample.item is plot:
                    sample.update()

    def set_file_display(self, key, mode):
        state = self.spectrograms[key]
        parent, plots = self.files[key]
        if (mode == 'Spectrogram') == isinstance(plots[0], SpectrogramItem):
            return
        # Build lazily, before removing the existing representation.
        if mode == 'Spectrogram' and state.image is None:
            state.image = SpectrogramItem(state.data, state.geometry)
            if state.image_settings is not None:
                image = state.image_settings
                state.image.set_colors(image.get('colormap', 'hot'),
                                       image.get('levels') or state.image.default_levels)
                state.image.setVisible(image.get('visible', True))
            else:
                state.image.setVisible(any(plot.isVisible() for plot in plots))
        time_range = self.plot.getViewBox().viewRange()[0]
        default_axis = self.curve_axes[plots[0]]
        self.save_axis_preferences()
        with QSignalBlocker(self.tree):
            for plot in plots:
                self.remove_curve(plot)
            parent.takeChildren()
            if mode == 'Spectrogram':
                entries = [('Spectrogram', state.image)]
            else:
                entries = list(zip(map(str, state.data.columns), state.curves))
            active = []
            for label, plot in entries:
                if self.is_interval_curve(plot):
                    shading = self.interval_states[plot]
                    self.curve_axes[plot] = shading.axis
                    shading.attach(self.plot_views.values(), plot.opts['pen'].color(), plot.isVisible())
                else:
                    self.plot.addItem(plot)
                    self.curve_axes[plot] = 'Top left'
                active.append(plot)
                child = QTreeWidgetItem(parent, [label, ''])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked if plot.isVisible()
                                    else Qt.CheckState.Unchecked)
                child.setData(0, Qt.ItemDataRole.UserRole, plot)
                if not isinstance(plot, SpectrogramItem):
                    child.setForeground(0, plot.opts['pen'].color())
                self.add_color_button(child, plot)
                axis = self.axis_preferences.get(key, {}).get(label, default_axis)
                self.add_axis_selector(child, plot, axis)
            self.files[key] = (parent, active)
        self.update_axis_labels()
        self.sync_legends()
        for plot in active:
            if not self.is_interval_curve(plot):
                self.axis_targets[self.curve_axes[plot]][1].enableAutoRange(y=True)
        self.plot.setXRange(*time_range, padding=0)
        self.save_axis_preferences()
        self.settings.sync()

    def update_axis_labels(self):
        for axis, (plot, _) in self.axis_targets.items():
            side = 'left' if axis.endswith('left') else 'right'
            image = any(isinstance(item, SpectrogramItem) and destination == axis
                        for item, destination in self.curve_axes.items())
            plot.setLabel(side, 'Frequency' if image else f'{side.title()} axis',
                          units='Hz' if image else '',
                          **({'color': '#00008b'} if side == 'right' else {}))

    def update_views(self):
        for plot, right_view in self.plot_views.items():
            left_view = plot.getViewBox()
            right_view.setGeometry(left_view.sceneBoundingRect())
            right_view.linkedViewChanged(left_view, right_view.XAxis)

    def is_interval_curve(self, plot):
        shading = self.interval_states.get(plot)
        return shading is not None and shading.enabled

    def set_curve_axis(self, plot, axis, persist=True):
        intervals = axis == 'Intervals'
        shading = self.interval_states.get(plot)
        if intervals and shading is None:
            return
        if ((intervals and self.is_interval_curve(plot))
                or (not intervals and not self.is_interval_curve(plot) and self.curve_axes[plot] == axis)):
            return
        visible = plot.isVisible()
        previous_axis = self.curve_axes[plot]
        self.remove_curve(plot)
        if shading is not None:
            shading.enabled = intervals
            shading.axis = previous_axis if intervals else axis
        if intervals:
            shading.attach(self.plot_views.values(), plot.opts['pen'].color(), visible)
            self.curve_axes[plot] = previous_axis
        else:
            target_plot, target_view = self.axis_targets[axis]
            if target_view is target_plot.getViewBox():
                target_plot.addItem(plot)
            else:
                target_view.addItem(plot)
                if isinstance(plot, pg.PlotDataItem):
                    target_plot.plotItem.legend.addItem(plot, plot.name())
            self.curve_axes[plot] = axis
        plot.setVisible(visible)
        self.sync_legends()
        self.update_views()
        self.update_axis_labels()
        if persist:
            self.save_axis_preferences()

    def remove_curve(self, plot):
        axis = self.curve_axes.pop(plot, 'Top left')
        shading = self.interval_states.get(plot)
        if shading is not None:
            shading.axis = axis
            if shading.enabled:
                shading.remove()
                return
        target_plot, target_view = self.axis_targets[axis]
        if target_view is target_plot.getViewBox():
            target_plot.removeItem(plot)
        else:
            target_view.removeItem(plot)
            target_plot.plotItem.legend.removeItem(plot)

    def set_curve_visibility(self, item, column):
        plot = item.data(0, Qt.ItemDataRole.UserRole)
        if column == 0 and plot is not None:
            plot.setVisible(item.checkState(0) == Qt.CheckState.Checked)
            if self.is_interval_curve(plot):
                self.interval_states[plot].set_visible(plot.isVisible())
            self.sync_legends()

    def close_file(self, key):
        parent, plots = self.files.pop(key)
        state = self.spectrograms.get(key)
        curves = state.curves if state is not None else plots
        for plot in plots:
            self.remove_curve(plot)
        for plot in curves:
            self.interval_states.pop(plot, None)
        self.spectrograms.pop(key, None)
        self.file_data.pop(key, None)
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(parent))
        self.update_axis_labels()
        self.sync_legends()
        self.update_title()

    def update_title(self):
        titles = [item.text(0) for item, _ in self.files.values()]
        title = 'Pynapple NPZ viewer'
        if self.session_path is not None:
            title += ' – ' + os.path.basename(self.session_path)
        if titles:
            title += ' – ' + ', '.join(titles[:2])
            if len(titles) > 2:
                title += f' (+{len(titles) - 2} more)'
        self.setWindowTitle(title)

    def closeEvent(self, event):
        self.finish_measurement()
        self.settings.setValue('window/geometry', self.saveGeometry())
        self.settings.setValue('window/splitter', self.splitter.saveState())
        self.settings.setValue('window/plot_splitter', self.plot_splitter.saveState())
        self.settings.sync()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Pynapple NPZ viewer')
    viewer = NpzViewer()
    viewer.show()
    if len(sys.argv) > 1:
        viewer.add_files(sys.argv[1:])
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
