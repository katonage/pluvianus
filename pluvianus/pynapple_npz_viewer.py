#!/usr/bin/env python3

import math
import os
import sys

import pynapple as nap
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QLineF, QPointF, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGraphicsLineItem, QHeaderView, QHBoxLayout, QLabel, QMenu,
    QMessageBox, QPushButton, QSizePolicy,
    QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from pluvianus.GripSplitter import GripSplitter

try:
    from pluvianus import __version__
except ImportError:
    __version__ = '0.0.0-dev'

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class NpzViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('pluvianus', 'pynapple_npz_viewer')
        self.recent_files = self.settings.value('files/recent', [], type=list)[:20]
        self.axis_preferences = self.settings.value('files/axis_preferences', {})
        self.files = {}
        self.curve_axes = {}
        self.color_index = 0
        self.measurement_line = None
        self.measurement_start = None
        self.measurement_plot = None
        self.setWindowTitle('Pynapple NPZ viewer')
        self.resize(1100, 700)

        layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        open_button = QPushButton('Open files…')
        open_button.clicked.connect(self.open_files)
        open_layout = QHBoxLayout()
        open_layout.addWidget(open_button)
        self.recent_button = QToolButton()
        self.recent_button.setText('Open recent')
        self.recent_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_menu = QMenu(self.recent_button)
        self.recent_menu.aboutToShow.connect(self.update_recent_menu)
        self.recent_button.setMenu(self.recent_menu)
        self.recent_button.setEnabled(bool(self.recent_files))
        open_layout.addWidget(self.recent_button)
        about_button = QToolButton()
        about_button.setText('?')
        about_button.setFixedSize(24, 24)
        about_button.setToolTip('About NPZ viewer')
        about_button.setAccessibleName('About NPZ viewer')
        about_button.clicked.connect(self.show_about)
        open_layout.addWidget(about_button)
        sidebar_layout.addLayout(open_layout)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Files / curves', 'Axis', ''])
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
        plot.addLegend(offset=(10, 10))
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

    def update_time_scale_bar(self, *args):
        lower, upper = self.plot.getViewBox().viewRange()[0]
        target = (upper - lower) / 5
        if not math.isfinite(target) or target <= 0:
            return
        magnitude = 10 ** math.floor(math.log10(target))
        size = max(step * magnitude for step in (1, 2, 5) if step * magnitude <= target)
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
        self.coordinate_status.setText(
            f'{panel} plot\nx: {left.x():.6f} s\ny1 (left): {left.y():.6g}\ny2 (right): {right.y():.6g}'
        )

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

    def update_recent_menu(self):
        self.recent_menu.clear()
        for path in self.recent_files:
            action = self.recent_menu.addAction(path.replace('&', '&&'))
            action.triggered.connect(
                lambda checked=False, path=path: self.add_files([path])
            )

    def remember_file(self, path):
        self.settings.setValue('files/last_directory', os.path.dirname(path))
        key = os.path.normcase(path)
        self.recent_files = [path] + [
            recent for recent in self.recent_files
            if os.path.normcase(recent) != key
        ]
        self.recent_files = self.recent_files[:20]
        self.settings.setValue('files/recent', self.recent_files)
        self.save_axis_preferences()
        self.settings.sync()
        self.recent_button.setEnabled(True)

    def save_axis_preferences(self):
        for key, (parent, plots) in self.files.items():
            self.axis_preferences[key] = {
                parent.child(index).text(0): self.curve_axes[plot]
                for index, plot in enumerate(plots)
            }
        recent_keys = {os.path.normcase(path) for path in self.recent_files}
        self.axis_preferences = {
            key: axes for key, axes in self.axis_preferences.items() if key in recent_keys
        }
        self.settings.setValue('files/axis_preferences', self.axis_preferences)

    def add_files(self, paths):
        errors = []
        for path in paths:
            path = os.path.realpath(os.path.abspath(path))
            key = os.path.normcase(path)
            if key in self.files:
                self.tree.setCurrentItem(self.files[key][0])
                self.remember_file(path)
                continue
            plots = []
            try:
                data = nap.load_file(path)
                if isinstance(data, nap.Tsd):
                    curves = [('Tsd', data.data())]
                elif isinstance(data, nap.TsdFrame):
                    curves = [(str(col), data.loc[col].data()) for col in data.columns]
                else:
                    raise ValueError(f'Unsupported format: {type(data).__name__}')
                times = data.times()
                filename = os.path.basename(path)
                parent = QTreeWidgetItem([filename, '', ''])
                parent.setToolTip(0, path)
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
                                | Qt.ItemFlag.ItemIsAutoTristate)
                parent.setCheckState(0, Qt.CheckState.Checked)
                for label, values in curves:
                    color = COLORS[self.color_index % len(COLORS)]
                    name = filename if isinstance(data, nap.Tsd) else f'{filename}/{label}'
                    plot = self.plot.plot(times, values, pen=pg.mkPen(color), name=name)
                    plots.append(plot)
                    self.curve_axes[plot] = 'Top left'
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
                    axis_selector = QComboBox()
                    axis_selector.addItems(list(self.axis_targets))
                    axis_selector.setToolTip('Choose the plot and vertical axis for this curve')
                    axis = self.axis_preferences.get(key, {}).get(parent.child(index).text(0), 'Top left')
                    axis = {'Left': 'Top left', 'Right': 'Top right'}.get(axis, axis)
                    if axis not in self.axis_targets:
                        axis = 'Top left'
                    self.set_curve_axis(plot, axis, persist=False)
                    axis_selector.setCurrentText(axis)
                    axis_selector.currentTextChanged.connect(
                        lambda axis, plot=plot: self.set_curve_axis(plot, axis)
                    )
                    self.tree.setItemWidget(parent.child(index), 1, axis_selector)
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
                self.remember_file(path)
            except Exception as exc:
                for plot in plots:
                    self.remove_curve(plot)
                errors.append(f'{path}\n{exc}')
        self.update_title()
        if errors:
            QMessageBox.warning(self, 'Could not open files', '\n\n'.join(errors))

    def update_views(self):
        for plot, right_view in self.plot_views.items():
            left_view = plot.getViewBox()
            right_view.setGeometry(left_view.sceneBoundingRect())
            right_view.linkedViewChanged(left_view, right_view.XAxis)

    def set_curve_axis(self, plot, axis, persist=True):
        if self.curve_axes[plot] == axis:
            return
        visible = plot.isVisible()
        self.remove_curve(plot)
        target_plot, target_view = self.axis_targets[axis]
        if target_view is target_plot.getViewBox():
            target_plot.addItem(plot)
        else:
            target_view.addItem(plot)
            target_plot.plotItem.legend.addItem(plot, plot.name())
        self.curve_axes[plot] = axis
        plot.setVisible(visible)
        self.update_views()
        if persist:
            self.save_axis_preferences()

    def remove_curve(self, plot):
        axis = self.curve_axes.pop(plot, 'Top left')
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
            # External visibility changes do not repaint pyqtgraph's legend samples.
            target_plot, _ = self.axis_targets[self.curve_axes[plot]]
            for sample, _ in target_plot.plotItem.legend.items:
                if sample.item is plot:
                    sample.update()
                    break

    def close_file(self, key):
        parent, plots = self.files.pop(key)
        for plot in plots:
            self.remove_curve(plot)
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(parent))
        self.update_title()

    def update_title(self):
        titles = [item.text(0) for item, _ in self.files.values()]
        title = 'Pynapple NPZ viewer'
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
