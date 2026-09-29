#!/usr/bin/env python3

import os
import sys

import pynapple as nap
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHeaderView, QMessageBox, QPushButton,
    QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class NpzViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('pluvianus', 'pynapple_npz_viewer')
        self.files = {}
        self.color_index = 0
        self.setWindowTitle('Pynapple NPZ viewer')
        self.resize(1100, 700)

        layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        open_button = QPushButton('Open files…')
        open_button.clicked.connect(self.open_files)
        sidebar_layout.addWidget(open_button)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Files / curves', ''])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self.set_curve_visibility)
        sidebar_layout.addWidget(self.tree)
        self.splitter.addWidget(sidebar)

        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(10, 10))
        self.plot.setLabel('bottom', 'Time (s)')
        self.splitter.addWidget(self.plot)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([300, 800])
        geometry = self.settings.value('window/geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self.settings.value('window/splitter')
        if splitter_state is not None:
            self.splitter.restoreState(splitter_state)

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Open Pynapple NPZ containing Tsd or TsdFrame',
            self.settings.value('files/last_directory', '', type=str),
            'NPZ files (*.npz)',
        )
        self.add_files(paths)

    def add_files(self, paths):
        errors = []
        for path in paths:
            path = os.path.realpath(os.path.abspath(path))
            key = os.path.normcase(path)
            if key in self.files:
                self.tree.setCurrentItem(self.files[key][0])
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
                parent = QTreeWidgetItem([filename, ''])
                parent.setToolTip(0, path)
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
                                | Qt.ItemFlag.ItemIsAutoTristate)
                parent.setCheckState(0, Qt.CheckState.Checked)
                for label, values in curves:
                    color = COLORS[self.color_index % len(COLORS)]
                    name = filename if isinstance(data, nap.Tsd) else f'{filename}/{label}'
                    plot = self.plot.plot(times, values, pen=pg.mkPen(color), name=name)
                    plots.append(plot)
                    self.color_index += 1
                    child = QTreeWidgetItem(parent, [label, ''])
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Checked)
                    child.setForeground(0, QColor(color))
                    child.setData(0, Qt.ItemDataRole.UserRole, plot)
                self.tree.addTopLevelItem(parent)
                close_button = QPushButton('Close')
                close_button.setToolTip(f'Close {path}')
                close_button.clicked.connect(lambda checked=False, key=key: self.close_file(key))
                self.tree.setItemWidget(parent, 1, close_button)
                parent.setExpanded(True)
                self.files[key] = (parent, plots)
                self.settings.setValue('files/last_directory', os.path.dirname(path))
            except Exception as exc:
                for plot in plots:
                    self.plot.removeItem(plot)
                errors.append(f'{path}\n{exc}')
        self.update_title()
        if errors:
            QMessageBox.warning(self, 'Could not open files', '\n\n'.join(errors))

    def set_curve_visibility(self, item, column):
        plot = item.data(0, Qt.ItemDataRole.UserRole)
        if column == 0 and plot is not None:
            plot.setVisible(item.checkState(0) == Qt.CheckState.Checked)
            # External visibility changes do not repaint pyqtgraph's legend samples.
            for sample, _ in self.plot.plotItem.legend.items:
                if sample.item is plot:
                    sample.update()
                    break

    def close_file(self, key):
        parent, plots = self.files.pop(key)
        for plot in plots:
            self.plot.removeItem(plot)
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
        self.settings.setValue('window/geometry', self.saveGeometry())
        self.settings.setValue('window/splitter', self.splitter.saveState())
        self.settings.sync()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Pynapple NPZ viewer')
    viewer = NpzViewer()
    viewer.show()
    if len(sys.argv) > 1:
        viewer.add_files(sys.argv[1:])
    else:
        viewer.open_files()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
