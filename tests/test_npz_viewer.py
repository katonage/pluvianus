import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pynapple as nap
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from pluvianus.GripSplitter import GripSplitterHandle
from pluvianus.pynapple_npz_viewer import NpzViewer


class NpzViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = QSettings(
            str(Path(self.directory.name) / 'settings.ini'), QSettings.Format.IniFormat,
        )
        settings_patch = patch('pluvianus.pynapple_npz_viewer.QSettings', return_value=self.settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.viewer = self.make_viewer()
        self.path = str(Path(self.directory.name) / 'traces.npz')
        times = np.linspace(0, 10, 101)
        nap.TsdFrame(t=times, d=np.column_stack((np.sin(times), np.cos(times))),
                     columns=['a', 'b']).save(self.path)
        self.key = os.path.normcase(os.path.realpath(self.path))

    def make_viewer(self):
        viewer = NpzViewer()
        viewer.show()
        self.app.processEvents()
        self.addCleanup(viewer.close)
        return viewer

    def load_curves(self):
        with patch.object(QMessageBox, 'warning') as warning:
            self.viewer.add_files([self.path])
        warning.assert_not_called()
        self.app.processEvents()
        return self.viewer.files[self.key]

    def test_curve_destinations_visibility_legends_and_close(self):
        parent, curves = self.load_curves()
        curve = curves[0]
        child = parent.child(0)
        selector = self.viewer.tree.itemWidget(child, 1)
        choices = ['Top left', 'Top right', 'Bottom left', 'Bottom right']
        self.assertEqual([selector.itemText(i) for i in range(selector.count())], choices)
        child.setCheckState(0, Qt.CheckState.Unchecked)
        for axis in choices:
            selector.setCurrentText(axis)
            target_plot, target_view = self.viewer.axis_targets[axis]
            self.assertIn(curve, target_view.addedItems)
            self.assertFalse(curve.isVisible())
            for plot, right_view in self.viewer.plot_views.items():
                for view in (plot.getViewBox(), right_view):
                    if view is not target_view:
                        self.assertNotIn(curve, view.addedItems)
                legend_curves = [sample.item for sample, _ in plot.plotItem.legend.items]
                self.assertEqual(curve in legend_curves, plot is target_plot)
        child.setCheckState(0, Qt.CheckState.Checked)
        self.assertTrue(curve.isVisible())
        self.viewer.close_file(self.key)
        for plot, right_view in self.viewer.plot_views.items():
            self.assertFalse(plot.getViewBox().addedItems)
            self.assertFalse(right_view.addedItems)
            self.assertFalse(plot.plotItem.legend.items)

    def test_time_ranges_link_both_directions_and_survive_splitter_drag(self):
        viewer = self.viewer
        viewer.plot.setXRange(2, 7, padding=0)
        self.app.processEvents()
        viewer.bottom_plot.setXRange(3, 5.5, padding=0)
        self.app.processEvents()
        for view in (viewer.plot.getViewBox(), viewer.right_view,
                     viewer.bottom_plot.getViewBox(), viewer.bottom_right_view):
            np.testing.assert_allclose(view.viewRange()[0], [3, 5.5])
        viewer.plot.setYRange(-10, 10, padding=0)
        viewer.bottom_plot.setYRange(100, 200, padding=0)
        self.assertEqual(viewer.plot.getViewBox().viewRange()[1], [-10, 10])
        handle = viewer.plot_splitter.handle(1)
        self.assertIsInstance(handle, GripSplitterHandle)
        self.assertEqual(viewer.plot_splitter.orientation(), Qt.Orientation.Vertical)
        before = viewer.plot_splitter.sizes()
        start = handle.rect().center()
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(handle, start + QPointF(0, 60).toPoint())
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=start + QPointF(0, 60).toPoint())
        self.app.processEvents()
        self.assertNotEqual(before, viewer.plot_splitter.sizes())
        for plot, right_view in viewer.plot_views.items():
            np.testing.assert_allclose(plot.getViewBox().viewRange()[0], [3, 5.5])
            np.testing.assert_allclose(right_view.viewRange()[0], [3, 5.5])
            self.assertEqual(right_view.geometry(), plot.getViewBox().sceneBoundingRect())

    def test_scale_bar_matches_time_units_and_stays_in_upper_right(self):
        viewer = self.viewer
        for span, size, label in [(5, 1, '1s'), (2.5, 0.5, '500ms'), (0.005, 0.001, '1ms')]:
            viewer.bottom_plot.setXRange(0, span, padding=0)
            self.app.processEvents()
            bar = viewer.time_scale_bar
            self.assertAlmostEqual(bar.size, size)
            self.assertEqual(bar.text.toPlainText(), label)
            bounds = viewer.plot.getViewBox().sceneBoundingRect()
            bar_bounds = bar.bar.sceneBoundingRect()
            self.assertGreater(bar_bounds.center().x(), bounds.center().x())
            self.assertLess(bar_bounds.bottom(), bounds.center().y())
            self.assertTrue(bounds.contains(bar.text.sceneBoundingRect()))
            left = viewer.plot.getViewBox().mapSceneToView(bar_bounds.topLeft())
            right = viewer.plot.getViewBox().mapSceneToView(bar_bounds.topRight())
            # Include the one-pixel pen in the rendered rectangle's bounds.
            self.assertAlmostEqual(right.x() - left.x(), size,
                                   delta=span / bounds.width() * 2)
            self.assertEqual(bar.bar.brush().color().name(), '#000000')
            self.assertGreaterEqual(bar.bar.rect().height(), 5)

    def test_measurements_use_each_plots_own_y_axes(self):
        viewer = self.viewer
        for index, (plot, right_view) in enumerate(viewer.plot_views.items()):
            plot.setXRange(0, 10, padding=0)
            plot.setYRange(0, 10 * (index + 1), padding=0)
            right_view.setYRange(0, 100 * (index + 1), padding=0)
            self.app.processEvents()
            view = plot.getViewBox()
            position = view.mapViewToScene(QPointF(5, 5 * (index + 1)))
            viewer.update_coordinates(position, plot)
            self.assertIn(f'y1 (left): {5 * (index + 1)}', viewer.coordinate_status.text())
            self.assertIn(f'y2 (right): {50 * (index + 1)}', viewer.coordinate_status.text())
            start = plot.mapFromScene(view.mapViewToScene(QPointF(2, 2 * (index + 1))))
            end = plot.mapFromScene(view.mapViewToScene(QPointF(7, 7 * (index + 1))))
            QTest.mousePress(plot.viewport(), Qt.MouseButton.MiddleButton, pos=start)
            self.assertIs(viewer.measurement_plot, plot)
            QTest.mouseMove(plot.viewport(), end)
            QTest.mouseRelease(plot.viewport(), Qt.MouseButton.MiddleButton, pos=end)
            self.assertIsNone(viewer.measurement_line)
            lines = viewer.coordinate_status.text().splitlines()
            self.assertEqual(lines[0], 'Top plot' if index == 0 else 'Bottom plot')
            deltas = [float(line.split(': ')[1].split()[0]) for line in lines[1:]]
            np.testing.assert_allclose(deltas, [5, 5 * (index + 1), 50 * (index + 1)], rtol=0.03)

    def test_legacy_preferences_and_bottom_destination_survive_reopen(self):
        viewer = self.viewer
        viewer.axis_preferences[self.key] = {'a': 'Right', 'b': 'Left'}
        parent, curves = self.load_curves()
        first_selector = viewer.tree.itemWidget(parent.child(0), 1)
        second_selector = viewer.tree.itemWidget(parent.child(1), 1)
        self.assertEqual(first_selector.currentText(), 'Top right')
        self.assertEqual(second_selector.currentText(), 'Top left')
        first_selector.setCurrentText('Bottom right')
        viewer.plot_splitter.setSizes([200, 450])
        sizes = viewer.plot_splitter.sizes()
        viewer.close()
        self.viewer = self.make_viewer()
        parent, curves = self.load_curves()
        self.assertEqual(self.viewer.tree.itemWidget(parent.child(0), 1).currentText(), 'Bottom right')
        self.assertIn(curves[0], self.viewer.bottom_right_view.addedItems)
        self.assertEqual(self.viewer.plot_splitter.sizes(), sizes)

    def test_restoring_curves_only_in_bottom_plot_shows_full_time_range(self):
        self.viewer.axis_preferences[self.key] = {'a': 'Bottom left', 'b': 'Bottom right'}
        self.load_curves()
        for plot in self.viewer.plot_views:
            lower, upper = plot.getViewBox().viewRange()[0]
            self.assertLessEqual(lower, 0)
            self.assertGreaterEqual(upper, 10)


if __name__ == '__main__':
    unittest.main()
