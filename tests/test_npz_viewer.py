import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pynapple as nap
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from pluvianus.GripSplitter import GripSplitterHandle
from pluvianus.npz_intervals import high_state_epochs
from pluvianus.pynapple_npz_viewer import NpzViewer
from pluvianus.npz_spectrogram import (
    SpectrogramColorsDialog, SpectrogramItem, spectrogram_geometry,
)
from pluvianus import npz_viewer_session as session


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

    def load_spectrogram(self, times=None, values=None, columns=None, support=None):
        times = np.array([.5, 1.5, 2.5, 6.5, 7.5]) if times is None else times
        values = np.arange(len(times) * 3, dtype=float).reshape(-1, 3) if values is None else values
        columns = [1., 2., 3.] if columns is None else columns
        support = nap.IntervalSet(start=[0., 6.], end=[3., 8.]) if support is None else support
        path = str(Path(self.directory.name) / 'spectrum.npz')
        data = nap.TsdFrame(t=times, d=values, columns=columns, time_support=support)
        data.save(path)
        with patch.object(QMessageBox, 'warning') as warning:
            self.viewer.add_files([path])
        warning.assert_not_called()
        key = os.path.normcase(os.path.realpath(path))
        return key, data, self.viewer.files[key][0]

    def test_spectrogram_pieces_preserve_coordinates_and_leave_gaps_blank(self):
        key, data, parent = self.load_spectrogram()
        selector = self.viewer.tree.itemWidget(parent, 1)
        curves = self.viewer.files[key][1]
        # Curve mode also leaves the recording gap disconnected.
        np.testing.assert_array_equal(curves[0].opts['connect'], [True, True, False, True, False])
        selector.setCurrentText('Spectrogram')
        self.app.processEvents()
        self.assertEqual(parent.childCount(), 1)
        layer = self.viewer.files[key][1][0]
        self.assertIsInstance(layer, SpectrogramItem)
        self.assertEqual(len(layer.images), 2)
        self.assertFalse(self.viewer.plot.plotItem.legend.items)
        for image, rows in zip(layer.images, [slice(0, 3), slice(3, 5)]):
            np.testing.assert_array_equal(image.image, data.values[rows].T)
            for column, time in enumerate(data.times()[rows]):
                center = image.mapToItem(layer, QPointF(column + .5, 1.5))
                self.assertAlmostEqual(center.x(), time)
                self.assertAlmostEqual(center.y(), 2.)
        self.viewer.plot.setXRange(0, 8, padding=0)
        self.viewer.plot.setYRange(.5, 3.5, padding=0)
        self.app.processEvents()
        view = self.viewer.plot.getViewBox()
        snapshot = self.viewer.plot.viewport().grab().toImage()
        for time, blank in [(1.5, False), (4.5, True), (6.5, False)]:
            position = self.viewer.plot.mapFromScene(view.mapViewToScene(QPointF(time, 2.)))
            color = snapshot.pixelColor(position)
            self.assertEqual(color.name() == '#ffffff', blank)

    def test_spectrogram_moves_hides_and_closes_on_every_axis(self):
        key, _, parent = self.load_spectrogram()
        curves = self.viewer.files[key][1]
        self.viewer.tree.itemWidget(parent, 1).setCurrentText('Spectrogram')
        layer = self.viewer.files[key][1][0]
        child = parent.child(0)
        selector = self.viewer.tree.itemWidget(child, 1)
        child.setCheckState(0, Qt.CheckState.Unchecked)
        for axis, (plot, target) in self.viewer.axis_targets.items():
            selector.setCurrentText(axis)
            self.assertIn(layer, target.addedItems)
            self.assertFalse(layer.isVisible())
            self.assertTrue(all(not image.isVisible() for image in layer.images))
            self.assertEqual(plot.getAxis('left' if axis.endswith('left') else 'right').labelText,
                             'Frequency')
            for other_plot, other_right in self.viewer.plot_views.items():
                self.assertFalse(other_plot.plotItem.legend.items)
                for view in (other_plot.getViewBox(), other_right):
                    if view is not target:
                        self.assertNotIn(layer, view.addedItems)
        child.setCheckState(0, Qt.CheckState.Checked)
        self.assertTrue(all(image.isVisible() for image in layer.images))
        self.viewer.close_file(key)
        self.assertNotIn(key, self.viewer.spectrograms)
        for plot, right in self.viewer.plot_views.items():
            self.assertFalse(plot.getViewBox().addedItems)
            self.assertFalse(right.addedItems)
        self.assertTrue(all(curve.scene() is None for curve in curves))
        self.assertIsNone(layer.scene())

    def test_spectrogram_switching_preserves_zoom_destinations_visibility_and_colors(self):
        key, _, parent = self.load_spectrogram()
        viewer = self.viewer
        curves = viewer.files[key][1]
        viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom left')
        parent.child(1).setCheckState(0, Qt.CheckState.Unchecked)
        viewer.plot.setXRange(1, 7, padding=0)
        display = viewer.tree.itemWidget(parent, 1)
        display.setCurrentText('Spectrogram')
        layer = viewer.files[key][1][0]
        viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom right')
        layer.set_colors('viridis', (2., 9.))
        display.setCurrentText('Curves')
        self.assertEqual(parent.childCount(), 3)
        self.assertEqual(viewer.files[key][1], curves)
        self.assertIn(curves[0], viewer.bottom_plot.getViewBox().addedItems)
        self.assertFalse(curves[1].isVisible())
        self.assertEqual(parent.child(1).checkState(0), Qt.CheckState.Unchecked)
        self.assertIsNone(layer.scene())
        display.setCurrentText('Spectrogram')
        self.app.processEvents()
        self.assertIs(viewer.files[key][1][0], layer)
        self.assertIn(layer, viewer.bottom_right_view.addedItems)
        self.assertEqual(layer.color_name, 'viridis')
        self.assertEqual(layer.levels, (2., 9.))
        for plot in viewer.plot_views:
            np.testing.assert_allclose(plot.getViewBox().viewRange()[0], [1, 7])

    def test_shared_colors_use_finite_global_percentiles_and_dialog_edits(self):
        values = np.arange(15, dtype=float).reshape(5, 3)
        values[0, 0], values[-1, -1] = np.nan, np.inf
        key, _, parent = self.load_spectrogram(values=values)
        self.viewer.tree.itemWidget(parent, 1).setCurrentText('Spectrogram')
        layer = self.viewer.files[key][1][0]
        expected = np.percentile(values[np.isfinite(values)], [1, 99])
        np.testing.assert_allclose(layer.levels, expected)
        self.assertEqual(layer.color_name, 'hot')
        dialog = SpectrogramColorsDialog(layer, self.viewer)
        self.addCleanup(dialog.close)
        dialog.colormap.setCurrentText('plasma')
        dialog.low.setValue(3.)
        dialog.high.setValue(10.)
        for image in layer.images:
            np.testing.assert_allclose(image.getLevels(), [3., 10.])
            np.testing.assert_array_equal(image.getColorMap().getLookupTable(),
                                          layer.images[0].getColorMap().getLookupTable())
        dialog.high.setValue(2.)
        self.assertEqual(layer.levels, (3., 10.))
        dialog.reset_colors()
        self.assertEqual(layer.color_name, 'hot')
        for image in layer.images:
            np.testing.assert_allclose(image.getLevels(), expected)

    def test_image_eligibility_requires_linear_frequencies_and_regular_intervals(self):
        for columns, times in [(['a', 'b', 'c'], [0., 1., 2.]),
                               ([1., 2., 4.], [0., 1., 2.]),
                               ([1., 1., 2.], [0., 1., 2.]),
                               ([1., 2., 3.], [0., 1., 2.5])]:
            with self.subTest(columns=columns, times=times):
                frame = nap.TsdFrame(t=times, d=np.ones((3, 3)), columns=columns)
                self.assertIsNone(spectrogram_geometry(frame))
        parent, _ = self.load_curves()
        self.assertIsNone(self.viewer.tree.itemWidget(parent, 1))

    def test_image_edges_are_clipped_without_shifting_centers(self):
        times = [0., 1., 2., 5., 7., 9.]
        support = nap.IntervalSet(start=[0., 5.], end=[2., 9.])
        key, _, parent = self.load_spectrogram(times=times, support=support, columns=[3., 2., 1.])
        self.viewer.tree.itemWidget(parent, 1).setCurrentText('Spectrogram')
        layer = self.viewer.files[key][1][0]
        self.assertEqual([image.parentItem().rect().getCoords() for image in layer.images],
                         [(0., .5, 2., 3.5), (5., .5, 9., 3.5)])
        for image, interval_times in zip(layer.images, [times[:3], times[3:]]):
            for column, time in enumerate(interval_times):
                center = image.mapToItem(layer, QPointF(column + .5, .5))
                self.assertAlmostEqual(center.x(), time)
                self.assertAlmostEqual(center.y(), 1.)
        np.testing.assert_array_equal(layer.images[0].image[0], [2., 5., 8.])

    def test_constant_and_missing_spectrograms_have_finite_color_limits(self):
        for value in [0., 1e-20, np.nan]:
            with self.subTest(value=value):
                data = nap.TsdFrame(t=[0., 1.], d=np.full((2, 2), value), columns=[1., 2.])
                layer = SpectrogramItem(data, spectrogram_geometry(data))
                self.assertTrue(np.isfinite(layer.levels).all())
                self.assertLess(*layer.levels)

    def test_session_round_trip_restores_files_modes_styles_views_and_layout(self):
        parent, curves = self.load_curves()
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom left')
        self.viewer.tree.itemWidget(parent.child(1), 1).setCurrentText('Top right')
        parent.child(1).setCheckState(0, Qt.CheckState.Unchecked)
        curves[0].setPen(QColor('#80334455'))
        key, _, parent = self.load_spectrogram()
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom left')
        parent.child(1).setCheckState(0, Qt.CheckState.Unchecked)
        self.viewer.tree.itemWidget(parent, 1).setCurrentText('Spectrogram')
        layer = self.viewer.files[key][1][0]
        layer.set_colors('plasma', (2., 9.))
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom right')
        parent.child(0).setCheckState(0, Qt.CheckState.Unchecked)
        signal = str(Path(self.directory.name) / 'signal.npz')
        nap.Tsd(t=[0., 1.], d=[1., 2.]).save(signal)
        self.viewer.add_files([signal])
        self.viewer.splitter.setSizes([250, 850])
        self.viewer.plot_splitter.setSizes([260, 400])
        self.viewer.plot.setXRange(1., 7., padding=0)
        for index, (_, target) in enumerate(self.viewer.axis_targets.values()):
            target.setYRange(-index - 1., index + 3., padding=0)
        self.viewer.bottom_plot.getViewBox().enableAutoRange(y=True)
        self.app.processEvents()
        path = str(Path(self.directory.name) / 'display.npzview.json')
        self.assertTrue(self.viewer.save_session(path))
        saved = session.read_session(path)
        restored = self.make_viewer()
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(restored.load_session(path))
        warning.assert_not_called()
        self.app.processEvents()
        self.assertEqual(list(restored.files), list(self.viewer.files))
        self.assertEqual(restored.session_document(path)['files'], saved['files'])
        self.assertIsInstance(restored.files[key][1][0], SpectrogramItem)
        self.assertEqual(restored.tree.itemWidget(restored.files[key][0], 1).currentText(), 'Spectrogram')
        for axis, (_, target) in restored.axis_targets.items():
            settings = saved['view']['axes'][axis.lower().replace(' ', '_')]
            self.assertEqual(bool(target.autoRangeEnabled()[1]), settings['auto_range'])
            if not settings['auto_range']:
                np.testing.assert_allclose(target.viewRange()[1], settings['range'])
            np.testing.assert_allclose(target.viewRange()[0], [1., 7.])
        for name, splitter in [('sidebar', restored.splitter), ('plots', restored.plot_splitter)]:
            sizes = splitter.sizes()
            np.testing.assert_allclose(np.array(sizes) / sum(sizes),
                                       saved['view']['splitters'][name], atol=.005)

    def test_session_preserves_inactive_image_settings_without_building_images(self):
        key, _, parent = self.load_spectrogram()
        display = self.viewer.tree.itemWidget(parent, 1)
        display.setCurrentText('Spectrogram')
        layer = self.viewer.files[key][1][0]
        layer.set_colors('viridis', (1., 12.))
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom right')
        parent.child(0).setCheckState(0, Qt.CheckState.Unchecked)
        display.setCurrentText('Curves')
        path = str(Path(self.directory.name) / 'display.npzview.json')
        self.assertTrue(self.viewer.save_session(path))
        restored = self.make_viewer()
        self.assertTrue(restored.load_session(path))
        self.assertIsNone(restored.spectrograms[key].image)
        self.assertEqual(restored.session_document(path)['files'], session.read_session(path)['files'])
        restored.tree.itemWidget(restored.files[key][0], 1).setCurrentText('Spectrogram')
        image = restored.files[key][1][0]
        self.assertEqual(image.color_name, 'viridis')
        self.assertEqual(image.levels, (1., 12.))
        self.assertFalse(image.isVisible())
        self.assertIn(image, restored.bottom_right_view.addedItems)

    def test_session_restores_time_autoscale_and_can_load_an_empty_session(self):
        self.load_curves()
        self.viewer.plot.getViewBox().enableAutoRange(x=True)
        self.app.processEvents()
        path = Path(self.directory.name) / 'display.npzview.json'
        self.assertTrue(self.viewer.save_session(path))
        restored = self.make_viewer()
        self.assertTrue(restored.load_session(path))
        self.app.processEvents()
        main = restored.plot.getViewBox()
        self.assertTrue(main.autoRangeEnabled()[0])
        self.assertLessEqual(main.viewRange()[0][0], 0.)
        self.assertGreaterEqual(main.viewRange()[0][1], 10.)
        session.write_session(path, {'format': session.FORMAT, 'version': 1, 'files': [],
                                     'view': {'time_range': [-2., 2.], 'time_auto_range': False}})
        self.assertTrue(restored.load_session(path))
        self.assertFalse(restored.files)
        self.assertFalse(restored.file_data)
        self.assertFalse(restored.spectrograms)
        np.testing.assert_allclose(main.viewRange()[0], [-2., 2.])
        self.assertFalse(main.autoRangeEnabled()[0])

    def test_session_relative_paths_work_after_moving_a_bundle(self):
        original = Path(self.directory.name) / 'bundle'
        (original / 'data').mkdir(parents=True)
        source = original / 'data' / 'traces.npz'
        shutil.copyfile(self.path, source)
        self.viewer.add_files([str(source)])
        self.assertTrue(self.viewer.save_session(original / 'display.npzview.json'))
        moved = Path(self.directory.name) / 'moved'
        shutil.copytree(original, moved)
        restored = self.make_viewer()
        self.assertTrue(restored.load_session(moved / 'display.npzview.json'))
        self.assertEqual(list(restored.files), [os.path.normcase(os.path.realpath(moved / 'data' / 'traces.npz'))])

    def test_bad_or_entirely_missing_sessions_leave_current_data_intact(self):
        _, curves = self.load_curves()
        path = Path(self.directory.name) / 'display.npzview.json'
        invalid = [dict(format=session.FORMAT, version=2, files=[]),
                   dict(format=session.FORMAT, version=1, files=[], view={'time_range': [2., 1.]}),
                   dict(format=session.FORMAT, version=1, files=[{'path': 'missing.npz'}])]
        for document in invalid:
            path.write_text(json.dumps(document), encoding='utf-8')
            with self.subTest(document=document), patch.object(QMessageBox, 'warning') as warning:
                self.assertFalse(self.viewer.load_session(path))
                warning.assert_called_once()
            self.assertIs(self.viewer.files[self.key][1][0], curves[0])
            self.assertIsNone(self.viewer.session_path)

    def test_partial_load_reports_missing_sources_together(self):
        self.load_curves()
        path = Path(self.directory.name) / 'display.npzview.json'
        document = self.viewer.session_document(path)
        document['files'].extend([{'path': 'missing1.npz'}, {'path': 'missing2.npz'}])
        session.write_session(path, document)
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(self.viewer.load_session(path))
        warning.assert_called_once()
        message = warning.call_args.args[2]
        self.assertIn('missing1.npz', message)
        self.assertIn('missing2.npz', message)
        self.assertEqual(list(self.viewer.files), [self.key])

    def test_changed_column_order_restores_by_label_and_reports_the_change(self):
        parent, curves = self.load_curves()
        curves[0].setPen('#ff0000')
        curves[1].setPen('#0000ff')
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom left')
        self.viewer.tree.itemWidget(parent.child(1), 1).setCurrentText('Bottom right')
        path = Path(self.directory.name) / 'display.npzview.json'
        self.assertTrue(self.viewer.save_session(path))
        times = np.linspace(0, 10, 101)
        nap.TsdFrame(t=times, d=np.column_stack((np.cos(times), np.sin(times))),
                     columns=['b', 'a']).save(self.path)
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(self.viewer.load_session(path))
        self.assertIn('reordered', warning.call_args.args[2])
        parent, curves = self.viewer.files[self.key]
        self.assertEqual(curves[0].opts['pen'].color().name(), '#0000ff')
        self.assertEqual(curves[1].opts['pen'].color().name(), '#ff0000')
        self.assertEqual(self.viewer.curve_axes[curves[0]], 'Bottom right')
        self.assertEqual(self.viewer.curve_axes[curves[1]], 'Bottom left')

    def test_save_actions_choose_extensions_reuse_target_and_rebase_paths(self):
        self.load_curves()
        path = Path(self.directory.name) / 'display'
        with patch('pluvianus.pynapple_npz_viewer.QFileDialog.getSaveFileName',
                   return_value=(str(path), '')):
            self.viewer.save_settings_as()
        first = Path(str(path) + session.SUFFIX)
        self.assertTrue(first.exists())
        with patch('pluvianus.pynapple_npz_viewer.QFileDialog.getSaveFileName') as dialog:
            self.viewer.files[self.key][1][0].setPen('#ff0000')
            self.viewer.save_settings()
            dialog.assert_not_called()
        self.assertEqual(session.read_session(first)['files'][0]['curves'][0]['color'], '#ffff0000')
        destination = Path(self.directory.name) / 'settings'
        destination.mkdir()
        second = destination / 'another.npzview.json'
        with patch('pluvianus.pynapple_npz_viewer.QFileDialog.getSaveFileName',
                   return_value=(str(second), '')):
            self.viewer.save_settings_as()
        self.assertEqual(session.read_session(second)['files'][0]['path'], '../traces.npz')
        with patch('pluvianus.pynapple_npz_viewer.session.write_session', side_effect=OSError('failed')):
            with patch.object(QMessageBox, 'warning'):
                self.assertFalse(self.viewer.save_session(destination / 'failed.npzview.json'))
        self.assertEqual(self.viewer.session_path, str(second))

    def test_missing_sources_can_be_located_or_loading_cancelled(self):
        self.load_curves()
        path = Path(self.directory.name) / 'display.npzview.json'
        session.write_session(path, {'format': session.FORMAT, 'version': 1,
                                     'files': [{'path': 'missing.npz'}]})
        with patch('pluvianus.pynapple_npz_viewer.MissingSessionFilesDialog.exec',
                   return_value=QDialog.DialogCode.Rejected):
            self.assertFalse(self.viewer.load_session(path, locate_missing=True))
        self.assertIn(self.key, self.viewer.files)

        def locate(dialog):
            dialog.locate_selected()
            return QDialog.DialogCode.Accepted

        with patch('pluvianus.pynapple_npz_viewer.MissingSessionFilesDialog.exec', new=locate):
            with patch('pluvianus.pynapple_npz_viewer.QFileDialog.getOpenFileName',
                       return_value=(self.path, '')):
                with patch.object(QMessageBox, 'warning') as warning:
                    self.assertTrue(self.viewer.load_session(path, locate_missing=True))
        warning.assert_not_called()
        self.assertEqual(list(self.viewer.files), [self.key])
        self.assertEqual(self.viewer.session_document(path)['files'][0]['path'], 'traces.npz')

    def test_changed_spectrogram_grid_and_unknown_colormap_have_explicit_fallbacks(self):
        key, _, parent = self.load_spectrogram()
        self.viewer.tree.itemWidget(parent, 1).setCurrentText('Spectrogram')
        path = Path(self.directory.name) / 'display.npzview.json'
        document = self.viewer.session_document(path)
        document['files'][0]['spectrogram']['colormap'] = 'unavailable'
        session.write_session(path, document)
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(self.viewer.load_session(path))
        self.assertIn('Unknown colormap', warning.call_args.args[2])
        self.assertEqual(self.viewer.files[key][1][0].color_name, 'hot')
        data = self.viewer.file_data[key]
        nap.TsdFrame(t=data.times(), d=data.values, columns=[1., 2., 4.],
                     time_support=data.time_support).save(str(Path(self.directory.name) / 'spectrum.npz'))
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(self.viewer.load_session(path))
        self.assertIn('using curves', warning.call_args.args[2])
        self.assertNotIn(key, self.viewer.spectrograms)

    def load_annotations(self, columns=None):
        times = np.array([0., 1., 2., 3., 4., 5., 8., 9., 10., 11.])
        values = np.array([[0., 0.], [1., 0.], [1., 1.], [np.nan, 1.], [1., 0.],
                           [1., 0.], [1., 0.], [0., 0.], [0., 1.], [1., 1.]])
        support = nap.IntervalSet(start=[0., 8.], end=[6., 12.])
        data = nap.TsdFrame(t=times, d=values, columns=['a', 'b'] if columns is None else columns,
                            time_support=support)
        path = str(Path(self.directory.name) / 'annotations.npz')
        data.save(path)
        self.viewer.add_files([path])
        key = os.path.normcase(os.path.realpath(path))
        return key, data, self.viewer.files[key][0], self.viewer.files[key][1]

    def test_high_state_epochs_respect_gaps_missing_samples_and_binary_levels(self):
        _, data, _, _ = self.load_annotations()
        epochs = high_state_epochs(data.times(), data.values[:, 0], data.time_support)
        np.testing.assert_array_equal(epochs, [[1., 3.], [4., 6.], [8., 9.], [11., 12.]])
        times = np.array([0., .25, 1., 2.])
        support = nap.IntervalSet(start=0., end=3.)
        for values, expected in [(np.array([False, True, True, False]), [[.25, 2.]]),
                                 (np.array([-1., -1., 0., 0.]), [[1., 3.]]),
                                 (np.zeros(4), []), (np.ones(4), [[0., 3.]])]:
            with self.subTest(values=values):
                np.testing.assert_array_equal(high_state_epochs(times, values, support),
                                              np.asarray(expected).reshape(-1, 2))
        for values in [np.array([0., 1., 2., 0.]), np.full(4, 7.), np.full(4, np.nan)]:
            self.assertIsNone(high_state_epochs(times, values, support))

    def test_intervals_shade_both_panels_with_distinct_transparent_colors(self):
        key, _, parent, curves = self.load_annotations()
        viewer = self.viewer
        viewer.plot.setXRange(0., 12., padding=0)
        for index, (_, target) in enumerate(viewer.axis_targets.values()):
            target.setYRange(-100. * (index + 1), 100. * (index + 1), padding=0)
        ranges = [target.viewRange()[1] for _, target in viewer.axis_targets.values()]
        for index in range(2):
            viewer.tree.itemWidget(parent.child(index), 1).setCurrentText('Intervals')
        self.app.processEvents()
        colors = []
        for curve in curves:
            shading = viewer.interval_states[curve]
            self.assertTrue(shading.enabled)
            self.assertEqual(len(shading.items), 2)
            colors.append(shading.items[0].brush.color().name())
            for view, item in zip(viewer.plot_views.values(), shading.items):
                self.assertIs(item.getViewBox(), view)
                self.assertAlmostEqual(item.brush.color().alphaF(), .14, delta=.005)
                self.assertFalse(item.acceptedMouseButtons())
            for plot, right in viewer.plot_views.items():
                self.assertNotIn(curve, plot.getViewBox().addedItems)
                self.assertNotIn(curve, right.addedItems)
                self.assertFalse(plot.plotItem.legend.items)
        self.assertNotEqual(*colors)
        self.assertEqual([target.viewRange()[1] for _, target in viewer.axis_targets.values()], ranges)
        # Render actual pixels: both panels are shaded at every height; low
        # periods and the recording gap stay white, and overlaps remain light.
        for plot in viewer.plot_views:
            view = plot.getViewBox()
            image = plot.viewport().grab().toImage()
            sampled = {}
            for time in [.5, 1.5, 2.5, 3.5, 7.]:
                for fraction in [.2, .8]:
                    lower, upper = view.viewRange()[1]
                    point = view.mapViewToScene(QPointF(time, lower + fraction * (upper - lower)))
                    color = image.pixelColor(plot.mapFromScene(point))
                    self.assertEqual(color.name() == '#ffffff', time in (.5, 7.))
                    self.assertGreater(min(color.red(), color.green(), color.blue()), 175)
                    sampled[time] = color.name()
            self.assertNotEqual(sampled[1.5], sampled[3.5])
            self.assertNotEqual(sampled[2.5], sampled[1.5])

    def test_interval_visibility_axis_switching_zoom_and_close(self):
        key, _, parent, curves = self.load_annotations()
        viewer = self.viewer
        selector = viewer.tree.itemWidget(parent.child(0), 1)
        selector.setCurrentText('Bottom right')
        selector.setCurrentText('Intervals')
        shading = viewer.interval_states[curves[0]]
        self.assertEqual(shading.axis, 'Bottom right')
        parent.child(0).setCheckState(0, Qt.CheckState.Unchecked)
        self.assertTrue(all(not item.isVisible() for item in shading.items))
        selector.setCurrentText('Top left')
        self.assertIn(curves[0], viewer.plot.getViewBox().addedItems)
        self.assertFalse(curves[0].isVisible())
        self.assertTrue(all(item.scene() is None for item in shading.items))
        selector.setCurrentText('Intervals')
        parent.child(0).setCheckState(0, Qt.CheckState.Checked)
        viewer.plot.setXRange(1., 4., padding=0)
        viewer.right_view.setYRange(-2000., 4000., padding=0)
        viewer.plot_splitter.setSizes([450, 200])
        self.app.processEvents()
        for view, item in zip(viewer.plot_views.values(), shading.items):
            np.testing.assert_allclose(item.boundingRect().getCoords(), view.viewRect().getCoords())
        viewer.close_file(key)
        self.assertTrue(all(item.scene() is None for item in shading.items))
        self.assertTrue(all(curve not in viewer.interval_states for curve in curves))
        for plot, right in viewer.plot_views.items():
            self.assertFalse(plot.getViewBox().addedItems)
            self.assertFalse(right.addedItems)

    def test_intervals_are_visible_over_a_spectrogram_on_the_right_axis(self):
        _, _, spectrum = self.load_spectrogram()
        viewer = self.viewer
        viewer.tree.itemWidget(spectrum, 1).setCurrentText('Spectrogram')
        viewer.tree.itemWidget(spectrum.child(0), 1).setCurrentText('Top right')
        _, _, parent, _ = self.load_annotations()
        parent.setCheckState(0, Qt.CheckState.Unchecked)
        viewer.plot.setXRange(0., 8., padding=0)
        viewer.plot.setYRange(-100., 100., padding=0)
        viewer.right_view.setYRange(.5, 3.5, padding=0)
        self.app.processEvents()

        def colors():
            image = viewer.plot.viewport().grab().toImage()
            return [image.pixelColor(viewer.plot.mapFromScene(
                viewer.plot.getViewBox().mapViewToScene(QPointF(time, 0.)))).name()
                    for time in [.5, 1.5]]

        before = colors()
        viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Intervals')
        parent.child(0).setCheckState(0, Qt.CheckState.Checked)
        self.app.processEvents()
        after = colors()
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])

    def test_interval_session_round_trip_and_changed_data_fallback(self):
        key, data, parent, curves = self.load_annotations()
        curves[0].setPen(QColor('#ff228844'))
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Bottom right')
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Intervals')
        parent.child(0).setCheckState(0, Qt.CheckState.Unchecked)
        path = Path(self.directory.name) / 'intervals.npzview.json'
        self.assertTrue(self.viewer.save_session(path))
        document = session.read_session(path)
        self.assertEqual(document['version'], 1)
        self.assertEqual(document['files'][0]['curves'][0]['representation'], 'intervals')
        self.assertEqual(document['files'][0]['curves'][0]['axis'], 'bottom_right')
        restored = self.make_viewer()
        self.assertTrue(restored.load_session(path))
        self.assertEqual(restored.session_document(path)['files'], document['files'])
        parent, curves = restored.files[key]
        self.assertEqual(restored.tree.itemWidget(parent.child(0), 1).currentText(), 'Intervals')
        self.assertEqual(restored.interval_states[curves[0]].items[0].brush.color().name(), '#228844')
        self.assertTrue(all(not item.isVisible() for item in restored.interval_states[curves[0]].items))
        # Older files omit representation and continue to restore ordinary curves.
        for curve in document['files'][0]['curves']:
            curve.pop('representation')
        session.write_session(path, document)
        self.assertTrue(restored.load_session(path))
        self.assertFalse(restored.is_interval_curve(restored.files[key][1][0]))
        document['files'][0]['curves'][0]['representation'] = 'intervals'
        session.write_session(path, document)
        values = data.values.copy()
        values[0, 0] = 2.
        nap.TsdFrame(t=data.times(), d=values, columns=data.columns,
                     time_support=data.time_support).save(str(Path(self.directory.name) / 'annotations.npz'))
        with patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(restored.load_session(path))
        self.assertIn('no longer has two levels', warning.call_args.args[2])
        self.assertFalse(restored.is_interval_curve(restored.files[key][1][0]))

    def test_interval_choices_survive_inactive_spectrogram_representation(self):
        key, _, parent, curves = self.load_annotations(columns=[1., 2.])
        self.viewer.tree.itemWidget(parent.child(0), 1).setCurrentText('Intervals')
        shading = self.viewer.interval_states[curves[0]]
        display = self.viewer.tree.itemWidget(parent, 1)
        display.setCurrentText('Spectrogram')
        self.assertTrue(all(item.scene() is None for item in shading.items))
        path = Path(self.directory.name) / 'intervals.npzview.json'
        self.assertTrue(self.viewer.save_session(path))
        restored = self.make_viewer()
        self.assertTrue(restored.load_session(path))
        state = restored.spectrograms[key]
        self.assertTrue(restored.is_interval_curve(state.curves[0]))
        self.assertTrue(all(item.scene() is None for item in restored.interval_states[state.curves[0]].items))
        restored.tree.itemWidget(restored.files[key][0], 1).setCurrentText('Curves')
        self.assertEqual(restored.tree.itemWidget(restored.files[key][0].child(0), 1).currentText(), 'Intervals')
        self.assertTrue(all(item.scene() is not None for item in restored.interval_states[state.curves[0]].items))


if __name__ == '__main__':
    unittest.main()
