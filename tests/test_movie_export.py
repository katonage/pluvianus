import contextlib
import io
import os
import subprocess
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import imageio_ffmpeg
import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QAction, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from pluvianus.movie_export import Mp4Writer, SpatialMovieRenderer
from pluvianus.pluvianus_app import MainWindow, SpatialWidget


def qimage_rgb(image):
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )
    return pixels[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


class MovieExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, directory, dims=(19, 13)):
        window = QMainWindow()
        window.cnm = SimpleNamespace()
        window.dims = dims
        window.num_frames = 6
        window.framerate = 29.97002997002997
        window.frame_window = 1
        window.selected_frame = 3
        window.selected_component = 0
        x, y, t = np.indices((*window.dims, window.num_frames))
        data = (x * 4 + y * 2 + t * 7).astype(np.float32)
        data[8, 5, 2] += 80
        window.data_array = data.reshape(-1, window.num_frames)
        window.hdf5_file = str(Path(directory) / 'results.hdf5')
        spatial = SimpleNamespace(
            mainwindow=window,
            channel_combo=SimpleNamespace(currentText=lambda: 'Data'),
            spatial_avr_spinbox=SimpleNamespace(value=lambda: 1),
            contur_items=[],
        )
        spatial.get_spatial_view_image = MethodType(SpatialWidget.get_spatial_view_image, spatial)
        image_data, _ = spatial.get_spatial_view_image(2, 5)
        spatial.spatial_image = pg.ImageItem(
            image_data, axisOrder='col-major', levels=(0, 160),
            lut=pg.colormap.get('viridis').getLookupTable(nPts=256),
        )
        window.spatial_widget = spatial
        return window

    def test_renderer_matches_spatial_lut_levels_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            spatial = window.spatial_widget
            spatial.spatial_image.render()
            expected = qimage_rgb(spatial.spatial_image.qimage)
            renderer = SpatialMovieRenderer(spatial)
            actual = renderer.render(spatial.spatial_image.image)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.shape, (13, 19, 3))
            # Later source LUT changes must not affect the export snapshot.
            spatial.spatial_image.setLookupTable(np.zeros((256, 3), dtype=np.uint8))
            np.testing.assert_array_equal(renderer.render(spatial.spatial_image.image), expected)

    def test_roi_colors_alpha_width_and_visibility(self):
        image = pg.ImageItem(np.zeros((19, 13)), levels=(0, 1))
        selected = pg.PlotCurveItem(x=[2, 16], y=[3, 3], pen=pg.mkPen((180, 255, 60), width=2))
        faint = pg.PlotCurveItem(x=[2, 16], y=[7, 7], pen=pg.mkPen((255, 0, 0, 100), width=1))
        hidden = pg.PlotCurveItem(x=[2, 16], y=[10, 10], pen=pg.mkPen((0, 255, 0, 0)))
        renderer = SpatialMovieRenderer(SimpleNamespace(spatial_image=image, contur_items=[selected, faint, hidden]))
        selected.setPen(pg.mkPen('r'))
        frame = renderer.render(image.image)
        np.testing.assert_array_equal(frame[3, 8], (180, 255, 60))
        self.assertGreater(frame[6:8, 8, 0].max(), 0)
        self.assertLessEqual(frame[6:8, 8, 0].max(), 100)
        self.assertEqual(frame[6:8, 8, 1:].max(), 0)
        self.assertEqual(frame[9:12, 8].max(), 0)

    def test_movie_menu_availability_tracks_data_array(self):
        # Exercise the real action-state update with unrelated panels mocked.
        window = Mock(cnm=None, data_array=None, num_frames=0,
                      save_movie_action=QAction('Movie to MP4'))
        MainWindow.update_all(window)
        self.assertFalse(window.save_movie_action.isEnabled())
        window.data_array = np.zeros((1, 1))
        MainWindow.update_all(window)
        self.assertTrue(window.save_movie_action.isEnabled())
        window.data_array = None
        MainWindow.update_all(window)
        self.assertFalse(window.save_movie_action.isEnabled())

    def test_export_decodes_at_native_resolution_fps_and_matches_averaging(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory, dims=(65, 49))
            expected = []
            renderer = SpatialMovieRenderer(window.spatial_widget)
            kernel = cv2.getGaussianKernel(3, 1)
            for t in range(window.num_frames):
                mean = window.data_array[:, max(0, t-1):min(6, t+2)].mean(axis=1)
                smooth = cv2.filter2D(mean.reshape(window.dims), -1, np.outer(kernel, kernel))
                expected.append(renderer.render(smooth))
            with patch.object(QMessageBox, 'critical') as critical, contextlib.redirect_stdout(io.StringIO()) as output:
                MainWindow.save_movie(window)
            critical.assert_not_called()
            filename = Path(directory) / 'results.mp4'
            self.assertIn(str(filename), output.getvalue())
            frames = imageio_ffmpeg.read_frames(str(filename))
            metadata = next(frames)
            self.assertEqual(metadata['source_size'], (66, 50))
            self.assertAlmostEqual(metadata['fps'], window.framerate, places=2)
            decoded = [np.frombuffer(frame, dtype=np.uint8).reshape(50, 66, 3) for frame in frames]
            self.assertEqual(len(decoded), window.num_frames)
            for actual, reference in zip(decoded, expected):
                # 4:2:0 mixes chroma between neighboring pixels, especially at
                # the padded right/bottom edge, even with low compression.
                self.assertLess(np.abs(actual[:49, :65].astype(float) - reference).mean(), 5)
            info = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), '-i', str(filename), '-f', 'null', '-'],
                capture_output=True, text=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            self.assertEqual(info.returncode, 0, info.stderr)
            self.assertIn('h264 (High)', info.stderr)
            self.assertIn('yuv420p', info.stderr)
            # Verify the precise fractional rate, beyond FFmpeg's rounded log.
            reader = cv2.VideoCapture(str(filename))
            try:
                self.assertAlmostEqual(reader.get(cv2.CAP_PROP_FPS), window.framerate, places=8)
            finally:
                reader.release()
            self.assertEqual(window.selected_frame, 3)
            self.assertEqual(list(Path(directory).iterdir()), [filename])

    def test_even_dimensions_are_not_padded_or_resized(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'even.mp4')
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[16:32, 24:40] = 255
            writer = Mp4Writer(filename, 64, 48, 12.345678)
            try:
                writer.write(frame)
                writer.finish()
            finally:
                writer.abort()
            frames = imageio_ffmpeg.read_frames(filename)
            self.assertEqual(next(frames)['source_size'], (64, 48))
            decoded = np.frombuffer(next(frames), dtype=np.uint8).reshape(48, 64, 3)
            self.assertLess(np.abs(decoded.astype(float) - frame).mean(), 2)
            frames.close()

    def test_cancel_preserves_existing_movie_and_removes_partial_export(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            filename = Path(directory) / 'results.mp4'
            filename.write_bytes(b'previous movie')
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                 patch('pluvianus.pluvianus_app.QProgressDialog.wasCanceled', side_effect=[False, True]), \
                 patch.object(QMessageBox, 'critical') as critical:
                MainWindow.save_movie(window)
            critical.assert_not_called()
            self.assertEqual(filename.read_bytes(), b'previous movie')
            self.assertEqual(list(Path(directory).iterdir()), [filename])

    def test_encoder_failure_preserves_existing_movie_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            window = self.make_window(directory)
            filename = Path(directory) / 'results.mp4'
            filename.write_bytes(b'previous movie')
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                 patch('pluvianus.pluvianus_app.Mp4Writer.write', side_effect=RuntimeError('encoder failed')), \
                 patch.object(QMessageBox, 'critical') as critical:
                MainWindow.save_movie(window)
            critical.assert_called_once()
            self.assertEqual(filename.read_bytes(), b'previous movie')
            self.assertEqual(list(Path(directory).iterdir()), [filename])


if __name__ == '__main__':
    unittest.main()
