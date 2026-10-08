"""Render native-resolution movie frames and stream them to FFmpeg."""

import subprocess
import tempfile

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPen


class SpatialMovieRenderer:
    """Snapshot the first spatial image's LUT, levels and contour styles."""

    def __init__(self, spatial_widget):
        source = spatial_widget.spatial_image
        lut = source.lut
        if callable(lut):
            lut = lut(source.image, 256)
        self.image_item = pg.ImageItem(
            axisOrder=source.axisOrder,
            autoDownsample=False,
            lut=None if lut is None else np.array(lut, copy=True),
            levels=np.array(source.getLevels(), copy=True),
        )
        self.contours = [
            (QPainterPath(curve.getPath()), QPen(curve.opts['pen']), curve.opts['antialias'])
            for curve in spatial_widget.contur_items
            if curve.isVisible() and curve.opts['pen'].color().alpha() > 0
        ]

    def render(self, image_data):
        # ImageItem performs exactly the same LUT mapping and axis-order
        # conversion as the spatial view, without taking a screen capture.
        self.image_item.setImage(image_data, autoLevels=False)
        self.image_item.render()
        source = self.image_item.qimage
        frame = QImage(source.size(), QImage.Format.Format_RGB888)
        frame.fill(0)
        painter = QPainter(frame)
        try:
            painter.drawImage(0, 0, source)
            for path, pen, antialias in self.contours:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
                painter.setPen(pen)
                painter.drawPath(path)
        finally:
            painter.end()
        # QImage scanlines can contain padding, which raw RGB video cannot.
        pixels = np.frombuffer(frame.constBits(), dtype=np.uint8).reshape(
            frame.height(), frame.bytesPerLine()
        )
        # Own the memory after the local QImage is destroyed, even when its
        # scanlines have no padding and are already contiguous.
        return pixels[:, :frame.width() * 3].reshape(
            frame.height(), frame.width(), 3
        ).copy()


class Mp4Writer:
    """Stream RGB frames without resizing or rounding the source frame rate."""

    def __init__(self, filename, width, height, fps):
        from imageio_ffmpeg import get_ffmpeg_exe

        command = [
            get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s:v', f'{width}x{height}', '-framerate', str(float(fps)),
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '10', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0',
            '-movflags', '+faststart',
            filename,
        ]
        # Windows' built-in decoder supports 4:2:0. Pad odd dimensions by one
        # pixel on the right/bottom, keeping every source pixel without scaling.
        self.log = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
        except Exception:
            self.log.close()
            raise

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def finish(self):
        try:
            self.process.stdin.close()
        except OSError:
            pass  # Report the encoder's error below.
        returncode = self.process.wait()
        self.log.seek(0)
        error = self.log.read().decode('utf-8', errors='replace').strip()
        self.log.close()
        if returncode:
            raise RuntimeError(error or f'FFmpeg exited with code {returncode}.')

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        try:
            if not self.process.stdin.closed:
                self.process.stdin.close()
        except OSError:
            pass
        finally:
            self.log.close()
