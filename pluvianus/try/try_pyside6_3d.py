import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
from PySide6.Qt3DExtras import Qt3DWindow, QOrbitCameraController, QSphereMesh, QPhongMaterial, QFirstPersonCameraController
from PySide6.Qt3DCore import QEntity
from PySide6.Qt3DExtras import QTransform
from PySide6.Qt3DRender import QObjectPicker, QPickEvent
from PySide6.QtGui import QVector3D, QColor

class Scatter3D(QEntity):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = np.random.normal(size=(50, 3), scale=5)
        self.create_scatter()

    def create_scatter(self):
        for point in self.points:
            sphereEntity = QEntity(self)

            sphereMesh = QSphereMesh()
            sphereMesh.setRadius(0.15)

            sphereMaterial = QPhongMaterial(self)
            sphereMaterial.setDiffuse(QColor('dodgerblue'))

            sphereTransform = QTransform()
            sphereTransform.setTranslation(QVector3D(*point))

            picker = QObjectPicker(sphereEntity)
            picker.clicked.connect(self.on_point_clicked)

            sphereEntity.addComponent(sphereMesh)
            sphereEntity.addComponent(sphereMaterial)
            sphereEntity.addComponent(sphereTransform)
            sphereEntity.addComponent(picker)
            
            # Store coordinates for easy retrieval
            sphereEntity.setProperty('coordinates', QVector3D(*point))

    def on_point_clicked(self, event: QPickEvent):
        coords = event.entity().property('coordinates')
        print(f"Clicked: ({coords.x():.2f}, {coords.y():.2f}, {coords.z():.2f})")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    view = Qt3DWindow()
    container = QWidget.createWindowContainer(view)
    widget = QWidget()
    hLayout = QHBoxLayout(widget)
    hLayout.addWidget(container)

    rootEntity = QEntity()

    scatter = Scatter3D(rootEntity)

    camera = view.camera()
    camera.lens().setPerspectiveProjection(45.0, 16/9, 0.1, 1000.0)
    camera.setPosition(QVector3D(0, 0, 20))
    camera.setViewCenter(QVector3D(0, 0, 0))

    camController = QOrbitCameraController(rootEntity)
    camController.setCamera(camera)

    view.setRootEntity(rootEntity)

    widget.resize(800, 600)
    widget.setWindowTitle("Native PySide6 Qt3D Scatter")
    widget.show()

    sys.exit(app.exec())
