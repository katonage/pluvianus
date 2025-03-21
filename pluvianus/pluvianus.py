import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QLabel, QComboBox, QPushButton, QProgressDialog
)
from PySide6.QtGui import QAction
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QObject, Signal, Slot, Qt
from PySide6.QtWebChannel import QWebChannel
import pyqtgraph as pg

import plotly.graph_objects as go
import json
import os
import numpy as np

import caiman as cm
from caiman.source_extraction import cnmf # type: ignore

class OptsWindow(QMainWindow):
    def __init__(self, opts):
        super().__init__()
        self.setWindowTitle("Options")
        self.textedit = QTextEdit()
        self.textedit.setReadOnly(True)
        self.setCentralWidget(self.textedit)
        self.textedit.setText(repr(opts))
        self.resize(500, 800)

class ShiftsWindow(QMainWindow):
    def __init__(self, shifts):
        super().__init__()
        self.setWindowTitle("Shifts")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        if True:
            self.browser = pg.PlotWidget()
            self.browser.plot(x=np.arange(shifts.shape[0]), y=shifts[:, 0], pen=pg.mkPen(color='b', width=2), name='x shifts')
            self.browser.plot(x=np.arange(shifts.shape[0]), y=shifts[:, 1], pen=pg.mkPen(color='r', width=2), name='y shifts')
            self.browser.setLabel('bottom', 'Frame Number')
            self.browser.setLabel('left', 'Shift')
            self.browser.showGrid(x=True, y=True)
            self.browser.setWindowTitle("Shifts per Frame")
    
        else:
            fig = go.Figure(data=[
                go.Scatter(x=np.arange(shifts.shape[0]), y=shifts[:, 0], name='x shifts', line=dict(color='blue', width=2)),
                go.Scatter(x=np.arange(shifts.shape[0]), y=shifts[:, 1], name='y shifts', line=dict(color='red', width=2))
            ])
            fig.update_layout(title="Shifts per Frame", xaxis_title="Frame Number", yaxis_title="Shift")
            self.browser = QWebEngineView()
            #self.browser.setContextMenuPolicy(Qt.NoContextMenu)       
            self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))
        
        layout.addWidget(self.browser)
        self.resize(700, 500)
        
"""
class PluvianusPlotbox_plotly(QWebEngineView):
    def __init__(self):
        super().__init__()
        
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.fig = go.Figure()
        
    def add_trace(x, y, name):
"""     
        
        
class BackgroundWindow(QMainWindow):
    def __init__(self, b, f, dims):
        super().__init__()
        self.setWindowTitle("Background Components")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # top row
        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignLeft)
        label = QLabel("Component: ")
        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(0)
        self.spin_box.setMaximum(b.shape[-1] - 1)
        self.spin_box.valueChanged.connect(self.update_plot)
        top_row.addWidget(label)
        top_row.addWidget(self.spin_box)
        layout.addLayout(top_row)
        
                
        self.b = b
        self.f = f
        self.dims = dims
        
        # bottom row
        bottom_row = QHBoxLayout()
        
        if True:
            self.tempopral_widget = pg.PlotWidget()
        else:
            self.tempopral_widget = QWebEngineView()
            self.tempopral_widget.setContextMenuPolicy(Qt.NoContextMenu)
        bottom_row.addWidget(self.tempopral_widget)  

        self.spatial_widget = pg.ImageView()
        self.spatial_widget.getView().setAspectLocked(True)
        self.spatial_widget.getView().invertY(False)
        self.spatial_widget.getView().setBackgroundColor((255, 0, 255))

        
        self.spatial_widget.setColorMap(pg.colormap.get('viridis', source='matplotlib'))
        self.spatial_widget.ui.menuBtn.hide()
        self.spatial_widget.ui.roiBtn.hide()
        self.spatial_widget.getView().autoRange()
        region = self.spatial_widget.ui.histogram.item.region
        alpha = 50
        region.setBrush((200, 200, 200, alpha))  # subtle gray for clarity
        for line in region.lines:
            line.setPen((100, 100, 100, 255))  # neutral gray outline
        
       
        bottom_row.addWidget(self.spatial_widget)
        
    
        layout.addLayout(bottom_row)
        self.update_plot(0)        
        self.resize(1200, 600)
    
    def update_plot(self, value):
        index = self.spin_box.value()
        
        vec = self.b[:, index]
        print("Component", index, "vector length:", len(vec), "dims product:", np.prod(self.dims))

        # Temporal plot
        if True:
            self.tempopral_widget.plot(x=np.arange(self.f.shape[1]), y=self.f[index, :], pen=pg.mkPen(color='b', width=2), name='data')
            self.tempopral_widget.setLabel('bottom', 'Frame Number')
            self.tempopral_widget.setLabel('left', 'Fluorescence')
            self.tempopral_widget.showGrid(x=True, y=True)
            self.tempopral_widget.setWindowTitle(f"Temporal Component {index}")
        else:
            temporal_fig = go.Figure(data=[go.Scatter(x=np.arange(self.f.shape[1]), y=self.f[index, :], line=dict(color='blue', width=2))])
            temporal_fig.update_layout(title=f'Temporal Component {index}', xaxis_title='Frame', yaxis_title='Fluorescence')
            self.tempopral_widget.setHtml(temporal_fig.to_html(include_plotlyjs='cdn'))

        
        # Spatial plot using pyqtgraph
        self.spatial_widget.setImage(vec.reshape(self.dims).T, autoLevels=True)
        

    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1000, 700)
        #os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
        # Setup file menu with Open, Save, Save As
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open", self)
        self.save_action = QAction("Save", self)
        self.save_as_action = QAction("Save As", self)
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        
        comp_menu = self.menuBar().addMenu("Compute")
        self.detr_action = QAction("Detrend df/f", self)
        comp_menu.addAction(self.detr_action)

        view_menu = self.menuBar().addMenu("View")
        self.opts_action = QAction("Opts", self)
        view_menu.addAction(self.opts_action)
        self.bg_action = QAction("Background", self)
        view_menu.addAction(self.bg_action)
        self.shifts_action = QAction("Shifts", self)
        view_menu.addAction(self.shifts_action)
        
        exp_menu = self.menuBar().addMenu("Export")
        self.npz_action = QAction("Pynapple npz", self)
        exp_menu.addAction(self.npz_action)
        
        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About", self)
        license_action = QAction("License", self)
        source_action = QAction("Source", self)
        help_menu.addAction(about_action)
        help_menu.addAction(license_action)
        help_menu.addAction(source_action)
        
        open_action.triggered.connect(self.open_file)
        self.save_action.triggered.connect(self.save_file)
        self.save_as_action.triggered.connect(self.save_file_as)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top plot: Temporal (full width)
        self.temporal_view = QWebEngineView()
        self.temporal_view.setContextMenuPolicy(Qt.NoContextMenu)
            
        self.component_spinbox = QSpinBox()
        self.component_spinbox.setMinimum(0)
        self.component_spinbox.setValue(0)
        self.component_spinbox.setFixedWidth(100)
        
        self.component_params_r = QLabel("R: --")
        self.component_params_SNR = QLabel("SNR: --")
        self.component_params_CNN = QLabel("CNN: --")
     
        self.component_type = QComboBox()
        self.component_type.addItem("All")
        self.component_type.addItem("Good")
        self.component_type.addItem("Bad")
       
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        head_label=QLabel("Component:")
        head_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(head_label)
        left_layout.addWidget(self.component_spinbox)
        left_layout.addWidget(QLabel("Limit to:"))
        left_layout.addWidget(self.component_type)
        
        head_label=QLabel("Plot:")
        head_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(head_label)
        self.array_selector = QComboBox()
        left_layout.addWidget(self.array_selector)
        
        head_label=QLabel("Metrics:")
        head_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(head_label)
        left_layout.addWidget(self.component_params_r)
        left_layout.addWidget(self.component_params_SNR)
        left_layout.addWidget(self.component_params_CNN)
        
        head_label=QLabel("Accept:")
        head_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(head_label)


        # Create a layout for the toggle buttons
        toggle_button_layout = QVBoxLayout()
        # Create the "Good" toggle button
        good_toggle_button = QPushButton("Good")
        good_toggle_button.setCheckable(True)
        #good_toggle_button.setStyleSheet("background-color: white; color: green;")
        toggle_button_layout.addWidget(good_toggle_button)
        self.good_toggle_button = good_toggle_button
        # Create the "Bad" toggle button
        bad_toggle_button = QPushButton("Bad")
        bad_toggle_button.setCheckable(True)
        bad_toggle_button.setStyleSheet("background-color: white; color: red;")
        toggle_button_layout.addWidget(bad_toggle_button)
        self.bad_toggle_button = bad_toggle_button
        # Add the toggle button layout to the left layout
        left_layout.addLayout(toggle_button_layout)

        
        top_layout = QHBoxLayout()
        top_layout.addLayout(left_layout)
        top_layout.addWidget(self.temporal_view, stretch=1)
        main_layout.addLayout(top_layout, stretch=1)
        # Bottom layout for Spatial and Parameters plots
        threshold_layout = QVBoxLayout()
        threshold_layout.setAlignment(Qt.AlignTop)
        head_label=QLabel("Components:")
        head_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        threshold_layout.addWidget(head_label)
        self.total_label = QLabel("Total: --")
        threshold_layout.addWidget(self.total_label)
        self.good_label = QLabel("Good: --")
        threshold_layout.addWidget(self.good_label)
        self.bad_label = QLabel("Bad: --")
        threshold_layout.addWidget(self.bad_label)
        
        head_label=QLabel("Thresholds:")
        head_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        
        threshold_layout.addWidget(head_label)
        threshold_layout.addWidget(QLabel("  SNR_lowest:"))
        self.SNR_lowest_spinbox = QDoubleSpinBox()
        self.SNR_lowest_spinbox.setToolTip("Minimum required trace SNR. Traces with SNR below this will get rejected")
        threshold_layout.addWidget(self.SNR_lowest_spinbox)
        threshold_layout.addWidget(QLabel("  min_SNR:"))
        self.min_SNR_spinbox = QDoubleSpinBox()
        self.min_SNR_spinbox.setToolTip("Trace SNR threshold. Traces with SNR above this will get accepted")
        threshold_layout.addWidget(self.min_SNR_spinbox)
        threshold_layout.addWidget(QLabel("  cnn_lowest:"))
        self.cnn_lowest_spinbox = QDoubleSpinBox()
        self.cnn_lowest_spinbox.setToolTip("Minimum required CNN threshold. Components with score lower than this will get rejected")
        threshold_layout.addWidget(self.cnn_lowest_spinbox)
        threshold_layout.addWidget(QLabel("  min_cnn_thr:"))
        self.min_cnn_thr_spinbox = QDoubleSpinBox()
        self.min_cnn_thr_spinbox.setToolTip("CNN classifier threshold. Components with score higher than this will get accepted")
        threshold_layout.addWidget(self.min_cnn_thr_spinbox)
        threshold_layout.addWidget(QLabel("  rval_lowest:"))
        self.rval_lowest_spinbox = QDoubleSpinBox()
        self.rval_lowest_spinbox.setToolTip("Minimum required space correlation. Components with correlation below this will get rejected")
        threshold_layout.addWidget(self.rval_lowest_spinbox)
        threshold_layout.addWidget(QLabel("  rval_thr:"))
        self.rval_thr_spinbox = QDoubleSpinBox()
        self.rval_thr_spinbox.setToolTip("Space correlation threshold. Components with correlation higher than this will get accepted")
        threshold_layout.addWidget(self.rval_thr_spinbox)
        bottom_layout = QHBoxLayout()
        self.spatial_view = QWebEngineView()
        self.parameters_view = QWebEngineView()
        self.parameters_view.setContextMenuPolicy(Qt.NoContextMenu)
        self.spatial_view.setContextMenuPolicy(Qt.NoContextMenu)
        bottom_layout.addLayout(threshold_layout)
        bottom_layout.addWidget(self.parameters_view, stretch=2)
        bottom_layout.addWidget(self.spatial_view, stretch=2)
        main_layout.addLayout(bottom_layout, stretch=1)
        
        # Initialize variables
        self.hdf5_file = None
        self.hdf5_path = None
        self.file_changed = False
        self.online = False
        self.cnm = None
        self.selected_component = 0
        self.selected_component_on_scatter = -1
        self.time = 0
        self.limit='All'
        
        # Connect events
        self.component_spinbox.valueChanged.connect(self.on_component_spinbox_changed)
        self.resizeEvent = self.on_resize_figure
        self.component_type.currentTextChanged.connect(self.on_limit_component_type_changed)
        self.array_selector.currentTextChanged.connect(self.on_array_selector_changed)  
        self.detr_action.triggered.connect(self.on_detrend_action)
        self.npz_action.triggered.connect(self.on_npz_action)
        for widget in (self.SNR_lowest_spinbox, self.min_SNR_spinbox, self.cnn_lowest_spinbox, self.min_cnn_thr_spinbox, self.rval_lowest_spinbox, self.rval_thr_spinbox):
            widget.valueChanged.connect(self.on_threshold_spinbox_changed)
        self.opts_action.triggered.connect(self.on_opts_action)
        self.shifts_action.triggered.connect(self.on_shifts_action)
        self.bg_action.triggered.connect(self.on_bg_action)
        self.closeEvent = self.on_mainwindow_closing
        
        # Create a Python bridge and web channel for JS communication
        self.bridge = PythonBridge(self)
        self.channel = QWebChannel()
        self.channel.registerObject("pythonBridge", self.bridge)
        self.parameters_view.page().setWebChannel(self.channel)
        
        # Update figure and state
        self.load_state()
        self.update_all()
    
    def valid_component_select(self, value):
        """
        Valid component select. This function is called when the component number is changed.
        The component number is checked against the type of components that are selected in the combo box.

        Parameters
        ----------
        value : int
            The new value of the component.

        Returns
        -------
        None, sets the self.selected_component to the new value.
        """
        def neares_in_dir(array, previous, new):
            direction = np.sign(new - previous)
            if direction == 0:
                idx = (np.abs(array - new)).argmin()
            elif direction == 1:
                idx = np.where(array > new)[0][0] if len(np.where(array > new)[0]) > 0 else -1
            else: # direction == -1
                idx = np.where(array < new)[0][-1] if len(np.where(array < new)[0]) > 0 else 0
            return array[idx]
        
        previous_value = self.selected_component
        if self.limit == 'All':
            self.selected_component = value
        elif self.limit == 'Good':
            if self.cnm is None or self.cnm.estimates.idx_components is None:
                self.component_type.setCurrentText('All')
                self.selected_component = value
            else:                
                if value in self.cnm.estimates.idx_components:
                    self.selected_component = value
                else:
                    self.selected_component = neares_in_dir(self.cnm.estimates.idx_components, previous_value, value)
        elif self.limit == 'Bad':
            if self.cnm is None or self.cnm.estimates.idx_components is None:
                self.component_type.setCurrentText('All')
                self.selected_component = value
            else: 
                if value in self.cnm.estimates.idx_components_bad:
                    self.selected_component = value
                else:
                    self.selected_component = neares_in_dir(self.cnm.estimates.idx_components_bad, previous_value, value)
        #print('selector: ', previous_value, value, self.selected_component )      
    
    # Event handlers
    def on_component_spinbox_changed(self, value):
        self.valid_component_select(value)
        self.update_component_spinbox(self.selected_component)
        self.update_selected_component_on_scatterplot(self.selected_component)
        self.plot_temporal()
        
    def update_component_spinbox(self, value):
        if self.component_spinbox.value() != value:
            self.component_spinbox.blockSignals(True)
            self.component_spinbox.setValue(value)
            self.component_spinbox.blockSignals(False)
    
    def on_resize_figure(self, event):
        self.update_title() 
        self.save_state()
    
    def on_mainwindow_closing(self, event):
        self.close_child_windows()
        
    def close_child_windows(self):
        if hasattr(self, 'opts_window') and self.opts_window.isVisible():
            self.opts_window.close()
        if hasattr(self, 'shifts_window') and self.shifts_window.isVisible():
            self.shifts_window.close()
        if hasattr(self, 'background_window') and self.background_window.isVisible():
            self.background_window.close()
            
    def on_limit_component_type_changed(self, text):
        self.limit = text
        self.on_component_spinbox_changed(self.selected_component)
        
    def on_array_selector_changed(self, text):
        #[ "F_dff", "C",  "S", "YrA", "R", "noisyC", "C_on"]
        tooltips={'C': 'Temporal traces', 
                  'F_dff': 'DF/F normalized activity trace', 
                  'S': 'Deconvolved neural activity trace', 
                  'YrA': 'Trace residuals', 
                  'R': 'Trace residuals', 
                  'noisyC': 'Temporal traces (including residuals plus background)', 
                  'C_on': '?'}
        self.plot_temporal()
        self.array_selector.setToolTip(tooltips[text])
    
    def construct_threshold_gridline_data(self):
        if self.cnm is None:
            return {}
        x = self.cnm.estimates.cnn_preds
        y = self.cnm.estimates.r_values
        z = self.cnm.estimates.SNR_comp
        lines = {
                'X = min_cnn_thr': {'x': [float(self.cnm.params.quality['min_cnn_thr'])]*5, 'y': [float(min(y)), float(max(y)), float(max(y)), float(min(y)), float(min(y))], 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'green'},
                'X = cnn_lowest': {'x': [float(self.cnm.params.quality['cnn_lowest'])]*5, 'y': [float(min(y)), float(max(y)), float(max(y)), float(min(y)), float(min(y))], 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'green'},
                'Y = rval_thr': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(self.cnm.params.quality['rval_thr'])]*5, 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'blue'},
                'Y = rval_lowest': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(self.cnm.params.quality['rval_lowest'])]*5, 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'blue'},
                'Z = min_SNR': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(min(y)), float(min(y)), float(max(y)), float(max(y)), float(min(y))], 'z': [float(self.cnm.params.quality['min_SNR'])]*5, 'color': 'magenta'},
                'Z = SNR_lowest': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(min(y)), float(min(y)), float(max(y)), float(max(y)), float(min(y))], 'z': [float(self.cnm.params.quality['SNR_lowest'])]*5, 'color': 'magenta'},
            }
        return lines
    
    def on_threshold_spinbox_changed(self, value):
        if self.cnm.estimates.idx_components is None:
            return
        self.cnm.params.quality['SNR_lowest'] = self.SNR_lowest_spinbox.value()
        self.cnm.params.quality['min_SNR'] = self.min_SNR_spinbox.value()
        self.cnm.params.quality['cnn_lowest'] = self.cnn_lowest_spinbox.value()
        self.cnm.params.quality['min_cnn_thr'] = self.min_cnn_thr_spinbox.value()
        self.cnm.params.quality['rval_lowest'] = self.rval_lowest_spinbox.value()
        self.cnm.params.quality['rval_thr'] = self.rval_thr_spinbox.value()
        self.file_changed = True
        self.update_title()
        self.update_threshold_lines_on_scatterplot()
        self.update_treshold_spinboxes()
        
    def update_treshold_spinboxes(self):
        if self.cnm is None:
            self.SNR_lowest_spinbox.setEnabled(False)
            self.min_SNR_spinbox.setEnabled(False)
            self.cnn_lowest_spinbox.setEnabled(False)
            self.min_cnn_thr_spinbox.setEnabled(False)
            self.rval_lowest_spinbox.setEnabled(False)
            self.rval_thr_spinbox.setEnabled(False)
            return
        if self.cnm.estimates.cnn_preds is None:
            cnn_range = (0, 1)
        else:
            cnn_range = (np.min(self.cnm.estimates.cnn_preds), np.max(self.cnm.estimates.cnn_preds))
        if self.cnm.estimates.r_values is None:
            rval_range = (-1, 1)
        else:   
            rval_range = (np.min(self.cnm.estimates.r_values), np.max(self.cnm.estimates.r_values))
        if self.cnm.estimates.SNR_comp is None:
            snr_range = (0, 100)
        else:
            snr_range = (np.min(self.cnm.estimates.SNR_comp), np.max(self.cnm.estimates.SNR_comp))       
        
        self.SNR_lowest_spinbox.blockSignals(True)
        self.SNR_lowest_spinbox.setEnabled(True)
        self.SNR_lowest_spinbox.setRange(*snr_range)
        self.SNR_lowest_spinbox.setSingleStep(0.1)
        self.SNR_lowest_spinbox.setValue(self.cnm.params.quality['SNR_lowest'])
        self.SNR_lowest_spinbox.blockSignals(False)
        self.min_SNR_spinbox.blockSignals(True)
        self.min_SNR_spinbox.setEnabled(True)
        self.min_SNR_spinbox.setRange(*snr_range)
        self.min_SNR_spinbox.setSingleStep(0.1)        
        self.min_SNR_spinbox.setValue(self.cnm.params.quality['min_SNR'])
        self.min_SNR_spinbox.blockSignals(False)
        self.cnn_lowest_spinbox.blockSignals(True)
        self.cnn_lowest_spinbox.setEnabled(True)
        self.cnn_lowest_spinbox.setRange(*cnn_range)
        self.cnn_lowest_spinbox.setSingleStep(0.1)     
        self.cnn_lowest_spinbox.setValue(self.cnm.params.quality['cnn_lowest'])
        self.cnn_lowest_spinbox.blockSignals(False)
        self.min_cnn_thr_spinbox.blockSignals(True)
        self.min_cnn_thr_spinbox.setEnabled(True)
        self.min_cnn_thr_spinbox.setRange(*cnn_range)
        self.min_cnn_thr_spinbox.setSingleStep(0.1)
        self.min_cnn_thr_spinbox.setValue(self.cnm.params.quality['min_cnn_thr'])
        self.min_cnn_thr_spinbox.blockSignals(False)
        self.rval_lowest_spinbox.blockSignals(True)
        self.rval_lowest_spinbox.setEnabled(True)
        self.rval_lowest_spinbox.setRange(*rval_range)
        self.rval_lowest_spinbox.setSingleStep(0.1)       
        self.rval_lowest_spinbox.setValue(self.cnm.params.quality['rval_lowest'])
        self.rval_lowest_spinbox.blockSignals(False)
        self.rval_thr_spinbox.blockSignals(True)
        self.rval_thr_spinbox.setEnabled(True)
        self.rval_thr_spinbox.setRange(*rval_range)
        self.rval_thr_spinbox.setSingleStep(0.1)
        self.rval_thr_spinbox.setValue(self.cnm.params.quality['rval_thr'])
        self.rval_thr_spinbox.blockSignals(False)
        
        
    def on_scatter_point_clicked(self, index):
        #print(f"Point clicked: {index}", end='')
        if self.selected_component_on_scatter == index:
            #print(" (same as before)")
            return
        self.selected_component_on_scatter = index
        #print(f" (new value: {index})")
        self.valid_component_select(index)
        self.update_component_spinbox(self.selected_component)
        self.update_selected_component_on_scatterplot(self.selected_component)
        self.plot_temporal()
    
    def update_threshold_lines_on_scatterplot(self):
        if self.cnm.estimates.idx_components is None:
            return
        lines=self.construct_threshold_gridline_data()
        lines_json = json.dumps(lines)
        # Build the JS command calling updateThresholdLines, ensuring proper quoting
        js_cmd = f'window.updateThresholdLines({json.dumps(lines_json)});'
        #print(f"Calling JS for threshold lines: {js_cmd}")
        self.parameters_view.page().runJavaScript(js_cmd)
        
    def update_selected_component_on_scatterplot(self, index):
        if self.cnm is None or self.cnm.estimates.r_values is None:
            return
        # Compute 3D coordinates and pass them directly as floats
        #print(f"Updating selected component on scatter plot: {index}")
        x_val = float(self.cnm.estimates.cnn_preds[index])
        y_val = float(self.cnm.estimates.r_values[index])
        z_val = float(self.cnm.estimates.SNR_comp[index])
        if index in self.cnm.estimates.idx_components:
            color = 'green'
        else:
            color = 'magenta'
        js_cmd = f'window.updateSelectedTrace({x_val}, {y_val}, {z_val}, "{color}");'
        #print(f'Calling JS: {js_cmd}')
        self.parameters_view.page().runJavaScript(js_cmd)
        
    def on_detrend_action(self):
        """
        Detrending using the `detrend_df_f` method of the CNMF object.
        After detrending is complete, the `file_changed` flag is set qand the estimates.f_dff array is filled.
        """
        if self.cnm is None or self.cnm.estimates.F_dff is not None:
            return
        print("Detrending...")
        waitDlg = QProgressDialog("Detrending in progress...", None, 0, 0, self)
        waitDlg.setWindowModality(Qt.ApplicationModal)  # Blocks input to main window
        waitDlg.setCancelButton(None)  # No cancel button
        waitDlg.setWindowTitle("Please Wait")
        waitDlg.setMinimumDuration(0)  # Show immediately
        waitDlg.setRange(0, 0)  # Indeterminate progress
        waitDlg.show()

        # Change cursor to wait cursor
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # Allow UI updates while detrending
        QApplication.processEvents()
        
        try:
            # Detrend traces
            self.cnm.estimates.detrend_df_f() #quantileMin=8, frames_window=250
            self.file_changed = True
            self.update_all()
        finally:
            waitDlg.close()
            QApplication.restoreOverrideCursor()
        
    
    def on_npz_action(self):
        pass

    def on_opts_action(self):
        if self.cnm is None:
            return
        if hasattr(self, 'opts_window') and self.opts_window.isVisible():
            self.opts_window.raise_()
            self.opts_window.show()
        else:
            self.opts_window = OptsWindow(self.cnm.params)
            self.opts_window.show()
            if self.online:
                self.opts_window.setWindowTitle("Options (OnlineCNMF)")
            else:
                self.opts_window.setWindowTitle("Options (CNMF)")
                
    def on_shifts_action(self):
        if self.cnm is None or self.cnm.estimates.shifts is None or len(self.cnm.estimates.shifts) == 0:
            return
        if hasattr(self, 'shifts_window') and self.shifts_window.isVisible():
            self.shifts_window.raise_()
            self.shifts_window.show()
            return
        shifts=self.cnm.estimates.shifts
        if self.online:
            #epochs=cnm.params.online['epochs']
            total_frames=self.cnm.estimates.C.shape[-1]
            shifts=shifts[-(total_frames):,:]
        self.shifts_window = ShiftsWindow(shifts)
        self.shifts_window.show()
    
    def on_bg_action(self):
        if self.cnm is None:
            return
        if hasattr(self, 'background_window') and self.background_window.isVisible():
            self.background_window.raise_()
            self.background_window.show()
            return
        print(self.cnm.dims)
        self.background_window = BackgroundWindow(self.cnm.estimates.b, self.cnm.estimates.f, self.cnm.dims)
        self.background_window.show()
        
    def open_file(self):        
        if self.hdf5_path is not None and os.path.exists(self.hdf5_path):
            filename, _ = QFileDialog.getOpenFileName(self, "Open CaImAn hdf5 File", self.hdf5_path, "HDF5 Files (*.hdf5)")
        else:
            filename, _ = QFileDialog.getOpenFileName(self, "Open CaImAn hdf5 File", ".", "HDF5 Files (*.hdf5)")
        if filename:
            print("Open file:", filename)
            try:
                self.cnm = cnmf.online_cnmf.load_OnlineCNMF(filename) 
                #check
                if self.cnm.params.online['movie_name_online'] == 'online_movie.mp4':
                    raise Exception("Not an OnlineCNMF file")
                self.online = True
                print("File loaded (OnlineCNMF):", filename)
            except Exception as e:
                try:
                    self.cnm = cnmf.cnmf.load_CNMF(filename)
                    self.online = False
                    print("File loaded (CNMF):", filename)
                except Exception as e:
                    print("Could not load file")
                    QMessageBox.critical(self, "Error opening file", "File could not be opened: " + filename)
                    return
            self.hdf5_file = filename
            self.hdf5_path = os.path.abspath(filename)
            self.file_changed = False
            
            #some corrections
            #if self.cnm.params.data['dims'] is None:
            #    if hasattr(self.cnm.estimates, 'Cn'):
            #        self.cnm.params.data['dims'] = self.cnm.estimates.Cn.shape
            #        print("dims corrected from Cn")
            
            self.save_state()
            self.update_all()
                        
    def update_all(self):
        
        self.close_child_windows()
                    
        if  self.cnm is None:
            self.component_spinbox.setEnabled(False)
            self.detr_action.setEnabled(False)
            self.npz_action.setEnabled(False)
            self.array_selector.setEnabled(False)
            self.bg_action.setEnabled(False)
            self.shifts_action.setEnabled(False)
            self.opts_action.setEnabled(False)
            self.update_title()
            self.plot_temporal()
            self.plot_spatial()
            self.plot_parameters()
            self.update_title()
            self.update_treshold_spinboxes()
            return
        
        selectable_array_names=[]
        possible_array_names = [ "F_dff", "C",  "S", "YrA", "R", "noisyC", "C_on"]
        for array_name in possible_array_names:
            value = getattr(self.cnm.estimates, array_name)
            if (value is not None) :
                selectable_array_names.append(array_name)
        print("Selectable array names:", selectable_array_names)
        previous_selected_array = self.array_selector.currentText()
        self.array_selector.blockSignals(True)
        self.array_selector.clear()
        for array_name in selectable_array_names:
            self.array_selector.addItem(array_name)
        if previous_selected_array in selectable_array_names:
            self.array_selector.setCurrentText(previous_selected_array)
        self.array_selector.setEnabled(True) 
        self.array_selector.blockSignals(False)
         
        numcomps=self.cnm.estimates.A.shape[-1]
        self.component_spinbox.setMaximum(numcomps - 1)
        self.selected_component = min(numcomps-1, self.selected_component)
        self.update_component_spinbox(self.selected_component)
        self.total_label.setText(f"    Total: {numcomps}")
        self.component_spinbox.setEnabled(True)
                    
        self.detr_action.setEnabled(self.cnm.estimates.F_dff is None)
        self.npz_action.setEnabled(True)
        self.bg_action.setEnabled(True)
        self.shifts_action.setEnabled(self.cnm.estimates.shifts is not None and len(self.cnm.estimates.shifts) > 0)
        self.opts_action.setEnabled(True)
                          
        self.update_title()
        self.update_treshold_spinboxes()
        self.plot_temporal()
        self.plot_spatial()
        self.plot_parameters()

    def save_file(self):
        print("Save file")
        # Implement save logic here

    def save_file_as(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save File As", "", "All Files (*)")
        if filename:
            print("Save file as:", filename)
            # Implement save as logic here

    def plot_temporal(self):
        if self.cnm is None:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(
                text="No data loaded yet.\nOpen CaImAn HDF5 file using the file menu.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )])
            html = fig.to_html(include_plotlyjs='cdn')
            self.temporal_view.setHtml(html)
            self.component_type.setEnabled(False)
            self.good_toggle_button.setEnabled(False)
            self.bad_toggle_button.setEnabled(False)
            return
        
        index = self.selected_component
        
        
        #array_names = ["C", "f", "YrA", "F_dff", "R", "S", "noisyC", "C_on"]
        array_text=self.array_selector.currentText()
        length=self.cnm.estimates.C.shape[-1] #always present
        y=getattr(self.cnm.estimates, array_text)[index, :]
        if len(y) > length:
            y = y[-length:] # in case of noisyC or C_on
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y = y,
            mode = 'lines',
            line = dict(color='blue'),
            name = f'Temporal Component {array_text} {index}'
        ))
        fig.update_layout(
            yaxis=dict(range=[0, None]),
            margin=dict(l=1, r=1, t=10, b=1)
        )
        html = fig.to_html(include_plotlyjs='cdn')
        self.temporal_view.setHtml(html)    

        if not self.cnm.estimates.r_values is None:
            r = self.cnm.estimates.r_values[index]
            max_r = np.max(self.cnm.estimates.r_values)
            min_r = np.min(self.cnm.estimates.r_values)
            color = f"rgb({int(255*(1-(r-min_r)/(max_r-min_r)))}, {int(255*(r-min_r)/(max_r-min_r))}, 0)"
            self.component_params_r.setText(f"    Rval: {np.format_float_positional(r, precision=2)}")
            self.component_params_r.setToolTip(f"cnm.estimates.r_values[{index}]")
            self.component_params_r.setStyleSheet(f"color: {color}")
            
            self.component_params_SNR.setText(f"    SNR: {np.format_float_positional(self.cnm.estimates.SNR_comp[index], precision=2)}")
            self.component_params_SNR.setToolTip(f"cnm.estimates.SNR_comp[{index}]")
            self.component_params_CNN.setText(f"    CNN: {np.format_float_positional(self.cnm.estimates.cnn_preds[index], precision=2)}")
            self.component_params_CNN.setToolTip(f"cnm.estimates.cnn_preds[{index}]")
        else:
            self.component_params_r.setText("    R: --")
            self.component_params_r.setToolTip("use evaluate components to compute r values")
            self.component_params_r.setStyleSheet("color: black")
            self.component_params_SNR.setText("    SNR: --")
            self.component_params_SNR.setToolTip("use evaluate components to compute SNR")
            self.component_params_SNR.setStyleSheet("color: black")
            self.component_params_CNN.setText("    CNN: --")
            self.component_params_CNN.setToolTip("use evaluate components to compute CNN predictions")
            self.component_params_CNN.setStyleSheet("color: black")
            
        if not self.cnm.estimates.idx_components is None:
            self.component_type.setEnabled(True)
            if index in self.cnm.estimates.idx_components:
                self.good_toggle_button.setChecked(True)
                self.bad_toggle_button.setChecked(False)                
            else:
                self.good_toggle_button.setChecked(False)
                self.bad_toggle_button.setChecked(True)
            self.good_toggle_button.setEnabled(True)
            self.bad_toggle_button.setEnabled(True)
            self.good_label.setText(f"    Good: {len(self.cnm.estimates.idx_components)}")
            self.good_label.setEnabled(True)
            self.bad_label.setText(f"    Bad: {len(self.cnm.estimates.idx_components_bad)}")
            self.bad_label.setEnabled(True)  
        else:
            self.good_toggle_button.setChecked(False)
            self.good_toggle_button.setEnabled(False)
            self.bad_toggle_button.setChecked(False)
            self.bad_toggle_button.setEnabled(False)
            self.component_type.setEnabled(False)
            self.component_type.setCurrentText('All')
            self.good_label.setText("    Good: --")
            self.good_label.setEnabled(False)
            self.bad_label.setText("    Bad: --")
            self.bad_label.setEnabled(False)   
            
    def plot_spatial(self):
        if self.cnm is None:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(
                text="No data loaded yet",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )])
            html = fig.to_html(include_plotlyjs='cdn')
            self.spatial_view.setHtml(html)
            return

        index =0
        
        vec = self.cnm.estimates.b[:, index]
        print("Component", index, "vector length:", len(vec), "dims produ")
        print("Has NaNs:", np.isnan(vec).any())
        print("Has infs:", np.isinf(vec).any())
        print("Min:", np.nanmin(vec), "Max:", np.nanmax(vec))
        vec=vec.reshape(self.cnm.dims)
        vec=vec*1000
        print(vec.shape)
        x = np.linspace(0, 4*np.pi, 512)
        y = np.linspace(0, 2*np.pi, 512)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.sin(Y)  # Calculating the wave
        #Z=vec
        # Create a heatmap trace using Plotly
        fig = go.Figure(data=go.Heatmap(z=Z, colorscale='viridis'))
        fig.update_layout(
            title=f'Spatial Component {index}',
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Generate HTML and set it in the QWebEngineView widget.
        html = fig.to_html(include_plotlyjs='cdn', config={"responsive": True})
        self.spatial_view.setHtml(html)
        
                
    def plot_parameters(self):
        if self.cnm is None or self.cnm.estimates.r_values is None:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(
                text="No evaluated components.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )])
            html = fig.to_html(include_plotlyjs='cdn')
            self.parameters_view.setHtml(html)
            return
        
        num_components = self.cnm.estimates.r_values.shape[0]
        fig = go.Figure()
        # Main scatter trace (trace 0)
        if self.cnm.estimates.idx_components is not None:
            color=np.zeros(num_components)
            color[self.cnm.estimates.idx_components] = 1
            colorscale = ['red', 'green']  
        else:
            color = np.arange(num_components)
            colorscale = 'viridis'
        
        x = self.cnm.estimates.cnn_preds
        y = self.cnm.estimates.r_values
        z = self.cnm.estimates.SNR_comp
        fig.add_trace(go.Scatter3d(
            x = x,
            y = y,
            z = z,
            mode = 'markers',
            marker = dict(
                size = 3,
                color = color,
                colorscale = colorscale,
                opacity = 0.5
            ),
            text = [str(i) for i in range(num_components)],
            hovertemplate = "<br>".join([
                "Index: %{text}",
                "CNN prediction: %{x:.2f}",
                "R value: %{y:.2f}",
                "SNR: %{z:.2f}"
            ]) + "<extra></extra>"
        ))
        
        # Selected point trace (trace 1)
        selected_index = self.selected_component if self.selected_component is not None else 0
        fig.add_trace(go.Scatter3d(
            x = [self.cnm.estimates.cnn_preds[selected_index]],
            y = [self.cnm.estimates.r_values[selected_index]],
            z = [self.cnm.estimates.SNR_comp[selected_index]],
            mode = 'markers',
            marker = dict(
                size = 6,
                color = 'magenta',
                opacity = 0.7
            ),
            name = 'Selected Point',
            hoverinfo = 'skip'
        ))
        
        # Add gridlines for the selected point                  
        if self.cnm.estimates.idx_components is not None:
            lines=self.construct_threshold_gridline_data()
            for name, line in lines.items():
                fig.add_trace(go.Scatter3d(
                        x = line['x'],
                        y = line['y'],
                        z = line['z'],
                        mode = 'lines',
                        line = dict(color = line['color'], width = 1, dash = 'dash'),
                        name = name
                ))
        
        fig.update_layout(scene=dict(
            xaxis_title='CNN prediction',
            yaxis_title='R value',
            zaxis=dict(type='log'),
            zaxis_title='SNR'
        ))
        fig.update_layout(hovermode='closest', hoverdistance=22, margin=dict(l=1, r=1, t=10, b=1), showlegend=False)
  
        html = fig.to_html(include_plotlyjs='cdn')
        html += """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        if (!window._qtcInitialized) {
            window._qtcInitialized = true;
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.bridge = channel.objects.pythonBridge;
                var plot = document.getElementsByClassName('plotly-graph-div')[0];
                plot.on('plotly_click', function(data) {
                    if (data.points.length > 0) {
                        var pointIndex = data.points[0].text;
                        bridge.pointClicked(pointIndex);
                    }
                });
                // Global function to update the selected trace using direct coordinates
                window.updateSelectedTrace = function(newX, newY, newZ, rgb) {
                    console.log("JS: Updating selected trace with coordinates: " + newX + ", " + newY + ", " + newZ + " and color: " + rgb);
                    Plotly.restyle(plot, {
                        x: [[newX]],
                        y: [[newY]],
                        z: [[newZ]],
                        marker: { color: rgb } 
                    }, [1]);
                };
                // ---  updateThresholdLines global function ---
                window.updateThresholdLines = function(linesJson) {
                    var linesObj = JSON.parse(linesJson);
                    console.log("JS: Updating threshold lines", linesObj);
                    // Assume threshold line traces start at index 2
                    var traceIndex = 2;
                    for (var key in linesObj) {
                        var line = linesObj[key];
                        Plotly.restyle(plot, {
                            x: [[].concat(line.x)],
                            y: [[].concat(line.y)],
                            z: [[].concat(line.z)],
                            line: { color: line.color, width: 1, dash: 'dash' }
                        }, [traceIndex]);
                        traceIndex++;
                    }
                };
            });
        }
        </script>
        """
        self.parameters_view.setHtml(html)
        
    def update_title(self):
        if self.hdf5_file is None:
            self.setWindowTitle("Pluvianus: CaImAn result browser")
            self.save_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
        else:
            filestr = str(self.hdf5_file)
            wchar = int(round((self.width() - 100) / 9))
            if len(filestr) > (wchar + 3):
                filestr = "..." + filestr[-wchar:]
            if self.file_changed:
                self.save_action.setEnabled(True)
                filestr = filestr + " *"
            else:
                self.save_action.setEnabled(False)
            self.save_as_action.setEnabled(True)
            self.setWindowTitle("Pluvianus - " + filestr)
            
    def save_state(self):
        filename = "pluvianus_state.json"
        state = {"figure_size": (self.width(), self.height()), "path": self.hdf5_path}
        with open(filename, 'w') as f:
            json.dump(state, f)
        
    def load_state(self):
        filename = "pluvianus_state.json"
        if not os.path.exists(filename):
            return
        with open(filename, 'r') as f:
            state = json.load(f)
            self.resize(state["figure_size"][0], state["figure_size"][1])
            if state["path"] is not None and os.path.exists(state["path"]):
                self.hdf5_path = state["path"]

class PythonBridge(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    @Slot(str)
    def pointClicked(self, index):
        parent = self.parent()
        if parent and index != '':
            parent.on_scatter_point_clicked(int(index))
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
