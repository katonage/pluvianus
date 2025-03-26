import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QSplitter,QScrollArea, QCheckBox,QSlider,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QLabel, QComboBox, QPushButton, QProgressDialog, QSizePolicy
)
from PySide6.QtGui import QAction, QColor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QObject, Signal, Slot, Qt
from PySide6.QtWebChannel import QWebChannel
import pyqtgraph as pg

import plotly.graph_objects as go
import json
import os
import tempfile
import numpy as np
import glob

import pynapple as nap
from scipy.signal.windows import gaussian

import inspect
import time

import caiman as cm # type: ignore
from caiman.source_extraction import cnmf # type: ignore
from caiman.utils.visualization import get_contours as caiman_get_contours # type: ignore

class OptsWindow(QMainWindow):
    def __init__(self, opts, title='Options'):
        def custom_pretty_print(d, indent=0, spacing=1):
            """
            Recursively pretty-print nested dictionaries with extra spacing.
            
            Parameters:
            d (dict): dictionary to print
            indent (int): current indentation level
            spacing (int): number of extra empty lines between levels
            """
            stri=''
            for key, value in d.items():
                stri+='    ' * indent + str(key) + ':'
                if isinstance(value, dict):
                    stri+='\n' * spacing
                    stri+=custom_pretty_print(value, indent + 1, spacing)
                    stri+='\n'
                else:
                    stri+=' ' + str(value) +'\n'
            #print(stri)
            return stri
            
        super().__init__()
        self.setWindowTitle(title)
        self.textedit = QTextEdit()
        self.textedit.setReadOnly(True)
        self.setCentralWidget(self.textedit)
        if isinstance(opts, dict):
            stris=custom_pretty_print(opts)
        else:
            stris=repr(opts)
        self.textedit.setText(stris)
        self.resize(500, 800)
        

class ShiftsWindow(QMainWindow):
    def __init__(self, shifts):
        super().__init__()
        self.setWindowTitle('Shifts')
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.temporal_widget = pg.PlotWidget()
        self.temporal_widget.setDefaultPadding( 0.0 )
        self.temporal_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.temporal_widget.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.temporal_widget.getPlotItem().setContentsMargins(0, 0, 10, 0)  # add margin to the right
        self.temporal_widget.getPlotItem().setTitle(f'Motion correction shifts per frame')
        self.temporal_widget.setLabel('bottom', 'Frame Number')
        self.temporal_widget.setLabel('left', 'Shift (pixels)')
        self.temporal_widget.plot(x=np.arange(shifts.shape[0]), y=shifts[:, 0], pen=pg.mkPen(color='b', width=2), name='x shifts')
        self.temporal_widget.plot(x=np.arange(shifts.shape[0]), y=shifts[:, 1], pen=pg.mkPen(color='r', width=2), name='y shifts')
        
        layout.addWidget(self.temporal_widget)
        self.resize(700, 500)
        
class BackgroundWindow(QMainWindow):
    def __init__(self, b, f, dims):
        super().__init__()
        self.setWindowTitle('Background Components')
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # top row
        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignLeft)
        label = QLabel('Component: ')
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
        #bottom_row = QHBoxLayout()
        bottom_row = QSplitter( childrenCollapsible=False)
        bottom_row.setStyleSheet('QSplitter::handle { background-color: lightgray; }')
        
        self.temporal_widget = pg.PlotWidget()
        self.temporal_widget.setDefaultPadding( 0.0 )
        self.temporal_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.temporal_widget.getPlotItem().setMenuEnabled(False)
        self.temporal_widget.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.temporal_widget.getPlotItem().setContentsMargins(0, 0, 10, 0)  # add margin to the right
        self.temporal_widget.getPlotItem().setLabel('bottom', 'Frame Number')
        self.temporal_widget.getPlotItem().setLabel('left', 'Fluorescence')
        bottom_row.addWidget(self.temporal_widget)

        self.spatial_widget = pg.PlotWidget()
        #p1 = self.spatial_widget.addPlot(title='interactive')
        # Basic steps to create a false color image with color bar:
        self.spatial_image = pg.ImageItem()
        self.spatial_widget.addItem( self.spatial_image )
        self.colorbar_item=self.spatial_widget.getPlotItem().addColorBar( self.spatial_image, colorMap='viridis', rounding=0.00000000001) # , interactive=False)
        self.spatial_widget.setAspectLocked(True)
        self.spatial_widget.getPlotItem().showAxes(True, showValues=(True,False,False,True) )
        for side in ( 'top', 'right'):
            ax = self.spatial_widget.getPlotItem().getAxis(side)
            ax.setStyle(tickLength=0) 
        for side in ('left', 'bottom'):
            ax = self.spatial_widget.getPlotItem().getAxis(side)
            ax.setStyle(tickLength=10)         
        self.spatial_widget.getPlotItem().setMenuEnabled(False)
        self.spatial_widget.setDefaultPadding( 0.0 )
        bottom_row.addWidget(self.spatial_widget)
        
        layout.addWidget(bottom_row)
        self.update_plot(0)        
        self.resize(1200, 600)
    
    def update_plot(self, value):
        component_idx = self.spin_box.value()
        # Update image data
        img_data = self.b[:, component_idx].reshape(self.dims)
        self.spatial_image.setImage(img_data, autoLevels=False)
        self.spatial_widget.getPlotItem().setTitle(f'Spatial component {component_idx}')
        # Update colorbar limits explicitly
        min_val, max_val = np.min(img_data), np.max(img_data)
        #self.spatial_image.setLevels([min_val, max_val])
        self.colorbar_item.setLevels(values=[min_val, max_val])
        # Update temporal plot (if needed)
        temporal_data = self.f[component_idx, :]
        self.temporal_widget.clear()
        self.temporal_widget.plot(temporal_data, pen='b')
        self.temporal_widget.getPlotItem().setTitle(f'Temporal component {component_idx}')
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1000, 700)
        pg.setConfigOptions(background='w', foreground='k')
        
        # Setup file menu with Open, Save, Save As
        file_menu = self.menuBar().addMenu('File')
        open_action = QAction('Open CaImAn HDF5 File...', self)
        open_action.setShortcut('Ctrl+O')

        self.save_action = QAction('Save', self)
        self.save_action.setShortcut('Ctrl+S')
        self.save_as_action = QAction('Save As...', self)
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        self.open_data_action = QAction('Open Data Array...', self)
        self.open_data_action.setShortcut('Ctrl+D')
        file_menu.addAction(self.open_data_action)
        file_menu.addSeparator()
        self.open_cn_image_action = QAction('Open Local Correlation Image...', self)
        file_menu.addAction(self.open_cn_image_action)
        self.open_mean_image_action = QAction('Open Mean Image...', self)
        file_menu.addAction(self.open_mean_image_action)
        self.open_max_image_action = QAction('Open Max Image...', self)
        file_menu.addAction(self.open_max_image_action)
        self.open_std_image_action = QAction('Open Std Image...', self)
        file_menu.addAction(self.open_std_image_action)
        self.save_cn_image_action = QAction('Save Local Correlation Image...', self)
        file_menu.addAction(self.save_cn_image_action)
        self.save_mean_image_action = QAction('Save Mean Image...', self)
        file_menu.addAction(self.save_mean_image_action)
        self.save_max_image_action = QAction('Save Max Image...', self)
        file_menu.addAction(self.save_max_image_action)
        self.save_std_image_action = QAction('Save Std Image...', self)
        file_menu.addAction(self.save_std_image_action)
        
        comp_menu = self.menuBar().addMenu('Compute')
        self.detr_action = QAction('Detrend df/f', self)
        comp_menu.addAction(self.detr_action)
        self.compute_component_evaluation_action = QAction('Compute Component Metrics', self)
        comp_menu.addAction(self.compute_component_evaluation_action)
        self.compute_projections_action = QAction('Compute Projections (heavy)', self)
        comp_menu.addAction(self.compute_projections_action)
        self.compute_cn_action = QAction('Compute Local Correlation Image (heavy)', self)
        comp_menu.addAction(self.compute_cn_action)
        self.compute_origtrace_action = QAction('Compute Original Fluorescence Traces', self)
        comp_menu.addAction(self.compute_origtrace_action)
        
        view_menu = self.menuBar().addMenu('View')
        self.info_action = QAction('Info', self)
        view_menu.addAction(self.info_action)
        self.opts_action = QAction('CaImAn Parameters', self)
        view_menu.addAction(self.opts_action)
        self.bg_action = QAction('Background Components', self)
        view_menu.addAction(self.bg_action)
        self.shifts_action = QAction('Movement Correction Shifts', self)
        view_menu.addAction(self.shifts_action)
        
        exp_menu = self.menuBar().addMenu('Export')
        self.save_trace_action_c_g_n = QAction('C to Pynapple NPZ (Good)...', self)
        exp_menu.addAction(self.save_trace_action_c_g_n)
        self.save_trace_action_c_a_n = QAction('C to Pynapple NPZ (All)...', self)
        exp_menu.addAction(self.save_trace_action_c_a_n)
        self.save_trace_action_f_g_n = QAction('\u0394F/F to Pynapple NPZ (Good)...', self)
        exp_menu.addAction(self.save_trace_action_f_g_n)
        self.save_trace_action_f_a_n = QAction('\u0394F/F to Pynapple NPZ (All)...', self)
        exp_menu.addAction(self.save_trace_action_f_a_n)
        exp_menu.addSeparator()
        self.save_mescroi_action = QAction('Contours to MEScROI...', self)
        exp_menu.addAction(self.save_mescroi_action)
        
        help_menu = self.menuBar().addMenu('Help')
        about_action = QAction('About', self)
        license_action = QAction('License', self)
        source_action = QAction('Source', self)
        help_menu.addAction(about_action)
        help_menu.addAction(license_action)
        help_menu.addAction(source_action)
        
        open_action.triggered.connect(self.open_file)
        self.save_action.triggered.connect(self.save_file)
        self.save_as_action.triggered.connect(self.save_file_as)
        self.open_data_action.triggered.connect(self.open_data_file)
        self.open_cn_image_action.triggered.connect(lambda: self.open_image_file('cn'))
        self.open_mean_image_action.triggered.connect(lambda: self.open_image_file('mean'))
        self.open_max_image_action.triggered.connect(lambda: self.open_image_file('max'))
        self.open_std_image_action.triggered.connect(lambda: self.open_image_file('std'))
        self.save_cn_image_action.triggered.connect(lambda: self.save_image_file('cn'))
        self.save_mean_image_action.triggered.connect(lambda: self.save_image_file('mean'))
        self.save_max_image_action.triggered.connect(lambda: self.save_image_file('max'))
        self.save_std_image_action.triggered.connect(lambda: self.save_image_file('std'))
        self.save_trace_action_c_g_n.triggered.connect(lambda: self.save_trace('C', 'Good', 'npz'))
        self.save_trace_action_c_a_n.triggered.connect(lambda: self.save_trace('C', 'All', 'npz'))
        self.save_trace_action_f_g_n.triggered.connect(lambda: self.save_trace('F_dff', 'Good', 'npz'))
        self.save_trace_action_f_a_n.triggered.connect(lambda: self.save_trace('F_dff', 'All', 'npz'))
        self.save_mescroi_action.triggered.connect(self.save_MEScROI)
        
        self.detr_action.triggered.connect(self.on_detrend_action)
        self.compute_component_evaluation_action.triggered.connect(self.on_compute_evaluate_components_action)
        self.compute_projections_action.triggered.connect(self.on_compute_projections_action)
        self.compute_cn_action.triggered.connect(self.on_compute_cn_action)
        self.compute_origtrace_action.triggered.connect(self.on_compute_origtrace_action)
        self.opts_action.triggered.connect(self.on_opts_action)
        
        self.info_action.triggered.connect(self.on_info_action)
        self.shifts_action.triggered.connect(self.on_shifts_action)
        self.bg_action.triggered.connect(self.on_bg_action)
        self.resizeEvent = self.on_resize_figure
        self.closeEvent = self.on_mainwindow_closing
        
        # Create central widget and layout       
        main_layout = QSplitter(Qt.Vertical)
        self.setCentralWidget(main_layout)
        main_layout.setStyleSheet('QSplitter::handle { background-color: lightgray; }')
        
        self.temporal_widget = TopWidget(self, self)
        main_layout.addWidget(self.temporal_widget)
        
        bottom_layout_splitter = QSplitter( )
        bottom_layout_splitter.setStyleSheet('QSplitter::handle { background-color: lightgray; }')
        
        self.scatter_widget= ScatterWidget(self, self)
        bottom_layout_splitter.addWidget(self.scatter_widget)
        
        self.spatial_widget = SpatialWidget(self, self)
        bottom_layout_splitter.addWidget(self.spatial_widget)
        self.spatial_widget2 = SpatialWidget(self, self)
        bottom_layout_splitter.addWidget(self.spatial_widget2)
        
        main_layout.addWidget(bottom_layout_splitter)
        bottom_layout_splitter.setSizes([1, 2])
        
        # Initialize variables
        self.cnm = None #caiman object
        self.hdf5_file = None # flie name of caiman hdf5 file
        self.file_changed = False # flag for storing if file has changed
        self.online = False # flag for OnACID files
        self.selected_component = 0 # index of selected component
        self.selected_frame = 0 # index of selected frame
        self.num_frames = 0  # number of frames in movie
        self.frame_window = 0 # temporal window of displaying movie frames (half window, in frames)
        self.limit='All' # restriction on selecting components according to their good/bad assignment
        self.manual_acceptance_assigment_has_been_made = False # flag for storing if manual component assignment has been made
        self.data_file = '' # file name of data array file (mmap)
        self.data_array = None # data array if loaded
        self.mean_projection_array = None # mean projection array
        self.max_projection_array = None # max projection array
        self.std_projection_array = None # std projection array
        self.orig_trace_array = None # computed original fluorescence traces
        self.orig_trace_array_neuropil = None # computed original fluorescence traces' neuropil
        # correlation image is stored in the cnm object
        

        

         
        # Update figure and state
        self.load_state()
        self.update_all()
    
    def set_selected_component(self, value, method):
        '''
            The component number is checked against the type of components that are selected in the combo box.
        '''

        def nearest_in_dir(array, previous, new):
            direction = np.sign(new - previous)
            if direction == 0:
                idx = (np.abs(array - new)).argmin()
            elif direction == 1:
                idx = np.where(array > new)[0][0] if len(np.where(array > new)[0]) > 0 else -1
            else: # direction == -1
                idx = np.where(array < new)[0][-1] if len(np.where(array < new)[0]) > 0 else 0
            return array[idx]
        
        def valid_component_select(self, value):
            '''
            Valid component select. This function is called when the component number is changed from the spinbox.
            
            Parameters
            ----------
            value : int
                The new value of the component.

            Returns
            -------
            None, sets the self.selected_component to the new value.
            '''
            
            if self.cnm is None or self.cnm.estimates.idx_components is None:
                self.limit = 'All'
                return value
            
            cnme=self.cnm.estimates
            previous_value = self.selected_component
            if self.limit == 'All':
                return value
            elif self.limit == 'Good':                      
                if value in cnme.idx_components:
                    return value
                else:
                    return nearest_in_dir(cnme.idx_components, previous_value, value)
            elif self.limit == 'Bad':
                if value in cnme.idx_components_bad:
                    return value
                else:
                    return nearest_in_dir(cnme.idx_components_bad, previous_value, value)
            #never reach here
        if self.cnm is None:
            return
        #print('set_selected_component', value, method)
        cnme=self.cnm.estimates
        numcomps=cnme.A.shape[-1]
        if self.cnm is None:
            value = min(numcomps-1, value)
        if method == 'direct':
            if self.limit == 'All' or cnme.idx_components is None:
                self.selected_component = value
                self.limit = 'All'
            elif self.limit == 'Good':
                self.selected_component =  nearest_in_dir(cnme.idx_components, value, value)
            elif self.limit == 'Bad':
                self.selected_component =  nearest_in_dir(cnme.idx_components_bad, value, value)           
        elif method == 'scatter':
            if self.limit == 'All' or cnme.idx_components is None:
                self.selected_component = value
            else:
                xyz = np.array([cnme.cnn_preds, cnme.r_values, np.log(cnme.SNR_comp)]).T
                current=xyz[value]
                if self.limit == 'Good':
                    idxs=cnme.idx_components
                elif self.limit == 'Bad':
                    idxs=cnme.idx_components_bad
                xyz = xyz[idxs]
                dist = np.linalg.norm(xyz - current, axis=1)
                idx = np.argmin(dist)
                self.selected_component = idxs[idx]
        elif method == 'spinbox':
            self.selected_component = valid_component_select(self, value)
        elif method == 'spatial':
            if self.limit == 'All' or cnme.idx_components is None or cnme.coordinates is None:
                self.selected_component = value
            else:
                xyz = self.component_centers #is Nx2 shape
                current = xyz[value,:]
                if self.limit == 'Good':
                    idxs = cnme.idx_components
                elif self.limit == 'Bad':
                    idxs = cnme.idx_components_bad
                xyz = xyz[idxs,:]
                dist = np.linalg.norm(xyz - current, axis=1)
                idx = np.argmin(dist)
                self.selected_component = idxs[idx]
        else:
            raise ValueError(f'Invalid method: {method}')
        
        #update
        self.temporal_widget.update_component_spinbox(self.selected_component)
        self.scatter_widget.update_selected_component_on_scatterplot(self.selected_component)
        self.temporal_widget.update_temporal_view()
        self.spatial_widget.update_spatial_view()
        self.spatial_widget2.update_spatial_view()
    

    def set_component_assignment_manually(self, state, component=None):
        # Handle the toggle button click event
        print(f"The {state} button was toggled")
        if component is None:
            component=self.selected_component
        changed=False
        if self.cnm is not None and self.cnm.estimates.idx_components is not None:
            numcomps=self.cnm.estimates.A.shape[-1]
            if state == 'Good':
                if component not in self.cnm.estimates.idx_components:
                    self.cnm.estimates.idx_components = np.unique(np.append(self.cnm.estimates.idx_components, component))
                    self.cnm.estimates.idx_components_bad= np.array(np.setdiff1d(range(numcomps),self.cnm.estimates.idx_components))
                    changed=True
            elif state == 'Bad':
                if component not in self.cnm.estimates.idx_components_bad:
                    self.cnm.estimates.idx_components_bad = np.unique(np.append(self.cnm.estimates.idx_components_bad, component))
                    self.cnm.estimates.idx_components= np.array(np.setdiff1d(range(numcomps),self.cnm.estimates.idx_components_bad))
                    changed=True
        if changed:
            self.manual_acceptance_assigment_has_been_made=True
            self.file_changed=True
            self.update_title()
            self.plot_parameters()
            self.set_selected_component(component, 'direct')
            self.scatter_widget.update_totals()
            self.scatter_widget.update_selected_component_on_scatterplot(self.selected_component)
    
    def set_selected_frame(self, value, window=None):
        if self.cnm is None:
            return
        if value is not None:
            value = max(min(self.num_frames-1, value), 0)
            value=round(value)
            self.selected_frame=value
        if window is not None:
            self.frame_window=int(window)
            
        self.spatial_widget.update_spatial_view_image()
        self.spatial_widget2.update_spatial_view_image()        
        self.temporal_widget.update_temporal_widget()
        self.temporal_widget.update_time_selector_line()
        
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
        if hasattr(self, 'info_window') and self.info_window.isVisible():
            self.info_window.close()
            
    
 
    def on_threshold_spinbox_changed(self):
        if self.cnm.estimates.idx_components is None:
            return
        self.file_changed = True
        self.update_title()
            
        
    def on_detrend_action(self):
        '''
        Detrending using the `detrend_df_f` method of the CNMF object.
        After detrending is complete, the `file_changed` flag is set qand the estimates.f_dff array is filled.
        '''
        if self.cnm is None or self.cnm.estimates.F_dff is not None:
            return
        print('Detrending...')
        waitDlg = QProgressDialog('Detrending in progress...', None, 0, 0, self)
        waitDlg.setWindowModality(Qt.ApplicationModal)  # Blocks input to main window
        waitDlg.setCancelButton(None)  # No cancel button
        waitDlg.setWindowTitle('Please Wait')
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
            self.temporal_widget.update_array_selector(value='F_dff')
            self.update_all()
        finally:
            waitDlg.close()
            QApplication.restoreOverrideCursor()

    def on_opts_action(self):
        if self.cnm is None:
            return
        if hasattr(self, 'opts_window') and self.opts_window.isVisible():
            self.opts_window.raise_()
            self.opts_window.show()
        else:
            if self.online:
                title='Options (OnlineCNMF)'
            else:
                title='Options (CNMF)'     
            self.opts_window = OptsWindow(self.cnm.params, title=title)
            self.opts_window.show()

                
    def on_info_action(self):
        if self.cnm is None:
            return
        if hasattr(self, 'info_window') and self.info_window.isVisible():
            self.info_window.raise_()
            self.info_window.show()
        else:
            info={'Data information': {
                'Data dimensions': self.dims,
                'Number of frames': self.num_frames,
                'Frame rate (Hz)': self.framerate, 
                'Decay time (s)': self.decay_time, 
                'Pixel size (um)': self.pixel_size,
                'Neuron diameter (pix)': self.neuron_diam,
                'Number of components': self.cnm.estimates.A.shape[-1],
                'OnACID': self.online
                }, 
                'Paths': {
                'HDF5 path': self.hdf5_file, 
                'Data path': self.data_file
                }}
            self.info_window = OptsWindow(info, title='Info')
            self.info_window.show()
                
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
            shifts=shifts[-(self.num_frames):,:]
        self.shifts_window = ShiftsWindow(shifts)
        self.shifts_window.show()
    
    def on_bg_action(self):
        if self.cnm is None:
            return
        if hasattr(self, 'background_window') and self.background_window.isVisible():
            self.background_window.raise_()
            self.background_window.show()
            return
        print(self.dims)
        self.background_window = BackgroundWindow(self.cnm.estimates.b, self.cnm.estimates.f, self.dims)
        self.background_window.show()
        
    def open_file(self):        
        if self.hdf5_file is None:
            previ='.'
        elif os.path.exists(self.hdf5_file):
            previ=self.hdf5_file
        elif os.path.exists(os.path.dirname(self.hdf5_file)):
            previ=os.path.dirname(self.hdf5_file)
        else:
            previ='.'
        
        filename, _ = QFileDialog.getOpenFileName(self, 'Open CaImAn HDF5 File', previ, 'HDF5 Files (*.hdf5)')

        if not filename:
            return
        
        print('Open file:', filename)
        progress_dialog = QProgressDialog('Opening file', None, 0, 100, self)
        progress_dialog.setWindowTitle('Loading CaImAn file...')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(300)
        progress_dialog.show()
        QApplication.processEvents()
        try:
            self.cnm = cnmf.online_cnmf.load_OnlineCNMF(filename) 
            #check
            if self.cnm.params.online['movie_name_online'] == 'online_movie.mp4':
                raise Exception('Not an OnlineCNMF file')
            self.online = True
            print('File loaded (OnlineCNMF):', filename)
        except Exception as e:
            progress_dialog.setValue(12)
            progress_dialog.setLabelText('Opening CNMF file...')
            QApplication.processEvents()
            try:
                self.cnm = cnmf.cnmf.load_CNMF(filename)
                self.online = False
                print('File loaded (CNMF):', filename)
            except Exception as e:
                print('Could not load file')
                self.cnm=None
                QMessageBox.critical(self, 'Error opening file', 'File could not be opened: ' + filename)
                return
        self.hdf5_file = filename
        self.file_changed = False
        self.data_file = ''
        self.data_array = None
        self.mean_projection_array = None
        self.max_projection_array = None
        self.std_projection_array = None
        self.orig_trace_array = None
        self.orig_trace_array_neuropil = None
        # cn, std_projection?
        progress_dialog.setValue(25)
        progress_dialog.setLabelText('Processing data...')
        QApplication.processEvents()
        
        if self.online:
            self.dims=self.cnm.params.data['dims']
        else:         
            self.dims=self.cnm.dims
        self.dims=(self.dims[1], self.dims[0])
        self.num_frames=self.cnm.estimates.C.shape[-1]
        self.numcomps=self.cnm.estimates.A.shape[-1]
        print(f'Data frame dimensions: {self.dims} x {self.num_frames} frames')
        self.framerate=self.cnm.params.data['fr'] #Hz
        self.decay_time=self.cnm.params.data['decay_time'] #sec
        self.frame_window=int(round(self.decay_time*self.framerate))
        self.neuron_diam=np.mean(self.cnm.params.init['gSiz'])*2 #pixels
        self.pixel_size=self.cnm.params.data['dxy'] #um
        print(f'Frame rate: {self.framerate} Hz, decay time: {self.decay_time} sec, neuron diameter: {self.neuron_diam} pixels, pixel size: {self.pixel_size} um')
        
        #ensuring A is dense
        self.cnm.estimates.A = self.cnm.estimates.A.toarray()
        #ensuring contours are calculated
        if self.cnm.estimates.coordinates is None:
            thr=0.9
            print(f'Calculating component contours with threshold {thr}...')
            progress_dialog.setValue(30)
            progress_dialog.setLabelText('Calculating component contours...')
            QApplication.processEvents()
            self.cnm.estimates.coordinates=caiman_get_contours(self.cnm.estimates.A, self.dims, swap_dim=True, thr=thr)     
        self.component_contour_coords = [self.cnm.estimates.coordinates[idx]['coordinates'] for idx in range(self.numcomps)]
        self.component_centers = np.array([self.cnm.estimates.coordinates[idx]['CoM'] for idx in range(self.numcomps)])
        
        progress_dialog.setValue(50)
        progress_dialog.setLabelText(f'Rendering {self.numcomps} components...')
        QApplication.processEvents()
        self.save_state()
        self.update_all()
        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
        

                        
    def update_all(self):
        
        self.close_child_windows()
        if self.cnm is None:
            self.selected_component=0
        else:
            self.selected_component=min(self.selected_component, self.numcomps-1)
        
        self.temporal_widget.update_limit_component_type()
        self.temporal_widget.update_array_selector()
        self.spatial_widget.recreate_spatial_view()
        self.spatial_widget2.recreate_spatial_view()
        self.scatter_widget.update_treshold_spinboxes()
                  
        self.opts_action.setEnabled(self.cnm is not None)
        self.info_action.setEnabled(self.cnm is not None)
        self.open_data_action.setEnabled(self.cnm is not None)
        self.open_cn_image_action.setEnabled(self.cnm is not None)
        self.open_mean_image_action.setEnabled(self.cnm is not None)
        self.open_max_image_action.setEnabled(self.cnm is not None)
        self.open_std_image_action.setEnabled(self.cnm is not None)
        self.save_cn_image_action.setEnabled(self.cnm is not None and hasattr(self.cnm.estimates, 'Cn') and self.cnm.estimates.Cn is not None)
        self.save_mean_image_action.setEnabled(self.mean_projection_array is not None)
        self.save_max_image_action.setEnabled(self.max_projection_array is not None)
        self.save_std_image_action.setEnabled(self.std_projection_array is not None)       
        
        self.detr_action.setEnabled(self.cnm is not None and self.cnm.estimates.F_dff is None)
        self.compute_component_evaluation_action.setEnabled(self.data_array is not None)
        self.compute_projections_action.setEnabled(self.data_array is not None)
        self.compute_cn_action.setEnabled(self.data_array is not None)
        self.compute_origtrace_action.setEnabled(self.data_array is not None)
        self.save_trace_action_c_a_n.setEnabled(self.cnm is not None)
        self.save_trace_action_c_g_n.setEnabled(self.cnm is not None and self.cnm.estimates.idx_components is not None)
        self.save_trace_action_f_a_n.setEnabled(self.cnm is not None and self.cnm.estimates.F_dff is not None)
        self.save_trace_action_f_g_n.setEnabled(self.cnm is not None and self.cnm.estimates.idx_components is not None and self.cnm.estimates.F_dff is not None)
        self.save_mescroi_action.setEnabled(self.cnm is not None and self.cnm.estimates.idx_components is not None)
        self.bg_action.setEnabled(self.cnm is not None)
        self.shifts_action.setEnabled(self.cnm is not None and self.cnm.estimates.shifts is not None and len(self.cnm.estimates.shifts) > 0)
        self.temporal_widget.time_slider.setEnabled(self.cnm is not None)
        self.temporal_widget.time_spinbox.setEnabled(self.cnm is not None)
        self.temporal_widget.time_window_spinbox.setEnabled(self.cnm is not None)
        self.temporal_widget.rlavr_spinbox.setEnabled(self.cnm is not None)
        self.temporal_widget.time_slider.setRange(0, self.num_frames-1)
        self.temporal_widget.time_spinbox.setRange(0, self.num_frames-1)        
        
        if  self.cnm is None:
            self.temporal_widget.component_spinbox.setEnabled(False)
            self.temporal_widget.array_selector.setEnabled(False)
            self.update_title()
            self.temporal_widget.recreate_temporal_view()
            self.plot_parameters()
            self.scatter_widget.update_totals()
            self.update_title()
            return
         
        self.temporal_widget.update_component_spinbox(self.selected_component)
        self.scatter_widget.update_totals()
                
        self.update_title()
        self.temporal_widget.recreate_temporal_view()
        self.plot_parameters()
        self.set_selected_component(self.selected_component, 'direct')           

    def save_file(self):
        print('Save file')
        # Implement save logic here

    def save_file_as(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File As', '', 'All Files (*)')
        if filename:
            print('Save file as:', filename)
            # Implement save as logic here

    def open_data_file(self):
        # Logic to open a data file
        if self.cnm is None:
            return
        suggested_file = None
        for ext in ['.mmap', '.h5']:
            previ=os.path.dirname(self.hdf5_file)
            suggested_file = next((f for f in glob.glob(os.path.join(previ, '*' + ext)) if os.path.isfile(f)), None)
            if suggested_file is not None:
                break
        if suggested_file is None:
            suggested_file = os.path.dirname(self.hdf5_file)
        data_file, _  = QFileDialog.getOpenFileName(self, 'Open Movement Corrected Data Array File (.mmap, .h5)', suggested_file, 'Memory mapped files (*.mmap);;Movie file (*.h5);;All Files (*)')
        
        if not data_file:
            return
        
        progress_dialog = QProgressDialog('Opening data file...', None, 0, 100, self)
        progress_dialog.setWindowTitle('Loading Data Array File')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(300)
        progress_dialog.show()
        QApplication.processEvents()
        
        print(f'Loading mmap ({os.path.basename(data_file)})')
        Yr, dims, T = cm.load_memmap(data_file)
        if T != self.num_frames or dims[0] != self.dims[1] or dims[1] != self.dims[0]:
            progress_dialog.close()
            QMessageBox.critical(self, 'Error loading data', f'Incompatible data dimensions: expected {self.num_frames} frames x {self.dims[0]} x {self.dims[1]} pixels, but got {T} frames x {dims[0]} x {dims[1]} pixels.')
            print(f'Incompatible data dimensions: expected {self.num_frames} frames x {self.dims[0]} x {self.dims[1]} pixels, but got {T} frames x {dims[1]} x {dims[0]} pixels.')
            return
        self.data_array = Yr
        self.data_file = data_file
        
        progress_dialog.setValue(50)
        progress_dialog.setLabelText(f'Rendering windows...')
        self.spatial_widget.update_spatial_view(array_text='Data')
        self.update_all()
        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
    
    def look_for_image_file(self, type, save=False):
        if type == 'cn':
            file_signature = 'Cn.'
        else:
            file_signature = type+'_projection.'

        previ=os.path.dirname(self.hdf5_file)
        suggested_file = None
        for ext in ('npy', 'npz'):
            suggested_file = next((f for f in glob.glob(os.path.join(previ, '*'+file_signature+ext)) if os.path.isfile(f)), None)
            if suggested_file:
                break

        if suggested_file is None:
            suggested_file = os.path.dirname(self.hdf5_file)
            if save:
                suggested_file = os.path.join(os.path.dirname(self.hdf5_file), file_signature+'npy')
        return suggested_file
            
    def open_image_file(self, type):
        if self.cnm is None:
            return
        suggested_file=self.look_for_image_file(type)
        print(suggested_file)
        image_filep, _ = QFileDialog.getOpenFileName(self, 'Open file containing ' + type + ' image', suggested_file, 'NPY/NPZ files (*.npy *.npz);;All Files (*)')
        
        if not image_filep:
            return 
        image=np.load(image_filep)
        image=image.T
        if image.shape[0] != self.dims[0] or image.shape[1] != self.dims[1]:
            QMessageBox.critical(self, 'Error loading image', f'Incompatible image dimensions: expected {self.dims[0]} x {self.dims[1]} pixels, but got {image.shape[0]} x {image.shape[1]} pixels.')
            print(f'Incompatible image dimensions: expected {self.dims[0]} x {self.dims[1]} pixels, but got {image.shape[0]} x {image.shape[1]} pixels.')
            return  

        if type == 'cn':
            self.cnm.estimates.Cn = image
            self.file_changed = True
        else:
            setattr(self, type + '_projection_array', image)
        
        self.spatial_widget.update_spatial_view(array_text=type.capitalize())
        self.update_all()

    def save_image_file(self, ptype):
        if self.cnm is None:
            return
        
        if ptype == 'cn':
            image=self.cnm.estimates.Cn 
        else:
            image=getattr(self, ptype + '_projection_array')
        
        suggested_file=self.look_for_image_file(ptype, save=True)
        print(suggested_file)
        image_filep, _ = QFileDialog.getSaveFileName(self, 'Save ' + ptype + ' image', suggested_file, 'NPY files (*.npy);;')
        if not image_filep: #overwrite confirmation has been madde
            return
        np.save(str(image_filep), image.T)
        print( ptype.capitalize() + ' image saved to: ' + image_filep )
        
    def save_trace(self, trace, filtering, filetype):
        if self.cnm is None:
            return
        cnme=self.cnm.estimates
        data=getattr(cnme, trace)
        if data is None:
            raise Exception('No ' + trace + ' data available')
        if filtering == 'All':
            idx=range(self.numcomps)
        elif filtering == 'Good':
            idx=cnme.idx_components
            if idx is None:
                raise Exception('No component metrics available')
        else:
            raise Exception('Unknown filtering: ' + filtering)
        if len(idx) == 0:
            QMessageBox.critical(self, 'Error', 'No selected components available')
            return
        
        if filetype=='npz':
            suggestion=os.path.join(os.path.dirname(self.hdf5_file), trace+'_traces_'+filtering+ '.npz')
            data_filep, _ = QFileDialog.getSaveFileName(self, 'Save ' + filtering.lower() + ' ' + trace + ' traces', suggestion, 'NPZ files (*.npz);;')
            if not data_filep: #overwrite confirmation has been madde
                return
            data=data[idx, :]
            time=np.arange(data.shape[1])/self.framerate #sec
            component_centers = np.array([cnme.coordinates[idxe]['CoM'] for idxe in idx])
            metadict={
                'original_index': list(idx), 
                'X_pix': list(component_centers[:,0]),
                'Y_pix': list(component_centers[:,1])
                } 
            if cnme.idx_components is not None:
                metadict={
                    'original_index': list(idx), 
                    'X_pix': list(component_centers[:,0]),
                    'Y_pix': list(component_centers[:,1]),
                    'r_value': list(cnme.r_values[idx]),
                    'cnn_preds': list(cnme.cnn_preds[idx]), 
                    'SNR_comp': list(cnme.SNR_comp[idx]), 
                    'assignment': ['Good' if idx[i] in cnme.idx_components else 'Bad' for i in range(len(idx))]
                    } 
            output_data=nap.TsdFrame(t=time, d=data.T, metadata=metadict)
            output_data.save(str(data_filep))
            print(data_filep + ' saved.')
        else:
            raise Exception('Unknown filetype: ' + filetype)
         
    def save_MEScROI(self, filtering='Good'):
        cnme=self.cnm.estimates
        if filtering == 'All':
            idx=range(self.numcomps)
        elif filtering == 'Good':
            idx=cnme.idx_components
            if idx is None:
                raise Exception('No component metrics available')
        else:
            raise Exception('Unknown filtering: ' + filtering)
        if len(idx) == 0:
            QMessageBox.critical(self, 'Error', 'No selected components available')
            return
        
        suggestion=os.path.join(os.path.dirname(self.hdf5_file), 'selection_' + filtering + '.MEScROI')
        data_filep, _ = QFileDialog.getSaveFileName(self, 'Save ' + filtering.lower() + ' component contours', suggestion, 'MEScROI files (*.mescroi);;')
        if not data_filep: #overwrite confirmation has been made
            return
        data=data[idx, :]
        time=np.arange(data.shape[1])/self.framerate #sec
        self.component_contour_coords 
        
        #todo
        print('todo')
        #output_data.save(str(data_filep))
        #print(data_filep + ' saved.')
   
        
    def on_compute_evaluate_components_action(self):
        if self.data_array is None:
            return
        
        progress_dialog = QProgressDialog('Transposing data...', None, 0, 100, self)
        progress_dialog.setWindowTitle('Computing Component Metrics')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(300)
        progress_dialog.show()
        QApplication.processEvents()
        
        Yr=self.data_array 
        print('Evaluating components... Transposing data...')
        images = np.reshape(Yr.T, [self.num_frames] + [self.dims[1]] + [self.dims[0]], order='F')
        
        progress_dialog.setValue(10)
        progress_dialog.setLabelText(f'Evaluating components (may take a while)...')
        QApplication.processEvents()
        print('Evaluating components (estimates.evaluate_components)...')
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        self.cnm.estimates.evaluate_components(images, self.cnm.params, dview=dview)
        self.file_changed = True
        dview.terminate()
        print('Done evaluating components.')
        
        progress_dialog.setValue(90)
        progress_dialog.setLabelText(f'Rendering windows...')
        QApplication.processEvents()
        self.update_all()
        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
    
    def on_evaluate_button_clicked(self):
        if self.data_array is None:
            QMessageBox.information(self, 'Information', 'Open data array first to evaluate components')
            return
        if self.cnm.estimates.r_values is None or self.cnm.estimates.SNR_comp is None or self.cnm.estimates.cnn_preds is None:
            QMessageBox.information(self, 'Information', 'Component metrics are missing. Use Compute Component Metrics from the file menu.')
            return
        if self.manual_acceptance_assigment_has_been_made:
            reply = QMessageBox.question(self, 'Confirm', 'This operation will overwrite manual component assignment made. Do you want to continue?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        #CaImAn do the job:
        self.cnm.estimates.filter_components(imgs=self.data_array, params=self.cnm.params)
        self.manual_acceptance_assigment_has_been_made=False
        
        self.update_all()
        
    def on_compute_projections_action(self):
        if self.data_array is None:
            return

        print('Calculating projections... Transposing data...')
        progress_dialog = QProgressDialog('Transposing data...', None, 0, 100, self)
        progress_dialog.setWindowTitle('Computing Projection Images')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(300)
        progress_dialog.show()
        
        Yr=self.data_array 
        images = np.reshape(Yr.T, [self.num_frames] + [self.dims[1]] + [self.dims[0]], order='F')
        
        ii=0
        for proj_type in ["mean", "std", "max"]:
            progress_dialog.setValue(10+ii*30)
            ii+=1
            progress_dialog.setLabelText(f'Calculating {proj_type} projection image...')
            QApplication.processEvents()
            print(f'Calculating {proj_type} projection image...')
            p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
            p_img[np.isnan(p_img)] = 0
            p_img = p_img.T
            setattr(self, proj_type + '_projection_array', p_img)
        
        progress_dialog.setValue(90)
        progress_dialog.setLabelText(f'Rendering windows...')
        QApplication.processEvents()
        self.update_all()
        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
        
    def on_compute_cn_action(self):
        if self.data_array is None:
            return    
        from caiman.summary_images import local_correlations_movie_offline # type: ignore
        
        progress_dialog = QProgressDialog('Creating local correlation movie (may take a while)...', None, 0, 100, self)
        progress_dialog.setWindowTitle('Computing Correlation Image')
        progress_dialog.setModal(True)
        progress_dialog.setFixedWidth(300)
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.processEvents()
        
        print('Calculating correlation image (local_correlations_movie_offline)...')   
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        Cns = local_correlations_movie_offline(
                str(self.data_file),
                remove_baseline=True,
                window=1000,
                stride=1000,
                winSize_baseline=100,
                quantil_min_baseline=10,
                dview=dview,
            )
        dview.terminate()
        
        print('Movie created.')
        progress_dialog.setValue(80)
        progress_dialog.setLabelText(f'Projecting to correlation image...')
        QApplication.processEvents()
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        Cn = Cn.T
        self.cnm.estimates.Cn = Cn
        self.file_changed = True
        
        progress_dialog.setValue(90)
        progress_dialog.setLabelText(f'Rendering windows...')
        QApplication.processEvents()
        self.update_all()
        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
            
    def on_compute_origtrace_action(self):
        if self.data_array is None:
            return
        
        progress_dialog = QProgressDialog('Processing data', None, 0, 100, self)
        progress_dialog.setWindowTitle('Calculating fluorescence traces from data file...')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(400)
        progress_dialog.show()
        QApplication.processEvents()
        
        #masks for polygons (enery threshold)
        threshold=0.9
        print(f'Calculating fluorescence traces from data file, using component masks at threshold {threshold}...')
        A_masked = self.cnm.estimates.A >= 1
        for i in range(self.numcomps):
            vec=self.cnm.estimates.A[:,i]
            vec[np.isnan(vec)]=0
            vec_sorted = np.sort(vec)[::-1]
            cum_sum = np.cumsum(vec_sorted)
            weight=cum_sum[-1]  # equivalent to sum(vec)
            
            idx = np.searchsorted(cum_sum, threshold * weight)
            pixthresh=vec_sorted[min(idx, len(vec_sorted)-1)]
            A_masked[:,i]=vec>pixthresh
            if not i%100:
                #print(f'mask calculated. comp {i}, enery {pixthresh}, idx {idx}, {vec.shape}, {cum_sum.shape} ')
                progress_dialog.setValue(i/self.numcomps*20)
                progress_dialog.setLabelText(f'Calculating masks ({i}/{self.numcomps})...')
                QApplication.processEvents()
        
        # Prepare output
        output = np.zeros((self.numcomps, self.num_frames))

        # For each component, compute mean of pixels within mask over time
        for i in range(self.numcomps):
            masked_values = self.data_array[A_masked[:, i], :]         # shape: [masked_pixels, num_frames]
            vec = np.nanmean(masked_values, axis=0)  
            output[i, :]=vec
            if not i%100:
                #print(f'Number of nan elements in vec: {np.isnan(vec).sum()}  i: {i}')
                progress_dialog.setValue(i/self.numcomps*60+20)
                progress_dialog.setLabelText(f'Calculating traces ({i}/{self.numcomps})...')
                QApplication.processEvents()
        self.orig_trace_array=output
        print(f'Processed {self.numcomps} components.')

        progress_dialog.setValue(80)
        progress_dialog.setLabelText(f'Calculating neuropil fluorescence...')
        QApplication.processEvents()
                
        #neuropil
        print('Calculating neuropil trace: ', end='')
        neuropi_mask=A_masked[:,0]
        for i in range(1,self.numcomps):
            neuropi_mask=np.logical_or(neuropi_mask, A_masked[:,1])
        neuropi_mask = ~neuropi_mask
        
        mask_cout=np.count_nonzero(neuropi_mask)
        print('mask size: ', mask_cout, end='')
        if mask_cout>5200:
            true_idxs = np.flatnonzero(neuropi_mask)
            num_to_clear = mask_cout-5000
            to_clear = np.random.choice(true_idxs, size=num_to_clear, replace=False)
            neuropi_mask[to_clear] = False
            print(', downsampled mask size: ', np.count_nonzero(neuropi_mask))

        masked_values = self.data_array[neuropi_mask, :]
        vec = np.nanmean(masked_values, axis=0)
        self.orig_trace_array_neuropil=vec
        
        print(f'Neuropil trace calculated.')
        progress_dialog.setValue(90)
        progress_dialog.setLabelText(f'Rendering windows...')
        QApplication.processEvents()
        
        self.temporal_widget.update_array_selector('Data')
        self.update_all()
        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
        
    def plot_parameters(self):
        if self.cnm is None or self.cnm.estimates.r_values is None:
            fig = go.Figure()
            fig.update_layout(annotations=[dict(
                text='No evaluated components.\n Open data array to compute component metrics.',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False
            )])
            html = fig.to_html(include_plotlyjs='cdn')
            self.scatter_widget.parameters_view.setHtml(html)
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
            hovertemplate = '<br>'.join([
                'Index: %{text}',
                'CNN prediction: %{x:.2f}',
                'R value: %{y:.2f}',
                'SNR: %{z:.2f}'
            ]) + '<extra></extra>'
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
            lines=self.scatter_widget.construct_threshold_gridline_data()
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
        html += '''
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
        '''
        self.scatter_widget.parameters_view.setHtml(html)
        
    def perform_temporal_zoom(self):
        cnme=self.cnm.estimates
        component_index=self.selected_component
        vec=cnme.C[component_index, :]
        max_index = np.argmax(vec)
        
        zoomwindow=self.decay_time*self.framerate*10
        xrange=max_index-zoomwindow,max_index+zoomwindow
        self.temporal_widget.temporal_view.setRange(xRange=xrange, padding=0.0)    
        self.set_selected_frame(max_index)
     
    def update_title(self):
        if self.cnm is None:
            self.setWindowTitle('Pluvianus: CaImAn result browser')
            self.save_action.setEnabled(False)
            self.save_as_action.setEnabled(False)
        else:
            filestr = str(self.hdf5_file)
            wchar = int(round((self.width() - 100) / 9))
            if len(filestr) > (wchar + 3):
                filestr = '...' + filestr[-wchar:]
            if self.file_changed:
                self.save_action.setEnabled(True)
                filestr = filestr + ' *'
            else:
                self.save_action.setEnabled(False)
            self.save_as_action.setEnabled(True)
            self.setWindowTitle('Pluvianus - ' + filestr)
            
    def save_state(self):
        filename = 'pluvianus_state.json'
        filename = os.path.join(tempfile.gettempdir(), filename)
        state = {'figure_size': (self.width(), self.height()), 'path': self.hdf5_file}
        with open(filename, 'w') as f:
            json.dump(state, f)
        
    def load_state(self):
        filename = 'pluvianus_state.json'
        filename = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(filename):
            return
        with open(filename, 'r') as f:
            state = json.load(f)
            self.resize(state['figure_size'][0], state['figure_size'][1])
            if state['path'] is not None and os.path.exists(state['path']):
                self.hdf5_file = state['path']

class TopWidget(QWidget):
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.mainwindow = main_window
        
        my_layot=QHBoxLayout(self)
        
                # Top plot: Temporal (full width)
        self.temporal_view = pg.PlotWidget()
          
        # Create a container widget inside scroll area
        scroll_content = QWidget()
        left_layout = QVBoxLayout(scroll_content)

        #left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(0)
        
        head_label=QLabel('Component:')
        head_label.setStyleSheet('font-weight: bold;')
        left_layout.addWidget(head_label)
        
        self.component_spinbox = QSpinBox()
        self.component_spinbox.setMinimum(0)
        self.component_spinbox.setValue(0)
        self.component_spinbox.setToolTip('Select component')
        self.component_spinbox.setFixedWidth(90)
        left_layout.addWidget(self.component_spinbox)
        
        left_layout.addWidget(QLabel('Limit to:'))
        self.limit_component_type_combo = QComboBox()
        self.limit_component_type_combo.setFixedWidth(90)
        self.limit_component_type_combo.addItem('All')
        self.limit_component_type_combo.addItem('Good')
        self.limit_component_type_combo.addItem('Bad')
        self.limit_component_type_combo.setToolTip('Limit selection to component group')
        left_layout.addWidget(self.limit_component_type_combo)
        
        head_label=QLabel('Plot:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 10px;')
        left_layout.addWidget(head_label)
        self.array_selector = QComboBox()
        self.array_selector.setFixedWidth(90)
        self.array_selector.setToolTip('Select temporal array to plot')
        left_layout.addWidget(self.array_selector)
        
        self.rlavr_spinbox = QSpinBox()
        self.rlavr_spinbox.setMinimum(0)
        self.rlavr_spinbox.setMaximum(100)
        self.rlavr_spinbox.setValue(0)
        self.rlavr_spinbox.setToolTip('Sets running average Gauss kernel on the displayed data')
        self.rlavr_spinbox.setFixedWidth(90)
        self.rlavr_spinbox.setPrefix('Avr: ')
        left_layout.addWidget(self.rlavr_spinbox)
        
        self.temporal_zoom_button = QPushButton('Zoom')
        self.temporal_zoom_button.setToolTip('Centers view on largest peak on C, with zoom corresponding to decay time')
        left_layout.addWidget(self.temporal_zoom_button)
        self.temporal_zoom_auto_checkbox = QCheckBox('Auto')
        self.temporal_zoom_auto_checkbox.setToolTip('Centers view on largest peak on C, with zoom corresponding to decay time')
        left_layout.addWidget(self.temporal_zoom_auto_checkbox)
        head_label=QLabel('Metrics:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 10px;')
        left_layout.addWidget(head_label)
                
        self.component_params_r = QLabel('R: --')
        self.component_params_SNR = QLabel('SNR: --')
        self.component_params_CNN = QLabel('CNN: --')
        left_layout.addWidget(self.component_params_r)
        left_layout.addWidget(self.component_params_SNR)
        left_layout.addWidget(self.component_params_CNN)
        
        head_label=QLabel('Accept:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 10px;')
        left_layout.addWidget(head_label)

        # Create a layout for the toggle buttons
        toggle_button_layout = QHBoxLayout()
        toggle_button_layout.setSpacing(0)
        good_toggle_button = QPushButton('Good')
        good_toggle_button.setFixedWidth(45)
        good_toggle_button.setCheckable(True)
        good_toggle_button.setContentsMargins(0, 0, 0, 0)
        good_toggle_button.setToolTip('Accept component manually as good')
        #good_toggle_button.setStyleSheet('background-color: white; color: green;')
        toggle_button_layout.addWidget(good_toggle_button)
        self.good_toggle_button = good_toggle_button
        bad_toggle_button = QPushButton('Bad')
        bad_toggle_button.setFixedWidth(45)
        bad_toggle_button.setContentsMargins(0, 0, 0, 0)
        bad_toggle_button.setCheckable(True)
        bad_toggle_button.setStyleSheet('background-color: white; color: red;')
        bad_toggle_button.setToolTip('Reject component manually to bad')
        toggle_button_layout.addWidget(bad_toggle_button)
        self.bad_toggle_button = bad_toggle_button
        # Add the toggle button layout to the left layout
        left_layout.addLayout(toggle_button_layout)        
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Makes the scroll area resize with the window
        #scroll_area.setStyleSheet('background-color: red;')
        #scroll_content.setStyleSheet('background-color: yellow;')
        #head_label.setStyleSheet('background-color: green;')
        #self.temporal_zoom_auto_checkbox.setStyleSheet('background-color: green;')
        #head_label.setStyleSheet('background-color: blue;')
        left_layout.setContentsMargins(6, 1, 10, 1)
        my_layot.setContentsMargins(1 , 1, 6, 1)
    
        # Set scroll content as scroll area widget
        scroll_area.setWidget(scroll_content)
        my_layot.addWidget(scroll_area)
        
        right_layout=QVBoxLayout()
        left_layout.setContentsMargins(0, 1, 9, 9)
       
        right_layout.addWidget(self.temporal_view, stretch=1) 
        time_layout=QHBoxLayout()
        left_layout.setContentsMargins(0, 0,5,5)
       
        time_label=QLabel('Time:')
        time_layout.addWidget(time_label)
        self.time_spinbox = QSpinBox()
        self.time_spinbox.setPrefix('frame ')
        time_layout.addWidget(self.time_spinbox)
        self.time_slider = QSlider(Qt.Horizontal)
        time_layout.addWidget(self.time_slider)
        self.time_label=QLabel('--')
        time_layout.addWidget(self.time_label)
        self.time_window_spinbox = QSpinBox()
        self.time_window_spinbox.setMaximum(300)
        self.time_window_spinbox.setPrefix('±')
        self.time_window_spinbox.setFixedWidth(100)
        time_layout.addWidget(self.time_window_spinbox)
        
        # Set the initial value
        self.time_slider.valueChanged.connect(lambda value: self.on_time_widget_changed(value, 'slider'))
        self.time_spinbox.valueChanged.connect(lambda value: self.on_time_widget_changed(value, 'spinbox'))
        self.time_window_spinbox.valueChanged.connect(self.on_time_window_changed)
        
        right_layout.addLayout(time_layout)        
        my_layot.addLayout(right_layout, stretch=10)        
        
        self.setLayout(my_layot)
        
        #event hadlers
        self.component_spinbox.valueChanged.connect(self.on_component_spinbox_changed)
        self.rlavr_spinbox.valueChanged.connect(self.on_rlavr_spinbox_changed)
        self.limit_component_type_combo.currentTextChanged.connect(self.on_limit_component_type_changed)
        self.array_selector.currentTextChanged.connect(self.on_array_selector_changed)  
        self.temporal_zoom_button.clicked.connect(self.on_temporal_zoom)
        self.temporal_zoom_auto_checkbox.stateChanged.connect(self.on_temporal_zoom_auto_changed)
        self.good_toggle_button.clicked.connect(lambda: self.mainwindow.set_component_assignment_manually('Good'))
        self.bad_toggle_button.clicked.connect(lambda: self.mainwindow.set_component_assignment_manually('Bad'))
    
    def on_time_widget_changed(self, value, source):
        #print(f'{inspect.stack()[1][3]} called with value {value}{source}')
        self.mainwindow.set_selected_frame(value)
 
    def on_time_window_changed(self, value):
        self.mainwindow.set_selected_frame(None, window=value)
        
    def on_rlavr_spinbox_changed(self):
        self.update_temporal_view()
        
    def update_component_spinbox(self, value):
        self.component_spinbox.blockSignals(True)
        if self.mainwindow.cnm is None:
            self.component_spinbox.setEnabled(False)
        else:
            self.component_spinbox.setEnabled(True)
            self.component_spinbox.setMaximum(self.mainwindow.numcomps - 1)
        if self.component_spinbox.value() != value:
            self.component_spinbox.setValue(value)
        self.component_spinbox.blockSignals(False)
        
    def update_limit_component_type(self):
        value=self.mainwindow.limit
        cnm=self.mainwindow.cnm
        self.limit_component_type_combo.blockSignals(True)
        if cnm is None:
            self.limit_component_type_combo.setEnabled(False)
        elif cnm.estimates.idx_components is None:
            self.limit_component_type_combo.setEnabled(False)
            self.limit_component_type_combo.setCurrentText('All')
        else:
            self.limit_component_type_combo.setEnabled(True)
            self.limit_component_type_combo.setCurrentText(value)
        self.limit_component_type_combo.blockSignals(False)
        
    def on_component_spinbox_changed(self, value):
        self.mainwindow.set_selected_component(value, 'spinbox')

    def on_limit_component_type_changed(self, text):
        self.mainwindow.limit = text
        self.mainwindow.set_selected_component(self.mainwindow.selected_component, 'direct')
    
    def on_array_selector_changed(self, text):
        #[ 'F_dff', 'C',  'S', 'YrA', 'R', 'noisyC', 'C_on']
        tooltips={'C': 'Temporal traces', 
                  'F_dff': '\u0394F/F normalized activity trace', 
                  'S': 'Deconvolved neural activity trace', 
                  'YrA': 'Trace residuals', 
                  'R': 'Trace residuals', 
                  'noisyC': 'Temporal traces (including residuals plus background)', 
                  'C_on': '?', 
                  'Data': 'Original fluorescence trace calculated from contour polygons', 
                  'Data neuropil': 'Original fluorescence trace neuropil mean'}
        self.update_temporal_view()
        self.array_selector.setToolTip(tooltips[text])

    def on_temporal_zoom_auto_changed(self, state):
        if self.temporal_zoom_auto_checkbox.isChecked():
            self.mainwindow.perform_temporal_zoom()
    
    def on_temporal_zoom(self):
        self.mainwindow.perform_temporal_zoom()
    
   
  
    def update_array_selector(self, value=None):
        cnm=self.mainwindow.cnm
        if cnm is None:
            self.array_selector.setEnabled(False)
            return
        selectable_array_names=[]
        possible_array_names = [ 'F_dff', 'C',  'S', 'YrA', 'R', 'noisyC', 'C_on']
        for array_name in possible_array_names:
            temparr = getattr(cnm.estimates, array_name)
            if (temparr is not None) :
                selectable_array_names.append(array_name)
        if self.mainwindow.orig_trace_array is not None:
            selectable_array_names.append('Data')
        if self.mainwindow.orig_trace_array_neuropil is not None:
            selectable_array_names.append('Data neuropil')
        print('Selectable array names:', selectable_array_names)
        if value is None:
            previous_selected_array = self.array_selector.currentText()
        else:
            previous_selected_array = value
        self.array_selector.blockSignals(True)
        self.array_selector.clear()
        self.array_selector.addItems(selectable_array_names)
        if previous_selected_array in selectable_array_names:
            self.array_selector.setCurrentText(previous_selected_array)
        else:
            self.array_selector.setCurrentIndex(0)
        self.array_selector.setEnabled(True) 
        self.array_selector.blockSignals(False)
    
    def recreate_temporal_view(self):
        #.update_all esetén
        self.temporal_zoom_auto_checkbox.setEnabled(self.mainwindow.cnm is not None)
        self.temporal_zoom_button.setEnabled(self.mainwindow.cnm is not None)
        self.good_toggle_button.setEnabled(self.mainwindow.cnm is not None)
        self.bad_toggle_button.setEnabled(self.mainwindow.cnm is not None)
        #self.temporal_zoom_auto_checkbox.setChecked(False)
        
        self.temporal_view.clear()
        
        if self.mainwindow.cnm is None:
            text='No data loaded yet.\nOpen CaImAn HDF5 file using the file menu.'
            text = pg.TextItem(text=text, anchor=(0.5, 0.5), color='k')
            self.temporal_view.addItem(text)
            self.temporal_view.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
            self.temporal_view.getPlotItem().showGrid(False)
            self.temporal_view.getPlotItem().showAxes(False)
            self.temporal_view.setBackground(QColor(200, 200, 210, 127))
            return
        
        cnme=self.mainwindow.cnm.estimates
        
        self.temporal_view.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        self.temporal_view.setBackground(None)
        self.temporal_view.setDefaultPadding( 0.0 )
        self.temporal_view.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.temporal_view.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.temporal_view.getPlotItem().setContentsMargins(0, 0, 10, 0)  # add margin to the right
  
        if not cnme.idx_components is None:
            self.good_toggle_button.setEnabled(False)
            self.bad_toggle_button.setEnabled(False)

        self.mainwindow.scatter_widget.update_totals()
        
        self.temporal_line1=self.temporal_view.plot(x=[0,1 ], y=[0,3], pen=pg.mkPen('r', width=1), name='component')
        
        self.temporal_marker_P = pg.InfiniteLine(pos=2, angle=90, movable=False, pen=pg.mkPen('darkgrey', width=2))
        self.temporal_view.addItem(self.temporal_marker_P)
        self.temporal_marker_N = pg.InfiniteLine(pos=2, angle=90, movable=False, pen=pg.mkPen('darkgrey', width=2))
        self.temporal_view.addItem(self.temporal_marker_N)
        self.temporal_marker = pg.InfiniteLine(pos=2, angle=90, movable=True, pen=pg.mkPen('r', width=2), hoverPen=pg.mkPen('m', width=4))
        self.temporal_view.addItem(self.temporal_marker)
        self.temporal_marker.sigPositionChangeFinished.connect(lambda line=self.temporal_marker: line.setPen(pg.mkPen('r', width=2)))
        self.temporal_marker.sigDragged.connect(self.on_temporal_marker_dragged)
        self.update_temporal_view()

    def update_temporal_view(self):
        
        index = self.mainwindow.selected_component
        array_text=self.array_selector.currentText()
        cnme=self.mainwindow.cnm.estimates
        
        if array_text == 'C':
            ctitle=f'Temporal Component ({index})'
        elif array_text == 'F_dff':
            ctitle=f'\u0394F/F ({index})'
        elif array_text == 'YrA':
            ctitle=f'Residual ({index})'
        elif array_text == 'S':
            ctitle=f'Spike count estimate ({index})'
        elif array_text== 'Data':
            ctitle=f'Original fluorescence trace ({index})'
        elif array_text== 'Data neuropil':
            ctitle=f'Original fluorescence trace neuropil mean'
        else:
            ctitle=f'Temporal Component ({array_text}, {index})'
        
        self.temporal_view.getPlotItem().setTitle(ctitle)
        self.temporal_view.setLabel('bottom', 'Frame Number')
        self.temporal_view.setLabel('left', f'{array_text} value')
        
        #array_names = ['C', 'f', 'YrA', 'F_dff', 'R', 'S', 'noisyC', 'C_on']
        if array_text == 'Data':
            y=self.mainwindow.orig_trace_array[index, :]
        elif array_text== 'Data neuropil':
            y=self.mainwindow.orig_trace_array_neuropil
        else:
            y=getattr(cnme, array_text)[index, :]
            if len(y) > self.mainwindow.num_frames:
                y = y[-self.mainwindow.num_frames:] # in case of noisyC or C_on   
        
        w=self.rlavr_spinbox.value()
        if w>0:
            kernel = gaussian(2*w+1, w)
            kernel = kernel / np.sum(kernel)
            y = np.convolve(y, kernel, mode='same')
        self.temporal_line1.setData(x=np.arange(len(y)), y=y, pen=pg.mkPen(color='b', width=2), name=f'data {array_text} {index}')
        
        if not cnme.r_values is None:
            r = cnme.r_values[index]
            max_r = np.max(cnme.r_values)
            min_r = np.min(cnme.r_values)
            color = f'rgb({int(255*(1-(r-min_r)/(max_r-min_r)))}, {int(255*(r-min_r)/(max_r-min_r))}, 0)'
            self.component_params_r.setText(f'    Rval: {np.format_float_positional(r, precision=2)}')
            self.component_params_r.setToolTip(f'cnm.estimates.r_values[{index}]')
            self.component_params_r.setStyleSheet(f'color: {color}')
            
            max_SNR = np.max(cnme.SNR_comp)
            min_SNR = np.min(cnme.SNR_comp)
            color = f'rgb({int(255*(1-(cnme.SNR_comp[index]-min_SNR)/(max_SNR-min_SNR)))}, {int(255*(cnme.SNR_comp[index]-min_SNR)/(max_SNR-min_SNR))}, 0)'
            self.component_params_SNR.setText(f'    SNR: {np.format_float_positional(cnme.SNR_comp[index], precision=2)}')
            self.component_params_SNR.setToolTip(f'cnm.estimates.SNR_comp[{index}]')
            self.component_params_SNR.setStyleSheet(f'color: {color}')
            
            max_cnn = np.max(cnme.cnn_preds)
            min_cnn = np.min(cnme.cnn_preds)
            color = f'rgb({int(255*(1-(cnme.cnn_preds[index]-min_cnn)/(max_cnn-min_cnn)))}, {int(255*(cnme.cnn_preds[index]-min_cnn)/(max_cnn-min_cnn))}, 0)'
            self.component_params_CNN.setText(f'    CNN: {np.format_float_positional(cnme.cnn_preds[index], precision=2)}')
            self.component_params_CNN.setToolTip(f'cnm.estimates.cnn_preds[{index}]')
            self.component_params_CNN.setStyleSheet(f'color: {color}')
        else:
            self.component_params_r.setText('    R: --')
            self.component_params_r.setToolTip('use evaluate components to compute r values')
            self.component_params_r.setStyleSheet('color: black')
            self.component_params_SNR.setText('    SNR: --')
            self.component_params_SNR.setToolTip('use evaluate components to compute SNR')
            self.component_params_SNR.setStyleSheet('color: black')
            self.component_params_CNN.setText('    CNN: --')
            self.component_params_CNN.setToolTip('use evaluate components to compute CNN predictions')
            self.component_params_CNN.setStyleSheet('color: black')
            
        if not cnme.idx_components is None:
            if index in cnme.idx_components:
                self.good_toggle_button.setChecked(True)
                self.bad_toggle_button.setChecked(False)                
            else:
                self.good_toggle_button.setChecked(False)
                self.bad_toggle_button.setChecked(True)
            self.good_toggle_button.setEnabled(True)
            self.bad_toggle_button.setEnabled(True)
        else:
            self.good_toggle_button.setChecked(False)
            self.good_toggle_button.setEnabled(False)
            self.bad_toggle_button.setChecked(False)
            self.bad_toggle_button.setEnabled(False)  
        
        if self.temporal_zoom_auto_checkbox.isChecked():
            self.mainwindow.perform_temporal_zoom()
        else:
            self.update_time_selector_line()
            self.update_temporal_widget()
        
    def on_temporal_marker_dragged(self, line):
        line.setPen(pg.mkPen('m', width=4))
        self.mainwindow.set_selected_frame(int(line.value()))
     
    def update_temporal_widget(self):
        value=self.mainwindow.selected_frame
        w=self.mainwindow.frame_window
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(value)
        self.time_slider.blockSignals(False)
        self.time_spinbox.blockSignals(True)
        self.time_spinbox.setValue(value)
        self.time_spinbox.blockSignals(False)
        self.time_window_spinbox.blockSignals(True)
        self.time_window_spinbox.setValue(self.mainwindow.frame_window)
        self.time_window_spinbox.setSuffix(' frames' if self.mainwindow.frame_window > 1 else ' frame')
        self.time_window_spinbox.blockSignals(False)
        
        strin=f'{value/self.mainwindow.framerate:.3f} s'
        if w>0:
            strin=strin+f' ±{w/self.mainwindow.framerate:.3f} s'
        self.time_label.setText(strin)
        
    def update_time_selector_line(self):
        value=self.mainwindow.selected_frame
        w=self.mainwindow.frame_window
        tmin=value-w if value-w>0 else 0
        tmax=value+w if value+w<self.mainwindow.num_frames-1 else self.mainwindow.num_frames-1
        self.temporal_marker.setValue(value)
        self.temporal_marker_N.setValue(tmin)
        self.temporal_marker_P.setValue(tmax)
        

class ScatterWidget(QWidget):
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.mainwindow = main_window
        
        my_layout = QHBoxLayout(self)
        # Bottom layout for Spatial and Parameters plots
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(0)
        
        threshold_layout.setAlignment(Qt.AlignTop)
        head_label=QLabel('Components:')
        head_label.setStyleSheet('font-weight: bold;')
        threshold_layout.addWidget(head_label)
        self.total_label = QLabel('Total: --')
        threshold_layout.addWidget(self.total_label)
        self.good_label = QLabel('Good: --')
        threshold_layout.addWidget(self.good_label)
        self.bad_label = QLabel('Bad: --')
        threshold_layout.addWidget(self.bad_label)
        
        head_label=QLabel('Thresholds:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 10px;')
        
        threshold_layout.addWidget(head_label)
        threshold_layout.addWidget(QLabel('  SNR_lowest:'))
        self.SNR_lowest_spinbox = QDoubleSpinBox()
        self.SNR_lowest_spinbox.setToolTip('Minimum required trace SNR. Traces with SNR below this will get rejected')
        threshold_layout.addWidget(self.SNR_lowest_spinbox)
        threshold_layout.addWidget(QLabel('  min_SNR:'))
        self.min_SNR_spinbox = QDoubleSpinBox()
        self.min_SNR_spinbox.setToolTip('Trace SNR threshold. Traces with SNR above this will get accepted')
        threshold_layout.addWidget(self.min_SNR_spinbox)
        threshold_layout.addWidget(QLabel('  cnn_lowest:'))
        self.cnn_lowest_spinbox = QDoubleSpinBox()
        self.cnn_lowest_spinbox.setToolTip('Minimum required CNN threshold. Components with score lower than this will get rejected')
        threshold_layout.addWidget(self.cnn_lowest_spinbox)
        threshold_layout.addWidget(QLabel('  min_cnn_thr:'))
        self.min_cnn_thr_spinbox = QDoubleSpinBox()
        self.min_cnn_thr_spinbox.setToolTip('CNN classifier threshold. Components with score higher than this will get accepted')
        threshold_layout.addWidget(self.min_cnn_thr_spinbox)
        threshold_layout.addWidget(QLabel('  rval_lowest:'))
        self.rval_lowest_spinbox = QDoubleSpinBox()
        self.rval_lowest_spinbox.setToolTip('Minimum required space correlation. Components with correlation below this will get rejected')
        threshold_layout.addWidget(self.rval_lowest_spinbox)
        threshold_layout.addWidget(QLabel('  rval_thr:'))
        self.rval_thr_spinbox = QDoubleSpinBox()
        self.rval_thr_spinbox.setToolTip('Space correlation threshold. Components with correlation higher than this will get accepted')
        threshold_layout.addWidget(self.rval_thr_spinbox)
        self.evaluate_button = QPushButton('Evaluate')
        self.evaluate_button.setToolTip('Accept or reject components based on these threshold values (filter_components())')
        threshold_layout.addWidget(self.evaluate_button)
                
        my_layout.addLayout(threshold_layout)
        
        self.parameters_view = QWebEngineView()
        self.parameters_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.parameters_view.setContextMenuPolicy(Qt.NoContextMenu)
        my_layout.addWidget(self.parameters_view )
        self.setLayout(my_layout)
        
        # Create a Python bridge and web channel for JS communication
        self.bridge = PythonBridge(self)
        self.channel = QWebChannel()
        self.channel.registerObject('pythonBridge', self.bridge)
        self.parameters_view.page().setWebChannel(self.channel)
        
        self.selected_component_on_scatter = -1
        
        for widget in (self.SNR_lowest_spinbox, 
                self.min_SNR_spinbox, 
                self.cnn_lowest_spinbox, 
                self.min_cnn_thr_spinbox, 
                self.rval_lowest_spinbox, 
                self.rval_thr_spinbox):
            widget.valueChanged.connect(self.on_threshold_spinbox_changed)
        self.evaluate_button.clicked.connect(self.mainwindow.on_evaluate_button_clicked)
        
        
    def update_treshold_spinboxes(self):
        cnm=self.mainwindow.cnm
        self.evaluate_button.setEnabled(cnm is not None and cnm.estimates.r_values is not None and self.mainwindow.data_array is not None)
        if cnm is None:
            self.SNR_lowest_spinbox.setEnabled(False)
            self.min_SNR_spinbox.setEnabled(False)
            self.cnn_lowest_spinbox.setEnabled(False)
            self.min_cnn_thr_spinbox.setEnabled(False)
            self.rval_lowest_spinbox.setEnabled(False)
            self.rval_thr_spinbox.setEnabled(False)
            return
        if cnm.estimates.cnn_preds is None:
            cnn_range = (0, 1)
        else:
            cnn_range = (np.min(cnm.estimates.cnn_preds), np.max(cnm.estimates.cnn_preds))
        if cnm.estimates.r_values is None:
            rval_range = (-1, 1)
        else:   
            rval_range = (np.min(cnm.estimates.r_values), np.max(cnm.estimates.r_values))
        if cnm.estimates.SNR_comp is None:
            snr_range = (0, 100)
        else:
            snr_range = (np.min(cnm.estimates.SNR_comp), np.max(cnm.estimates.SNR_comp))       
        
        self.SNR_lowest_spinbox.blockSignals(True)
        self.SNR_lowest_spinbox.setEnabled(True)
        self.SNR_lowest_spinbox.setRange(*snr_range)
        self.SNR_lowest_spinbox.setSingleStep(0.1)
        self.SNR_lowest_spinbox.setValue(cnm.params.quality['SNR_lowest'])
        self.SNR_lowest_spinbox.blockSignals(False)
        self.min_SNR_spinbox.blockSignals(True)
        self.min_SNR_spinbox.setEnabled(True)
        self.min_SNR_spinbox.setRange(*snr_range)
        self.min_SNR_spinbox.setSingleStep(0.1)        
        self.min_SNR_spinbox.setValue(cnm.params.quality['min_SNR'])
        self.min_SNR_spinbox.blockSignals(False)
        self.cnn_lowest_spinbox.blockSignals(True)
        self.cnn_lowest_spinbox.setEnabled(True)
        self.cnn_lowest_spinbox.setRange(*cnn_range)
        self.cnn_lowest_spinbox.setSingleStep(0.1)     
        self.cnn_lowest_spinbox.setValue(cnm.params.quality['cnn_lowest'])
        self.cnn_lowest_spinbox.blockSignals(False)
        self.min_cnn_thr_spinbox.blockSignals(True)
        self.min_cnn_thr_spinbox.setEnabled(True)
        self.min_cnn_thr_spinbox.setRange(*cnn_range)
        self.min_cnn_thr_spinbox.setSingleStep(0.1)
        self.min_cnn_thr_spinbox.setValue(cnm.params.quality['min_cnn_thr'])
        self.min_cnn_thr_spinbox.blockSignals(False)
        self.rval_lowest_spinbox.blockSignals(True)
        self.rval_lowest_spinbox.setEnabled(True)
        self.rval_lowest_spinbox.setRange(*rval_range)
        self.rval_lowest_spinbox.setSingleStep(0.1)       
        self.rval_lowest_spinbox.setValue(cnm.params.quality['rval_lowest'])
        self.rval_lowest_spinbox.blockSignals(False)
        self.rval_thr_spinbox.blockSignals(True)
        self.rval_thr_spinbox.setEnabled(True)
        self.rval_thr_spinbox.setRange(*rval_range)
        self.rval_thr_spinbox.setSingleStep(0.1)
        self.rval_thr_spinbox.setValue(cnm.params.quality['rval_thr'])
        self.rval_thr_spinbox.blockSignals(False)
       
    def on_threshold_spinbox_changed(self, value):
        cnm=self.mainwindow.cnm
        if cnm.estimates.idx_components is None:
            return
        cnm.params.quality['SNR_lowest'] = self.SNR_lowest_spinbox.value()
        cnm.params.quality['min_SNR'] = self.min_SNR_spinbox.value()
        cnm.params.quality['cnn_lowest'] = self.cnn_lowest_spinbox.value()
        cnm.params.quality['min_cnn_thr'] = self.min_cnn_thr_spinbox.value()
        cnm.params.quality['rval_lowest'] = self.rval_lowest_spinbox.value()
        cnm.params.quality['rval_thr'] = self.rval_thr_spinbox.value()
        self.update_treshold_spinboxes()
        self.mainwindow.on_threshold_spinbox_changed() 
        self.update_threshold_lines_on_scatterplot()
  
    def on_scatter_point_clicked(self, index):
        #print(f'Point clicked: {index}', end='')
        if self.selected_component_on_scatter == index:
            #print(' (same as before)')
            return
        self.selected_component_on_scatter = index
        #print(f' (new value: {index})')
        self.mainwindow.set_selected_component(index, 'scatter')
    
    
    def construct_threshold_gridline_data(self):
        cnm=self.mainwindow.cnm
        if cnm is None:
            return {}
        x = cnm.estimates.cnn_preds
        y = cnm.estimates.r_values
        z = cnm.estimates.SNR_comp
        lines = {
                'X = min_cnn_thr': {'x': [float(cnm.params.quality['min_cnn_thr'])]*5, 'y': [float(min(y)), float(max(y)), float(max(y)), float(min(y)), float(min(y))], 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'green'},
                'X = cnn_lowest': {'x': [float(cnm.params.quality['cnn_lowest'])]*5, 'y': [float(min(y)), float(max(y)), float(max(y)), float(min(y)), float(min(y))], 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'green'},
                'Y = rval_thr': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(cnm.params.quality['rval_thr'])]*5, 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'blue'},
                'Y = rval_lowest': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(cnm.params.quality['rval_lowest'])]*5, 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'blue'},
                'Z = min_SNR': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(min(y)), float(min(y)), float(max(y)), float(max(y)), float(min(y))], 'z': [float(cnm.params.quality['min_SNR'])]*5, 'color': 'magenta'},
                'Z = SNR_lowest': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(min(y)), float(min(y)), float(max(y)), float(max(y)), float(min(y))], 'z': [float(cnm.params.quality['SNR_lowest'])]*5, 'color': 'magenta'},
            }
        return lines
    
    def update_threshold_lines_on_scatterplot(self):
        if self.mainwindow.cnm.estimates.idx_components is None:
            return
        lines=self.construct_threshold_gridline_data() 
        lines_json = json.dumps(lines)
        # Build the JS command calling updateThresholdLines, ensuring proper quoting
        js_cmd = f'window.updateThresholdLines({json.dumps(lines_json)});'
        #print(f'Calling JS for threshold lines: {js_cmd}')
        self.parameters_view.page().runJavaScript(js_cmd)
        
    def update_selected_component_on_scatterplot(self, index):
        cnm=self.mainwindow.cnm
        if cnm is None or cnm.estimates.r_values is None:
            return
        # Compute 3D coordinates and pass them directly as floats
        #print(f'Updating selected component on scatter plot: {index}')
        x_val = float(cnm.estimates.cnn_preds[index])
        y_val = float(cnm.estimates.r_values[index])
        z_val = float(cnm.estimates.SNR_comp[index])
        if index in cnm.estimates.idx_components:
            color = 'green'
        else:
            color = 'magenta'
        js_cmd = f'window.updateSelectedTrace({x_val}, {y_val}, {z_val}, "{color}");'
        #print(f'Calling JS: {js_cmd}')
        self.parameters_view.page().runJavaScript(js_cmd)
        
    def update_totals(self):
        cnm=self.mainwindow.cnm
        if cnm is None:
            self.good_label.setEnabled(False)
            self.bad_label.setEnabled(False) 
            self.total_label.setEnabled(False)
            return
        cnme=cnm.estimates
        self.total_label.setText(f'    Total: {self.mainwindow.numcomps}')
        self.total_label.setEnabled(True)
        if not cnme.idx_components is None:
            self.good_label.setText(f'    Good: {len(cnme.idx_components)}')
            self.good_label.setEnabled(True)
            self.bad_label.setText(f'    Bad: {len(cnme.idx_components_bad)}')
            self.bad_label.setEnabled(True)  
        else:
            self.good_label.setText('    Good: --')
            self.good_label.setEnabled(False)
            self.bad_label.setText('    Bad: --')
            self.bad_label.setEnabled(False)   
        
class PythonBridge(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)

    @Slot(str)
    def pointClicked(self, index):
        parent = self.parent()
        if parent and index != '':
            parent.on_scatter_point_clicked(int(index))
            

class SpatialWidget(QWidget):
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.mainwindow = main_window
        
        my_layot=QHBoxLayout(self)
        
        self.spatial_view = pg.PlotWidget()
          
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignTop)

        head_label=QLabel('Display:')
        head_label.setStyleSheet('font-weight: bold;')
        left_layout.addWidget(head_label)

        self.channel_combo = QComboBox()
        self.channel_combo.addItem('A')
        self.channel_combo.setToolTip('Select spatial data to display')
        left_layout.addWidget(self.channel_combo)  
        
        self.spatial_zoom_button = QPushButton('Zoom')
        self.spatial_zoom_button.setToolTip('Centers view on selected component, with zoom corresponding to neuron diameter')
        left_layout.addWidget(self.spatial_zoom_button)
        self.spatial_zoom_auto_checkbox = QCheckBox('Auto')
        self.spatial_zoom_auto_checkbox.setToolTip('Centers view on selected component, with zoom corresponding to neuron diameter')
        left_layout.addWidget(self.spatial_zoom_auto_checkbox)
     
        head_label=QLabel('Contours:')
        head_label.setStyleSheet('font-weight: bold;')
        left_layout.addWidget(head_label)
        
        self.contour_combo = QComboBox()
        self.contour_combo.addItem('--')
        self.contour_combo.setToolTip('Select contour groups to draw')
        left_layout.addWidget(self.contour_combo)  
        
        my_layot.addLayout(left_layout)
        my_layot.addWidget(self.spatial_view, stretch=1)        
        
        self.setLayout(my_layot)        
        self.ctime=0
        #event hadlers
        self.channel_combo.currentIndexChanged.connect(self.on_channel_combo_changed)
        self.spatial_zoom_button.clicked.connect(self.on_spatial_zoom)
        self.spatial_zoom_auto_checkbox.stateChanged.connect(self.on_spatial_zoom_auto_changed)
        self.contour_combo.currentIndexChanged.connect(self.on_contour_combo_changed)

    def on_contour_combo_changed(self, index):
        # called on change of contour selector combo box
        self.update_spatial_view()

    def on_spatial_zoom(self):
        self.perform_spatial_zoom_on_component(self.mainwindow.selected_component)

    def on_spatial_zoom_auto_changed(self, state):
        if self.spatial_zoom_auto_checkbox.isChecked():
            self.perform_spatial_zoom_on_component(self.mainwindow.selected_component)

        
    def on_channel_combo_changed(self, index):
        #called on change of channel selector combo box
        self.update_spatial_view(setLUT=True)
        
    def perform_spatial_zoom_on_component(self, index):
        zoomwindow=self.mainwindow.neuron_diam*1.5
        coord=self.mainwindow.component_centers[index,:]
        xrange=coord[0]-zoomwindow,coord[0]+zoomwindow
        yrange=coord[1]-zoomwindow,coord[1]+zoomwindow
        self.spatial_view.setRange(xRange=xrange, yRange=yrange, padding=0.0)
        
    def recreate_spatial_view(self):
        self.channel_combo.setEnabled(self.mainwindow.cnm is not None)
        
        if self.mainwindow.cnm is None:
            text='No data loaded yet'
            text = pg.TextItem(text=text, anchor=(0.5, 0.5), color='k')
            self.spatial_view.clear()
            self.spatial_view.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
            self.spatial_view.addItem(text)
            self.spatial_view.getPlotItem().showGrid(False)
            self.spatial_view.getPlotItem().showAxes(False)
            self.spatial_view.setBackground(QColor(200, 200, 210, 127))
            self.spatial_zoom_auto_checkbox.setEnabled(False)
            self.spatial_zoom_button.setEnabled(False)
            self.contour_combo.setEnabled(False)
            return
        
        print('Rendering spatial view, contours...')
        self.spatial_view.disableAutoRange()
        self.spatial_view.clear()
        self.spatial_view.setBackground(None)
        self.spatial_view.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        # Explicitly remove previous colorbar if exists
        if hasattr(self, 'colorbar_item'):
            self.spatial_view.getPlotItem().layout.removeItem(self.colorbar_item)
            self.colorbar_item.deleteLater()
            del self.colorbar_item
                   
        self.spatial_image = pg.ImageItem()
        self.spatial_view.addItem( self.spatial_image )
        self.colorbar_item=self.spatial_view.getPlotItem().addColorBar( self.spatial_image, colorMap='viridis', rounding=1e-10) # , interactive=False)
        
        plot_item = self.spatial_view.getPlotItem()
        plot_item.setAspectLocked(True)
        plot_item.showAxes(True, showValues=(True, False, False, True))
        plot_item.showGrid(x=False, y=False)
        plot_item.setMenuEnabled(False)

        # Configure axis tick lengths explicitly
        for side in ('top', 'right'):
            ax = plot_item.getAxis(side)
            ax.setStyle(tickLength=0)
        for side in ('left', 'bottom'):
            ax = plot_item.getAxis(side)
            ax.setStyle(tickLength=10)      
        self.spatial_view.setDefaultPadding( 0.0 )
        
        cnme=self.mainwindow.cnm.estimates
        numcomps=self.mainwindow.numcomps
        if cnme.coordinates is None:
            self.spatial_zoom_auto_checkbox.setEnabled(False)
            self.spatial_zoom_button.setEnabled(False)
        else:
            self.spatial_zoom_auto_checkbox.setEnabled(True)
            #self.spatial_zoom_auto_checkbox.setChecked(False)
            self.spatial_zoom_button.setEnabled(True)

        
        if cnme.idx_components is None:
            selectable_combo_names=['All', 'Selected', 'None']
        else:
            selectable_combo_names=['All', 'Good+T', 'Bad+T', 'Good', 'Bad', 'Selected', 'None']
        
        previous_selected_text = self.contour_combo.currentText()
        self.contour_combo.blockSignals(True)
        self.contour_combo.clear()
        self.contour_combo.addItems(selectable_combo_names)
        if previous_selected_text in selectable_combo_names:
            self.contour_combo.setCurrentText(previous_selected_text)
        else:
            self.contour_combo.setCurrentIndex(0)
        self.contour_combo.setEnabled(True) 
        self.contour_combo.blockSignals(False)

        #plotting contours
        self.goodpen=pg.mkPen(color='g', width=1)
        self.badpen=pg.mkPen(color='r', width=1)
        self.selectedpen=pg.mkPen(color='y', width=2)
        self.contur_items=[]
        for idx_to_plot in range(len(self.mainwindow.component_contour_coords)):
            component_contour = self.mainwindow.component_contour_coords[idx_to_plot]
            component_contour=component_contour[1:-1,:]
            curve = pg.PlotCurveItem(x=component_contour[:, 1], y=component_contour[:, 0], name=f'{idx_to_plot}', pen=self.goodpen,  clickable=True, fillLevel=0.5)
            curve.sigClicked.connect(self.on_contour_click)
            curve.setClickable(True, 10)
            self.contur_items.append(curve)
            self.spatial_view.addItem(curve)
        self.spatial_view.enableAutoRange()
        
        self.update_spatial_view(setLUT=True)

        
    def update_spatial_view(self, array_text=None, setLUT=False):
        #update combo with available channels
        #update image view, titles, etc
        cnme=self.mainwindow.cnm.estimates
        
        possible_array_text=['A']
        if self.mainwindow.data_array is not None:
            possible_array_text.append('Data')
            possible_array_text.append('Residuals')
        possible_array_text.append('RCM')
        possible_array_text.append('RCB')
        numbackround=cnme.b.shape[-1]
        for i in range(numbackround):
            possible_array_text.append(f'B{i}')
        if hasattr(cnme, 'Cn') and cnme.Cn is not None:
            possible_array_text.append(f'Cn')
        for type in ['mean', 'max', 'std']:
            if getattr(self.mainwindow, type+'_projection_array') is not None:
                possible_array_text.append(type.capitalize())
        if hasattr(cnme, 'sn') and cnme.sn is not None:
            possible_array_text.append(f'sn')
        
        if array_text is None:
            previous_text=self.channel_combo.currentText()
        else:
            previous_text=array_text
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(possible_array_text)
        if previous_text not in possible_array_text:
            self.channel_combo.setCurrentIndex(0)
        else:
            self.channel_combo.setCurrentText(previous_text)
        self.channel_combo.blockSignals(False)
        array_text=self.channel_combo.currentText()
        
        component_idx = self.mainwindow.selected_component
        
        plot_item = self.spatial_view.getPlotItem()
        if array_text == 'A':
            ctitle=f'Spatial component footprint ({component_idx})'
        elif array_text == 'Data':
            ctitle=f'Original data (movie)'
        elif array_text == 'RCM':
            ctitle=f'Reconstructed movie (A ⊗ C) (movie)'
        elif array_text == 'RCB':
            ctitle=f'Reconstructed background (b ⊗ f) (movie)'
        elif array_text == 'Residuals':
            ctitle=f'Residuals (Y - (A ⊗ C) - (b ⊗ f)) (movie)'
        elif array_text[0] == 'B':
            ctitle=f'Background component {int(array_text[1:])}'
        elif array_text == 'Cn':
            ctitle=f'Correlation image'
        elif array_text in ['Mean', 'Max', 'Std']:
            ctitle=f'{array_text} projection image'
        else:
            ctitle=f'Array: {array_text}'
        plot_item.setTitle(ctitle)
        
        if self.spatial_zoom_auto_checkbox.isChecked():
            self.perform_spatial_zoom_on_component(component_idx)
            
        self.update_spatial_view_image(setLUT=setLUT)
        
        #plotting contours
        #for idx_to_plot in [component_idx]:
        contour_mode=self.contour_combo.currentText()
        transparency=100
        if cnme.idx_components is None or component_idx  in cnme.idx_components:
            cursor_color=(180, 255, 60, 255)
        else:
            cursor_color=(255, 180, 60, 255)
        if contour_mode == 'All':
            self.goodpen.setColor(pg.mkColor(0, 255, 0, 255))
            self.badpen.setColor(pg.mkColor(255, 0, 0, 255))
            self.selectedpen.setColor(pg.mkColor(cursor_color))
        elif contour_mode == 'Good':
            self.goodpen.setColor(pg.mkColor(0, 255, 0, 255))
            self.badpen.setColor(pg.mkColor(0, 0, 0, 0))
            self.selectedpen.setColor(pg.mkColor(cursor_color))
        elif contour_mode == 'Bad':
            self.goodpen.setColor(pg.mkColor(0, 0, 0, 0))
            self.badpen.setColor(pg.mkColor(255, 0, 0, 255))
            self.selectedpen.setColor(pg.mkColor(cursor_color))
        elif contour_mode == 'Good+T':
            self.goodpen.setColor(pg.mkColor(0, 255, 0, 255))
            self.badpen.setColor(pg.mkColor(255, 0, 0, transparency))
            self.selectedpen.setColor(pg.mkColor(cursor_color))
        elif contour_mode == 'Bad+T':
            self.goodpen.setColor(pg.mkColor(0, 255, 0, transparency))
            self.badpen.setColor(pg.mkColor(255, 0, 0, 255))    
            self.selectedpen.setColor(pg.mkColor(cursor_color))
        elif contour_mode == 'Selected':
            self.goodpen.setColor(pg.mkColor(0, 0, 0, 0))    
            self.badpen.setColor(pg.mkColor(0, 0, 0, 0))    
            self.selectedpen.setColor(pg.mkColor(cursor_color))
        elif contour_mode == 'None':
            self.goodpen.setColor(pg.mkColor(0, 0, 0, 0))
            self.badpen.setColor(pg.mkColor(0, 0, 0, 0))
            self.selectedpen.setColor(pg.mkColor(0, 0, 0, 0))
        else:
            raise ValueError(f'Invalid contour mode: {contour_mode}')
        
        #setting component contour graphics properties
        if  cnme.idx_components is not None:
            if self.mainwindow.limit == 'All':
                clickarray=[True]*self.mainwindow.numcomps
            elif self.mainwindow.limit == 'Good':
                clickarray=np.zeros(self.mainwindow.numcomps, dtype=bool)
                clickarray[cnme.idx_components]=True
            else:
                clickarray=np.zeros(self.mainwindow.numcomps, dtype=bool)
                clickarray[cnme.idx_components_bad]=True
        else:
            clickarray=[True]*self.mainwindow.numcomps
        for idx_to_plot in range(self.mainwindow.numcomps):
            if idx_to_plot==component_idx:
                peny=self.selectedpen
            elif cnme.idx_components is None or idx_to_plot in cnme.idx_components:
                peny=self.goodpen
            else:    
                peny=self.badpen
            self.contur_items[idx_to_plot].setPen(peny)
            self.contur_items[idx_to_plot].setClickable(clickarray[idx_to_plot])                                       
            
        
    def update_spatial_view_image(self, setLUT=False):
        #update only the image according to t
        
        array_text=self.channel_combo.currentText()
        component_idx = self.mainwindow.selected_component
        t=self.mainwindow.selected_frame
        w=self.mainwindow.frame_window
        tmin=t-w if t-w>0 else 0
        tmax=t+w+1 if t+w+1<self.mainwindow.num_frames else self.mainwindow.num_frames        
        #residuals = self._raw_movie[indices] - self._rcm[indices] - self._rcb[indices]
        
        if array_text == 'A':
            image_data = np.reshape(self.mainwindow.cnm.estimates.A[:, component_idx], self.mainwindow.dims) #(self.dims[1], self.dims[0]), order='F').T
        elif array_text == 'Data':
            #print('display: elapsed off {:.2f}'.format((time.perf_counter()-self.ctime)))
            #self.ctime=time.perf_counter()
            res=self.mainwindow.data_array[:,tmin:tmax]
            res=np.mean(res, axis=1)

            image_data = res.reshape(self.mainwindow.dims)
        elif array_text == 'RCM':
            res=np.dot(self.mainwindow.cnm.estimates.A[:, :] , self.mainwindow.cnm.estimates.C[:, tmin:tmax])
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
        elif array_text == 'RCB':
            res=np.dot(self.mainwindow.cnm.estimates.b[:, :] , self.mainwindow.cnm.estimates.f[:, tmin:tmax])
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
        elif array_text == 'Residuals':
            res=self.mainwindow.data_array[:,tmin:tmax]
            res=np.mean(res, axis=1)
            rcm=np.dot(self.mainwindow.cnm.estimates.A[:, :] , self.mainwindow.cnm.estimates.C[:, tmin:tmax])
            rcm=np.mean(rcm, axis=1)
            rcb=np.dot(self.mainwindow.cnm.estimates.b[:, :] , self.mainwindow.cnm.estimates.f[:, tmin:tmax])
            rcb=np.mean(rcb, axis=1)
            res=res-rcm-rcb
            image_data = res.reshape(self.mainwindow.dims)
        elif array_text[0] == 'B':
            try:
                bgindex = int(array_text[1:])
            except ValueError:
                raise ValueError(f'Invalid array text: {array_text}')
            image_data = self.mainwindow.cnm.estimates.b[:, bgindex].reshape(self.mainwindow.dims)
        elif array_text == 'Cn':
            image_data = self.mainwindow.cnm.estimates.Cn
        elif array_text in ['Mean', 'Max', 'Std']:
            image_data = getattr(self.mainwindow, array_text.lower()+'_projection_array')
        elif array_text == 'sn':
            image_data = self.mainwindow.cnm.estimates.sn.reshape(self.mainwindow.dims)
        else:
            raise NotImplementedError   
        
        # Update image data
        self.spatial_image.setImage(image_data, autoLevels=False)
   
        if setLUT:
            # Update colorbar limits explicitly
            min_val, max_val = np.min(image_data), np.max(image_data)
            self.colorbar_item.setLevels(values=[min_val, max_val])       
    
    def on_contour_click(self, ev):
        index = int(ev.name())
        if self.mainwindow.cnm.estimates.idx_components is not None:
            if self.mainwindow.limit == 'Good':
                if index in self.mainwindow.cnm.estimates.idx_components_bad:
                    return
            elif self.mainwindow.limit == 'Bad':
                if index in self.mainwindow.cnm.estimates.idx_components:
                    return
        self.mainwindow.set_selected_component(index, 'spatial') 
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    
