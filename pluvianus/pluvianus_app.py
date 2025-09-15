import glob
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

import caiman as cm # type: ignore
from caiman.source_extraction import cnmf # type: ignore
from caiman.utils.visualization import get_contours as caiman_get_contours # type: ignore

import cv2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import pyqtgraph as pg
from PySide6 import __version__ as PySide6_version
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QShortcut 
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QScrollArea, QCheckBox,QSlider,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QLabel, QComboBox, QPushButton, QProgressDialog, QSizePolicy,
    QPlainTextEdit, QDialog, QFrame
)
from scipy.signal.windows import gaussian

from pluvianus.GripSplitter import GripSplitter
from pluvianus.CaImAnFileChecker import CaImAnFileChecker

print(f'PySide6 {PySide6_version} loaded.')
print(f'CaImAn { cm.__version__} loaded.')

try:
    from pluvianus import __version__
except ImportError:
    __version__ = "0.0.0-dev"
__date__ = time.strftime("%Y-%m-%d")


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
        bottom_row = GripSplitter( childrenCollapsible=False)
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
        self.colorbar_item=self.spatial_widget.getPlotItem().addColorBar(self.spatial_image, colorMap='viridis', rounding=0.00000000001) # , interactive=False)
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
        self.spatial_widget.getPlotItem().invertY(True)
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
           
class PlotWidgetWithRightAxis(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        """
        Initialize a PlotWidgetWithRightAxis instance, a child of pg.PlotWidget.

        This constructor sets up the plot widget with an additional right axis.
        It creates a new ViewBox for the right axis, links it to the main plot,
        and applies styling to the right axis. The right axis color is set to 
        dark blue by default. The view geometry is updated to synchronize with 
        the main plot's ViewBox when resized.
        
        You can acces the right axis by calling self.RightViewBox, e.g.:
            self.RightViewBox.addItem(pg.PlotCurveItem(...))   
        You can set the right axis color by calling self.setRightColor(...) e.g.:
            self.setRightColor('#ff008b') 

        Parameters (passed to the PlotWidget parent class constructor):
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super(PlotWidgetWithRightAxis, self).__init__(*args, **kwargs)

        self.showAxis('right')
        self.RightViewBox = pg.ViewBox()
        self.plotItem.scene().addItem(self.RightViewBox)

        right_axis = self.getAxis('right')
        right_axis.linkToView(self.RightViewBox)
        self.RightViewBox.setXLink(self)
        self.RightColor = '#00008b' #default dark blue
        right_axis.setStyle(showValues=True)
        self.setRightColor(self.RightColor)

        self._updateViews()
        self.plotItem.vb.sigResized.connect(self._updateViews)

    def _updateViews(self):
        self.RightViewBox.setGeometry(self.plotItem.vb.sceneBoundingRect())
        self.RightViewBox.linkedViewChanged(self.plotItem.vb, self.RightViewBox.XAxis)
        
    def setRightColor(self, color):
        self.RightColor = color
        right_axis = self.getAxis('right')
        right_axis.setPen(pg.mkPen(self.RightColor))       
        right_axis.setLabel('axis2', color=self.RightColor)
        right_axis.setTextPen(pg.mkPen(self.RightColor))
        
    
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
        file_menu.addAction(self.save_std_image_action) # Need to add maximum residuals here later
        
        # Compute menu
        comp_menu = self.menuBar().addMenu('Compute')
        self.detr_action = QAction('Detrend ΔF/F', self)
        comp_menu.addAction(self.detr_action)
        self.compute_component_evaluation_action = QAction('Compute Component Metrics', self)
        comp_menu.addAction(self.compute_component_evaluation_action)
        self.compute_projections_action = QAction('Compute Projections (heavy)', self)
        comp_menu.addAction(self.compute_projections_action)
        self.compute_cn_action = QAction('Compute Local Correlation Image (heavy)', self)
        comp_menu.addAction(self.compute_cn_action)
        self.compute_origtrace_action = QAction('Compute Original Fluorescence Traces', self)
        comp_menu.addAction(self.compute_origtrace_action)
        self.compute_residual_maximums_action = QAction('Compute Temporal Maximum of Residuals', self)
        comp_menu.addAction(self.compute_residual_maximums_action)
        self.compute_data_array_action = QAction('Compute Data Array for OnACID Files')
        comp_menu.addAction(self.compute_data_array_action)
        
        # View menu
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
        self.save_mescroi_action = QAction('Contours to MEScROI (Good)...', self)
        exp_menu.addAction(self.save_mescroi_action)
        
        help_menu = self.menuBar().addMenu('Help')
        about_action = QAction('About...', self)
        license_action = QAction('License...', self)
        source_action = QAction('Documentation && Source...', self)
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
        self.compute_residual_maximums_action.triggered.connect(self.on_compute_residual_maximums)
        self.compute_data_array_action.triggered.connect(self.on_compute_data_array_action)
        self.opts_action.triggered.connect(self.on_opts_action)
        
        self.info_action.triggered.connect(self.on_info_action)
        self.shifts_action.triggered.connect(self.on_shifts_action)
        self.bg_action.triggered.connect(self.on_bg_action)
        
        about_action.triggered.connect(self.on_about_action)
        license_action.triggered.connect(self.on_license_action)
        source_action.triggered.connect(self.on_source_action)
        
        #Keyboard shortcuts
        shortcut_g = QShortcut(QKeySequence("G"), self) # g key for good
        shortcut_g.activated.connect(lambda: self.set_component_assignment_manually('Good'))
        shortcut_b = QShortcut(QKeySequence("B"), self) # b key for bad
        shortcut_b.activated.connect(lambda: self.set_component_assignment_manually('Bad'))       
        shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self) # Up arrow
        shortcut_up.activated.connect(lambda: self.set_nav_component_pressed('next'))
        shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self) # Down arrow
        shortcut_down.activated.connect(lambda: self.set_nav_component_pressed('prev'))

        self.resizeEvent = self.on_resize_figure
        self.closeEvent = self.on_mainwindow_closing
        
        # Create central widget and layout       
        main_layout = GripSplitter(Qt.Vertical)
        self.setCentralWidget(main_layout)
        main_layout.setStyleSheet('QSplitter::handle { background-color: lightgray; }')
        
        self.temporal_widget = TopWidget(self, self)
        main_layout.addWidget(self.temporal_widget)
        
        bottom_layout_splitter = GripSplitter()
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
        self.num_frames = 0  # number of frames in movie
        self.selected_frame = 0 # index of selected frame
        self.frame_window = 0 # temporal window of displaying movie frames (half window, in frames)
        self.order='index (All)' # order of components in combo box
        self.order_indexes = [] # indexes of components in order
        self.manual_acceptance_assigment_has_been_made = False # flag for storing if manual component assignment has been made
        self.data_file = '' # file name of data array file (mmap)
        self.data_array = None # data array if loaded
        self.mean_projection_array = None # mean projection array
        self.max_projection_array = None # max projection array
        self.std_projection_array = None # std projection array
        self.orig_trace_array = None # computed original fluorescence traces
        self.orig_trace_array_neuropil = None # computed original fluorescence traces' neuropil
        # correlation image is stored in the cnm object
        self.max_res_none = None
        self.max_res_none_idx = None # maximum residuals after bg subtraction time indexes
        self.max_res_good = None
        self.max_res_good_idx = None # maximum residuals for good components time indexes
        self.max_res_all = None 
        self.max_res_all_idx = None  # maximum residuals for all components time indexes
    
         
        # Update figure and state
        self.load_state()
        self.update_all()
    
    def set_nav_order_by(self, text):
        self.order = text
        if text == 'index (All)':
            self.order_indexes = np.arange(self.numcomps)
        elif text == 'index (Good)':
            self.order_indexes = self.cnm.estimates.idx_components
        elif text == 'index (Bad)':
            self.order_indexes = self.cnm.estimates.idx_components_bad
        elif text == 'SNR':
            self.order_indexes = np.argsort(self.cnm.estimates.SNR_comp)#[::-1]
        elif text == 'R value':
            self.order_indexes = np.argsort(self.cnm.estimates.r_values)#[::-1]
        elif text == 'CNN':
            self.order_indexes = np.argsort(self.cnm.estimates.cnn_preds)#[::-1]
        elif text == 'Compound':
            compound_metrics=np.log10(self.cnm.estimates.SNR_comp)/3 + self.cnm.estimates.r_values * self.cnm.estimates.cnn_preds
            self.order_indexes = np.argsort(compound_metrics)#[::-1]
        else:
            raise ValueError(f'Invalid order: {text}')
        
        current_component=self.selected_component
        if current_component not in self.order_indexes:
            idx = (np.abs(self.order_indexes - current_component)).argmin()
            current_component = self.order_indexes[idx]
                    
        self.set_selected_component(current_component, 'direct')
        self.temporal_widget.update_nav_order_by()
        
    def set_nav_component_pressed(self, direction):
        '''
        Navigate to the next or previous component in the order defined by the combo box.
        Parameters:  direction : str  'next' or 'prev' to navigate to the next or previous component.
        '''
        if self.cnm is None:
            return
        
        idx = (np.abs(self.order_indexes - self.selected_component)).argmin()            
        if direction == 'next':
            idx=idx+1
        elif direction == 'prev':
            idx=idx-1
        idx = max(0, min(idx, len(self.order_indexes)-1))
        self.set_selected_component(self.order_indexes[idx], 'direct')
        
    def set_selected_component(self, value, method):
        '''
            Component number is set. combo is reset if setting outside of good/bad. method is disregarded.
        '''
        if self.cnm is None:
            return
        #print('set_selected_component', value, method)
        if self.cnm is None:
            value = min(self.numcomps-1, value)
        if method == 'spinbox': 
            method = 'direct' 
        if method == 'spatial':
            method = 'direct'
        if method == 'scatter':
            method = 'direct'
            
        if method == 'direct':
            if value not in self.order_indexes:
                self.order= 'index (All)'
                self.set_nav_order_by(self.order)
            self.selected_component = value 
        else:
            raise ValueError(f'Invalid method: {method}')
        
        #update
        self.temporal_widget.update_component_spinbox(self.selected_component)
        self.temporal_widget.update_nav_button_enabled()
        self.scatter_widget.update_selected_component_on_scatterplot(self.selected_component)
        self.temporal_widget.update_temporal_view()
        self.spatial_widget.update_spatial_view()
        self.spatial_widget2.update_spatial_view()
        self.scatter_widget.update_component_assignment_buttons()

    def set_component_assignment_manually(self, state, component=None):
        #print(f"The {state} button was toggled") # Handle the toggle button click event
        if component is None:
            component=self.selected_component
        changed=False
        if self.cnm is not None and self.cnm.estimates.idx_components is not None:
            numcomps=self.numcomps
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
            self.scatter_widget.update_scatterplot()
            self.set_selected_component(component, 'direct')
            self.scatter_widget.update_totals()
            self.scatter_widget.update_selected_component_on_scatterplot(self.selected_component)
        else:
            self.scatter_widget.update_component_assignment_buttons()
        
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
            
    def on_compute_data_array_action(self):
        movie_paths, _ = QFileDialog.getOpenFileNames(self, "Open original movie(s) used for the OnACID analysis", self.hdf5_file, "HDF5 Files (*.hdf5);;TIFF files (*.tif *.tiff);;All files (*)")
        # The files HAVE to be in order!
        if not movie_paths:
            return

        progress_dialog = QProgressDialog('Computing Data Array', None, 0, 100, self)
        progress_dialog.setWindowTitle('Computing Data Array')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(450)
        progress_dialog.setLabelText(f'Motion correcting movie(s) (apply_shift_online()) ...')
        progress_dialog.show()
        QApplication.processEvents()

        images = cm.load(movie_paths) # Documentation says hdf5, tiff, npy, and mmap formats are supported. Only tested on hdf5 so far though!
        shifts = self.cnm.estimates.shifts[-self.cnm.estimates.C.shape[-1]:]

        try:
            mmap_path = cm.motion_correction.apply_shift_online(images, shifts, save_base_name="memmap", order='C')
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(self, 'Motion Correction Error', f'Error during motion correction:\n{str(e)}')
            print(f'Error during motion correction:\n{str(e)}')
            return

        progress_dialog.setValue(50)
        progress_dialog.setLabelText(f'Loading memmap file...')
        print('Motion corrected mmap file created: ', mmap_path)

        Yr, dims, T = cm.load_memmap(mmap_path)
        if T != self.num_frames or dims[0] != self.dims[1] or dims[1] != self.dims[0]:
            progress_dialog.close()
            QMessageBox.critical(self, 'Error loading data', f'Incompatible data dimensions: expected {self.num_frames} frames x {self.dims[0]} x {self.dims[1]} pixels, but got {T} frames x {dims[0]} x {dims[1]} pixels.')
            print(f'Incompatible data dimensions: expected {self.num_frames} frames x {self.dims[0]} x {self.dims[1]} pixels, but got {T} frames x {dims[1]} x {dims[0]} pixels.')
            return
        
        self.data_array = Yr
        self.data_file = mmap_path

        progress_dialog.setValue(75)
        progress_dialog.setLabelText(f'Rendering windows...')
        
        self.spatial_widget.update_spatial_view(array_text='Data')
        self.update_all()

        progress_dialog.setValue(100)
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
        
    def on_detrend_action(self):
        '''
        Detrending using the `detrend_df_f` method of the CNMF object.
        After detrending is complete, the `file_changed` flag is set qand the estimates.f_dff array is filled.
        '''
        if self.cnm is None or self.cnm.estimates.F_dff is not None:
            return
        print('Detrending...')
        waitDlg = QProgressDialog('Detrending in progress (detrend_df_f())...', None, 0, 0, self)
        waitDlg.setWindowModality(Qt.ApplicationModal)  # Blocks input to main window
        waitDlg.setCancelButton(None)  # No cancel button
        waitDlg.setWindowTitle('Please Wait')
        waitDlg.setMinimumDuration(0)  # Show immediately
        waitDlg.setRange(0, 0)  # Indeterminate progress
        waitDlg.setFixedWidth(450)
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
                'Number of components': self.numcomps,
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
    

    def on_about_action(self):
        text = f"""
            Pluvianus: CaImAn Result Browser
            A standalone GUI for browsing, editing, 
            and manually verifying CaImAn results..

            by Gergely Katona

            Version: {__version__}
            Date: {__date__}
            """
        QMessageBox.about(self, "About Pluvianus", text)
    
    def on_license_action(self):

        def read_license():
            try:
                if hasattr(sys, '_MEIPASS'):
                    # PyInstaller: bundled location
                    base_path = sys._MEIPASS
                else:
                    # Dev or installed: use repo root (assumes this file is in pluvianus/)
                    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

                license_path = os.path.join(base_path, "LICENSE")

                with open(license_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"Could not load LICENSE file:\n{e}"
              
        dialog = QDialog(self)
        dialog.setWindowTitle("License")
        layout = QVBoxLayout(dialog)

        text_edit = QPlainTextEdit()
        text_edit.setReadOnly(True)        
        text_edit.setPlainText(read_license())

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)

        layout.addWidget(text_edit)
        layout.addWidget(close_button)

        dialog.resize(600, 500)
        dialog.exec()
        
    def on_source_action(self):
        url = QUrl("https://github.com/katonage/pluvianus")
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "Open URL", "Could not open URL: " + url.toString())
        
    def open_file(self, file_path=None): 
        if file_path is None or file_path is False:       
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
        else:
            if not os.path.exists(file_path):
                print('Error: File does not exist: ' + file_path)
                return
            filename = file_path
        
        print('Open file:', filename)
        progress_dialog = QProgressDialog('Opening file', None, 0, 100, self)
        progress_dialog.setWindowTitle('Loading CaImAn file...')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(300)
        progress_dialog.show()
        QApplication.processEvents()

        # Use file checker before loading to test if file is a valid CaImAn file and to it is an OnACID or CNMF file.
        file_checker = CaImAnFileChecker()
        file_checker.check_file(filename)
        if not file_checker.is_likely_correct:
            print('Error: File is not a valid CaImAn file:', filename)
            progress_dialog.close()
            QMessageBox.critical(self, 'Error opening file', 'File is not a valid CaImAn file: ' + filename)
            return
        
        progress_dialog.setValue(12)
        progress_dialog.setLabelText('Opening CNMF file...')
        QApplication.processEvents()
        
        try:      
            if(file_checker.is_online):
                loaded_cnm = cnmf.online_cnmf.load_OnlineCNMF(filename)
                print('File loaded (OnlineCNMF):', filename)
            else:
                loaded_cnm = cnmf.cnmf.load_CNMF(filename)
                print('File loaded (CNMF):', filename)
        except Exception as e:
            print('Could not load file')
            progress_dialog.close()
            QMessageBox.critical(self, 'Error opening file', 'File could not be opened: ' + filename)
            return

        self.cnm= loaded_cnm
        self.online = file_checker.is_online
        self.hdf5_file = filename
        self.file_changed = False
        self.data_file = ''
        self.data_array = None
        self.mean_projection_array = None
        self.max_projection_array = None
        self.std_projection_array = None
        self.orig_trace_array = None
        self.orig_trace_array_neuropil = None
        self.max_res_none = None
        self.max_res_none_idx = None # maximum residuals after bg subtraction time indexes
        self.max_res_good = None
        self.max_res_good_idx = None # maximum residuals for good components time indexes
        self.max_res_all = None 
        self.max_res_all_idx = None  # maximum residuals for all components time indexes
        
        progress_dialog.setValue(25)
        progress_dialog.setLabelText('Processing data...')
        QApplication.processEvents()
        
        if self.online:
            self.dims=self.cnm.params.data['dims']
        else:         
            self.dims=self.cnm.dims
        self.dims=(self.dims[1], self.dims[0])
        self.num_frames=self.cnm.estimates.C.shape[-1]
        self.selected_frame=int(self.num_frames/2)
        self.numcomps=self.cnm.estimates.A.shape[-1]
        print(f'Data frame dimensions: {self.dims} x {self.num_frames} frames')
        self.framerate=self.cnm.params.data['fr'] #Hz
        self.decay_time=self.cnm.params.data['decay_time'] #sec
        self.frame_window=int(round(self.decay_time*self.framerate))
        self.neuron_diam=np.mean(self.cnm.params.init['gSiz'])*2 #pixels
        self.pixel_size=self.cnm.params.data['dxy'] #um
        print(f'Frame rate: {self.framerate:.3f} Hz, decay time: {self.decay_time} sec, neuron diameter: {self.neuron_diam} pixels, pixel size: {self.pixel_size} um')
        
        #creating copy of A dense
        self.A_array = self.cnm.estimates.A.toarray()
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
        
        self.temporal_widget.update_nav_order_by()
        self.temporal_widget.update_array_selector()
        self.temporal_widget.update_array_selector(right_axis=True)
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
        self.compute_residual_maximums_action.setEnabled(self.data_array is not None and self.cnm is not None and self.cnm.estimates.idx_components is not None) 
        self.compute_data_array_action.setEnabled(self.cnm is not None and self.online is True and self.data_array is None)
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
        self.temporal_widget.rlavr_spinbox2.setEnabled(self.cnm is not None)
        self.temporal_widget.time_slider.setRange(0, self.num_frames-1)
        self.temporal_widget.time_spinbox.setRange(0, self.num_frames-1)        
        self.temporal_widget.range_to_all_button2.setEnabled(self.cnm is not None)
        self.temporal_widget.range_to_all_button.setEnabled(self.cnm is not None)       
            
        if  self.cnm is None:
            self.temporal_widget.component_spinbox.setEnabled(False)
            self.update_title()
            self.temporal_widget.recreate_temporal_view()
            self.scatter_widget.recreate_scatterplot()
            self.scatter_widget.update_totals()
            self.update_title()
            self.scatter_widget.update_component_assignment_buttons()
            return
         
        self.temporal_widget.update_component_spinbox(self.selected_component)
        self.scatter_widget.update_totals()
                
        self.update_title()
        self.temporal_widget.recreate_temporal_view()
        self.scatter_widget.recreate_scatterplot()
        self.set_nav_order_by(self.order)
        #self.set_selected_component(self.selected_component, 'direct')           

    def save_file(self, ignore_overwrite_warning=False, target_filename=None):
        if target_filename is None:
            target_filename=self.hdf5_file
        if target_filename:
            if not ignore_overwrite_warning and os.path.exists(target_filename):
                msg_box = QMessageBox(self)
                msg_box.setText("File already exists. Overwrite?")
                msg_box.setWindowTitle('Saving CaImAn file...')
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.No)
                result = msg_box.exec()
                if result == QMessageBox.No:
                    return
            # Implement save logic here
            self.cnm.save(target_filename)
            self.hdf5_file = target_filename
            self.file_changed = False
            print('File saved as:', target_filename)
            self.save_state()
            self.update_title()

    def save_file_as(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save CaImAn File As', self.hdf5_file, 'HDF5 Files (*.hdf5)')
        if filename:
            self.save_file(target_filename=filename, ignore_overwrite_warning=True)

    def open_data_file(self, data_path=None):
        # Logic to open a data file
        if self.cnm is None:
            print('No CaImAn file loaded. Cannot open data file.')
            return
        if data_path is None or data_path is False:        
            suggested_file = None
            for ext in ['.mmap', '.h5']:
                previ=os.path.dirname(self.hdf5_file)
                suggested_file = next((f for f in glob.glob(os.path.join(previ, '*' + ext)) if os.path.isfile(f)), None)
                if suggested_file is not None:
                    break
            if suggested_file is None:
                suggested_file = os.path.dirname(self.hdf5_file)
            data_file, _  = QFileDialog.getOpenFileName(self, 'Open Movement Corrected Data Array File (.mmap, .h5)', suggested_file, 'Memory mapped files (*.mmap);;Movie file (*.h5);;All Files (*)')
        else:
            if not os.path.exists(data_path):
                print(f'Error: Data path {data_path} does not exist.')
                return
            data_file = data_path   
        
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
            
            try:
                subprocess.Popen([sys.executable, '-m', 'pluvianus.pynapple_npz_viewer', str(data_filep)])
                print('pynapple_npz_viewer launched.')
            except Exception as e:
                print(f'Could not launch pynapple_npz_viewer: {e}')
            
        else:
            raise Exception('Unknown filetype: ' + filetype)
         
    def save_MEScROI(self, _, filtering='Good'):
        #inspired from  Kata5/FemtoOnAcid
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
        mescroi_name, _ = QFileDialog.getSaveFileName(self, 'Save ' + filtering.lower() + ' component contours', suggestion, 'MEScROI files (*.mescroi);;')
        if not mescroi_name: #overwrite confirmation has been made
            return
        
        contours = [self.component_contour_coords[int(i)] for i in idx]
        layer_index = 0

            
        def pixel_coords_to_local(pixel_coords, size_x, size_y, pixel_size_x, pixel_size_y):
            # converts the array indices used by CaImAn (measured in pixels)
            # to local coordinates used by MESc GUI (usually measured in micrometers)
            # CaImAn (and OpenCV) use a coordinate system where the origin is the upper left corner and the y coordinate points downwards
            # MESc GUI uses a coordinate system where the origin is in the center and the y coordinate points upwards
            local_x = (pixel_coords[0] - (size_x / 2)) * pixel_size_x
            local_y = (-1) * (pixel_coords[1] - (size_y / 2)) * pixel_size_y
            return np.array([ local_x, local_y])

        rng = np.random.default_rng(12345)
        mescroi_data = {"rois": []}
        
        for roi_index, contour in enumerate(contours):
            # lighter colours are more visible in the MESc GUI
            r = 128 + int(rng.integers(128))
            g = 128 + int(rng.integers(128))
            b = 128 + int(rng.integers(128))
            
            roi_data = {
                "color": "#ff%02x%02x%02x" % (r, g, b),
                "firstZPlane": layer_index,
                "label": str(roi_index + 1),
                "lastZPlane": layer_index,
                "role": "standard",
                "type": "polygonXY",
                "uniqueID": ("{" + str(uuid.uuid4()) + "}"),
                "vertices": [],
                }
            
            # the component might be non-contiguous, which means the contour may be multiple disconnected loops
            # for each region (thus, contour loop) the coordinates array contains a block of points, delimited by NaNs. the first and last rows are also NaNs.
            # here, we just need to filter out the NaNs
            contour=contour.copy()
            for point in contour:
                if not np.any(np.isnan(point)):
                    point[0], point[1] = point[1], point[0] #Transpose
                    local_coords = pixel_coords_to_local(point, self.dims[0], self.dims[1], self.pixel_size[0], self.pixel_size[1])
                    roi_data["vertices"].append(local_coords.tolist())
            
            mescroi_data["rois"].append(roi_data)
        
        with open(mescroi_name, "w") as f:
            f.write(json.dumps(mescroi_data, indent=4))

    
    
    def on_compute_evaluate_components_action(self):
        if self.data_array is None:
            return
        
        progress_dialog = QProgressDialog('Transposing data...', None, 0, 100, self)
        progress_dialog.setWindowTitle('Computing Component Metrics')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(450)
        progress_dialog.show()
        QApplication.processEvents()
        
        Yr=self.data_array 
        print('Evaluating components... Transposing data...')
        images = np.reshape(Yr.T, [self.num_frames] + [self.dims[1]] + [self.dims[0]], order='F')
        
        progress_dialog.setValue(10)
        progress_dialog.setLabelText(f'Evaluating components (evaluate_components()) (may take a while)...')
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
        progress_dialog.setFixedWidth(450)
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
        
        progress_dialog = QProgressDialog('Creating correlation movie (local_correlations_movie_offline()) (may take a while)...', None, 0, 100, self)
        progress_dialog.setWindowTitle('Computing Local Correlation Image')
        progress_dialog.setModal(True)
        progress_dialog.setFixedWidth(500)
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.processEvents()
        
        print('Calculating local correlation image (local_correlations_movie_offline)...')   
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
        progress_dialog.setLabelText(f'Projecting to image...')
        QApplication.processEvents()
        print(Cns.shape)
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        Cn = Cn.T
        Cn=np.array(Cn)
        self.cnm.estimates.Cn = Cn
        self.file_changed = True
        
        progress_dialog.setValue(90)
        progress_dialog.setLabelText(f'Rendering windows...')
        QApplication.processEvents()
        self.spatial_widget.update_spatial_view(array_text='Cn')
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
        progress_dialog.setFixedWidth(450)
        progress_dialog.show()
        QApplication.processEvents()
        
        #masks for polygons (enery threshold)
        threshold=0.9
        print(f'Calculating fluorescence traces from data file, using component masks at threshold {threshold}...')
        A_masked = self.A_array >= 1
        for i in range(self.numcomps):
            vec=self.A_array[:,i]
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

    def on_compute_residual_maximums(self):
        progress_dialog = QProgressDialog('Processing movie', None, 0, 100, self)
        progress_dialog.setWindowTitle('Calculating maximum of residuals...')
        progress_dialog.setModal(True)
        progress_dialog.setValue(0)
        progress_dialog.setFixedWidth(400)
        progress_dialog.show()
        QApplication.processEvents()

        # Initialize time range
        w = self.frame_window
        num_frames = self.num_frames
        t_range = range(0, num_frames, w)

        # Component indices and estimates
        idx_good_components = self.cnm.estimates.idx_components
        idx_bad_components = sorted(set(range(self.numcomps)) - set(idx_good_components))

        estimates = self.cnm.estimates
        A_array = self.A_array

        # Initialize residual maximums 
        dims = self.dims
        max_res_none = np.full(dims, -np.inf, dtype=np.float32)
        max_res_good = np.full(dims, -np.inf, dtype=np.float32)
        max_res_all  = np.full(dims, -np.inf, dtype=np.float32) # The 3 types of residuals depending on what we subtract from the difference between the original movie and the reconstructed background
        max_res_none_idx = np.zeros(dims, dtype=np.uint32) 
        max_res_good_idx = np.zeros(dims, dtype=np.uint32)
        max_res_all_idx  = np.zeros(dims, dtype=np.uint32) # The frame# in which the maximum residual is found for each pixel in the residual
        
        # Setting kernel for later spatial averaging
        avr = 1
        kernel = cv2.getGaussianKernel(2 * avr + 1, avr)
        kernel = np.outer(kernel, kernel)

        # Main loop for time intervals
        last_t = 0
        for t in t_range:
            t_min = t - w if t - w > 0 else 0
            t_max = t + w + 1 if t + w + 1 < num_frames else num_frames

            Y = self.data_array[:,t_min:t_max] # Original data
            BG = np.dot(estimates.b[:, :], estimates.f[:, t_min:t_max]) # Reconstructed background

            # Subtracting components and averaging
            data_res_none = np.mean(Y - BG, axis=1)
            data_res_none = data_res_none.reshape(dims)
            data_res_none = cv2.filter2D(data_res_none, -1, kernel)   
            max_res_none_idx = np.where(data_res_none > max_res_none, t, max_res_none_idx)
            max_res_none = np.maximum(max_res_none, data_res_none)

            GC = np.dot(A_array[:, idx_good_components], estimates.C[idx_good_components, t_min:t_max]) # Good components
            data_res_good = np.mean(Y - BG - GC, axis=1)
            data_res_good = data_res_good.reshape(dims)
            data_res_good = cv2.filter2D(data_res_good, -1, kernel)   
            max_res_good_idx = np.where(data_res_good > max_res_good, t, max_res_good_idx)
            max_res_good = np.maximum(max_res_good, data_res_good)

            BC = np.dot(A_array[:, idx_bad_components], estimates.C[idx_bad_components, t_min:t_max]) # Bad components
            data_res_all = np.mean(Y - BG - GC - BC, axis=1)
            data_res_all = data_res_all.reshape(dims)
            data_res_all = cv2.filter2D(data_res_all, -1, kernel)  
            max_res_all_idx = np.where(data_res_all > max_res_all, t, max_res_all_idx) 
            max_res_all = np.maximum(max_res_all, data_res_all)            
            
            if t > last_t + 10:
                progress_dialog.setValue(t / num_frames * 50)
                last_t = t
                progress_dialog.setLabelText(f'Calculating residual maximums... (frame {t}/{num_frames})')
                QApplication.processEvents()
            
        # Saving final images
        progress_dialog.setValue(100)
        progress_dialog.setLabelText(f'Rendering windows...')
        
        self.max_res_none = max_res_none
        self.max_res_none_idx = max_res_none_idx
        self.max_res_good = max_res_good
        self.max_res_good_idx = max_res_good_idx
        self.max_res_all = max_res_all
        self.max_res_all_idx = max_res_all_idx

        self.spatial_widget.update_spatial_view(array_text='MaxResAll')
        self.update_all()
        progress_dialog.setLabelText('Done.')
        progress_dialog.close()
            
    def perform_temporal_zoom(self):
        cnme=self.cnm.estimates
        component_index=self.selected_component
        vec=cnme.C[component_index, :]
        max_index = np.argmax(vec)
        
        #zoomwindow=self.decay_time*self.framerate*10 # zooming
        current_xrange = self.temporal_widget.temporal_view.viewRange()[0]
        zoomwindow = (current_xrange[1] - current_xrange[0]) / 2 # just centering

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
        
        # Top plot: Temporal 
        self.temporal_view = PlotWidgetWithRightAxis() #override of pg.PlotWidget
        self.temporal_view.setRightColor('#8b0000')
        # Create a container widget inside scroll area
        scroll_content = QWidget()
        left_layout = QVBoxLayout(scroll_content)

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
        
        left_layout.addWidget(QLabel('Order by:'))
        self.nav_order_by_combo = QComboBox()
        self.nav_order_by_combo.setFixedWidth(90)
        self.nav_order_by_combo.addItem('index (All)')
        self.nav_order_by_combo.addItem('index (Good)')
        self.nav_order_by_combo.addItem('index (Bad)')
        self.nav_order_by_combo.addItem('Compound')
        self.nav_order_by_combo.addItem('SNR')
        self.nav_order_by_combo.addItem('R value')
        self.nav_order_by_combo.addItem('CNN')
        self.nav_order_by_combo.setToolTip('Stepping components according to this metric. Limit selection to component group')
        left_layout.addWidget(self.nav_order_by_combo)
        
        nav_layout = QHBoxLayout()
        self.nav_prev_button = QPushButton("Down")
        self.nav_prev_button.setToolTip('Go to previous/smaller component according to the selected metrics. Key: Down Arrow')
        self.nav_prev_button.setFixedWidth(45)
        self.nav_prev_button.setContentsMargins(0, 0, 0, 0)        
        self.nav_next_button = QPushButton("Up")
        self.nav_next_button.setToolTip('Go to next/larger component according to the selected metrics. Key: Up Arrow')
        self.nav_next_button.setFixedWidth(45)
        self.nav_next_button.setContentsMargins(0, 0, 0, 0)        
        nav_layout.addWidget(self.nav_prev_button)
        nav_layout.addWidget(self.nav_next_button)
        left_layout.addLayout(nav_layout)

        head_label=QLabel('Plot:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 10px;')
        left_layout.addWidget(head_label)
        
        arr1 = QHBoxLayout()
        arr1.setContentsMargins(0, 0, 0, 0)
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Plain)
        vline.setLineWidth(1)  # thinner line
        vline.setStyleSheet("background-color: blue;")
        arr1.addWidget(vline)
        arr1_but=QVBoxLayout()
        arr1_but.setContentsMargins(0, 0, 0, 0)
        arr1_but.setSpacing(0)
        arr1_but.setAlignment(Qt.AlignTop)
        arr1.addLayout(arr1_but)
        left_layout.addLayout(arr1)
        
        self.array_selector = QComboBox()
        self.array_selector.setFixedWidth(90)
        self.array_selector.setToolTip('Select temporal array to plot')
        arr1_but.addWidget(self.array_selector)
        
        self.rlavr_spinbox = QSpinBox()
        self.rlavr_spinbox.setMinimum(0)
        self.rlavr_spinbox.setMaximum(100)
        self.rlavr_spinbox.setValue(0)
        self.rlavr_spinbox.setToolTip('Sets running average Gauss kernel on the displayed data')
        self.rlavr_spinbox.setFixedWidth(90)
        self.rlavr_spinbox.setPrefix('Avr: ')
        arr1_but.addWidget(self.rlavr_spinbox)
        
        self.range_to_all_button = QPushButton('Y fit all')
        self.range_to_all_button.setToolTip("Set range of Y axis to match all component's tarces.")
        arr1_but.addWidget(self.range_to_all_button)
        
        arr1 = QHBoxLayout()
        arr1.setContentsMargins(0, 4, 0, 4)
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Plain)
        vline.setLineWidth(0)  # thinner line
        vline.setStyleSheet("background-color: red;")
        arr1.addWidget(vline)
        arr1_but=QVBoxLayout()
        #arr1_but.setContentsMargins(0, 0, 0, 0)
        arr1_but.setSpacing(0)
        arr1_but.setAlignment(Qt.AlignTop)
        arr1.addLayout(arr1_but)
        left_layout.addLayout(arr1)
        
        self.array_selector2 = QComboBox()
        self.array_selector2.setFixedWidth(90)
        self.array_selector2.addItem('-')
        self.array_selector2.setToolTip('Select temporal array to plot on the right axis')
        arr1_but.addWidget(self.array_selector2)
        
        self.rlavr_spinbox2 = QSpinBox()
        self.rlavr_spinbox2.setMinimum(0)
        self.rlavr_spinbox2.setMaximum(100)
        self.rlavr_spinbox2.setValue(0)
        self.rlavr_spinbox2.setToolTip('Sets running average Gauss kernel on the displayed data (right axis)')
        self.rlavr_spinbox2.setFixedWidth(90)
        self.rlavr_spinbox2.setPrefix('Avr: ')
        arr1_but.addWidget(self.rlavr_spinbox2)
        
        self.range_to_all_button2 = QPushButton('Y fit all')
        self.range_to_all_button2.setToolTip("Set range of Y axis to match all component's tarces.")
        arr1_but.addWidget(self.range_to_all_button2)
        
        self.temporal_zoom_button = QPushButton('Center')
        self.temporal_zoom_button.setToolTip('Center view on the largest activity peak (max of C); scroll on the horizontal axis to adjust zoom')
        left_layout.addWidget(self.temporal_zoom_button)
        self.temporal_zoom_auto_checkbox = QCheckBox('Auto')
        self.temporal_zoom_auto_checkbox.setStyleSheet('margin-left: 8px;')   
        self.temporal_zoom_auto_checkbox.setToolTip('Automatically centers view on the largest activity peak (max of C); scroll on the horizontal axis to adjust zoom')
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
        left_layout.setContentsMargins(0, 0,5,5)
               
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Makes the scroll area resize with the window
        #scroll_area.setStyleSheet('background-color: red;')
        #scroll_content.setStyleSheet('background-color: yellow;')
        #head_label.setStyleSheet('background-color: green;')
        #self.temporal_zoom_auto_checkbox.setStyleSheet('background-color: green;')
        #head_label.setStyleSheet('background-color: blue;')
        ##left_layout.setContentsMargins(6, 1, 10, 1)
        my_layot.setContentsMargins(6, 1, 6, 1)
    
        # Set scroll content as scroll area widget
        scroll_area.setWidget(scroll_content)
        my_layot.addWidget(scroll_area)
        
        right_layout=QVBoxLayout()
        ##left_layout.setContentsMargins(0, 1, 9, 9)
       
        right_layout.addWidget(self.temporal_view, stretch=1) 
        time_layout=QHBoxLayout()
        
       
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
        self.nav_order_by_combo.currentTextChanged.connect(self.mainwindow.set_nav_order_by)
        self.nav_next_button.clicked.connect(lambda: self.mainwindow.set_nav_component_pressed('next'))
        self.nav_prev_button.clicked.connect(lambda: self.mainwindow.set_nav_component_pressed('prev'))
        self.rlavr_spinbox.valueChanged.connect(self.on_rlavr_spinbox_changed)
        self.array_selector.currentTextChanged.connect(self.on_array_selector_changed)  
        self.rlavr_spinbox2.valueChanged.connect(self.on_rlavr_spinbox_changed)
        self.array_selector2.currentTextChanged.connect(self.on_array_selector_changed)  
        self.temporal_zoom_button.clicked.connect(self.on_temporal_zoom)
        self.temporal_zoom_auto_checkbox.stateChanged.connect(self.on_temporal_zoom_auto_changed)
        self.range_to_all_button.clicked.connect(lambda: self.on_range_to_all_clicked(right_axis=False))
        self.range_to_all_button2.clicked.connect(lambda: self.on_range_to_all_clicked(right_axis=True))

        self.temporal_view.sceneObj.sigMouseClicked.connect(self.on_mouseClickEvent)
        self.temporal_view.getPlotItem().sigXRangeChanged.connect(self.on_range_changed)

    def on_range_to_all_clicked(self, right_axis):
        #sets left or right axis range to match all component's traces
        if right_axis:
            array_text=self.array_selector2.currentText()
            avr=self.rlavr_spinbox2.value()
            if array_text=='-':
                return
        else:
            array_text=self.array_selector.currentText()
            avr=self.rlavr_spinbox.value()
        ymin=np.inf
        ymax=-np.inf
        for index in range(self.mainwindow.numcomps):
            y, _ =self._get_trace(index, array_text, avr)
            ymin=min(ymin, np.min(y))
            ymax=max(ymax, np.max(y))
        if right_axis:
            self.temporal_view.RightViewBox.setRange(yRange= (ymin, ymax), padding=0.0)
        else:
            self.temporal_view.getViewBox().setRange(yRange= (ymin, ymax), padding=0.0)
        
        
    def on_time_widget_changed(self, value, source):
        #print(f'{inspect.stack()[1][3]} called with value {value}{source}')
        self.mainwindow.set_selected_frame(value)
 
    def on_time_window_changed(self, value):
        self.mainwindow.set_selected_frame(None, window=value)
        
    def on_rlavr_spinbox_changed(self):
        self.update_temporal_view()
        
    def update_component_spinbox(self, value):
        #print(f'update_component_spinbox called with value {value}')
        self.component_spinbox.blockSignals(True)
        if self.mainwindow.cnm is None:
            self.component_spinbox.setEnabled(False)
        else:
            self.component_spinbox.setEnabled(True)
            self.component_spinbox.setMaximum(self.mainwindow.numcomps - 1)
        if self.component_spinbox.value() != value:
            self.component_spinbox.setValue(value)
        self.component_spinbox.blockSignals(False)
        
    def update_nav_order_by(self):
        #print(f'update_nav_order_by called')
        value=self.mainwindow.order
        cnm=self.mainwindow.cnm
        self.nav_order_by_combo.blockSignals(True)
        if cnm is None:
            self.nav_order_by_combo.setEnabled(False)
        elif cnm.estimates.idx_components is None:
            self.nav_order_by_combo.setEnabled(False)
            self.nav_order_by_combo.setCurrentText('index (All)')
        else:
            self.nav_order_by_combo.setEnabled(True)
            self.nav_order_by_combo.setCurrentText(value)
        self.nav_order_by_combo.blockSignals(False)

    def update_nav_button_enabled(self):
        indexes=self.mainwindow.order_indexes 
        if self.mainwindow.cnm is None or indexes is None or len(indexes) == 0:
            self.nav_prev_button.setEnabled(False)
            self.nav_next_button.setEnabled(False)
            return
              
        current_component=self.mainwindow.selected_component
        idx = (np.abs(indexes - current_component)).argmin()
        if idx == 0:
            self.nav_prev_button.setEnabled(False)
        else:
            self.nav_prev_button.setEnabled(True)
        if idx == len(indexes) - 1:
            self.nav_next_button.setEnabled(False)
        else:
            self.nav_next_button.setEnabled(True)

    def on_component_spinbox_changed(self, value):
        self.mainwindow.set_selected_component(value, 'spinbox')
    
    def on_temporal_zoom_auto_changed(self, state):
        if self.temporal_zoom_auto_checkbox.isChecked():
            self.mainwindow.perform_temporal_zoom()
    
    def on_temporal_zoom(self):
        self.mainwindow.perform_temporal_zoom()
  
    def on_array_selector_changed(self, text):
        self.update_temporal_view()
        
    def update_array_selector(self, value=None, right_axis=False):
        if right_axis:
            combo=self.array_selector2
        else:
            combo=self.array_selector
        cnm=self.mainwindow.cnm
        if cnm is None:
            combo.setEnabled(False)
            return
        selectable_array_names=[]
        if right_axis:
            selectable_array_names.append('-')
        possible_array_names = [ 'F_dff', 'C',  'S', 'YrA', 'R', 'noisyC', 'C_on']
        for array_name in possible_array_names:
            temparr = getattr(cnm.estimates, array_name)
            if (temparr is not None) :
                selectable_array_names.append(array_name)
        if self.mainwindow.orig_trace_array is not None:
            selectable_array_names.append('Data')
        if self.mainwindow.orig_trace_array_neuropil is not None:
            selectable_array_names.append('Data neuropil')
        if value is None:
            previous_selected_array = combo.currentText()
        else:
            previous_selected_array = value 
            
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(selectable_array_names)
        if previous_selected_array in selectable_array_names:
            combo.setCurrentText(previous_selected_array)
        else:
            combo.setCurrentIndex(0)
        #print(f'Update_array_selector: right:{right_axis}, prev:{previous_selected_array}, curr:{combo.currentText()}, Selectable array names:', selectable_array_names)
        combo.setEnabled(True) 
        combo.blockSignals(False)      
        
    
    def recreate_temporal_view(self):
        #.update_all esetén
        self.temporal_zoom_auto_checkbox.setEnabled(self.mainwindow.cnm is not None)
        self.temporal_zoom_button.setEnabled(self.mainwindow.cnm is not None)

        #self.temporal_zoom_auto_checkbox.setChecked(False)
        
        self.temporal_view.clear()
        self.temporal_view.RightViewBox.clear()
        
        if self.mainwindow.cnm is None:
            text='No data loaded yet.\nOpen CaImAn HDF5 file using the file menu.'
            text = pg.TextItem(text=text, anchor=(0.5, 0.5), color='k')
            self.temporal_view.addItem(text)
            self.temporal_view.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
            self.temporal_view.getPlotItem().showGrid(False)
            self.temporal_view.getPlotItem().showAxes(False)
            self.temporal_view.setBackground(QColor(200, 200, 210, 127))
            self.temporal_view.getPlotItem().setMenuEnabled(False)
            self.update_nav_button_enabled()
            return
        
        self.temporal_view.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
        self.temporal_view.setBackground(None)
        self.temporal_view.setDefaultPadding( 0.0 )
        self.temporal_view.getPlotItem().showGrid(x=True, y=True, alpha=0.3)
        self.temporal_view.getPlotItem().showAxes(True, showValues=(True, False, False, True))
        self.temporal_view.getPlotItem().setContentsMargins(0, 0, 10, 0)  # add margin to the right
        self.temporal_view.getPlotItem().setMenuEnabled(True)
        
        self.temporal_line1=self.temporal_view.plot(x=[0,self.mainwindow.num_frames-1 ], y=[0,0], pen=pg.mkPen('r', width=1), name='component')
        self.temporal_line2 = pg.PlotCurveItem(x=[0,self.mainwindow.num_frames-1 ], y=[0,0], pen=pg.mkPen('g', width=1), name='component2')
        self.temporal_view.RightViewBox.addItem(self.temporal_line2)
        
        self.temporal_marker_P = pg.InfiniteLine(pos=2, angle=90, movable=False, pen=pg.mkPen('darkgrey', width=2))
        self.temporal_view.addItem(self.temporal_marker_P)
        self.temporal_marker_N = pg.InfiniteLine(pos=2, angle=90, movable=False, pen=pg.mkPen('darkgrey', width=2))
        self.temporal_view.addItem(self.temporal_marker_N)
        self.temporal_marker = pg.InfiniteLine(pos=2, angle=90, movable=True, pen=pg.mkPen('m', width=2), hoverPen=pg.mkPen('m', width=4))
        self.temporal_view.addItem(self.temporal_marker)
        self.temporal_marker.sigPositionChangeFinished.connect(lambda line=self.temporal_marker: line.setPen(pg.mkPen('m', width=2)))
        self.temporal_marker.sigDragged.connect(self.on_temporal_marker_dragged)

        self.temporal_view.setLabel('bottom', 'Frame Number')
        
        self.mainwindow.scatter_widget.update_totals()
        self.update_temporal_view()
        

    def _get_trace(self, index, array_text, rlavr_width):    
            if array_text == 'C':
                ctitle=f'Temporal Component ({index})'
            elif array_text == 'F_dff':
                ctitle=f'Detrended \u0394F/F ({index})'
            elif array_text == 'YrA':
                ctitle=f'Residual ({index})'
            elif array_text == 'S':
                ctitle=f'Spike count estimate ({index})'
            elif array_text== 'Data':
                ctitle=f'Mean fluorescence under contour ({index})'
            elif array_text== 'Data neuropil':
                ctitle=f'Mean fluorescence outside contours'
            else:
                ctitle=f'Temporal Component ({array_text}, {index})'
            
            #array_names = ['C', 'f', 'YrA', 'F_dff', 'R', 'S', 'noisyC', 'C_on']
            if array_text == 'Data':
                y=self.mainwindow.orig_trace_array[index, :]
            elif array_text== 'Data neuropil':
                y=self.mainwindow.orig_trace_array_neuropil
            else:
                y=getattr(self.mainwindow.cnm.estimates, array_text)[index, :]
                if len(y) > self.mainwindow.num_frames:
                    y = y[-self.mainwindow.num_frames:] # in case of noisyC or C_on   
            
            if rlavr_width>0:
                kernel = gaussian(2*rlavr_width+1, rlavr_width)
                kernel = kernel / np.sum(kernel)
                y = np.convolve(y, kernel, mode='same')
            return y, ctitle
        
    def update_temporal_view(self):
                        
        tooltips={'C': 'Temporal traces', 
            'F_dff': '\u0394F/F normalized activity trace', 
            'S': 'Deconvolved neural activity trace', 
            'YrA': 'Trace residuals', 
            'R': 'Trace residuals', 
            'noisyC': 'Temporal traces (including residuals plus background)', 
            'C_on': '?', 
            'Data': 'Original fluorescence trace calculated from contour polygons', 
            'Data neuropil': 'Original fluorescence trace neuropil mean', 
            '-': 'No right axis displayed'}
        
        index = self.mainwindow.selected_component
        array_text=self.array_selector.currentText()
        self.array_selector.setToolTip(tooltips[array_text]) 
        
        array_text2=self.array_selector2.currentText()
        self.array_selector2.setToolTip(tooltips[array_text2] + ' (right axis)') 

        y, ctitle=self._get_trace( index, array_text, self.rlavr_spinbox.value())
        self.temporal_view.setLabel('left', f'{array_text} value')       
        self.temporal_line1.setData(x=np.arange(len(y)), y=y, pen=pg.mkPen(color='b', width=2), name=f'data {array_text} {index}')
        if array_text2 != '-':
            y2, ctitle2=self._get_trace( index, array_text2, self.rlavr_spinbox2.value())
            self.temporal_line2.setData(x=np.arange(len(y2)), y=y2, pen=pg.mkPen(color='r', width=2), name=f'data(2) {array_text2} {index}')
            self.temporal_line2.setVisible(True)
            self.temporal_view.getPlotItem().setTitle(ctitle + ' - ' + ctitle2)
            self.temporal_view.setLabel('right', f'{array_text2} value')
            self.temporal_view.getAxis('right').setStyle(showValues=True)
            self.range_to_all_button2.setEnabled(True)
            self.rlavr_spinbox2.setEnabled(True) 
        else:
            self.temporal_view.getPlotItem().setTitle(ctitle)
            self.temporal_view.getAxis('right').setStyle(showValues=False)
            self.temporal_line2.setVisible(False)
            self.temporal_view.getAxis('right').setLabel('')
            self.range_to_all_button2.setEnabled(False)
            self.rlavr_spinbox2.setEnabled(False) 
        
        cnme=self.mainwindow.cnm.estimates
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
            
        if self.temporal_zoom_auto_checkbox.isChecked():
            self.mainwindow.perform_temporal_zoom()
        else:
            self.update_time_selector_line()
            self.update_temporal_widget()
        
    def on_temporal_marker_dragged(self, line):
        #dragging the time-line jogs the selected frame
        line.setPen(pg.mkPen('m', width=4))
        self.mainwindow.set_selected_frame(int(line.value()))
     
    def on_mouseClickEvent(self, event):
        #centering t, the selected frame, thus x axis on the location of the double click
        event.accept()
        if event.double():
            scenepos=event.scenePos()
            #print(repr(event))
            axpos= self.temporal_view.getViewBox().mapSceneToView(scenepos)
            self.mainwindow.set_selected_frame(int(axpos.x()))
            #print('mouseClickEvent setting to: ',int(axpos.x()), '  got:', self.mainwindow.selected_frame)
        
    def on_range_changed(self, viewbox, ev):
        #interacting with axis ranges sets the selected frame to the center
        range=viewbox.viewRange()[0]
        center=round((range[0]+range[1])/2)
        self.mainwindow.set_selected_frame(center)
         
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
        if not hasattr(self, 'temporal_marker'):
            return
        value=self.mainwindow.selected_frame
        w=self.mainwindow.frame_window
        tmin=value-w if value-w>0 else 0
        tmax=value+w if value+w<self.mainwindow.num_frames-1 else self.mainwindow.num_frames-1
        self.temporal_marker.setValue(value)
        self.temporal_marker_N.setValue(tmin)
        self.temporal_marker_P.setValue(tmax)
        
        xrange=self.temporal_view.viewRange()[0]
        xspan=xrange[1]-xrange[0]
        xrange=(value-xspan/2, value+xspan/2)
        self.temporal_view.getViewBox().setRange(xRange=xrange)
        

class ScatterWidget(QWidget):
    def __init__(self, main_window: MainWindow, parent=None):
        super().__init__(parent)
        self.mainwindow = main_window
        
        my_layout = QHBoxLayout(self)
        
        # Create a container widget inside scroll area
        scroll_content = QWidget()
        threshold_layout = QVBoxLayout(scroll_content)
        
        # Bottom layout for Spatial and Parameters plots
        #threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(0)
        threshold_layout.setAlignment(Qt.AlignTop)
        
        head_label=QLabel('Assignment:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 0px;')
        threshold_layout.addWidget(head_label)
        # Create a layout for the toggle buttons
        toggle_button_layout = QHBoxLayout()
        toggle_button_layout.setSpacing(0)
        good_toggle_button = QPushButton('Good')
        good_toggle_button.setFixedWidth(45)
        good_toggle_button.setCheckable(True)
        good_toggle_button.setContentsMargins(0, 0, 0, 0)
        good_toggle_button.setToolTip('Accept component manually, assign as good. Keyboard: "g"')
        #good_toggle_button.setStyleSheet('background-color: white; color: green;')
        toggle_button_layout.addWidget(good_toggle_button)
        self.good_toggle_button = good_toggle_button
        bad_toggle_button = QPushButton('Bad')
        bad_toggle_button.setFixedWidth(45)
        bad_toggle_button.setContentsMargins(0, 0, 0, 0)
        bad_toggle_button.setCheckable(True)
        bad_toggle_button.setStyleSheet('background-color: white; color: red;')
        bad_toggle_button.setToolTip('Reject component manually, assign as bad. Keyboard: "b"')
        toggle_button_layout.addWidget(bad_toggle_button)
        self.bad_toggle_button = bad_toggle_button
        # Add the toggle button layout to the left layout
        threshold_layout.addLayout(toggle_button_layout)        
        self.good_toggle_button.clicked.connect(lambda: self.mainwindow.set_component_assignment_manually('Good'))
        self.bad_toggle_button.clicked.connect(lambda: self.mainwindow.set_component_assignment_manually('Bad'))
        
        head_label=QLabel('Thresholds:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 0px;')
        threshold_layout.addWidget(head_label)
        self.evaluate_button = QPushButton('Evaluate')
        self.evaluate_button.setToolTip('Accept or reject components based on these threshold values (filter_components())')
        threshold_layout.addWidget(self.evaluate_button)
        
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

        head_label=QLabel('Totals:')
        head_label.setStyleSheet('font-weight: bold; margin-top: 10px;')
        threshold_layout.addWidget(head_label)
        self.total_label = QLabel('    Total: --')
        threshold_layout.addWidget(self.total_label)
        self.good_label = QLabel('    Good: --')
        threshold_layout.addWidget(self.good_label)
        self.bad_label = QLabel('    Bad: --')
        threshold_layout.addWidget(self.bad_label)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Makes the scroll area resize with the window
        scroll_area.setFixedWidth(100)
        #scroll_area.setStyleSheet('background-color: red;')
        #scroll_content.setStyleSheet('background-color: yellow;')
        #head_label.setStyleSheet('background-color: green;')
        #self.temporal_zoom_auto_checkbox.setStyleSheet('background-color: green;')
        #head_label.setStyleSheet('background-color: blue;')
        my_layout.setContentsMargins(5, 5, 10, 1)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
    
        # Set scroll content as scroll area widget
        scroll_area.setWidget(scroll_content)
        my_layout.addWidget(scroll_area)

        # Matplotlib canvas for scatter plot ---
        self.plt_figure = Figure()
        self.plt_canvas = FigureCanvasQTAgg(self.plt_figure)
        self.plt_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        my_layout.addWidget(self.plt_canvas)
         
        self.setLayout(my_layout)
        
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
    
    def update_component_assignment_buttons(self):
        self.good_toggle_button.setEnabled(self.mainwindow.cnm is not None and self.mainwindow.cnm.estimates.idx_components is not None)
        self.bad_toggle_button.setEnabled(self.mainwindow.cnm is not None and self.mainwindow.cnm.estimates.idx_components is not None)

        if self.mainwindow.cnm is None:
            return
        
        index=self.mainwindow.selected_component
        cnme=self.mainwindow.cnm.estimates
        self.good_toggle_button.blockSignals(True)
        self.bad_toggle_button.blockSignals(True)
        if cnme.idx_components is not None:
            if index in cnme.idx_components:
                self.good_toggle_button.setChecked(True)
                self.bad_toggle_button.setChecked(False)                
            else:
                self.good_toggle_button.setChecked(False)
                self.bad_toggle_button.setChecked(True)
        else:
            self.good_toggle_button.setChecked(False)
            self.good_toggle_button.setEnabled(False)
        self.bad_toggle_button.blockSignals(False)
        self.good_toggle_button.blockSignals(False)
        
        
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
  
    def on_scatter_point_clicked(self, event):
        index=event.ind[0]
        #print(f'Point clicked: {index}')
        self.mainwindow.set_selected_component(index, 'scatter')

    def on_scatter_hover(self, event):
        setvis=False
        if event.inaxes == self.plt_ax:
            cont, ind = self.plt_scatter.contains(event)
            if cont:
                index=ind["ind"][0]
                text=f"Component {index}"
                cnme=self.mainwindow.cnm.estimates
                if cnme.r_values is not None:
                    text+=f'\nRval={cnme.r_values[index]:.2f}'
                if cnme.SNR_comp is not None:
                    text+=f'\nSNR={cnme.SNR_comp[index]:.2f}'
                if cnme.cnn_preds is not None:
                    text+=f'\nCNN={cnme.cnn_preds[index]:.2f}'
                if cnme.idx_components is not None:
                    text+=f'\n{"Good" if index in cnme.idx_components else "Bad"}'
                self.plt_annot.set_text(text)
                self.plt_annot.set_visible(True)
                
                xvec = cnme.cnn_preds
                if xvec is None:
                    return
                yvec = cnme.r_values
                zvec = np.log10(cnme.SNR_comp)
                zvec = np.where(np.isfinite(zvec), zvec, 0)
                xlim=self.plt_ax.get_xlim()
                ylim=self.plt_ax.get_ylim()
                zlim=self.plt_ax.get_zlim()
                cnme=self.mainwindow.cnm.estimates
                x, y, z = xvec[ind["ind"][0]], yvec[ind["ind"][0]], zvec[ind["ind"][0]]
                
                x=[xlim[0], xlim[1],np.nan, x, x, np.nan,x,x]
                y=[y,y, np.nan,ylim[0], ylim[1], np.nan,y,y]
                z=[z, z, np.nan,z,z,np.nan,zlim[0], zlim[1]]
                
                self.plt_hover_marker._verts3d = (x, y, z)
                self.plt_hover_marker.set_visible(True)
                self.plt_canvas.draw_idle()
                setvis=True
        if not setvis and self.plt_annot.get_visible():
            self.plt_annot.set_visible(False)
            self.plt_hover_marker.set_visible(False)
            self.plt_canvas.draw_idle()
        
        
    def recreate_scatterplot(self):
        if self.mainwindow.cnm is None or self.mainwindow.cnm.estimates.r_values is None:
            self.plt_figure.clf()
            text='No evaluated components. Open data array to compute component metrics.'
            self.plt_ax = self.plt_figure.add_subplot(111)
            self.plt_ax.text(0.5, 0.5, text, transform=self.plt_ax.transAxes, ha='center', va='center', size=7.8, color='k')
            self.plt_figure.patch.set_facecolor((200.0/255, 200.0/255, 210.0/255, 127.0/255))
            self.plt_ax.axis('off')
            self.plt_canvas.draw()
            return
    
        plt.rcParams.update({
                'font.size': 10,           # default text size
                'axes.labelsize': 8,      # axis label size
                'axes.titlesize': 8,      # title size
                'xtick.labelsize': 8,      # x-axis tick label size
                'ytick.labelsize': 8,      # y-axis tick label size
                'legend.fontsize': 8,      # legend font size
                'figure.titlesize': 8,    # figure title size
                'axes.labelpad': 4,        # space between label and axis
                'xtick.major.size': 4,     # length of major ticks
                'ytick.major.size': 4,
                'xtick.minor.size': 2,     # length of minor ticks
                'ytick.minor.size': 2,
            })
        plt.rcParams.update({
                'figure.subplot.left': 0.05,   # adjust left margin (default is around 0.125)
                'figure.subplot.right': 0.95,  # adjust right margin (default is around 0.9)
                'figure.subplot.bottom': 0.05, # adjust bottom margin (default is around 0.1)
                'figure.subplot.top': 0.95,    # adjust top margin (default is around 0.88)
            })
        self.plt_figure.tight_layout()
        # Clear the figure, removing all subplots
        self.plt_figure.clf()
        self.plt_figure.patch.set_facecolor('w')
        self.plt_ax = self.plt_figure.add_subplot(111, projection='3d')
        self.plt_ax.set_box_aspect((1,1,1),  zoom=1.0)
        
        self.plt_canvas.mpl_connect("motion_notify_event", self.on_scatter_hover)
        self.plt_canvas.mpl_connect("pick_event", self.on_scatter_point_clicked)
        
        self.update_scatterplot()
        
    def update_scatterplot(self):
        #clears figure and creates plots inside, updates scatter's coords and color
        self.plt_ax.cla()
        self.plt_ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos=None: "{:.0f}".format(round(10**val)) if val >= 1 else "{:.1g}".format(10**val)))
        
        stubs = np.array([1,2,3,5])
        allstubs = np.concatenate([stubs * 10 ** m for m in np.arange(-5.0, 6.0, dtype=float)])
        self.plt_ax.zaxis.set_major_locator(plt.FixedLocator(np.log10(allstubs)))
        
        stubs = np.array([1,2,3,4,5,6,7,8,9])
        allstubs = np.concatenate([stubs * 10 ** m for m in np.arange(-5.0, 6.0, dtype=float)])
        self.plt_ax.zaxis.set_minor_locator(plt.FixedLocator(np.log10(allstubs)))
        
        self.plt_ax.set_xlabel("CNN prediction")
        self.plt_ax.set_ylabel("R value")
        self.plt_ax.set_zlabel("SNR")
        num_components = self.mainwindow.numcomps
        cnme=self.mainwindow.cnm.estimates
        if cnme.idx_components is not None:
            colors = ['green' if i in cnme.idx_components else 'red' for i in range(num_components)]
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, num_components))
        self.colors=colors
        self.x = cnme.cnn_preds
        self.y = cnme.r_values
        z = np.log10(cnme.SNR_comp)
        self.z = np.where(np.isfinite(z), z, 0)
        self.s=self.z*0+10
        self.alpha=self.z*0+0.3
        self.plt_scatter = self.plt_ax.scatter(self.x, self.y, self.z, c=self.colors, s=self.s, alpha=self.alpha, picker=5)
        
        self.plt_annot = self.plt_figure.text(0.00, 1.00, "",  va='top', ha='left')
        self.plt_annot.set_fontsize(8)
        self.plt_annot.set_visible(False)
        
        self.plt_selected_scatterpoint = self.plt_ax.scatter(0,0, 0, marker='o', c='magenta', s=30, alpha=1)
        self.update_selected_component_on_scatterplot(self.mainwindow.selected_component)
        
        self.plt_hover_marker = self.plt_ax.plot([1], [1], [1], color='k', linewidth=0.5)[0]
        self.plt_hover_marker.set_visible(False)
        
        self.threshold_lines = {}
        if cnme.idx_components is not None:
            lines = self.construct_threshold_gridline_data()
            for name, line in lines.items():
                ln, = self.plt_ax.plot(1, 1, 1, linestyle='dashed',
                                    color=line['color'], linewidth=1, label=name)
                self.threshold_lines[name]=ln

            self.update_threshold_lines_on_scatterplot()
            
        self.plt_ax.set_xlim(xmin=0, xmax=1) #cnn
        self.plt_ax.set_ylim(ymin=0, ymax=1) #rval
        self.plt_ax.set_zlim(bottom=np.nanmin(z), top=np.nanmax(z)) #SNR
        self.plt_canvas.draw()

        
    def construct_threshold_gridline_data(self):
        cnm=self.mainwindow.cnm
        if cnm is None:
            return {}
        x = cnm.estimates.cnn_preds
        y = cnm.estimates.r_values
        z = cnm.estimates.SNR_comp
        z = np.where(np.isfinite(z), z, 1)
        lines = {
                'min_cnn_thr': {'x': [float(cnm.params.quality['min_cnn_thr'])]*5, 'y': [float(min(y)), float(max(y)), float(max(y)), float(min(y)), float(min(y))], 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'green'},
                'cnn_lowest': {'x': [float(cnm.params.quality['cnn_lowest'])]*5, 'y': [float(min(y)), float(max(y)), float(max(y)), float(min(y)), float(min(y))], 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'green'},
                'rval_thr': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(cnm.params.quality['rval_thr'])]*5, 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'blue'},
                'rval_lowest': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(cnm.params.quality['rval_lowest'])]*5, 'z': [float(min(z)), float(min(z)), float(max(z)), float(max(z)), float(min(z))], 'color': 'blue'},
                'min_SNR': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(min(y)), float(min(y)), float(max(y)), float(max(y)), float(min(y))], 'z': [float(cnm.params.quality['min_SNR'])]*5, 'color': 'magenta'},
                'SNR_lowest': {'x': [float(min(x)), float(max(x)), float(max(x)), float(min(x)), float(min(x))], 'y': [float(min(y)), float(min(y)), float(max(y)), float(max(y)), float(min(y))], 'z': [float(cnm.params.quality['SNR_lowest'])]*5, 'color': 'magenta'},
            }
        return lines
    
    def update_threshold_lines_on_scatterplot(self):
        if self.mainwindow.cnm.estimates.idx_components is None:
            return
        lines = self.construct_threshold_gridline_data()
        for name, lineobj in self.threshold_lines.items():
            line=lines[name]
            lineobj.set_data(line['x'], line['y'])
            lineobj.set_3d_properties(np.log10(line['z']))
            lineobj.set_color(line['color'])
        self.plt_canvas.draw()
               
    def update_selected_component_on_scatterplot(self, index):
        cnm=self.mainwindow.cnm
        if cnm is None or cnm.estimates.r_values is None:
            return
        #print(f'Updating selected component on scatter plot: {index}')
        x_val = float(cnm.estimates.cnn_preds[index])
        y_val = float(cnm.estimates.r_values[index])
        z_val = np.log10(float(cnm.estimates.SNR_comp[index]))
        z_val = z_val if np.isfinite(z_val) else 0.0
        if cnm.estimates.idx_components is not None and index in cnm.estimates.idx_components:
            color = 'blue'#'green'
        else:
            color = 'magenta'     
        colors=self.colors.copy()
        colors[index]=color
        s=self.s.copy()
        s[index]=30
        #alpha=self.alpha.copy()
        #alpha[index]=1.0
        
        self.plt_selected_scatterpoint._offsets3d = ([x_val], [y_val], [z_val])
        self.plt_selected_scatterpoint.set_color(color)
        
        self.plt_scatter.set_color(colors)
        self.plt_scatter.set_sizes(s)
        #self.plt_scatter.set_alpha(alpha)
        #funny things happen here. plt_selected_scatterpoint is alway shown behind the scatter, often covered.
        #setting alpha and having alpha[index]=1 on the scatter makes 'random' points of the scatter nontransparent also depending of the display angle of the axes.
        #So we set alpha of the scatter to 0.3 and also use a separate point for the selected component - but this is an incomplete solution. 
        # better would be to draw something as selection mark whish is always on top. maybe we need matplotlib buxfix.
        

        self.plt_canvas.draw()
        #print(f'Updating selected component on scatter plot: {index}')
        
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
        
        self.spatial_avr_spinbox = QSpinBox()
        self.spatial_avr_spinbox.setMinimum(0)
        self.spatial_avr_spinbox.setMaximum(100)
        self.spatial_avr_spinbox.setValue(0)
        self.spatial_avr_spinbox.setToolTip('Sets spatial average Gauss kernel on the displayed images')
        self.spatial_avr_spinbox.setPrefix('Avr: ')
        left_layout.addWidget(self.spatial_avr_spinbox)
        
        self.spatial_zoom_button = QPushButton('Zoom')
        self.spatial_zoom_button.setToolTip('Center view on selected component, with zoom corresponding to neuron diameter')
        left_layout.addWidget(self.spatial_zoom_button)
        self.spatial_zoom_auto_checkbox = QCheckBox('Auto')
        self.spatial_zoom_auto_checkbox.setStyleSheet('margin-left: 8px;')        
        self.spatial_zoom_auto_checkbox.setToolTip('Automatically centers view on selected component, with zoom corresponding to neuron diameter')
        left_layout.addWidget(self.spatial_zoom_auto_checkbox)
     
        self.colorbar_reset_button = QPushButton('Reset Colorbar')
        self.colorbar_reset_button.setToolTip('Reset colorbar to min and max of displayed data')
        self.colorbar_reset_button.setEnabled(False)
        left_layout.addWidget(self.colorbar_reset_button) # Colorbar reset button initialized

        head_label=QLabel('Contours:')
        head_label.setStyleSheet('font-weight: bold;')
        left_layout.addWidget(head_label)
        
        self.contour_combo = QComboBox()
        self.contour_combo.addItem('--')
        self.contour_combo.setToolTip('Select contour groups to draw')
        left_layout.addWidget(self.contour_combo)
        
        self.data_info_label=QLabel('')
        self.data_info_label.setWordWrap(True)
        self.data_info_label.setFixedWidth(95)
        self.data_info_label.setFixedHeight(150)
        self.data_info_label.setAlignment(Qt.AlignTop)
        #self.data_info_label.setStyleSheet('background-color: yellow;')
        left_layout.addWidget(self.data_info_label)
        
        my_layot.addLayout(left_layout)
        my_layot.addWidget(self.spatial_view, stretch=1)        
        
        self.setLayout(my_layot)        
        self.ctime=0
        #event hadlers
        self.channel_combo.currentIndexChanged.connect(self.on_channel_combo_changed)
        self.spatial_zoom_button.clicked.connect(self.on_spatial_zoom)
        self.spatial_zoom_auto_checkbox.stateChanged.connect(self.on_spatial_zoom_auto_changed)
        self.contour_combo.currentIndexChanged.connect(self.on_contour_combo_changed)
        self.spatial_avr_spinbox.valueChanged.connect(self.on_spatial_avr_spinbox_changed)
        self.spatial_view.sceneObj.sigMouseMoved.connect(self.on_mouseMoveEvent)
        self.colorbar_reset_button.clicked.connect(self.on_colorbar_reset_button_clicked)
 
    
    def on_mouseMoveEvent(self, event):
        # called on mouse move over spatial view
        if self.mainwindow.cnm is None:
            return
        axpos= self.spatial_view.getViewBox().mapSceneToView(event)
        im=self.spatial_image.image
        x=int(np.floor(axpos.x()))
        y=int(np.floor(axpos.y()))
        if x < 0 or x >= im.shape[0] or y < 0 or y >= im.shape[1]:
            if self.mainwindow.data_array is None: 
                tex=f'Open data array in file menu to enable more options...'
            else:
                tex=""
            self.data_info_label.setText(tex)
            return
        current_value=im[x, y]
        x_um=(x-im.shape[0]/2)*self.mainwindow.pixel_size[0]        
        y_um=(y-im.shape[1]/2)*self.mainwindow.pixel_size[1]
        if abs(x_um) >= 1000:
            x_um_str = f'{x_um:.0f}'
        elif abs(x_um) >= 100:
            x_um_str = f'{x_um:.1f}'
        else:
            x_um_str = f'{x_um:.2f}'
        if abs(y_um) >= 1000:
            y_um_str = f'{y_um:.0f}'
        elif abs(y_um) >= 100:
            y_um_str = f'{y_um:.1f}'
        else:
            y_um_str = f'{y_um:.2f}'
        if abs(current_value) >= 1000:
            current_value_str = f'{current_value:.0f}'
        elif abs(current_value) >= 100:
            current_value_str = f'{current_value:.1f}'
        elif abs(current_value) >= 10:
            current_value_str = f'{current_value:.2f}'
        elif abs(current_value) >= 0.01:
            current_value_str = f'{current_value:.3f}'
        else:
            current_value_str = f'{current_value:.3g}'
        tex=f'X: {x} ({x_um_str} μm)\n' +\
            f'Y: {y} ({y_um_str} μm)\n' +\
            f'Value: {current_value_str}\n' 
        
        array_text=self.channel_combo.currentText()
        if array_text == 'MaxResNone':
            idx=self.mainwindow.max_res_none_idx[x,y]
            tex+=f'Time idx: {idx}\n'
        elif array_text == 'MaxResGood':
            idx=self.mainwindow.max_res_good_idx[x,y]
            tex+=f'Time idx: {idx}\n'
        elif array_text == 'MaxResAll':
            idx=self.mainwindow.max_res_all_idx[x,y]
            tex+=f'Time idx: {idx}\n'            
            
        self.data_info_label.setText(tex)
        
        
    def on_contour_combo_changed(self, index):
        # called on change of contour selector combo box
        self.update_spatial_view()

    def on_spatial_zoom(self):
        self.perform_spatial_zoom_on_component(self.mainwindow.selected_component)

    def on_spatial_zoom_auto_changed(self, state):
        if self.spatial_zoom_auto_checkbox.isChecked():
            self.perform_spatial_zoom_on_component(self.mainwindow.selected_component)

    def on_spatial_avr_spinbox_changed(self):
        self.update_spatial_view_image()
        
    def on_channel_combo_changed(self, index):
        #called on change of channel selector combo box
        self.update_spatial_view(setLUT=True)

    def on_colorbar_reset_button_clicked(self):
        self.update_spatial_view_image(setLUT=True)
        
    def perform_spatial_zoom_on_component(self, index):
        zoomwindow=self.mainwindow.neuron_diam*1.5
        coord=self.mainwindow.component_centers[index,:]
        xrange=coord[0]-zoomwindow,coord[0]+zoomwindow
        yrange=coord[1]-zoomwindow,coord[1]+zoomwindow
        self.spatial_view.setRange(xRange=xrange, yRange=yrange, padding=0.0)
        
    def recreate_spatial_view(self):
        self.channel_combo.setEnabled(self.mainwindow.cnm is not None)
        self.data_info_label.setEnabled(self.mainwindow.cnm is not None)
        self.spatial_avr_spinbox.setEnabled(self.mainwindow.cnm is not None)
        
        if self.mainwindow.cnm is None:
            text='No data loaded yet'
            text = pg.TextItem(text=text, anchor=(0.5, 0.5), color='k')
            self.spatial_view.clear()
            self.spatial_view.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
            self.spatial_view.addItem(text)
            self.spatial_view.getPlotItem().showGrid(False)
            self.spatial_view.getPlotItem().showAxes(False)
            self.spatial_view.getPlotItem().setMenuEnabled(False)
            self.spatial_view.setBackground(QColor(200, 200, 210, 127))
            self.spatial_zoom_auto_checkbox.setEnabled(False)
            self.spatial_zoom_button.setEnabled(False)
            self.contour_combo.setEnabled(False)
            return
        
        #print('Rendering spatial view, contours...')
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
        plot_item.setMenuEnabled(True)
        for item in {'Transforms', 'Downsample', 'Average','Alpha',  'Points'}:
            plot_item.setContextMenuActionVisible(item, False)
        plot_item.invertY(True)
        
        if self.mainwindow.data_array is None: 
            tex=f'Open data array in file menu to enable more options...'
        else:
            tex=""
        self.data_info_label.setText(tex)

        # Configure axis tick lengths explicitly
        for side in ('top', 'right'):
            ax = plot_item.getAxis(side)
            ax.setStyle(tickLength=0)
        for side in ('left', 'bottom'):
            ax = plot_item.getAxis(side)
            ax.setStyle(tickLength=10)      
        self.spatial_view.setDefaultPadding( 0.0 )
        
        cnme=self.mainwindow.cnm.estimates
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
            if self.mainwindow.cnm.estimates.idx_components is not None:
                possible_array_text.append('Residuals (Good)')            
        possible_array_text.append('RCM')
        if self.mainwindow.cnm.estimates.idx_components is not None:
            possible_array_text.append('RCM (Good)')            
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
        if self.mainwindow.max_res_none is not None:
            possible_array_text.append('MaxResNone')
        if self.mainwindow.max_res_good is not None:
            possible_array_text.append('MaxResGood')
        if self.mainwindow.max_res_all is not None:
            possible_array_text.append('MaxResAll')
        
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

        ctitle=self.update_spatial_view_image(setLUT=setLUT)
                
        plot_item = self.spatial_view.getPlotItem()
        plot_item.setTitle(ctitle)

        if self.spatial_zoom_auto_checkbox.isChecked():
            self.perform_spatial_zoom_on_component(component_idx)
            
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
        clickarray=[True]*self.mainwindow.numcomps # we may disable clickability of some components here
        for idx_to_plot in range(self.mainwindow.numcomps):
            if idx_to_plot==component_idx:
                peny=self.selectedpen
            elif cnme.idx_components is None or idx_to_plot in cnme.idx_components:
                peny=self.goodpen
            else:    
                peny=self.badpen
            self.contur_items[idx_to_plot].setPen(peny)
            self.contur_items[idx_to_plot].setClickable(clickarray[idx_to_plot])                                       


    def get_spatial_view_image(self, tmin, tmax, add_vars_to_title=False):    
        array_text=self.channel_combo.currentText()
        component_idx = self.mainwindow.selected_component       
        
        if array_text == 'A':
            image_data = np.reshape(self.mainwindow.A_array[:, component_idx], self.mainwindow.dims) 
            ctitle=f'Spatial footprint of component {component_idx} (static)'
        elif array_text == 'Data':
            #print('display: elapsed off {:.2f}'.format((time.perf_counter()-self.ctime)))
            #self.ctime=time.perf_counter()
            res=self.mainwindow.data_array[:,tmin:tmax]
            res=np.mean(res, axis=1)

            image_data = res.reshape(self.mainwindow.dims)
            ctitle=f'Original data (movie)'
        elif array_text == 'RCM':
            res=np.dot(self.mainwindow.A_array[:, :] , self.mainwindow.cnm.estimates.C[:, tmin:tmax])
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
            ctitle=f'Reconstructed movie (A ⊗ C)'
        elif array_text == 'RCM (Good)':
            res=np.dot(self.mainwindow.A_array[:, self.mainwindow.cnm.estimates.idx_components] , self.mainwindow.cnm.estimates.C[self.mainwindow.cnm.estimates.idx_components, tmin:tmax])
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
            ctitle=f'Reconstructed movie (A ⊗ C) using good comps.'
        elif array_text == 'RCB':
            res=np.dot(self.mainwindow.cnm.estimates.b[:, :] , self.mainwindow.cnm.estimates.f[:, tmin:tmax])
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
            ctitle=f'Reconstructed background (b ⊗ f)'
        elif array_text == 'Residuals':
            res=self.mainwindow.data_array[:,tmin:tmax]
            rcm=np.dot(self.mainwindow.A_array[:, :] , self.mainwindow.cnm.estimates.C[:, tmin:tmax])
            rcb=np.dot(self.mainwindow.cnm.estimates.b[:, :] , self.mainwindow.cnm.estimates.f[:, tmin:tmax])
            res=res-rcm-rcb
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
            ctitle=f'Residuals (Y - (A ⊗ C) - (b ⊗ f))'
        elif array_text == 'Residuals (Good)':
            res=self.mainwindow.data_array[:,tmin:tmax]
            rcm=np.dot(self.mainwindow.A_array[:, self.mainwindow.cnm.estimates.idx_components] , self.mainwindow.cnm.estimates.C[self.mainwindow.cnm.estimates.idx_components, tmin:tmax])
            rcb=np.dot(self.mainwindow.cnm.estimates.b[:, :] , self.mainwindow.cnm.estimates.f[:, tmin:tmax])
            res=res-rcm-rcb
            res=np.mean(res, axis=1)
            image_data = res.reshape(self.mainwindow.dims)
            ctitle=f'Residuals (Y - (A ⊗ C) - (b ⊗ f)) using good comps.'
        elif array_text[0] == 'B':
            try:
                bgindex = int(array_text[1:])
            except ValueError:
                raise ValueError(f'Invalid array text: {array_text}')
            image_data = self.mainwindow.cnm.estimates.b[:, bgindex].reshape(self.mainwindow.dims)
            ctitle=f'Background component {bgindex} (static)'
        elif array_text == 'Cn':
            image_data = self.mainwindow.cnm.estimates.Cn
            ctitle=f'Correlation image (static)'
        elif array_text in ['Mean', 'Max', 'Std']:
            image_data = getattr(self.mainwindow, array_text.lower()+'_projection_array')
            ctitle=f'{array_text} projection image (static)'
        elif array_text == 'sn':
            image_data = self.mainwindow.cnm.estimates.sn.reshape(self.mainwindow.dims)
            ctitle=f'Array: {array_text} (static)'
        elif array_text == 'MaxResNone':
            image_data = self.mainwindow.max_res_none
            ctitle = 'Temporal maximum of Y - BG (static)'
        elif array_text == 'MaxResGood':
            image_data = self.mainwindow.max_res_good
            ctitle = 'Temporal maximum of Y - BG - RCM,good (static)'
        elif array_text == 'MaxResAll':
            image_data = self.mainwindow.max_res_all
            ctitle = 'Temporal maximum of Y - BG - RCM,all (static)'
        else:
            raise NotImplementedError   
        
        # Update image data
        avr=self.spatial_avr_spinbox.value()

        kernel = cv2.getGaussianKernel(2*avr+1, avr)
        kernel = np.outer(kernel, kernel)
        image_data = cv2.filter2D(image_data, -1, kernel)
        
        if add_vars_to_title:
            ctitle+=f'(w={int((tmax-tmin-1)/2)},a={avr})'
        
        return image_data, ctitle
        
    def update_spatial_view_image(self, setLUT=False):
        
        #update only the image according to t
        t=self.mainwindow.selected_frame
        w=self.mainwindow.frame_window
        tmin=t-w if t-w>0 else 0
        tmax=t+w+1 if t+w+1<self.mainwindow.num_frames else self.mainwindow.num_frames 
        image_data, ctitle = self.get_spatial_view_image(tmin, tmax)
        
        self.spatial_image.setImage(image_data, autoLevels=False)
   
        if setLUT:
            # Update colorbar limits explicitly
            min_val, max_val = np.min(image_data), np.max(image_data)
            self.colorbar_item.setLevels(values=[min_val, max_val])

        # Colorbar reset button logic
        self.colorbar_reset_button.setEnabled(True)  
    
        return ctitle
    
    def on_contour_click(self, ev):
        index = int(ev.name())
        self.mainwindow.set_selected_component(index, 'spatial') 
        

def run_gui(file_path=None, data_path=None):
    app = QApplication(sys.argv)

    try:
        with importlib.resources.as_file(importlib.resources.files("pluvianus").joinpath("pluvianus.ico")) as icon_path:
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception as e:
        print(f"Error setting window icon: {e}")

    app.processEvents()
    
    window = MainWindow()
    window.showMaximized()
    app.processEvents()
        
    if file_path:
        window.open_file(file_path)
    if data_path:
        window.open_data_file(data_path)  # if filepath was not loaded, it will be ignored

    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()