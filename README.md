# Pluvianus: CaImAn Result Browser 
<img src="https://github.com/katonage/pluvianus/blob/main/pluvianus%20image.png" width="400" align="right">
A standalone GUI for browsing, editing, and manually verifying CaImAn results.

*"Seeing is believing..."*

## Highlights
- Standalone Qt GUI
- Spatial and temporal views
- Interactive scatter plot of component metrics
- Visualization of extracted traces alongside original data
- Currently supports only 2D movies

##  License
Pluvianus is distributed under the MIT License. If you use Pluvianus in your work, please cite our associated publication:

TODO: Paper reference here

## Related repositories

- [CaImAn](https://caiman.readthedocs.io/en/latest/) – Calcium imaging analysis toolkit with motion correction and source extraction.
- [mesmerize-core](https://github.com/nel-lab/mesmerize-core) – Batch management of CaImAn calculations and parameter search.
- [pynapple](https://pynapple.org/index.html) – Time series analysis for neuroscience data.
- [PyQtGraph](https://www.pyqtgraph.org/) – Interactive plotting library for scientific data.

## Installation

This package is intended for installation within an existing CaImAn conda environment. Please refer to the [CaImAn installation instructions](https://caiman.readthedocs.io/en/latest/Installation.html) to set up this environment. 
Do not install Pluvianus outside of the CaImAn environment!

### From Source
Steps:
1. Clone the repository:  
   ```bash
   git clone https://github.com/katonage/pluvianus.git
2. Navigate to the pluvianus directory:
   ```bash  
   cd pluvianus
3. Install Pluvianus in editable mode into your CaImAn environment:
   ```bash
   pip install -e .

### From distribution
* Install Pluvianus from PyPI:
   ```bash
   pip install pluvianus


### System Requirements
* Refer to CaImAn’s system requirements: https://caiman.readthedocs.io/en/latest/Installation.html
* At least an HD display is recommended.

## Usage
### Starting with an Empty GUI
To launch Pluvianus, type either of the following commands in your terminal:
`python pluvianus` or simply `pluvianus`

Use the File menu to open a `.hdf5` file exported by the CaImAn package (`Ctrl+O`) and its corresponding Data Array (`.mmap` file) (`Ctrl+D`). 

**Note:** Only `.hdf5` and `.mmap` files created by CaImAn are supported.

### Command Line Usage
You can also launch Pluvianus directly with the CaImAn `.hdf5` result file and `.mmap` data file by specifying them as command-line arguments: 
   ```bash
   pluvianus -f results.hdf5 -d movie.mmap
   ```
   or 
   ```bash
   python -m pluvianus --file results.hdf5 --data movie.mmap
   ```
### Opening Demo Files
#### CNMF
* Run all cells in the CaImAn [demo_pipeline.ipynb](https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_pipeline.ipynb) notebook (you can also find a copy of this file within your already installed CaImAn environment).
* This will create the two necessary files in the `caiman_data\temp` folder:
  * `demo_pipeline_results.hdf5`: result file of the CNMF algorithm.
  * `Sue_2x_3000_40_-46_els__d1_170_d2_170_d3_1_order_F_frames_3000.mmap`: movement corected memory mapped movie datafile.
* Launch Pluvianus as described above, and open these two files.
#### OnACID
* Run the [demo_OnACID_mesoscope.ipynb](https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_OnACID_mesoscope.ipynb) notebook.
* To save results, add  `cnm.save('demo_OnACID_mesoscope_results.hdf5')` to the end of the notebook.
* This will create the results file.
* Launch Pluvianus as described above, and open this file.
* In Pluvianus, launch the `Compute Data Array for OnACID Files` menu option.
   * Select all three .hdf5 files in the `caiman_data\example_movies\Mesoscope` folder. This will create a concatenated movement corrected .mmap file in the `temp` folder that can be used later to load the data.
* Launch the `Compute Component Metrics` and the `Detrend ΔF/F` menu options to complete the analysis.

## GUI Overview
The Pluvianus GUI consists of three main panels:

1. Temporal Widget (Top):
Displays the activity traces of selected components over time.

2. Scatter Widget (Bottom Left):
Visualizes each component in a 3D scatter plot according to three evaluation metrics.

3. Spatial Widget (Bottom Right):
Shows the outlines and spatial positions of components. An additional Spatial Widget can be opened by dragging the three dots on the right edge.

### Temporal Widget
**Component selection:**<br>
Select a component using the "Component" spinbox, or by clicking its outline in the Spatial Widget or its point in the Scatter Widget. <br>
Cycle through components with the "Up" and "Down" buttons, or with the keyboard arrow keys. The order in which components are shown can be changed via the "Order by" dropdown. This way you can:
* Iterate through only the good or bad components (`index (Good)` or `index (Bad)`)
* Go through the components according to their selected metrics 
  * `SNR`, `CNN`, `R value`: Refer to the CaImAn documentation for detailed definitions of these metrics. 
  * `Compound`: Calculated from the above metrics; it corresponds to a position along the diagonal of the scatter plot.

**Quantity selection:**<br>
Under the "Plot" section, you can select the quantities to be plotted for the selected component as a function of frame number. The selected quantities are shown on the vertical axes: the first (blue) on the left, the second (red) on the right.
* `C`: Denoised calcium trace (Temporal component)
* `S`: Spike count estimate
* `YrA`: Residual of the trace
* Others if computed (see "Compute" section)
    * `F_dff`: Estimated ΔF/F trace
    * `Data`: Mean fluorescence of the original movie under the component contour

A running average over the data can also be applied.<br>
"Y fit all": Rescales the plot vertically to include all traces. Use this to compare absolute amplitudes of the comonents.<br>

**Centering time on activity maximum:**<br>
"Center": Positions the selected time on the largest peak of the selected component’s activity (maximum of C). Use mouse scrol on the horizontal axis to adjust temporal zoom.<br>
"Auto": Centers time automatically when a new component is selected. <br>

**Graph interactions:** <br>
* On the center pane:
  * Drag with left mouse button to pan.
  * Drag with right mouse button to zoom.
  * Scroll with the mouse wheel to uniformly zoom.
* On the axes:
  * Drag with left mouse button to pan time and vertical axis independently.
  * Drag with right mouse button or scroll to zoom on a specific axis.
* "A" Button: Hovering inside the view reveals an "A" button in the lower left, which fits the plot to all data horizontally and vertically.
* Right-click on the plot to open a context menu allowing you to export the scene

**Timeline navigation:** <br>
Scroll through time using the slider below the plot (shows both frame number and absolute time). Right of the timeline, specify the time window for temporal averaging in the Spatial Widget.

### Spatial Widget
Shows the component contours (green for good, red for bad).<br>
Pan: Left mouse button; Zoom: Mouse wheel; Fit image: "A" button (lower left)<br>
Adjust the colormap using the colorbar on the right; right-click to select alternate color schemes. Reset it with the "Reset Colorbar" button.

Under "Display" select the quantity to display (at the selected timepoint):
* `A`: Spatial footprint of selected component
* `Data`: Original data
* `RCM`: Reconstructed movie
* `RCB`: Reconstructed background
* `Residuals`: Difference of the original data and the RCM and RCB
* `B0`, `B1`: Background components

"Zoom" centers the view on the selected component, with zoom corresponding to estimated neuron diameter. "Auto" will automatically center and zoom when a new component is selected. <br>

Under "Contours" choose which component outlines are visible:
* `All`: All components
* `Good + T`: Good components normally, bad faint
* `Bad + T`: Bad components normally, good faint
* `Good`: Only good components
* `Bad`: Only bad components
* `Selected`: Only the selected component
* `None`: No components

Drag from the right edge (three dots) to open a secondary Spatial Widget for side-by-side comparison of different quantities.<br>
Right-click on the plot to open a context menu allowing you to export the scene.

### Scatter Widget
A 3D scatter plot displays all components, using three evaluation metrics as axes. Clicking on a point selects it on the Temporal and Spatial Widgets. Good components are displayed as green points, whereas bad components are displayed as red points.

In the Assignment section, use "Good" and "Bad" to manually accept or reject the selected component, regardless of evaluation thresholds. Keyboard shortcuts: `g` for Good, `b` for Bad.<br>
In the Thresholds section you can set two thresholds ("lowest" and "min") per metric. A component is classified as good if it exceeds all "lowest" thresholds and at least one "min" threshold, unless manually overridden as above. Press "Evaluate" to apply the current thresholds (uses CaImAn's `filter_components()`).

### File menu
* `Open CaImAn HDF5 File`: Loads the saved estimates object from CNMF or OnACID analyses.
* `Save / Save as...`: Pluvianius can save CaimAn data back into an `.hdf5` file, preserving:
   * Results of the ΔF/F calculation (`.F_dff`)
   * Calculated component metrics (`.r_values`, `.SNR_comp`, `.cnn_preds`)
   * Component evaluation threshold levels (`SNR_lowest`, `min_SNR`, `cnn_lowest`, `min_cnn_thr`, `val_lowest`, `rval_thr`)
   * Manual component assigments (`.idx_components`)
* `Open Data Array`: Opens the movement corrected `.mmap` file containing the original fluorescence movie. This file is required for most computations and visualizations.
* `Load/Save Mean/Max/Std/Local correlations images`: these images can be computed from the data and displayed in the Spatial view. The file format is `.npy`. 

### Compute menu
Performs various calculations on the data. Calculations that invoke CaImAn functions inherit their parameters from the currently opened `.hdf5` file; you can view these parameters under View → CaImAn Parameters.
* `Detrend ΔF/F`: Calculates detrended relative fluorescence change. You can view this by selecting `F_dff` under "Plot" in the Temporal Widget. Calls the `detrend_df_f()` function.
* `Compute Component Metrics`: (Re)computes the three metrics used for component evaluation. Calls the `evaluate_components()` function. 
* `Compute Projections`: computes the temporal Mean, Max and STD images. You can also save them to `.npy` files in the File menu.
* `Compute Local Correlation Image`: Calls the `local_correlations_movie_offline()` function, and computes its maximum. Result can be viewed as `Cn` in the Spatial Widget. 
* `Compute Original Fluorescence Traces`: Calculates the mean fluorescence under each component's contour. The component traces can be viewed by selecting `Data` in the Temporal Widget. `Data neuropil` is calculated as the mean of all pixels not belonging to a component. These raw traces can be compared to the results of CaImAn.
* `Compute Temporal Maximum of Residuals`: Computes the temporal maximum of three types of residuals. You can view these under "Display" in the Spatial Widget. The types of residuals are:
    * `MaxResNone`: Subtracts the background (BG) from the original data (Y), and takes each pixel's temporal maximum
    * `MaxResGood`: Subtracts BG and good components' activity (RCM,good) from Y, and takes each pixel's temporal maximum
    * `MaxResAll`: Subtracts BG and all components' activity (RCM,all) from Y, and takes each pixel's temporal maximum
* `Compute Data Array for OnACID files`: Computes the data array from an OnACID result file and the original movie, and saves it as a `.mmap` file. Only results with the parameter `pw_rigid = False` can be used. The original movie's format can be `.hdf5`, `.tiff`, `.npy`, or `.mmap`. If the movie consists of multiple files, they must be selected and passed in temporal order! Calls the `apply_shift_online()` function on the original movie and the shifts stored in the result.

### Export menu
"Export" lets you export various data:
* C: Export component activity traces in `.npz` format, either for all components or only for good components.
* ΔF/F: Export detrended ΔF/F traces in `.npz` format, either for all components or only for good components.
* Countours: Export component contours in `.MEScROI` format, only for good components.

See also the Save functionality in the File menu. <br>
If you require additional export formats, please submit a pull request.

### Keyboard shortcuts
* `g`, `b`: Manually accept or reject component
* `Up arrow`, `Down arrow`: Step through components
* `Ctrl-O`: Open CaImAn HDF5 file.
* `Ctrl-S`: Save to CaImAn HDF5 file.
* `Ctrl-D`: Open motion-corrected data array (.mmap file).

## Recommended Usage
For step-by-step instructions please refer to the associated publication, which covers:
- Verifying signal-to-noise separation
- Inspecting the component’s highest activity
- Assessing the completeness of component extraction
- Component evaluation using thresholds and manual review

TODO: Paper reference here

## Future plans 
* Displaying all component's temporal curves together (gray), mouse click event on that to select component
* Optionaly synchronize axis of spatial widgets
* Support of 3D data
* Edit parameters of the compute menu options
* Merging RoIs
* Supporting comonent registration for chronic recordings

## Notes
Tested on PC, Windows, Anaconda3. CaImAn 1.11.4, PySide6 6.9.2. <br>
With PySide 6.9.1 the application freezes.