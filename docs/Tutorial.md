# Tutorial
Here we present how Pluvianus can be used to inspect results of the source extraction algorithm. We use output of the CaImAn’s demo_pipeline throughout.

Load the .hdf5 file of your CaImAn result and its data array (.mmap file). In the Compute menu generate both the original fluorescence traces (“Compute Original Fluorescence traces” ) and the ΔF/F traces (“Detrend ΔF/F “, if not previously computed).

## Inspecting Components
Visualizing calcium transients and their footprints is fundamental for evaluating the quality of extracted components. For this reason, it is important to inspect representative transients and efficiently cycle through them. Check the “Auto” option in both the temporal and spatial widgets so that they automatically focus on the selected component. This setting shifts the time axis to the period of maximal activity for that component and zooms the spatial widget to its footprint. 
* Select the “Compound” order at the component selection, so that you visit the cells from best to worst quality, corresponding to the diagonal of the scatterplot. Select the top right cell on the scatterplot, cycle cells with the “Down” button (keyboard “d”).
* Load the “Data” and “C” or “F_dff” curves in the temporal widget. This way you can compare shape and signal-to-noise ratio of the raw and the evaluated transients. By displaying each component with identical y-axis scaling (“Y fit all” buttons), it becomes easier to distinguish components that stand out from the noise from those that merely represent fitting noise fluctuations.
* Having the raw movie (“Data”) loaded, the spatial widget allows direct verification of whether the calculated footprint (contour) matches the morphology of the cell when it becomes active. 

<img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI.jpg" width="600" align="center">

TODO further text

For step-by-step instructions please refer to the associated publication, which covers:
- Verifying signal-to-noise separation
- Inspecting the component’s highest activity
- Assessing the completeness of component extraction
- Component evaluation using thresholds and manual review