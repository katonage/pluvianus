# Tutorial
Here we present how Pluvianus can be used to inspect results of the source extraction algorithm. We use output of the CaImAn’s demo_pipeline throughout.

Load the .hdf5 file of your CaImAn result and its data array (.mmap file). In the Compute menu generate both the original fluorescence traces (“Compute Original Fluorescence traces” ) and the ΔF/F traces (“Detrend ΔF/F “, if not previously computed).

## Inspecting Components
Visualizing calcium transients and their footprints is fundamental for evaluating the quality of extracted components. For this reason, it is important to inspect representative transients and efficiently cycle through them. Check the “Auto” option in both the temporal and spatial widgets so that they automatically focus on the selected component. This setting shifts the time axis to the period of maximal activity for that component and zooms the spatial widget to its footprint. 
* Select the “Compound” order at the component selection, so that you visit the cells from best to worst quality, corresponding to the diagonal of the scatterplot. Select the top right cell on the scatterplot, cycle cells with the “Down” button (keyboard “d”).
* Load the “Data” and “C” or “F_dff” curves in the temporal widget. This way you can compare shape and signal-to-noise ratio of the raw and the evaluated transients. By displaying each component with identical y-axis scaling (“Y fit all” buttons), it becomes easier to distinguish components that stand out from the noise from those that merely represent fitting noise fluctuations.
* Having the raw movie (“Data”) loaded, the spatial widget allows direct verification of whether the calculated footprint (contour) matches the morphology of the cell when it becomes active. 

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI.png" width="80%"> </p>

## Assessing Baseline Subtraction and Component Quality
Although temporal components identified by CNMF can be difficult to interpret in cases of overlap or high background, inspecting components with Pluvianus helps users assess whether the algorithm has correctly subtracted baseline fluctuations or overlaping components. 
At cases of interest, you can drag the time axis to a particular period of activity and examine both the component’s activity and the surrounding regions in the spatial widget. To suppress noise, you may average the data both spatially and temporally. For example, here you can see that the active smaller cell gives crosstalk to the larger cell’s data as seen in the blue original fluorescence trace, which has been correctly removed by the CNMF and not apparent in the red ΔF/F trace.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI_2.png" width="80%"> </p>

TODO further text

For step-by-step instructions please refer to the associated publication, which covers:
- Verifying signal-to-noise separation
- Inspecting the component’s highest activity
- Assessing the completeness of component extraction
- Component evaluation using thresholds and manual review