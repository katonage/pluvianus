# Tutorial
Here we present a few examples, how Pluvianus can be used to inspect results of the source extraction algorithm. We use output of the CaImAn’s demo_pipeline throughout.

Load the .hdf5 file of your CaImAn result and also load the corresponding movement corrected movie data array (.mmap file). In the Compute menu generate both the original fluorescence traces (“Compute Original Fluorescence traces” ) and the ΔF/F traces (“Detrend ΔF/F “, if not previously computed).

## Inspecting Components
Visualizing calcium transients and their footprints is fundamental for evaluating the quality of extracted components. For this reason, it is important to inspect representative transients and efficiently cycle through them. Check the “Auto” option in both the temporal and spatial widgets so that they automatically focus on the selected component. This setting shifts the time axis to the period of maximal activity for that component and zooms the spatial widget to its footprint. 
* Select the “Compound” order at the component selection, so that you visit the cells from best to worst quality, corresponding to the diagonal of the scatterplot. Select the top right cell on the scatterplot, cycle cells with the “Down” button (keyboard “d”).
* Load the “Data” and “C” or “F_dff” curves in the temporal widget. This way you can compare shape and signal-to-noise ratio of the raw and the evaluated transients. By displaying each component with identical y-axis scaling (“Y fit all” buttons), it becomes easier to distinguish components that stand out from the noise from those that merely represent fitting noise fluctuations.
* Having the raw movie (“Data”) loaded, the spatial widget allows direct verification of whether the calculated footprint (contour) matches the morphology of the cell when it becomes active. 

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI.png" width="85%"> </p>

## Assessing Baseline Subtraction and Component Quality
Although temporal components identified by CNMF can be difficult to interpret in cases of overlap or high background, inspecting components with Pluvianus helps users assess whether the algorithm has correctly subtracted baseline fluctuations or overlaping components. 
At cases of interest, you can drag the time axis to a particular period of activity and examine both the component’s activity and the surrounding regions in the spatial widget. To suppress noise, you may average the data both spatially and temporally. For example, here you can see that the active smaller cell gives crosstalk to the larger cell’s data as seen in the blue original fluorescence trace, which has been correctly removed by the CNMF and not apparent in the red ΔF/F trace.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI_2.png" width="85%"> </p>

## Assessing completeness of component extraction
Computing three different maximum residual images allows you to assess whether CaImAn’s algorithm performed correctly, and if calculation parameters were optimal. Use the “Compute Temporal Maximum of Residuals” calculation. This creates three images to display in the spatial widgets: 
* `MaxResNone`: Maximum of residuals, only background subtracted. (Y- BG)
* `MaxResGood`: Maximum of residuals having background and good components subtracted. (Y – BG – RCM(good))
* `MaxResAll`: Maximum of residuals having background and all components subtracted. (Y – BG – RCM(all))

Pull the second spatial widget from the lower right edge using the three dots to display two of the calculated images side-by-side. Adjust colorbars if necessary. Use the Sync Axes button.

### Subtraction
* Compare `MaxResNone` to `MaxResGood`. Check that all bright patches are delineated as good components and disappear on the subtracted image. Inspect the completeness of the subtraction. Where significant amount of activity remains on the subtracted image, look up and inspect original data at the Frame displayed with the cursor hovering over the feature.
<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI_3.png" width="85%"> </p>

### Completeness
* Compare `MaxResGood` to `MaxResAll`. Adjust colorbars if necessary. Check the typical amplitude of the residual associated with components labelled as "bad" (red). On the `MaxResAll` image, all components should have been removed, ideally leaving no structured activity. Check if there is still activity not delineated as a component: if there is activity surpassing the typical activity of a bad component, the CaImAn source extraction algorithm parameters should be adjusted.


## TODO further text

For step-by-step instructions please refer to the associated publication, which covers:
- Verifying signal-to-noise separation
- Inspecting the component’s highest activity
- Assessing the completeness of component extraction
- Component evaluation using thresholds and manual review

## See also
- [Usage](Usage.md)
- [Contributing](Contributing.md)
- [TOC](../README.md)
