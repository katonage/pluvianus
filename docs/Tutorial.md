# Tutorial
Here we present examples demonstrating how Pluvianus can be used to inspect the results of the source extraction algorithm. Throughout this tutorial, we use the output of the CaImAn demo_pipeline.

Load the `.hdf5` file containing your CaImAn results and also load the corresponding movement-corrected movie data array (`.mmap` file). In the `Compute menu` generate both the original fluorescence traces (`Compute Original Fluorescence traces` ) and the ΔF/F traces (`Detrend ΔF/F`, if not previously computed).

## Inspecting Components
Visualizing calcium transients and their spatial footprints is fundamental for evaluating the quality of extracted components. It is therefore important to inspect representative transients across temporal, spatial, and quantitative dimensions, and efficiently cycle through each of them.

Enable the `Auto` option in both the temporal and spatial widgets so that they automatically focus on the selected component. This setting shifts the time axis to the period of maximal activity for that component and zooms the spatial widget to its footprint. 

* Select the `Compound` order at the component selector so that cells are visited from best to worst quality, corresponding to the diagonal of the scatterplot. Select the top-right cell in the scatterplot and cycle through cells using the `Down` button (keyboard: down arrow).
* Load the `Data` and either `C` or `F_dff` curves in the temporal widget. This enables direct comparison of the temporal profile and signal-to-noise characteristics of the raw fluorescence trace and the inferred activity signal. By displaying each component using identical y-axis scaling (`Y fit all` buttons), it becomes easier to distinguish components that rise above noise from those representing fitted noise fluctuations.
* Having the raw movie (`Data`) loaded, the spatial widget enables verification of whether the calculated footprint (contour) matches the cell morphology during periods of activity.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI.png" width="85%"> </p>

## Assessing Overlaps and Baseline Subtraction
Temporal components identified by CNMF can be difficult to interpret in cases of spatial overlap or high background activity. Inspecting components in Pluvianus facilitates assessment of whether overlapping components or baseline fluctuations have been correctly separated across multiple complementary views.

For cases of interest, drag the time axis to a specific period of activity and examine both the selected component and the surrounding regions in the spatial widget. To suppress noise, spatial and temporal averaging can be applied. For example, in the case shown below, activity from a smaller, active cell introduces crosstalk into the larger cell’s raw fluorescence trace (blue). This contamination is correctly removed by CNMF and is no longer apparent in the red ΔF/F trace.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI_2.png" width="85%"> </p>

## Assessing completeness of component extraction
Computing three different temporal maximum residual images allows assessment of whether the CaImAn source extraction algorithm performed correctly and whether parameter selection was appropriate.

Use the `Compute Temporal Maximum of Residuals` function. This generates three images that can be displayed in the spatial widgets: 
* `MaxResNone`: Temporal maximum of residuals with only background subtracted. (Y- BG)
* `MaxResGood`: Temporal maximum of residuals with background and accepted (good) components subtracted. (Y – BG – RCM(good))
* `MaxResAll`: Temporal maximum of residuals with background and all components subtracted. (Y – BG – RCM(all))

Open the second spatial widget from the lower-right corner (pull three-dots) to display two residual images side-by-side. Adjust colorbars if necessary and use the `Sync Axes` button to ensure spatial alignment.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI_3.png" width="85%"> </p>
<p align="center"> GUI reshaped for comparison of the residuals. </p>

#### Quality of Subtraction
Compare `MaxResNone` and `MaxResGood`. Verify that bright activity patches in `MaxResNone` correspond to accepted components and disappear in `MaxResGood` (green arrows on example below). Assess whether subtraction is complete. If structured residual activity remains under an accepted component in `MaxResGood` (orange arrows), inspect the corresponding time point in the original data at the `Frame` displayed with the cursor hovering over the feature.

####   Completeness of Component Detection
Compare `MaxResGood` and `MaxResAll`. Adjust colorbars if necessary. Evaluate the typical residual amplitude associated with components labeled as bad (purple arrow on example below). In `MaxResAll`, all modeled components should be removed, ideally leaving no structured residual activity. Inspect residual activity on `MaxResAll` (red arrow). If activity remains that exceeds the typical residual level of rejected components, this suggests incomplete source extraction, and CaImAn parameter settings should be reconsidered.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusExample_5.png" width="60%"> </p>
<p align="center"> Example patch of a measurement demonstrating the three residuals. </p>

## Component Evaluation Using Thresholds and Manual Review
Use the scatterplot widget to refine component classification. The plot can be rotated interactively to examine projections of interest, allowing identification and selection of borderline components. Evaluating quality of these, the acceptance criteria thresholds can be refined.

After adjusting classification thresholds, recalculate the good/bad labels using the Evaluate button (a data array must be loaded).

For manual classification:
* Iterate through components in a defined order (e.g., highest to lowest SNR).
* Modify classification using keyboard shortcuts (`g` = good, `b` = bad; navigate with `Up` and `Down` arrows). 
Updated classifications can be written back to the CaImAn `.hdf5` file to support downstream analyses and subsequent pipeline steps.

If gray dots appear in the scatterplot, compute component metrics using `Compute Component Metrics` from the menu. This function needs a data array to be loaded first.

<p align="center"> <img src="https://github.com/katonage/pluvianus/blob/main/docs/img/pluvianusGUI_4.png" width="65%"> </p>
The figure was compiled from screenshot segments.

## See also
- [Usage](Usage.md)
- [Contributing](Contributing.md)
- [TOC](../README.md)
