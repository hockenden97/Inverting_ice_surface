# Inverting_ice_surface

This reposistory contains 6 code files (python jupyter notebooks). 

'Transfer functions.ipynb' and 'Transfer functions 2003.ipynb'
Gudmundsson (2003, 2008) derived a set of transfer functions which describe the relationship between the fourier transforms of the ice surface and the bed beneath the ice. These notebooks contain the code for those transfer functions for the full system equations (2003) and the shallow-ice-stream approximation (2008). 

'Synthetic tests.ipynb'
The transfer functions derived by Gudmundsson (2003, 2008) describe a forward model where if the bed topography and slipperiness are known then the surface elevation and velocity can be calculated. Using a least squares inversion, if the surface elevation and velocity are known then the bed topography and slipperiness can be calculated. This notebook contains the code for this inversion, and explores the capabilities of the inversion with synthetic datasets of the bed topography and slipperiness. 

'Topography Inversion Model.ipynb'
The inverse model tested on synthetic datasets can also be applied to real surface elevation and velocity data. This notebook contains the code to run the inversion over a region of surface data, which can be specified by the user in one of the cells. It reads in files called new.nc (REMA surface data, Bedmachine bed topography and ice thickness), and new_itslive.nc (ITSLIVE surface data), which should be provided on the same grid. 

'Model_output_Figures.ipynb'
The inversion model was run across a region 160 km by 280 km including the main trunk and central flow line of Thwaites Glacier. To speed up model running and reduce memory use, the model was run in 8 parts, generating 8 output files (number 1-8). To look at these output files all together, including the location of the existing radar grids collected by Holschuh et al. (2010), this notebook joins the files together and plots topography and slipperiness across the Thwaites Glacier region. 

'Model_output_2runs_figures.ipynb'
The inversion model was run for several values of non-dimensional mean slipperiness, $\bar{C}$, including C = 100 and C = 150. This notebook allows comparison of two different model runs, and compares them to the exisiting radar grids for Upper and Lower Thwaites (Radar_grid_upper_thwaites.nc, Radar_grid_lower_thwaites.nc, see Holschuh et al. 2020), and to radar flightlines collected by BAS in the 2019-2020 field season (Thwaites_19_20_PASIN_radar_V2.1.csv). 

This repository also contains the data files produced when the inversion procedure was carried out as described, with C = 100 and C = 150. For C = 100 these are output.nc, output2.nc, output3.nc, output4.nc, output5.nc, output6.nc, output7.nc and output8.nc. For C = 150 these are output_1_150.nc, output_2_150.nc, output_3_150.nc, output_4_150.nc, output_5_150.nc, output_6_150.nc, output_7_150.nc and output_8_150.nc.

Finally this repository contains some generic useful shape files for plotting model inputs and outputs.
Thw_lakes_outlines.shp
GroundingLine_Antarctica_v2.shp
Coastline_Antarctica_v2.shp
IceBoundaries_Antarctica_v2.shp
1dg_latitude.shp
2dg_longitude.shp
