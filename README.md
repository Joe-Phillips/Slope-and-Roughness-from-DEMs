# Slope-and-Roughness-from-DEMs

This project produces **slope** and **roughness** maps using **Digital Elevation Model** (DEM) data and **Singular Value Decomposition** (SVD).

Made by Joe Phillips.

[![Repo](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips) 
[![Repo](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp;✉️ j.phillips5@lancaster.ac.uk

## :toolbox: How it Works

Although there exist several sufficient and easy-to-apply methodologies for calculating slope from DEMs, commonly applied approaches for calculating roughness such as Terrain Ruggedness Index (TRI) and Topographic Position Index (TPI) contain several underlying issues, despite extensive application in GIS programs and packages such as GRASS, ArcGIS, and GDAL (used by QGIS). This is because these methods do not account for topographic slope when computing the variance in elevation, instead simply calculating the differences between a pixel and its immediate neighbours. This, for example, would return a non-zero roughness value over a monotonic surface set at an angle. As such, values attained for slope and roughness calculated this way encode each via complex, non-linear interactions.

Here, we instead calculate roughness independently of slope via the dispersion of orthogonal residuals from a plane fitted through a given DEM point and its neighbours. To do so, we use SVD, which we apply using a sliding-window approach.

To obtain slope and roughness values, we first centre the data by subtracting the means of their x, y and z coordinates. By applying SVD, which decomposes the now-centred points in each window (described by a column of x,y, and z coordinates) into three distinct matrices ($M = U \Sigma V^{T}$), we can take the 3x3 left singular matrix which contains three orthogonal unit vectors describing a plane of best fit. By calculating the partial derivatives $\frac{dz}{dx}$ and $\frac{dz}{dy}$, we can then determine the resulting gradient, and hence the surface slope in degrees. Roughness is then directly computed based on the standard deviation of the orthogonal residuals, which are calculated by taking the dot product of the points with the normal vector to the plane.

## 🛠️ Usage Guide

First, make sure you have installed the required packages. This can be done via **pip install -r requirements.txt**.

To generate the slope and roughness maps, simply run **generate_Slope_and_Roughness.py** from the command line with the following arguments:

- **DEM_PATH** (string): *The path to the DEM file.*
- **DEM_RES** (string): *The resolution of the DEM in meters.*
- **WINDOW_SIZE** (int): *The size of the window around each pixel in meters over which slope and roughness will be calculated.*
- **NUM_TILES** (int): *The number of tiles to split the DEM into during processing to minimise memory usage. This will depend on the size of the DEM and the amount of available memory.*

### Example:

- python generate_Slope_and_Roughness.py example_folder/DEM.tif 200 1000 400

<br>

## :camera: Images
**Example output over Antarctica using REMA** (https://www.pgc.umn.edu/data/rema/) **at 200m and a window size of 1000m**. *(Plot created separately).*

<br>

![alt text](https://github.com/Joe-Phillips/DEM-to-Slope-and-Roughness/blob/main/REMA_Example_Figure.png?raw=true)
