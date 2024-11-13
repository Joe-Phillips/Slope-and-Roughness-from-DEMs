# Slope-and-Roughness-from-DEMs

This project produces **slope** and **roughness** maps using **Digital Elevation Model** (DEM) data and **Singular Value Decomposition** (SVD).

Made by Joe Phillips.

[![Repo](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips) 
[![Repo](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp;✉️ j.phillips5@lancaster.ac.uk

## :toolbox: How it Works

Commonly applied slope algorithms, such as Horn's method, estimate slope using local elevation differences across a small, fixed window (3x3). By approximating partial derivatives across immediate neighbours, these methods tend to be sensitive to noise and elevation variability, often resulting in exaggerated slope values in areas with highly variable or noisy data.

Similarly, widely used roughness methods such as Terrain Ruggedness Index (TRI) and Topographic Position Index (TPI) fail to account for topographic slope when evaluating elevation variance. Instead, they calculate differences between a pixel and its immediate neighbors, yielding non-zero roughness values for a monotonic surface set at an angle. Consequently, the roughness values produced by these methods encode slope through complex, non-linear interactions.

Despite these limitations, these approaches are extensively used in GIS software like GRASS, ArcGIS, and GDAL (the backend for QGIS).

Here, we calculate slope more accurately by fitting a plane through a given DEM point and its neighbors using Singular Value Decomposition (SVD). Roughness is computed independently of slope by assessing the dispersion of orthogonal residuals from this fitted plane. Unlike traditional methods, our approach enables a sliding-window application that is not restricted to a 3x3 window; instead, it can accommodate a larger area (e.g., a 9x9 pixel window), incorporating more data points to produce smoother, more reliable slope and roughness estimates.

Specifically, for a pixel and its surrounding neighbors, the data is first centered by subtracting the mean of the x, y, and z coordinates. Applying SVD to the centered points in each window (organized as columns of x, y, and z coordinates) decomposes them into three distinct matrices:

$$
M = U \Sigma V^{T}
$$

From these, the 3x3 left singular matrix **U** contains three orthogonal unit vectors that describe the best-fit plane. By calculating the partial derivatives $\( \frac{dz}{dx} \)$ and $\( \frac{dz}{dy} \)$, we determine the resulting gradient and thus the surface slope in degrees. Roughness is then computed directly from the variation in orthogonal residuals, which are obtained by taking the dot product of the centered points with the normal vector to the plane. The roughness can currently be quantified using the (1) standard deviation, (2) median absolute deviation, or (3) peak-to-peak distance of the residuals.

## 🛠️ Usage Guide

First, make sure you have installed the required packages. This can be done via **pip install -r requirements.txt**.

To generate the slope and roughness maps, simply run **dem_to_slope_and_roughness.py** from the command line with the following arguments:

- **DEM_PATH** (string): *The path to the DEM file.*
- **DEM_RESOLUTION** (string): *The resolution of the DEM in meters.*
- **WINDOW_SIZE** (int): *The size of the window around each pixel in meters over which slope and roughness will be calculated.*
- **ROUGHNESS_METHOD** (int): *The method used to calculate roughness. Options: 'std' (standard deviation), 'mad' (median absolute deviation), 'p2p' (peak-to-peak). Defaults to 'std'.*

### Example:

- python dem_to_slope_and_roughness.py example_folder/DEM.tif 200 900 std

<br>

## :camera: Images
**Example output over Antarctica using REMA** (https://www.pgc.umn.edu/data/rema/) **at 200 m resolution and a window size of 1000 m (5x5)**. *(Plot created separately).*

<br>

![alt text](https://github.com/Joe-Phillips/DEM-to-Slope-and-Roughness/blob/main/example_output.png?raw=true)
