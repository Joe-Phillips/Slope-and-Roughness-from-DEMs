# Slope-and-Roughness-from-DEMs

This project produces **slope** and **roughness** maps using **Digital Elevation Model** (DEM) data and **Least Squares Regression**.

Made by Joe Phillips.

[![Repo](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips)
[![Repo](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp;✉️ j.phillips5@lancaster.ac.uk

## :toolbox: How it Works

Commonly used slope algorithms, such as Horn's method, estimate slope based on local elevation differences across a small, fixed window (typically 3×3 pixels). By approximating partial derivatives using immediate neighbors, these methods are highly sensitive to noise and elevation variability, often leading to exaggerated slope values in areas with rough or noisy terrain.

Similarly, standard roughness metrics like the Terrain Ruggedness Index (TRI) and Topographic Position Index (TPI) fail to account for underlying slope when assessing elevation variance. These methods calculate elevation differences between a central pixel and its neighbors, which means that a perfectly smooth but tilted surface produces non-zero roughness values. Consequently, the roughness values produced by these methods are affected by slope in a way that makes it difficult to isolate their individual contributions to topographic variation.

Despite these limitations, these approaches are widely used in GIS software such as GRASS, ArcGIS, and GDAL (the backend for QGIS).

This project improves upon traditional methods by calculating slope through Least Squares Regression, fitting a plane to a given DEM point and its surrounding pixels. The slope is then derived from the plane’s gradient in the x and y directions and expressed in degrees. Roughness is computed independently of slope by analysing the dispersion of _orthogonal_ residuals from the fitted plane. By measuring roughness in a direction perpendicular to the local slope trend, this approach ensures that roughness values truly reflect surface variability rather than being biased by overall terrain inclination. The roughness can be quantified using one of three methods:

- **Range** (range): *Maximum minus minimum residuals.*
- **Standard Deviation** (std): *The standard deviation of residuals.*
- **Median Absolute Deviation** (mad): *The median absolute deviation of residuals.*

Unlike traditional approaches, this method also allows for a flexible, sliding-window application beyond a fixed 3×3 neighborhood. Larger window sizes (e.g., 9×9 pixels) can be used to incorporate more data points, producing smoother and more reliable slope and roughness estimates.

## 🛠️ Usage Guide

First, make sure you have installed the required packages. This can be done via **pip install -r requirements.txt**.

To generate the slope and roughness maps, simply run **dem_to_slope_and_roughness.py** from the command line with the following arguments:

- **DEM_PATH** (string): *The path to the DEM file.*
- **DEM_RESOLUTION** (string): *The resolution of the DEM in meters.*
- **WINDOW_SIZE** (int): *The size of the window around each pixel in meters over which slope and roughness will be calculated.*
- **ROUGHNESS_METHOD** (int): *The method used to calculate roughness. Options: 'range' (minimum-to-maximum difference), 'std' (standard deviation), 'mad' (median absolute deviation). Defaults to 'range'.*
- --**N_PROCESSES** (optional, int): *The number of processes to use for computation. Defaults to using all available CPU cores.*
- --**TILE_SIZE** (optional, int): *The size of tiles in pixels for processing large DEMs efficiently. Defaults to 256.*

### Example:

```sh
python dem_to_slope_and_roughness.py example_folder/DEM.tif 200 900 std --n_processes 4 --tile_size 256
```

<br>

## :camera: Images
**Example output over Antarctica using REMA** (https://www.pgc.umn.edu/data/rema/) **at 200 m resolution and a window size of 1000 m (5x5)**. *(Plot created separately).*

<br>

![alt text](https://github.com/Joe-Phillips/DEM-to-Slope-and-Roughness/blob/main/example_output.png?raw=true)

