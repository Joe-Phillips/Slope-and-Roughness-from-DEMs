# Slope and Roughness from DEMs

Generates **slope**, **roughness**, and (optionally) **aspect** rasters from Digital Elevation Model (DEM) data by fitting a plane to local elevation windows. **Slope** is derived from the plane's gradient, **roughness** from the dispersion of the **orthogonal residuals**, and **aspect** from the direction of steepest descent.

---

Made by **Joe Phillips**  

[![GitHub](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips)
[![LinkedIn](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp; ✉️ j.phillips5@lancaster.ac.uk

---

## Overview

Commonly used slope algorithms such as Horn's method estimate slope from local elevation differences across a fixed 3×3 window. By approximating partial derivatives using immediate neighbours, these methods are highly sensitive to noise and can produce exaggerated slope values over rough terrain.

Similarly, standard roughness metrics like TRI and TPI calculate elevation differences between a central pixel and its neighbours without accounting for underlying slope - meaning a smooth but tilted surface will produce non-zero roughness values, making it difficult to isolate roughness from slope in topographic variation.

This project improves on these approaches by fitting a plane over a configurable local window around each pixel. Two fitting methods are available:

- **Least Squares (LS)** - fast and effective for most terrain.
- **Singular Value Decomposition (SVD)** - slower but more robust over rugged terrain. Unlike LS, which minimises vertical residuals, SVD minimises **orthogonal residuals**, making it less sensitive to terrain anisotropy.

Slope is derived from the plane's x and y gradients (in degrees). Roughness is computed from the **orthogonal residuals** to the fitted plane, ensuring it reflects true surface variability rather than being inflated by overall terrain inclination. Three roughness statistics are supported:

- **`range`** - maximum minus minimum residuals
- **`std`** - standard deviation of residuals  
- **`mad`** - median absolute deviation of residuals

Aspect (direction of steepest descent, degrees clockwise from North) can optionally be computed from the same plane fit.

## Usage

Install the required packages:
```bash
pip install -r requirements.txt
```

Then run from the command line:
```bash
python dem_to_slope_and_roughness.py <dem_path> <dem_resolution> <window_size> [roughness_method] [options]
```

### Positional Arguments

| Argument | Type | Description |
|---|---|---|
| `dem_path` | string | Path to the input DEM file. |
| `dem_resolution` | int | DEM pixel size in metres. |
| `window_size` | int | Analysis window size in metres. |
| `roughness_method` | string | Roughness statistic: `range` (default), `std`, or `mad`. |

### Optional Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `-a`, `--aspect` | flag | `false` | Also compute and save an aspect raster. |
| `-m`, `--method` | string | `svd` | Plane-fitting method: `svd` (minimises orthogonal residuals) or `ls` (least squares, faster). |
| `-p`, `--processes` | int | cpu_count | Number of parallel workers. Set to `1` to run in serial. |
| `-t`, `--tile_size` | int | `256` | Tile side length in pixels for parallel dispatch. |

### Example
```bash
python dem_to_slope_and_roughness.py example_folder/DEM.tif 100 900 range --method svd --processes 4 --aspect
```

## Preview

**Example output over Antarctica using REMA** (https://www.pgc.umn.edu/data/rema/) **at 200 m resolution and a window size of 1000 m (5×5)**. *(Plot created separately.)*

![Example output](https://github.com/Joe-Phillips/DEM-to-Slope-and-Roughness/blob/main/example_output.png?raw=true)
