# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import rasterio
import numpy as np
import PIL
import os
import pathlib
from numpy.lib.stride_tricks import sliding_window_view
import time
import argparse

os.chdir(pathlib.Path(__file__).parent.resolve())

# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------

def generate_Slope_and_Roughness(DEM_PATH, RESOLUTION, WINDOW_SIZE, NUM_TILES):
    """
    Calculate slope and roughness from a Digital Elevation Model (DEM).

    This function takes a Digital Elevation Model (DEM) raster file, divides it into tiles with
    overlap, calculates the slope and roughness for each tile using Singular Value Decomposition
    (SVD), and saves the resulting slope and roughness rasters.

    Parameters:
        DEM_PATH (str): The file path to the DEM raster.
        RESOLUTION (float): The spatial resolution of the DEM in meters.
        WINDOW_SIZE (int): The window size in meters for calculating slope and roughness.
        NUM_TILES (int): The number of tiles to divide the DEM into.

    Returns:
        None
    """

    # ----------------------------------------------------------------------
    # Read in DEM
    # ----------------------------------------------------------------------

    PIL.Image.MAX_IMAGE_PIXELS = (
        618750000  # increase maximum allowed pixels so PIL can read in raster
    )
    im = PIL.Image.open(DEM_PATH)  # open raster
    NUM_COLS, NUM_ROWS = im.size  # get raster extent

    dem = np.asarray(im, dtype="float32")  # get elevation data as 2d array
    dem = dem.copy()  # hotfix to avoid read-only issues

    im.close()  # close raster connection

    void_data = np.where(
        dem <= -9999
    )  # void elevation data commonly given as -9999 in rasters - set to nan
    if np.any(void_data):
        dem[void_data] = np.nan

    WINDOW_SIZE_pixels = int(WINDOW_SIZE / RESOLUTION)  # get window size in pixels

    # ----------------------------------------------------------------------
    # Split up raster into manageable pieces with overlap
    # ----------------------------------------------------------------------

    # pad dem with nans so we can obtain windows from outer values
    overlap = WINDOW_SIZE_pixels // 2
    dem = np.pad(
        dem,
        ((overlap, overlap), (overlap, overlap)),
        "constant",
        constant_values=np.nan,
    )

    # check dem can be divided into given tiles, adjust if not
    TRY_CHECK = 0
    while NUM_ROWS % NUM_TILES != 0:
        TRY_CHECK = 1
        NUM_TILES += 1
    if TRY_CHECK == 1:
        print(
            "Couldn't divide array evenly into given tiles. Using "
            + str(NUM_TILES)
            + " tiles instead..."
        )

    TILE_HEIGHT = NUM_ROWS // NUM_TILES  # obtain height of each tile, unpadded

    dem = [
        dem[y - overlap : y + TILE_HEIGHT + overlap, 0 : NUM_COLS + overlap * 2]
        for y in range(overlap, NUM_ROWS, TILE_HEIGHT)
    ]  # split raster into tiles, ensuring each overlaps in x and y
    DEM_SHAPE = np.shape(dem)

    # ----------------------------------------------------------------------
    # Iterate over tiles, calculating gradient and roughness
    # ----------------------------------------------------------------------

    # intialise outputs, with tile overlaps removed
    slope_output = np.full(
        (DEM_SHAPE[0], DEM_SHAPE[1] - overlap * 2, DEM_SHAPE[2] - overlap * 2), np.nan
    )
    roughness_output = np.full(np.shape(slope_output), np.nan)

    START_TIME = time.time()

    # loop through tiles
    for i in range(NUM_TILES):

        START_TIME_loop = time.time()

        # ----------------------------------------------------------------------
        # Get windows
        # ----------------------------------------------------------------------

        windows = sliding_window_view(
            dem[i], (WINDOW_SIZE_pixels, WINDOW_SIZE_pixels)
        )  # obtain windows at each point in DEM - results in 4d array
        windows = windows.copy()  # hotfix
        WINDOW_SHAPE = (np.shape(windows)[0], np.shape(windows)[1])

        # set windows to nan that have more than 50% nans (~arbritrary)
        insufficient_windows = (
            np.count_nonzero(np.isnan(windows), axis=(-1, -2))
            >= (WINDOW_SIZE_pixels**2) * 0.5
        )
        windows[insufficient_windows] = np.full(
            (WINDOW_SIZE_pixels, WINDOW_SIZE_pixels), np.nan
        )

        # ----------------------------------------------------------------------
        # Get slope and roughness using SVD
        # ----------------------------------------------------------------------

        # center data
        windows = windows - np.nanmean(windows, axis=-1)[:, :, :, np.newaxis]

        # faux window for x and y centered around 0, step size equal to RESOLUTION
        faux_window_x = np.tile(
            (np.arange(WINDOW_SIZE_pixels) - (WINDOW_SIZE_pixels - 1) / 2) * RESOLUTION,
            (WINDOW_SIZE_pixels, 1),
        )
        faux_window_y = np.flip(faux_window_x.T)

        # stack faux x,y coordinate windows and tile to size of tile
        faux_window = np.stack((faux_window_y, faux_window_x), axis=-1)
        faux_window = np.tile(
            faux_window[:, :, :, np.newaxis, np.newaxis], WINDOW_SHAPE
        )
        faux_window = np.moveaxis(faux_window, -1, 0)
        faux_window = np.moveaxis(faux_window, -1, 0)

        # stack x,y,z coordinates
        windows = np.append(faux_window, windows[:, :, :, :, np.newaxis], axis=-1)

        # flatten windows to array of x,y,z coords
        windows = np.reshape(
            windows, (WINDOW_SHAPE[0], WINDOW_SHAPE[1], WINDOW_SIZE_pixels**2, 3)
        )

        # get nan mask
        nan_mask = np.sum(windows, axis=-1)  # make x,y nan where z is nan
        nan_mask = np.isnan(nan_mask)  # get boolean for nan

        # loop through windows, getting fitted plane and residuals using singular value decomposition
        U = np.full((WINDOW_SHAPE[0], WINDOW_SHAPE[1], 3, 3), np.nan)
        residuals = np.full(
            (WINDOW_SHAPE[0], WINDOW_SHAPE[1], WINDOW_SIZE_pixels**2), np.nan
        )
        for j in range(WINDOW_SHAPE[0]):
            for k in range(WINDOW_SHAPE[1]):

                # skip if all window values are nan
                if nan_mask[j, k].all():
                    continue

                # https://docs.pyvista.org/_modules/pyvista/utilities/helpers.html#fit_plane_to_points
                # https://www.mbfys.ru.nl/~robvdw/CNP04/LAB_ASSIGMENTS/LAB05_CN05/MATLAB2007b/stats/html/orthoregdemo.html
                U[j, k], _, _ = np.linalg.svd(
                    windows[j, k][~nan_mask[j, k]].T
                )  # U gives orthogonal vectors of fitted plane
                residuals[j, k] = np.einsum(
                    "ij, ij->i",
                    windows[j, k],
                    np.tile(U[j, k, :, 2], (WINDOW_SIZE_pixels**2, 1)),
                )  # residuals given by dot product of plane normal unit vector with points (einsum ~2x faster than np.dot)

        # use normal vector of slope to solve slope equation and find dz along x and y axes, sum resulting vectors, take gradient
        dz_x = -(1 / U[:, :, 2, 2]) * (RESOLUTION * U[:, :, 1, 2])
        dz_y = -(1 / U[:, :, 2, 2]) * (RESOLUTION * U[:, :, 0, 2])
        grad = (dz_x + dz_y) / np.sqrt(RESOLUTION**2 + RESOLUTION**2)

        # take absoulte gradient angle as slope
        slope_output[i] = np.absolute(np.rad2deg(np.arctan(grad)))

        # take std of absolute residuals as roughness
        roughness_output[i] = np.nanstd(np.absolute(residuals), axis=-1)

        print(
            "Tile "
            + str(i + 1)
            + "/"
            + str(NUM_TILES)
            + " complete! Time taken: "
            + str(time.time() - START_TIME_loop)
            + "s."
        )

    # ----------------------------------------------------------------------
    # Finalise and save
    # ----------------------------------------------------------------------

    # combine tiles
    slope_output = np.array([np.concatenate(slope_output, axis=0)])
    roughness_output = np.array([np.concatenate(roughness_output, axis=0)])
    size = np.shape(slope_output)

    # Get affine transform
    with rasterio.open(DEM_PATH) as src:
        transform = src.profile["transform"]

    # save slope raster
    with rasterio.open(
        "REMA_Slope_" + str(RESOLUTION) + "m_" + str(WINDOW_SIZE) + "ws.tif",
        "w",
        driver="GTiff",
        height=size[1],
        width=size[2],
        count=1,
        dtype=np.float32,
        crs="epsg:3031",
        transform=transform,
    ) as dest_file:
        dest_file.write(slope_output)
    dest_file.close()

    # save roughness raster
    with rasterio.open(
        "REMA_Roughness_" + str(RESOLUTION) + "m_" + str(WINDOW_SIZE) + "ws.tif",
        "w",
        driver="GTiff",
        height=size[1],
        width=size[2],
        count=1,
        dtype=np.float32,
        crs="epsg:3031",
        transform=transform,
    ) as dest_file:
        dest_file.write(roughness_output)
        dest_file.close()

    print("\nDone! Time taken: " + str(time.time() - START_TIME) + "s.")


# ----------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

# Read in command-line variables
parser = argparse.ArgumentParser(
    description="Create slope and roughness maps from DEM data."
)

parser.add_argument("DEM_PATH", help="The path to the DEM file.")
parser.add_argument("DEM_RES", help="The resolution of the DEM in meters.")
parser.add_argument(
    "WINDOW_SIZE",
    help="The size of the window around each pixel in meters over which slope and roughness will be calculated.",
)
parser.add_argument(
    "NUM_TILES",
    help="The number of tiles to split the DEM into during processing to minimise memory usage. This will depend on the size of the DEM and the amount of available memory.",
)

args = parser.parse_args()

# Call the function with the provided command-line arguments
generate_Slope_and_Roughness(
    args.DEM_PATH,
    args.DEM_RES,
    args.WINDOW_SIZE,
    args.NUM_TILES,
)
