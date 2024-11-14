# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import argparse
import os
import pathlib
import time
import warnings
import sys

import numpy as np
import rasterio
from numpy.lib.stride_tricks import sliding_window_view
from psutil import virtual_memory
from tqdm import tqdm

os.chdir(pathlib.Path(__file__).parent.resolve())

# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------


def read_raster(dem_path):
    """
    Reads a raster from the specified path, converts it to a 2D numpy array,
    and replaces void data values (-9999) with NaN.

    Args:
        dem_path (str): Path to the raster file.

    Returns:
        numpy.ndarray: 2D array of values with voids set to NaN.
    """
    # Open raster file using rasterio
    with rasterio.open(dem_path) as src:
        # Read the data into a numpy array
        dem = src.read(1, masked=True).astype("float32")  # Read the first band

    # Void elevation data commonly given as -9999 in rasters - set to nan
    dem = np.where(dem == -9999, np.nan, dem)

    return dem


def get_factors(n):
    """
    Calculates all factors of a given integer.

    Args:
        n (int): The integer for which to find factors.

    Returns:
        numpy.ndarray: Sorted array of factors of the integer `n`.
    """
    # Loop through integers and find factors
    factors = set()  # use a set to avoid duplicate factors
    for i in range(1, int(n**0.5) + 1):

        # Check if i is a factor
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)

    return np.array(sorted(factors))


def uniform_chunk_2d(data, overlap, desired_num_chunks):
    """
    Divides a 2D array into a specified number of lengthwise chunks of equal size, allowing optional overlap.

    Args:
        data (numpy.ndarray): The 2D array to be split into chunks.
        overlap (int): The number of rows to overlap between adjacent chunks. Pads data with NaNs of
                       this width to ensure overlap at boundaries.
        desired_num_chunks (int): The intended number of chunks to divide the array into. The actual
                                  number may be adjusted based on array dimensions and overlap to ensure
                                  uniform chunk sizes.

    Returns:
        list: A list of numpy arrays, each representing a 2D chunk of the original data with specified overlap.
    """

    # Pad data with nans of size overlap (allows for seamless windowing around border values)
    num_rows, num_cols = np.shape(data)
    data = np.pad(
        data,
        ((overlap, overlap), (overlap, overlap)),
        "constant",
        constant_values=np.nan,
    )

    # Get number of chunks to evenly split data into
    possible_num_chunks = get_factors(num_rows)  # get possible candidates
    possible_num_chunks = possible_num_chunks[
        possible_num_chunks - desired_num_chunks >= 0
    ]  # filter out num_chunks lower than desired
    num_chunks = possible_num_chunks[
        np.argmin(np.absolute(possible_num_chunks - desired_num_chunks))
    ]  # get closest candidate that evenly divides the data

    print(f"Divided into {num_chunks} chunk(s).")

    # Split data into chunks, with overlaps in x and y
    chunk_height = num_rows // num_chunks
    data = [
        data[y - overlap : y + chunk_height + overlap, 0 : num_cols + overlap * 2]
        for y in range(overlap, num_rows, chunk_height)
    ]

    return data


def window_2d(data, window_size_pixels, center_values=True, nan_threshold=0.5):
    """
    Extracts square windows of data from a 2D array centered on each cell, with optional mean-centering
    and NaN thresholding.

    Args:
        data (numpy.ndarray): 2D array of data values.
        window_size_pixels (int): Width and height of each square window in pixels.
        center_values (bool): If True, centers the values in each window by subtracting the mean
                              of values in that window. Default is True.
        nan_threshold (float): Proportion of NaN values allowed per window (0 to 1). Windows
                               with a higher proportion of NaNs are excluded by setting all
                               elements in the window to NaN. Default is 0.5.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: 4D array (H, W, window_size_pixels, window_size_pixels) where each
              slice contains the data values for a window centered on each cell in the input array.
    """
    # Obtain windows at each point - results in 4d array
    windows = sliding_window_view(data, (window_size_pixels, window_size_pixels))
    windows = windows.copy()  # hotfix

    # Set windows to nan that have more than nan_threshold% nans (~arbritrary)
    insufficient_windows = (
        np.count_nonzero(np.isnan(windows), axis=(-1, -2))
        >= (window_size_pixels**2) * nan_threshold
    )
    windows[insufficient_windows] = np.full(
        (window_size_pixels, window_size_pixels), np.nan
    )

    # Center windows
    if center_values:
        with warnings.catch_warnings():  # catch "mean of empty slice" warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            windows = windows - np.nanmean(windows, axis=-1)[:, :, :, np.newaxis]

    return windows


def perform_svd(z, resolution):
    """
    Applies Singular Value Decomposition (SVD) on a 2D elevation array to fit a best-fit plane
    and calculate the residuals relative to this plane.

    References:
        - PyVista: Plane fitting using SVD: https://docs.pyvista.org/_modules/pyvista/utilities/helpers.html#fit_plane_to_points
        - Orthogonal Regression Demo: https://www.mbfys.ru.nl/~robvdw/CNP04/LAB_ASSIGMENTS/LAB05_CN05/MATLAB2007b/stats/html/orthoregdemo.html

    Args:
        z (numpy.ndarray): 2D array of elevation (height) values.
        resolution (float): Spatial resolution of the elevation data in meters.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Matrix of orthogonal vectors (3x3), where the last column is the normal vector of the fitted plane.
            - numpy.ndarray: Array of residuals, indicating the vertical distances of each point in the original elevation data
                             from the fitted plane, in the same shape as the input array.
    """

    # Get x and y coords centered around 0
    z_shape = np.shape(z)
    x = np.tile(
        (np.arange(z_shape[0]) - (z_shape[0] - 1) / 2) * resolution,
        (z_shape[0], 1),
    )
    y = np.flip(x.T)

    # Flatten x, y, and z coords
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Stack x,y,z coordinates
    xyz = np.column_stack((x, y, z))

    # Get orthogonal vectors of fitted plane
    nan_mask = np.isnan(z)
    U, _, _ = np.linalg.svd(xyz[~nan_mask].T)

    # Get residuals (dot product of plane normal unit vector with points - einsum ~2x faster than np.dot)
    residuals = np.einsum(
        "ij, ij->i",
        xyz,
        np.tile(U[:, 2], (z_shape[0] ** 2, 1)),
    )

    return U, residuals


def nanmad(data, axis=None):
    """
    Calculates the median absolute deviation (MAD) of an array, ignoring NaN values.

    Args:
        data (numpy.ndarray): Input data array.
        axis (int, optional): Axis along which to calculate the MAD. Default is None.

    Returns:
        numpy.ndarray or float: MAD of the input data along the specified axis.
    """
    # Ignore NaN values in the median calculation
    with warnings.catch_warnings():  # catch warning for all nan slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(data, axis=axis)

    # Calculate the absolute deviations from the median
    abs_deviation = np.abs(data - np.expand_dims(median, axis=axis))

    # Calculate the MAD, again ignoring NaN values
    with warnings.catch_warnings():  # catch warning for all nan slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mad = np.nanmedian(abs_deviation, axis=axis)

    return mad


def save_raster(dem_path, output, output_path, output_description):
    """
    Saves data as a raster, taking metadata from a source file.

    Args:
        dem_path (str): Path to the source DEM file.
        output (numpy.ndarray): Modified DEM data to be saved.
        output_path (str): Path where the output raster will be saved.
        output_description (str): Description of the output raster.

    Returns:
        None
    """
    # Open source DEM to get transform and metadata
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        metadata = src.meta
        height, width = src.height, src.width
        description = src.tags().get("description", "No description available.")

    # Combine the given description with the source DEM description
    description = f"{output_description}\n\nSource DEM Description:\n{description}"

    # Update metadata for the new file
    metadata.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": np.float32,
            "crs": crs,
            "transform": transform,
        }
    )

    # Save the output raster
    with rasterio.open(output_path, "w", **metadata) as dest_file:
        dest_file.update_tags(description=description)  # add the description as a tag
        dest_file.write(output)


def dem_to_slope_and_roughness(dem_path, resolution, window_size, roughness_method):
    """
    Takes a DEM raster file, divides it into chunks with
    overlap, calculates the slope and roughness for each chunk using Singular Value Decomposition
    (SVD), and saves the resulting slope and roughness rasters.

    Args:
        dem_path (str): The file path to the DEM raster.
        resolution (float): The spatial resolution of the DEM in meters.
        window_size (int): The window size in meters for calculating slope and roughness.
        roughness_method (str): The method used to calculate roughness. Options: 'std' (standard deviation), 'mad' (median absolute deviation), 'p2p' (peak-to-peak).

    Returns:
        None
    """

    # check window size produces odd number of pixels
    window_size_pixels = int(window_size / resolution)
    if window_size_pixels % 2 == 0:
        print(
            f"WARNING: Window size of {window_size} m produces an even number of pixels ({window_size_pixels}), but an odd number is required."
        )

        # Options for nearest odd window sizes
        option_1, option_2 = (window_size_pixels + 1) * resolution, (
            window_size_pixels - 1
        ) * resolution
        choice = None

        # Loop until valid choice is made
        while choice not in {"0", "1"}:
            choice = input(
                f"Would you prefer a window size of {option_1} m [0] or {option_2} m [1]? "
            )
            if choice not in {"0", "1"}:
                print("Invalid input. Please input 0 or 1.")

        # Update window size based on choice
        window_size = option_1 if choice == "0" else option_2
        window_size_pixels = int(window_size / resolution)

    # Read in DEM
    print("Reading in DEM...")
    dem = read_raster(dem_path)
    num_rows, num_cols = np.shape(dem)

    # Get available memory
    print("Chunking DEM to reduce memory overhead...")
    available_memory = virtual_memory().available
    available_memory = available_memory / (1024**3)

    # Confirm how much memory to use
    while True:
        choice = input(
            f"Available memory {available_memory:.2f} Gb. Do you want to use all available [0] or specify [1]? "
        )

        if choice == "0":
            available_memory *= 0.9  # apply buffer
            print(f"Using {available_memory:.2f} Gb...")
            break
        elif choice == "1":
            try:
                specified_memory = float(input("Please specify memory to use in Gb: "))
                if 0 < specified_memory <= available_memory:
                    available_memory = specified_memory
                    print(f"Using {available_memory:.2f} Gb...")
                    break
                else:
                    print(
                        f"Please enter a value between 0 and {available_memory:.2f} Gb."
                    )
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
        else:
            print("Invalid input. Please enter 0 or 1.")

    # Split up DEM into manageable chunks with overlap
    overlap = window_size_pixels // 2  # overlap half the size of the windows
    approx_total_memory_needed = (num_cols * num_rows * (window_size_pixels**2) * 4) / (
        1024**3
    )  # *very* rough approx (float32 = 4 bytes)
    desired_num_chunks = (
        approx_total_memory_needed / available_memory
    )  # desired number of chunks
    dem = uniform_chunk_2d(dem, overlap, desired_num_chunks)  # chunk DEM
    num_chunks = len(dem)
    chunked_dem_shape = np.shape(dem)

    # Intialise outputs, with chunk overlaps removed
    overlap = window_size_pixels // 2
    slope_output = np.full(
        (
            chunked_dem_shape[0],
            chunked_dem_shape[1] - overlap * 2,
            chunked_dem_shape[2] - overlap * 2,
        ),
        np.nan,
    )
    roughness_output = np.full(np.shape(slope_output), np.nan)
    start_time = time.time()  # set start time

    # Iterate over chunks, calculating slope and roughness
    for i in range(num_chunks):

        print(f"Processing chunk {i+1}/{num_chunks}...")

        # Get windows over current chunk
        windows = window_2d(dem[i], window_size_pixels)
        chunk_shape = np.shape(windows)

        # Initialise orthogonal vectors (U) and residual vectors
        U = np.full((chunk_shape[0], chunk_shape[1], 3, 3), np.nan)
        residuals = np.full(
            (chunk_shape[0], chunk_shape[1], window_size_pixels**2), np.nan
        )

        # Loop through windows, getting fitted plane and residuals using singular value decomposition
        for j in tqdm(range(chunk_shape[0])):
            for k in range(chunk_shape[1]):

                # Skip if all window values are nan
                if np.isnan(windows[j, k]).all():
                    continue

                # Perform SVD
                U[j, k], residuals[j, k] = perform_svd(windows[j, k], resolution)

        # Use normal vector of slope to solve slope equation and find dz along x and y axes, sum resulting vectors, take gradient
        dz_x = -(1 / U[:, :, 2, 2]) * (resolution * U[:, :, 1, 2])
        dz_y = -(1 / U[:, :, 2, 2]) * (resolution * U[:, :, 0, 2])
        grad = (dz_x + dz_y) / np.sqrt(resolution**2 + resolution**2)

        # Take absoulte gradient angle as slope
        slope_output[i] = np.absolute(np.rad2deg(np.arctan(grad)))

        # Get roughness from residuals
        if roughness_method == "std":
            with warnings.catch_warnings():  # catch warning for all nan slice
                warnings.simplefilter("ignore", category=RuntimeWarning)
                roughness_output[i] = np.nanstd(np.absolute(residuals), axis=-1)

        elif roughness_method == "mad":
            roughness_output[i] = nanmad(np.absolute(residuals), axis=-1)

        elif roughness_method == "p2p":
            with warnings.catch_warnings():  # catch warning for all nan slice
                warnings.simplefilter("ignore", category=RuntimeWarning)
                roughness_output[i] = np.nanmax(residuals, axis=-1) - np.nanmin(
                    residuals, axis=-1
                )

        else:
            sys.exit(f"{roughness_method} not a valid roughness method. Exiting...")

    # Combine chunks
    slope_output = np.array([np.concatenate(slope_output, axis=0)])
    roughness_output = np.array([np.concatenate(roughness_output, axis=0)])

    print(f"Saving...")

    # Save slope raster
    slope_output_path = dem_path.split("/")[-1].split(".")[0] + "_slope.tif"
    slope_output_description = f"Slope raster calculated from a DEM at a resolution of {resolution} meters, using a window size of {window_size} meters. Values represent the gradient of the DEM within each window in degrees.\nhttps://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"
    save_raster(dem_path, slope_output, slope_output_path, slope_output_description)

    # Save roughness raster
    roughness_output_path = dem_path.split("/")[-1].split(".")[0] + "_roughness.tif"
    roughness_method_name = {
        "std": "standard deviation",
        "mad": "median absolute deviation",
        "p2p": "peak-to-peak",
    }
    roughness_output_description = f"Roughness raster calculated from a DEM at a resolution of {resolution} meters, using a window size of {window_size} meters. Values represent the surface roughness variation ({roughness_method_name[roughness_method]}) of the DEM within each window in meters.\nhttps://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"
    save_raster(
        dem_path, roughness_output, roughness_output_path, roughness_output_description
    )

    print("\nDone! Time taken: " + str(time.time() - start_time) + "s.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Read in command-line variables
    parser = argparse.ArgumentParser(
        description="Create slope and roughness maps from DEM data using SVD."
    )

    parser.add_argument("dem_path", help="The path to the DEM file.", type=str)
    parser.add_argument(
        "dem_resolution", help="The resolution of the DEM in meters.", type=int
    )
    parser.add_argument(
        "window_size",
        help="The size of the window around each pixel in meters over which slope and roughness will be calculated.",
        type=int,
    )

    valid_methods = ["std", "mad", "p2p"]
    parser.add_argument(
        "roughness_method",
        help="The method used to calculate roughness. Options: 'std' (standard deviation), 'mad' (median absolute deviation), 'p2p' (peak-to-peak).",
        type=str,
        choices=valid_methods,
        nargs="?",
        default="std",
    )

    args = parser.parse_args()

    # Run
    dem_to_slope_and_roughness(
        args.dem_path, args.dem_resolution, args.window_size, args.roughness_method
    )
