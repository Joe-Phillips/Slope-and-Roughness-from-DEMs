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
import psutil
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
    ]  # filter out num_chunks lower than desired1
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
    windows = windows.copy()  # read-only hotfix

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


def manage_memory(dem_rows, dem_cols, window_size):
    """
    Manage memory allocation for DEM processing.

    Args:
        dem_rows (int): Number of rows in the DEM.
        dem_cols (int): Number of columns in the DEM.
        window_size (int): Processing window size in pixels.

    Returns:
        float: Desired number of chunks for DEM processing.
    """

    def prompt_user_memory_choice(total_memory_gb, min_memory_required_gb):
        """
        Prompt the user to select or specify memory allocation.

        Args:
            total_memory_gb (float): Total available system memory in GB.
            min_memory_required_gb (float): Minimum memory required for processing in GB.

        Returns:
            float: Allocated memory for processing in GB.
        """
        while True:
            user_choice = input(
                f"Available memory: {total_memory_gb:.2f} GB. Use all [0] or specify [1]? "
            )
            if user_choice == "0":
                return total_memory_gb * 0.90  # Use 90% of available memory
            elif user_choice == "1":
                try:
                    specified_memory = float(input("Specify memory to use (GB): "))
                    if 0 < specified_memory <= total_memory_gb:
                        if specified_memory >= min_memory_required_gb:
                            return specified_memory
                        else:
                            print(
                                f"Insufficient memory specified ({specified_memory:.2f} GB). "
                                f"At least {min_memory_required_gb:.2f} GB is required."
                            )
                    else:
                        print(f"Enter a value between 0 and {total_memory_gb:.2f} GB.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
            else:
                print("Invalid choice. Enter 0 or 1.")

    # Get total available system memory
    total_memory_gb = psutil.virtual_memory().available / (1024**3)
    print("Chunking DEM to reduce memory overhead...")

    # Constants for memory calculations
    bytes_per_pixel = 4  # float32 (4 bytes per pixel)
    overlap_pixels = window_size // 2
    safety_buffer = 1.1  # 10% safety margin

    # Minimum chunk dimensions and memory usage
    min_chunk_height = window_size
    min_chunk_pixels = dem_cols * min_chunk_height

    io_array_memory_gb = (
        3 * dem_rows * dem_cols * bytes_per_pixel / (1024**3)
    )  # Full input/output arrays
    min_chunk_memory_gb = (
        min_chunk_pixels * window_size**2 * bytes_per_pixel / (1024**3)
    )  # Smallest possible chunk in memory
    intermediate_memory_gb = (
        (min_chunk_pixels * 9 * bytes_per_pixel)  # U arrays
        + (min_chunk_pixels * 2 * window_size * bytes_per_pixel)  # Residual arrays
        + (3 * min_chunk_pixels * bytes_per_pixel)  # Gradient arrays
    ) / (1024**3)

    total_min_memory_gb = (
        io_array_memory_gb + min_chunk_memory_gb + intermediate_memory_gb
    ) * safety_buffer
    min_chunk_processing_memory_gb = (
        min_chunk_memory_gb + intermediate_memory_gb
    ) * safety_buffer

    # Check if system memory is sufficient
    if total_memory_gb < total_min_memory_gb:
        sys.exit(
            f"Insufficient system memory ({total_memory_gb:.2f} GB). "
            f"At least {total_min_memory_gb:.2f} GB is required. Exiting..."
        )

    # Prompt the user for memory allocation
    allocated_memory_gb = prompt_user_memory_choice(
    total_memory_gb, total_min_memory_gb
    )

    # Calculate number of chunks based on available memory
    available_memory_for_chunks_gb = allocated_memory_gb - io_array_memory_gb
    memory_ratio = available_memory_for_chunks_gb / min_chunk_processing_memory_gb
    desired_num_chunks = dem_rows / memory_ratio

    return desired_num_chunks


def dem_to_slope_and_roughness(dem_path, resolution, window_size, roughness_method):
    """
    Takes a DEM raster file, divides it into chunks with
    overlap, calculates the slope and roughness for each chunk using Singular Value Decomposition
    (SVD), and saves the resulting slope and roughness rasters.

    Args:
        dem_path (str): The file path to the DEM raster.
        resolution (float): The spatial resolution of the DEM in meters.
        window_size (int): The window size in meters for calculating slope and roughness.
        roughness_method (str): The method used to calculate roughness. Options: 'std' (standard deviation), 'mad' (median absolute deviation), 'p2t' (peak-to-trough).

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

    # Memory Management
    # Vectorizing over larger chunks improves speed but significantly increases memory usage, especially with a windowed view.
    # The nested SVD loop is the current bottleneck. Does increasing chunk size provide a proportional speedup,
    # or is it just unnecessarily memory-intensive?
    overlap = window_size_pixels // 2
    desired_num_chunks = manage_memory(num_rows, num_cols, window_size_pixels)

    # Chunk DEM
    # *Uniform* chunking not necessary, but done to make possible future SVD vectorisation solutions viable
    dem = uniform_chunk_2d(dem, overlap, desired_num_chunks)
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
        dtype=np.float32,
    )
    roughness_output = np.full(np.shape(slope_output), np.nan, dtype=np.float32)
    start_time = time.time()  # set start time

    # Iterate over chunks, calculating slope and roughness
    for i in range(num_chunks):

        print(f"Processing chunk {i+1}/{num_chunks}...")

        # Get windows over current chunk
        windowed_chunk = window_2d(dem[i], window_size_pixels)
        windowed_chunk_shape = np.shape(windowed_chunk)

        # Initialise orthogonal vectors (U) and residual vectors
        U = np.full(
            (windowed_chunk_shape[0], windowed_chunk_shape[1], 3, 3),
            np.nan,
            dtype=np.float32,
        )
        residuals = np.full(
            (windowed_chunk_shape[0], windowed_chunk_shape[1], window_size_pixels**2),
            np.nan,
            dtype=np.float32,
        )

        # Loop through windows, getting fitted plane and residuals using singular value decomposition
        # Frustrating this has to be a nested loop - can't figure out how to vectorise this
        for j in tqdm(range(windowed_chunk_shape[0])):
            for k in range(windowed_chunk_shape[1]):

                # Skip if all window values are nan
                if np.isnan(windowed_chunk[j, k]).all():
                    continue

                # Perform SVD
                U[j, k], residuals[j, k] = perform_svd(windowed_chunk[j, k], resolution)

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

        elif roughness_method == "p2t":
            with warnings.catch_warnings():  # catch warning for all nan slice
                warnings.simplefilter("ignore", category=RuntimeWarning)
                roughness_output[i] = np.nanmax(residuals, axis=-1) - np.nanmin(
                    residuals, axis=-1
                )

        else:
            sys.exit(f"{roughness_method} not a valid roughness method. Exiting...")

    # Combine chunks
    slope_output = np.asarray([np.concatenate(slope_output, axis=0)])
    roughness_output = np.asarray([np.concatenate(roughness_output, axis=0)])

    print(f"Saving...")

    # Save slope raster
    slope_output_path = dem_path.split("/")[-1].split(".")[0] + f"_slope_{window_size_pixels}x{window_size_pixels}.tif"
    slope_output_description = f"Slope raster derived from a DEM with a resolution of {resolution} meters, using a window size of {window_size} meters. Values represent the gradient magnitude (in degrees) of a plane fitted through the DEM points within each window.\nhttps://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"
    save_raster(dem_path, slope_output, slope_output_path, slope_output_description)

    # Save roughness raster
    roughness_output_path = (
        dem_path.split("/")[-1].split(".")[0] + f"_roughness_{roughness_method}_{window_size_pixels}x{window_size_pixels}.tif"
    )
    roughness_method_name = {
        "std": "standard deviation",
        "mad": "median absolute deviation",
        "p2t": "peak-to-trough",
    }
    roughness_output_description = f"Roughness raster derived from a DEM with a resolution of {resolution} meters, using a window size of {window_size} meters. Values represent the orthogonal variation ({roughness_method_name[roughness_method]}) of residuals from a plane fitted through the DEM points within each window (in meters)\nhttps://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"
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

    valid_methods = ["std", "mad", "p2t"]
    parser.add_argument(
        "roughness_method",
        help="The method used to calculate roughness. Options: 'std' (standard deviation), 'mad' (median absolute deviation), 'p2t' (peak-to-trough).",
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
