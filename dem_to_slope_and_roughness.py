# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import argparse
from itertools import product
import time
import warnings
import numpy as np
import rasterio
from tqdm import tqdm
import multiprocessing as mp
from itertools import product
from functools import partial
import tempfile
import os
import sys
from multiprocessing.shared_memory import SharedMemory
import logging
import atexit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# os.chdir(pathlib.Path(__file__).parent.resolve())

# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------


def read_raster(dem_path):
    """
    Reads the first band of a raster from the specified path, converts it to a 2D numpy array,
    and replaces void data values with NaN.

    Args:
        dem_path (str): Path to the raster file.

    Returns:
        numpy.ndarray: 2D array of values with voids set to NaN.
    """

    # Open raster file using rasterio
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype("float32")  # Read the first band

    # Replace void values with NaN
    nodata_value = src.nodata
    dem = np.where(dem == nodata_value, np.nan, dem)

    return dem


def fit_plane(x, y, z):
    """
    Fits a plane to a set of 3D points using least squares regression and extracts the surface gradients
    in the x and y directions.

    Args:
        x (numpy.ndarray): 1D array of x-coordinate values, obtained from flattening a zero-centered 2D array.
        y (numpy.ndarray): 1D array of y-coordinate values, obtained from flattening a zero-centered 2D array.
        z (numpy.ndarray): 1D array of elevation (height) values, obtained from flattening a zero-centered 2D array.

    Returns:
        tuple: A tuple containing:
            - a (float): Gradient of the fitted plane in the x direction (dz/dx).
            - b (float): Gradient of the fitted plane in the y direction (dz/dy).
    """

    # Remove NaN values
    valid_mask = ~np.isnan(z)
    x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

    # Design matrix for least squares (Ax + By + D = Z)
    A = np.column_stack((x, y, np.ones_like(x)))
    coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # Solve for a, b, d

    # Extract coefficients
    a, b, _ = coef  # a = dz/dx, b = dz/dy

    return a, b


def get_orthogonal_residuals(x, y, z, a, b):
    """
    Computes the orthogonal residuals (perpendicular distances) from a set of 3D points
    to a fitted plane.

    Args:
        x (numpy.ndarray): 1D array of x-coordinate values, obtained from flattening a zero-centered 2D array.
        y (numpy.ndarray): 1D array of y-coordinate values, obtained from flattening a zero-centered 2D array.
        z (numpy.ndarray): 1D array of elevation (height) values, obtained from flattening a zero-centered 2D array.
        a (float): Gradient of the fitted plane in the x direction (dz/dx).
        b (float): Gradient of the fitted plane in the y direction (dz/dy).

    Returns:
        numpy.ndarray: 1D array of orthogonal residuals, representing the perpendicular distances
                       of each point to the fitted plane.
    """

    # Compute residuals
    fitted_plane = a * x + b * y
    residuals = z - fitted_plane

    # Compute orthogonal residuals (perpendicular distances to the plane)
    norm = np.sqrt(a**2 + b**2 + 1)
    orthogonal_residuals = residuals / norm

    return orthogonal_residuals


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

    # Convert NaNs to -9999
    output[np.isnan(output)] = -9999

    # Reshape to give 1-length channel dimension
    output = output[np.newaxis, :, :]

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
            "nodata": -9999,
        }
    )

    # Save the output raster
    with rasterio.open(output_path, "w", **metadata) as dest_file:
        dest_file.update_tags(description=description)  # add the description as a tag
        dest_file.write(output)


def process_tile(tile_coords, x, y, roughness_method, window_size_pixels, resolution):
    """
    Processes a single tile of the DEM, computing the slope and roughness 
    values for each pixel over a defined window.

    Args:
        tile_coords (tuple): A tuple of the form (row_start, row_end, col_start, col_end), defining 
                              the boundaries of the tile in the DEM.
        x (numpy.ndarray): The x-coordinates (pixel indices) corresponding to the DEM data.
        y (numpy.ndarray): The y-coordinates (pixel indices) corresponding to the DEM data.
        roughness_method (str): The method used to compute the roughness of the surface. Options include:
                               'std' (standard deviation), 'mad' (median absolute deviation), 
                               'range' (range between the maximum and minimum residuals).
        window_size_pixels (int): The size of the moving window (in pixels) used for calculating slope and roughness.
        resolution (float): The spatial resolution of the DEM in meters per pixel.

    Returns:
        tuple: A tuple (row_start, col_start, local_slope, local_roughness), where:
            - row_start, col_start are the coordinates of the top-left corner of the tile.
            - local_slope (numpy.ndarray): The calculated slope values for the tile.
            - local_roughness (numpy.ndarray): The calculated roughness values for the tile.
    """

    try:
        # Get tile row and col start/end
        row_start, row_end, col_start, col_end = tile_coords

        # Initialise slope and roughness arrays at tile location
        local_slope = np.full((row_end - row_start, col_end - col_start), np.nan, dtype=np.float32)
        local_roughness = np.full((row_end - row_start, col_end - col_start), np.nan, dtype=np.float32)
        
        # Loop through pixels in tile
        for row, col in product(range(row_start, row_end), range(col_start, col_end)):

            # Get window around pixel
            window_size_pixels_half = window_size_pixels//2
            window = shared_dem[row - window_size_pixels_half : row + window_size_pixels_half + 1, col - window_size_pixels_half : col + window_size_pixels_half + 1]

            # Skip if 50% or more values are NaNs
            if np.isnan(window).sum() > (window_size_pixels ** 2) / 2: 
                continue
            
            z = (window - np.nanmean(window)).flatten()
            a, b = fit_plane(x, y, z)
            dz_x, dz_y = a * resolution, b * resolution
            local_slope[row - row_start, col - col_start] = np.abs(np.rad2deg(np.arctan(np.sqrt(dz_x**2 + dz_y**2))))
            
            residuals = get_orthogonal_residuals(x, y, z, a, b)
            if roughness_method == "std":
                local_roughness[row - row_start, col - col_start] = np.nanstd(np.abs(residuals))
            elif roughness_method == "mad":
                local_roughness[row - row_start, col - col_start] = nanmad(np.abs(residuals))
            elif roughness_method == "range":
                local_roughness[row - row_start, col - col_start] = np.nanmax(residuals) - np.nanmin(residuals)
        
        return row_start, col_start, local_slope, local_roughness
    
    except Exception as e:
        logger.error(f"Error at tile {tile_coords}: {e}")
        return None, None, None, None


def init_worker(shared_name, shape, dtype):
    """
    Initialise worker processes with access to shared memory.

    Args:
        shared_name (str): The name of the shared memory block.
        shape (tuple): The shape of the DEM data array to reshape the shared memory into.
        dtype (numpy.dtype): The data type of the shared DEM array.

    Returns:
        None
    """
    global shared_dem
    existing_shm = SharedMemory(name=shared_name) # Reconnect to shared memory by name
    shared_dem = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf) # Interpret it as a numpy array
    atexit.register(existing_shm.close) # Ensure workers close their connection when done


def dem_to_slope_and_roughness(dem_path, resolution, window_size, roughness_method, n_processes=None, tile_size=256):
    """
    Calculates slope and roughness from a DEM raster.

    Args:
        dem_path (str): The file path to the DEM raster.
        resolution (float): The spatial resolution of the DEM in meters.
        window_size (int): The window size in meters for calculating slope and roughness.
        roughness_method (str): The method used to calculate roughness.
                               Options: 'std' (standard deviation),
                                       'mad' (median absolute deviation),
                                       'range' (minimum-to-maximum difference).
        n_processes (int, optional): Number of processes to use. Defaults to None (uses CPU count).

    Returns:
        None
    """

    # Set start time
    start_time = time.time()

    # Check window size produces odd number of pixels
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

        # If running interactively, give choice to user
        if sys.stdin.isatty(): 

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

        else:
            window_size = option_1

    # Read in DEM
    print("Reading in DEM...")
    dem = read_raster(dem_path)
    num_rows, num_cols = dem.shape

    # Create a shared memory DEM to enable efficient multiprocessing without redundant copies
    # Add NaN padding to support edge computations during window-based operations 
    pad_width = window_size_pixels // 2
    shm = SharedMemory(create=True, size=(num_rows + 2 * pad_width) * (num_cols + 2 * pad_width) * np.float32().nbytes) # Define raw block of memory
    if n_processes == 1: # make global in case of serial - in parallel, processes are initialised with this
        global shared_dem
    shared_dem = np.ndarray((num_rows + 2 * pad_width, num_cols + 2 * pad_width), dtype=np.float32, buffer=shm.buf) # Interpret it as a numpy array
    shared_dem[:] = np.nan # Fill with NaN
    shared_dem[pad_width:-pad_width, pad_width:-pad_width] = dem # Fill non-padded region with input DEM
    del dem

    # Precompute x, y coordinates for each window
    x = (np.arange(window_size_pixels) - (window_size_pixels - 1) / 2) * resolution
    y = (np.arange(window_size_pixels) - (window_size_pixels - 1) / 2) * resolution
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    try:
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:

            # Define file paths for temporary disk-backed arrays storing slope and roughness data
            slope_path = os.path.join(tmpdir, 'slope_tmp.dat')
            roughness_path = os.path.join(tmpdir, 'roughness_tmp.dat')
            
            # Create memory-mapped arrays to store slope and roughness efficiently without loading them entirely into memory
            slope_array = np.memmap(slope_path, dtype=np.float32, mode="w+", shape=(num_rows, num_cols))
            roughness_array = np.memmap(roughness_path, dtype=np.float32, mode="w+", shape=(num_rows, num_cols))
            
            # Divide the DEM into smaller tiles
            n_processes = mp.cpu_count() if n_processes is None else n_processes
            valid_tile_rows = range(pad_width, num_rows + pad_width, tile_size)
            valid_tile_cols = range(pad_width, num_cols + pad_width, tile_size) 
            tiles = [(r, min(r + tile_size, num_rows + pad_width), c, min(c + tile_size, num_cols + pad_width))
                for r in valid_tile_rows for c in valid_tile_cols] # Defined with respect to the padded input DEM, with padded pixels ignored

            # Create a partial function by pre-loading arguments into process_tile 
            process_tile_partial = partial(process_tile, x=x, y=y, roughness_method=roughness_method, window_size_pixels=window_size_pixels, resolution=resolution)
            
            # Process tles
            if n_processes == 1: # serial
                for tile in tqdm(tiles, desc="Computing slope and roughness..."):
                    row_start, col_start, local_slope, local_roughness = process_tile_partial(tile)
                    row_start, col_start = row_start - pad_width, col_start - pad_width
                    slope_array[row_start:row_start+local_slope.shape[0], col_start:col_start+local_slope.shape[1]] = local_slope
                    roughness_array[row_start:row_start+local_roughness.shape[0], col_start:col_start+local_roughness.shape[1]] = local_roughness
            else: # parallel
                mp.set_start_method("spawn", force=True)
                with mp.Pool(processes=n_processes, initializer=init_worker, initargs=(shm.name, shared_dem.shape, np.float32)) as pool:
                    for row_start, col_start, local_slope, local_roughness in tqdm(pool.imap(process_tile_partial, tiles), total=len(tiles), desc="Computing slope and roughness..."):
                        row_start, col_start = row_start - pad_width, col_start - pad_width
                        slope_array[row_start:row_start+local_slope.shape[0], col_start:col_start+local_slope.shape[1]] = local_slope
                        roughness_array[row_start:row_start+local_roughness.shape[0], col_start:col_start+local_roughness.shape[1]] = local_roughness

            # Close and unlink shared memory to prevent memory leaks  
            shm.close()
            shm.unlink()

            # Save slope raster
            print(f"Saving...")
            slope_output_path = (
                dem_path.split("/")[-1].split(".")[0]
                + f"_slope_{window_size_pixels}x{window_size_pixels}.tif"
            )
            slope_output_description = f"Slope raster derived from a DEM with a resolution of {resolution} meters, using a window size of {window_size} meters. Values represent the gradient magnitude (in degrees) of a plane fitted through the DEM points within each window.\nhttps://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"
            save_raster(dem_path, slope_array, slope_output_path, slope_output_description)

            # Save roughness raster
            roughness_output_path = (
                dem_path.split("/")[-1].split(".")[0]
                + f"_roughness_{roughness_method}_{window_size_pixels}x{window_size_pixels}.tif"
            )
            roughness_method_name = {
                "std": "standard deviation",
                "mad": "median absolute deviation",
                "range": "minimum-to-maximum difference",
            }
            roughness_output_description = f"Roughness raster derived from a DEM with a resolution of {resolution} meters, using a window size of {window_size} meters. Values represent the orthogonal variation ({roughness_method_name[roughness_method]}) of residuals from a plane fitted through the DEM points within each window (in meters).\nhttps://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"
            save_raster(
                dem_path, roughness_array, roughness_output_path, roughness_output_description
            )

            print(f"Done! Time taken: {time.time() - start_time:.2f} s.")

    # Ensure shared memory is freed and temporary directory is removed
    except:
        shm.close()
        shm.unlink() 


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=(
            "Generate slope and roughness maps from DEM (Digital Elevation Model) data. "
            "Slope is computed by first fitting a plane to the elevation data within a local window around each pixel using a least-squares method, "
            "and then taking the magnitude of the resulting gradients in x and y. "
            "Roughness is determined by measuring the variability of the orthogonal residuals relative to the fitted plane."
        )
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
    valid_methods = ["range", "std", "mad"]
    parser.add_argument(
        "roughness_method",
        help="The method used to calculate roughness (applied over the orthogonal residuals to the fitted plane). Options: 'range' (minimum-to-maximum difference), 'std' (standard deviation), 'mad' (median absolute deviation).",
        type=str,
        choices=valid_methods,
        nargs="?",
        default="range",
    )
    parser.add_argument(
        "-p",
        "--processes",
        help="Number of processes to use for parallel computation. Defaults to number of CPU cores. Set to 1 to process in serial.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-t",
        "--tile_size",
        help="The size of the tiles (in pixels) to split the DEM into for parallel processing. Each process attends to a single tile, so expect memory usage to grow with the number of workers. The default is 512.",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    # Run
    dem_to_slope_and_roughness(
        args.dem_path,
        args.dem_resolution,
        args.window_size,
        args.roughness_method,
        n_processes=args.processes,
        tile_size=args.tile_size,
    )
