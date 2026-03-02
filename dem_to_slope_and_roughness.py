# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import argparse
import atexit
import gc
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings
from functools import partial
from itertools import product
from multiprocessing.shared_memory import SharedMemory

import multiprocessing as mp
import numpy as np
import rasterio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Raster I/O
# ----------------------------------------------------------------------

def read_raster(dem_path):
    """
    Read the first band of a raster file as a float32 array, replacing
    nodata values (including -9999) with NaN.

    Args:
        dem_path (str): Path to the raster file.

    Returns:
        numpy.ndarray: 2D float32 array with nodata as NaN.
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype("float32")
        nodata_value = src.nodata

    dem = np.where(dem == nodata_value, np.nan, dem)
    dem = np.where(dem == -9999, np.nan, dem)  # fallback for unset nodata
    return dem


def save_raster(dem_path, output, output_path, description):
    """
    Save a 2D array as a GeoTIFF, inheriting CRS and transform from a
    source DEM. NaN values are written as -9999.

    Args:
        dem_path (str): Path to the source DEM (for metadata).
        output (numpy.ndarray): 2D array to save.
        output_path (str): Destination file path.
        description (str): Description string written as a raster tag.
    """
    output = output.copy()
    output[np.isnan(output)] = -9999

    with rasterio.open(dem_path) as src:
        metadata = src.meta.copy()
        source_description = src.tags().get("description", "No description available.")

    metadata.update(driver="GTiff", count=1, dtype=np.float32, nodata=-9999)

    full_description = f"{description}\n\nSource DEM Description:\n{source_description}"

    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.update_tags(description=full_description)
        dst.write(output[np.newaxis, :, :])


# ----------------------------------------------------------------------
# Maths / plane fitting
# ----------------------------------------------------------------------

def fit_plane(x, y, z, method="svd"):
    """
    Fit a plane to 3D points and return gradients dz/dx and dz/dy.

    Args:
        x, y (numpy.ndarray): Flattened, zero-centred coordinate arrays.
        z (numpy.ndarray): Flattened elevation values (mean-centred).
        method (str): 'svd' (orthogonal fit) or 'ls' (least squares).

    Returns:
        tuple:
            - a (float): Gradient dz/dx.
            - b (float): Gradient dz/dy.
            - residuals (numpy.ndarray or None): Orthogonal residuals (SVD
              only), otherwise None.
    """
    valid = ~np.isnan(z)
    x, y, z = x[valid], y[valid], z[valid]

    if method == "ls":
        A = np.column_stack((x, y, np.ones_like(x)))
        coef, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return coef[0], coef[1], None

    elif method == "svd":
        xyz = np.column_stack((x, y, z))
        _, _, Vt = np.linalg.svd(xyz)
        normal = Vt[-1] / np.linalg.norm(Vt[-1])
        nx, ny, nz = normal
        a, b = -nx / nz, -ny / nz
        residuals = np.abs(np.einsum("ij,j->i", xyz, normal)) / np.linalg.norm(normal)
        return a, b, residuals

    else:
        sys.exit(f"Unknown plane-fitting method '{method}'. Use 'svd' or 'ls'.")


def get_orthogonal_residuals(x, y, z, a, b):
    """
    Compute perpendicular distances from 3D points to a plane defined by
    gradients a (dz/dx) and b (dz/dy), passing through the origin.

    Args:
        x, y, z (numpy.ndarray): Flattened, zero-centred coordinate arrays.
        a (float): Plane gradient in x.
        b (float): Plane gradient in y.

    Returns:
        numpy.ndarray: Orthogonal residuals.
    """
    return (z - (a * x + b * y)) / np.sqrt(a**2 + b**2 + 1)


def compute_aspect(dz_x, dz_y):
    """
    Compute aspect (direction of steepest descent) from plane gradients.

    Args:
        dz_x (float): Gradient in the x (East) direction.
        dz_y (float): Gradient in the y (North) direction.

    Returns:
        float: Aspect in degrees clockwise from North (0–360), or NaN for
               flat surfaces.
    """
    if dz_x == 0.0 and dz_y == 0.0:
        return np.nan
    return (np.rad2deg(np.arctan2(dz_x, dz_y)) + 180.0) % 360.0


def nanmad(data, axis=None):
    """
    Median absolute deviation (MAD), ignoring NaN values.

    Args:
        data (numpy.ndarray): Input array.
        axis (int, optional): Axis along which to compute. Default is None.

    Returns:
        float or numpy.ndarray: MAD value(s).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(data, axis=axis)
        return np.nanmedian(np.abs(data - np.expand_dims(median, axis=axis)), axis=axis)


# ----------------------------------------------------------------------
# Tile processing
# ----------------------------------------------------------------------

def process_tile(tile_coords, x, y, roughness_method, window_size_pixels,
                 method="svd", output_aspect=False):
    """
    Compute slope, roughness, and optionally aspect for one tile of the DEM.

    Reads elevation data from the global `shared_dem` array, which is
    populated by the parent process (serial) or init_worker (parallel).

    Args:
        tile_coords (tuple): (row_start, row_end, col_start, col_end) in
                             padded-DEM coordinates.
        x, y (numpy.ndarray): Flattened, zero-centred window coordinate arrays.
        roughness_method (str): 'std', 'mad', 'range', or
                                'non_orthogonal_residual_range'.
        window_size_pixels (int): Side length of the moving window in pixels.
        method (str): Plane-fitting method, 'svd' or 'ls'.
        output_aspect (bool): Whether to compute and return aspect. Default False.

    Returns:
        tuple: (row_start, col_start, local_slope, local_aspect, local_roughness).
               local_aspect is None if output_aspect is False.
               All values are None on error.
    """
    try:
        row_start, row_end, col_start, col_end = tile_coords
        tile_shape = (row_end - row_start, col_end - col_start)
        half = window_size_pixels // 2

        local_slope = np.full(tile_shape, np.nan, dtype=np.float32)
        local_roughness = np.full(tile_shape, np.nan, dtype=np.float32)
        local_aspect = np.full(tile_shape, np.nan, dtype=np.float32) if output_aspect else None

        for row, col in product(range(row_start, row_end), range(col_start, col_end)):
            window = shared_dem[row - half : row + half + 1, col - half : col + half + 1]

            if np.isnan(window).sum() > (window_size_pixels**2) / 2:
                continue

            z = (window - np.nanmean(window)).flatten()
            a, b, residuals = fit_plane(x, y, z, method=method)

            ri, ci = row - row_start, col - col_start
            local_slope[ri, ci] = np.degrees(np.arctan(np.sqrt(a**2 + b**2)))

            if output_aspect:
                local_aspect[ri, ci] = compute_aspect(a, b)

            if residuals is None:
                residuals = get_orthogonal_residuals(x, y, z, a, b)

            if roughness_method == "std":
                local_roughness[ri, ci] = np.nanstd(np.abs(residuals))
            elif roughness_method == "mad":
                local_roughness[ri, ci] = nanmad(np.abs(residuals))
            elif roughness_method == "range":
                local_roughness[ri, ci] = np.nanmax(residuals) - np.nanmin(residuals)
            elif roughness_method == "non_orthogonal_residual_range":
                nr = z - (a * x + b * y)
                local_roughness[ri, ci] = np.nanmax(nr) - np.nanmin(nr)

        return row_start, col_start, local_slope, local_aspect, local_roughness

    except Exception as e:
        logger.error(f"Error processing tile {tile_coords}: {e}")
        return None, None, None, None, None


# ----------------------------------------------------------------------
# Multiprocessing helpers
# ----------------------------------------------------------------------

def init_worker(shared_name, shape, dtype):
    """
    Attach a worker process to the shared DEM memory block.

    Args:
        shared_name (str): Name of the SharedMemory block.
        shape (tuple): Array shape to interpret the buffer as.
        dtype (numpy.dtype): Array data type.
    """
    global shared_dem
    shm = SharedMemory(name=shared_name)
    shared_dem = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    atexit.register(shm.close)


def cleanup(shm, tmpdir):
    """
    Release shared memory and remove the temporary working directory.

    Args:
        shm (SharedMemory): Block to close and unlink.
        tmpdir (str): Temporary directory to remove.
    """
    gc.collect()
    time.sleep(0.1)
    shm.close()
    shm.unlink()
    shutil.rmtree(tmpdir)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def dem_to_slope_and_roughness(
    dem_path,
    resolution,
    window_size,
    roughness_method,
    n_processes=None,
    tile_size=256,
    method="svd",
    output_aspect=False,
):
    """
    Compute slope, roughness, and optionally aspect rasters from a DEM and
    write them as GeoTIFFs alongside the input file.

    Args:
        dem_path (str): Path to the input DEM raster.
        resolution (float): Pixel size in metres.
        window_size (float): Analysis window size in metres.
        roughness_method (str): 'std', 'mad', or 'range'.
        n_processes (int, optional): Worker count; defaults to cpu_count().
        tile_size (int): Tile side length in pixels for parallel dispatch.
        method (str): Plane-fitting method, 'svd' or 'ls'.
        output_aspect (bool): Whether to produce an aspect raster. Default False.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Validate / adjust window size
    # ------------------------------------------------------------------
    window_size_pixels = int(window_size / resolution)
    if window_size_pixels % 2 == 0:
        option_1 = (window_size_pixels + 1) * resolution
        option_2 = (window_size_pixels - 1) * resolution
        print(
            f"WARNING: Window size of {window_size} m gives an even number of pixels "
            f"({window_size_pixels}); an odd number is required."
        )
        if sys.stdin.isatty():
            choice = None
            while choice not in {"0", "1"}:
                choice = input(f"Use {option_1} m [0] or {option_2} m [1]? ")
                if choice not in {"0", "1"}:
                    print("Please enter 0 or 1.")
            window_size = option_1 if choice == "0" else option_2
        else:
            window_size = option_1
        window_size_pixels = int(window_size / resolution)

    # ------------------------------------------------------------------
    # Load DEM into shared memory (padded to support edge windows)
    # ------------------------------------------------------------------
    print("Reading DEM...")
    dem = read_raster(dem_path)
    num_rows, num_cols = dem.shape
    pad = window_size_pixels // 2

    shm = SharedMemory(
        create=True,
        size=(num_rows + 2 * pad) * (num_cols + 2 * pad) * np.float32().nbytes,
    )

    n_processes = n_processes or mp.cpu_count()
    if n_processes == 1:
        global shared_dem

    shared_dem = np.ndarray(
        (num_rows + 2 * pad, num_cols + 2 * pad), dtype=np.float32, buffer=shm.buf
    )
    shared_dem[:] = np.nan
    shared_dem[pad:-pad, pad:-pad] = dem
    del dem

    # Pre-compute zero-centred window coordinates
    coords = (np.arange(window_size_pixels) - (window_size_pixels - 1) / 2) * resolution
    x, y = np.meshgrid(coords, coords)
    x, y = x.flatten(), y.flatten()

    # ------------------------------------------------------------------
    # Allocate memory-mapped output arrays
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(dir=os.getcwd())
    slope_array = np.memmap(
        os.path.join(tmpdir, "slope.dat"), dtype=np.float32, mode="w+", shape=(num_rows, num_cols)
    )
    roughness_array = np.memmap(
        os.path.join(tmpdir, "roughness.dat"), dtype=np.float32, mode="w+", shape=(num_rows, num_cols)
    )
    if output_aspect:
        aspect_array = np.memmap(
            os.path.join(tmpdir, "aspect.dat"), dtype=np.float32, mode="w+", shape=(num_rows, num_cols)
        )

    tiles = [
        (r, min(r + tile_size, num_rows + pad), c, min(c + tile_size, num_cols + pad))
        for r in range(pad, num_rows + pad, tile_size)
        for c in range(pad, num_cols + pad, tile_size)
    ]

    tile_fn = partial(
        process_tile,
        x=x, y=y,
        roughness_method=roughness_method,
        window_size_pixels=window_size_pixels,
        method=method,
        output_aspect=output_aspect,
    )

    # ------------------------------------------------------------------
    # Process tiles, then save outputs
    # ------------------------------------------------------------------
    try:
        def write_results(row_start, col_start, local_slope, local_aspect, local_roughness):
            rs, cs = row_start - pad, col_start - pad
            slope_array[rs : rs + local_slope.shape[0], cs : cs + local_slope.shape[1]] = local_slope
            roughness_array[rs : rs + local_roughness.shape[0], cs : cs + local_roughness.shape[1]] = local_roughness
            if output_aspect:
                aspect_array[rs : rs + local_aspect.shape[0], cs : cs + local_aspect.shape[1]] = local_aspect

        if n_processes == 1:
            for tile in tqdm(tiles, desc="Processing tiles"):
                write_results(*tile_fn(tile))
        else:
            mp.set_start_method("spawn", force=True)
            with mp.Pool(
                processes=n_processes,
                initializer=init_worker,
                initargs=(shm.name, shared_dem.shape, np.float32),
            ) as pool:
                for result in tqdm(pool.imap(tile_fn, tiles), total=len(tiles), desc="Processing tiles"):
                    write_results(*result)

        print("Saving...")
        stem = os.path.splitext(os.path.basename(dem_path))[0]
        tag = f"{method}_{window_size_pixels}x{window_size_pixels}"
        url = "https://github.com/Joe-Phillips/Slope-and-Roughness-from-DEMs"

        save_raster(
            dem_path, slope_array,
            f"{stem}_slope_{tag}.tif",
            f"Slope (degrees) derived from {dem_path}. Resolution {resolution} m, "
            f"window {window_size} m. Gradient magnitude of best-fit plane.\n{url}",
        )

        if output_aspect:
            save_raster(
                dem_path, aspect_array,
                f"{stem}_aspect_{tag}.tif",
                f"Aspect (degrees clockwise from North) derived from {dem_path}. "
                f"Resolution {resolution} m, window {window_size} m. "
                f"Direction of steepest descent; NaN for flat areas.\n{url}",
            )

        roughness_label = {
            "std": "standard deviation",
            "mad": "median absolute deviation",
            "range": "min-to-max difference",
            "non_orthogonal_residual_range": "non-orthogonal residual range",
        }[roughness_method]
        save_raster(
            dem_path, roughness_array,
            f"{stem}_roughness_{roughness_method}_{tag}.tif",
            f"Roughness ({roughness_label} of orthogonal plane residuals, metres) "
            f"derived from {dem_path}. Resolution {resolution} m, window {window_size} m.\n{url}",
        )

        print(f"Done! Time taken: {time.time() - start_time:.2f} s.")

    finally:
        del slope_array, roughness_array
        if output_aspect:
            del aspect_array
        cleanup(shm, tmpdir)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate slope and roughness maps from a DEM. Slope is the gradient "
            "magnitude of a plane fitted within a moving window; roughness is the "
            "variability of orthogonal residuals to that plane."
        )
    )
    parser.add_argument("dem_path", type=str, help="Path to the input DEM file.")
    parser.add_argument("dem_resolution", type=int, help="DEM pixel size in metres.")
    parser.add_argument("window_size", type=int, help="Analysis window size in metres.")
    parser.add_argument(
        "roughness_method",
        nargs="?", default="range",
        choices=["range", "std", "mad", "non_orthogonal_residual_range"],
        help="Roughness statistic applied to orthogonal plane residuals: 'range' (default), 'std', or 'mad'.",
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=None,
        help="Number of parallel workers. Defaults to cpu_count(). Use 1 for serial.",
    )
    parser.add_argument(
        "-t", "--tile_size", type=int, default=256,
        help="Tile side length in pixels for parallel dispatch (default: 256).",
    )
    parser.add_argument(
        "-m", "--method", choices=["svd", "ls"], default="svd",
        help="Plane-fitting method: 'svd' (default, minimises orthogonal residuals) or 'ls' (least squares, faster).",
    )
    parser.add_argument(
        "-a", "--aspect", action="store_true", default=False,
        help="Also compute and save an aspect raster.",
    )
    args = parser.parse_args()

    dem_to_slope_and_roughness(
        args.dem_path,
        args.dem_resolution,
        args.window_size,
        args.roughness_method,
        n_processes=args.processes,
        tile_size=args.tile_size,
        method=args.method,
        output_aspect=args.aspect,
    )