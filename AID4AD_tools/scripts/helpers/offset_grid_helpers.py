import os
import numpy as np

def load_offset_grids(basemap_names, npy_dir):
    """
    Loads offset grids from .npy files on disk and returns
    a dictionary mapping each basemap_name to (grid_offset_x, grid_offset_y).

    Parameters
    ----------
    basemap_names : list of str
        The list of basemap names to load. E.g.:
        ["singapore-queenstown", "singapore-onenorth", ...]
        It must match the file naming convention:
            {basemap_name}_interp_x.npy
            {basemap_name}_interp_y.npy
    npy_dir : str
        Directory path where the .npy files are stored.

    Returns
    -------
    offset_grids : dict
        A dictionary of the form:
            {
              "basemap_name": (grid_x, grid_y),
              ...
            }
        where grid_x and grid_y are 2D NumPy arrays containing
        the interpolated offset values in X and Y directions.
    """
    offset_grids = {}

    for basemap_name in basemap_names:
        # Construct the paths to the saved numpy files
        path_x = os.path.join(npy_dir, f"{basemap_name}_interp_x.npy")
        path_y = os.path.join(npy_dir, f"{basemap_name}_interp_y.npy")

        # Check if these files exist
        if not os.path.exists(path_x) or not os.path.exists(path_y):
            # Raise an error, or skip with a warning
            raise FileNotFoundError(
                f"Cannot find .npy files for basemap '{basemap_name}'. "
                f"Expected: {path_x}, {path_y}"
            )

        # Load the arrays
        grid_x = np.load(path_x)
        grid_y = np.load(path_y)

        # Store them in our dictionary
        offset_grids[basemap_name] = (grid_x, grid_y)

    return offset_grids


def get_offset_for_coordinate(orig_x, orig_y, basemap_name, offset_grids, resolution=10.0):
    """
    Given a point (orig_x, orig_y) in basemap meters and a basemap name,
    return the (offset_x, offset_y) from the stored offset grids.

    Parameters
    ----------
    orig_x : float
        X coordinate in the basemap's meter system.
    orig_y : float
        Y coordinate in the basemap's meter system.
    basemap_name : str
        Name of the basemap, e.g. "singapore-queenstown".
    offset_grids : dict
        Dictionary mapping basemap_name -> (offset_x_grid, offset_y_grid).
        Each offset_x_grid, offset_y_grid is a 2D array of shape (H, W).
    resolution : float
        The grid resolution in meters per cell (default=10.0).

    Returns
    -------
    (offset_x_val, offset_y_val) : tuple of float
        Offset in X and Y directions for the given coordinate.
        Could be NaN if the coordinate is out-of-bounds or no data was interpolated.
    """

    # Retrieve the 2D grids for this basemap
    if basemap_name not in offset_grids:
        raise ValueError(f"Basemap '{basemap_name}' is not in offset_grids.")

    offset_x_grid, offset_y_grid = offset_grids[basemap_name]

    # Convert the (orig_x, orig_y) into grid cell indices
    cell_x = int(orig_x // resolution)
    cell_y = int(orig_y // resolution)

    # Check bounds
    h, w = offset_x_grid.shape
    if not (0 <= cell_x < w and 0 <= cell_y < h):
        # You can decide how to handle out-of-bounds. For example:
        # return (np.nan, np.nan), or raise an exception.
        raise ValueError(
            f"Coordinate (x={orig_x}, y={orig_y}) is out of basemap grid bounds "
            f"(cell_x={cell_x}, cell_y={cell_y}, grid_w={w}, grid_h={h})."
        )

    # Get the offset values
    offset_x_val = offset_x_grid[cell_y, cell_x]
    offset_y_val = offset_y_grid[cell_y, cell_x]

    return offset_x_val, offset_y_val