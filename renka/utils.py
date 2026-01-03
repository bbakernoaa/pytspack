import numpy as np


def jitter_coordinates(lats, lons, amount=1e-6):
    """
    Adds a small amount of random noise to latitude and longitude arrays.

    This is used to prevent issues with degenerate triangles in triangulation
    algorithms that can occur when points are perfectly co-linear (e.g.,
    on a regular grid).

    Parameters
    ----------
    lats : np.ndarray
        Array of latitudes.
    lons : np.ndarray
        Array of longitudes.
    amount : float, optional
        The maximum amount of noise to add, by default 1e-6.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the jittered latitude and longitude arrays.
    """
    # Ensure inputs are numpy arrays
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    # Add random noise uniformly distributed in [-amount/2, amount/2]
    lat_noise = (np.random.rand(*lats.shape) - 0.5) * amount
    lon_noise = (np.random.rand(*lons.shape) - 0.5) * amount

    return lats + lat_noise, lons + lon_noise
