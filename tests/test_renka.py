import numpy as np
import pytest
from renka.renka import SphericalMesh


def test_sphericalmesh_interpolate():
    """
    Tests the SphericalMesh.interpolate method.

    This test verifies that the interpolation function produces a grid of
    the correct shape and that a value at a known location is within an
    expected range. This confirms the basic functionality of the STRIPACK
    and SSRFPACK wrappers.
    """
    # Define sample scattered data points (latitude, longitude, value)
    lats = np.array([30.0, 45.0, 35.0, 40.0])
    lons = np.array([-90.0, -85.0, -95.0, -90.0])
    values = np.array([10.0, 20.0, 15.0, 18.0])

    # Initialize the spherical mesh
    mesh = SphericalMesh(lats, lons)

    # Define the grid for interpolation
    grid_lats = np.linspace(30, 45, 16)
    grid_lons = np.linspace(-95, -85, 11)

    # Perform interpolation
    interpolated_data = mesh.interpolate(values, grid_lats, grid_lons)

    # 1. Check the output shape
    assert interpolated_data.shape == (16, 11), "Output array shape is incorrect."

    # 2. Check a specific interpolated value.
    # The point (40.0, -90.0) has a value of 18.0. The interpolation at a
    # nearby grid point should be close to this value.
    # The grid point closest to (40, -90) is at index [10, 5]
    lat_idx = np.abs(grid_lats - 40.0).argmin()
    lon_idx = np.abs(grid_lons - -90.0).argmin()

    # The interpolated value should be very close to the input value 18.0
    # because the interpolation grid point is very close to a data point.
    assert np.isclose(interpolated_data[lat_idx, lon_idx], 18.0, atol=1e-9), (
        "Interpolated value at a known point is not as expected."
    )

    # 3. Check for NaN values in the interior
    # The result may have NaNs on the boundary, but the interior should be finite.
    interior_view = interpolated_data[1:-1, 1:-1]
    assert not np.isnan(interior_view).any(), (
        "Interpolated data contains NaNs in its interior."
    )
