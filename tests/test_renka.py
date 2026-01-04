import numpy as np
import xarray as xr
import dask.array as da
from renka.renka import SphericalMesh
from datetime import datetime


def test_sphericalmesh_interpolate_numpy():
    """
    Tests the SphericalMesh.interpolate_to_numpy_grid method.

    This test verifies that the interpolation function produces a grid of
    the correct shape and that a value at a known location is within an
    expected range. This confirms the basic functionality of the STRIPACK
    and SSRFPACK wrappers for eager, NumPy-based computation.
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
    interpolated_data = mesh.interpolate_to_numpy_grid(values, grid_lats, grid_lons)

    # 1. Check the output shape
    assert interpolated_data.shape == (16, 11), "Output array shape is incorrect."

    # 2. Check a specific interpolated value.
    lat_idx = np.abs(grid_lats - 40.0).argmin()
    lon_idx = np.abs(grid_lons - -90.0).argmin()

    assert np.isclose(interpolated_data[lat_idx, lon_idx], 18.0, atol=1e-9), (
        "Interpolated value at a known point is not as expected."
    )

    # 3. Check for NaN values in the interior
    interior_view = interpolated_data[1:-1, 1:-1]
    assert not np.isnan(interior_view).any(), (
        "Interpolated data contains NaNs in its interior."
    )


def test_sphericalmesh_interpolate_dask():
    """
    Tests the Dask-aware SphericalMesh.interpolate method.

    This test ensures that the `xarray.apply_ufunc` wrapper correctly
    handles Dask-backed arrays, producing a lazy Dask array as output,
    and that the computed result is numerically identical to the eager
    NumPy-based implementation.
    """
    # Define sample scattered data points (latitude, longitude, value)
    lats = np.array([30.0, 45.0, 35.0, 40.0])
    lons = np.array([-90.0, -85.0, -95.0, -90.0])
    values = np.array([10.0, 20.0, 15.0, 18.0])

    # Wrap the source values in a Dask-backed xarray.DataArray
    values_da = xr.DataArray(
        da.from_array(values, chunks="auto"),
        dims=["points"],
        coords={"lat": ("points", lats), "lon": ("points", lons)},
    ).chunk({"points": -1})

    # Initialize the spherical mesh
    mesh = SphericalMesh(lats, lons)

    # Define the grid for interpolation
    grid_lats = np.linspace(30, 45, 16)
    grid_lons = np.linspace(-95, -85, 11)

    # Perform Dask-aware interpolation
    interpolated_da = mesh.interpolate(values_da, grid_lats, grid_lons)

    # 1. Check that the output is a Dask-backed xarray.DataArray
    assert isinstance(interpolated_da, xr.DataArray), (
        "Output is not an xarray.DataArray."
    )
    assert isinstance(interpolated_da.data, da.Array), "Output is not a Dask array."

    # 2. Trigger computation and get the result
    computed_data = interpolated_da.compute()

    # 3. Compare with the NumPy-based method for numerical consistency
    numpy_data = mesh.interpolate_to_numpy_grid(values, grid_lats, grid_lons)
    np.testing.assert_allclose(
        computed_data.values,
        numpy_data,
        atol=1e-9,
        err_msg="Dask-aware interpolation result does not match NumPy result.",
    )

    # 4. Check shape and coordinates
    assert computed_data.shape == (16, 11), "Computed array shape is incorrect."
    assert "lat" in computed_data.coords
    assert "lon" in computed_data.coords


def test_sphericalmesh_regrid_conservative():
    """
    Tests the SphericalMesh.regrid_conservative method.

    This test verifies that the conservative regridding function produces a
    grid of the correct shape and that the total sum of the regridded data
    is close to the sum of the original data, which is a key property of
    conservative regridding.
    """
    # Define sample scattered data points (latitude, longitude, value)
    lats = np.array([30.0, 45.0, 35.0, 40.0])
    lons = np.array([-90.0, -85.0, -95.0, -90.0])
    values = np.array([10.0, 20.0, 15.0, 18.0])

    # Initialize the spherical mesh
    mesh = SphericalMesh(lats, lons)

    # Define the grid for regridding
    grid_lats = np.linspace(30, 45, 16)
    grid_lons = np.linspace(-95, -85, 11)

    # Perform conservative regridding
    regridded_data = mesh.regrid_conservative(values, grid_lats, grid_lons)

    # 1. Check the output shape
    assert regridded_data.shape == (16, 11), "Output array shape is incorrect."

    # 2. Check for conservation
    # The sum of the regridded data should be close to the sum of the
    # original data. This is a fundamental property of conservative
    # regridding.
    # Note: This is an approximate conservation, so we use a tolerance.
    assert np.isclose(np.sum(regridded_data), np.sum(values), rtol=1e-6), (
        "Conservative regridding is not preserving the total sum."
    )

    # 3. Check for NaN values in the interior
    # The result may have NaNs on the boundary, but the interior should be finite.
    interior_view = regridded_data[1:-1, 1:-1]
    assert not np.isnan(interior_view).any(), (
        "Regridded data contains NaNs in its interior."
    )
