import numpy as np
import xarray as xr
import dask.array as da
from pytspack import interpolate_vertical


def test_interpolate_vertical_basic():
    # Create simple 1D profile
    levels = np.array([300.0, 500.0, 700.0, 850.0, 1000.0])
    data = np.array([230.0, 255.0, 268.0, 275.0, 280.0])

    da_src = xr.DataArray(data, coords=[levels], dims=["level"])

    target_levels = np.array([400.0, 600.0, 900.0])
    result = interpolate_vertical(da_src, target_levels)

    assert result.shape == (3,)
    assert np.allclose(result.level.values, target_levels)
    # Check that values are within plausible ranges
    assert 230 < result.values[0] < 255
    assert 255 < result.values[1] < 268
    assert 275 < result.values[2] < 280


def test_interpolate_vertical_dask():
    # N-D with dask
    levels = np.array([300.0, 500.0, 700.0, 850.0, 1000.0])
    # time, level, lat, lon
    data = np.random.rand(2, 5, 4, 5)

    da_src = xr.DataArray(
        da.from_array(data, chunks=(1, 5, 4, 5)),
        coords={"level": levels},
        dims=["time", "level", "lat", "lon"],
    )

    target_levels = np.array([400.0, 600.0, 900.0])
    result = interpolate_vertical(da_src, target_levels)

    assert isinstance(result.data, da.Array)
    computed = result.compute()
    assert computed.shape == (2, 3, 4, 5)
    assert "history" in result.attrs
    assert "pytspack" in result.attrs["history"]


def test_interpolate_vertical_decreasing():
    # Test with decreasing levels (common in atmospheric pressure levels)
    levels = np.array([1000.0, 850.0, 700.0])
    data = np.array([10.0, 20.0, 30.0])
    da_src = xr.DataArray(data, coords=[levels], dims=["level"])

    target_levels = np.array([925.0, 775.0])
    result = interpolate_vertical(da_src, target_levels)

    # Values should be roughly linear here
    assert 10 < result.sel(level=925.0) < 20
    assert 20 < result.sel(level=775.0) < 30


def test_interpolate_vertical_dataset():
    # Test with xarray.Dataset
    levels = np.array([300.0, 500.0, 700.0])
    temp = np.array([230.0, 255.0, 268.0])
    hum = np.array([10.0, 30.0, 50.0])

    ds = xr.Dataset(
        {
            "temperature": (["level"], temp),
            "humidity": (["level"], hum),
            "static_var": (["x"], [1, 2, 3]),
        },
        coords={"level": levels, "x": [10, 20, 30]},
    )

    target_levels = np.array([400.0, 600.0])
    result = interpolate_vertical(ds, target_levels)

    assert "temperature" in result.data_vars
    assert "humidity" in result.data_vars
    assert "static_var" in result.data_vars
    assert result.temperature.shape == (2,)
    assert result.static_var.shape == (3,)
    assert "level" in result.temperature.coords


def test_interpolate_vertical_nan_handling():
    # Test with NaNs in data
    levels = np.array([300.0, 500.0, 700.0])
    data = np.array([230.0, np.nan, 268.0])
    da_src = xr.DataArray(data, coords=[levels], dims=["level"])

    target_levels = np.array([400.0, 600.0])
    result = interpolate_vertical(da_src, target_levels)

    assert np.all(np.isnan(result.values))


def test_interpolate_vertical_unsorted_levels():
    # Test with unsorted levels
    levels = np.array([500.0, 300.0, 700.0])
    data = np.array([255.0, 230.0, 268.0])
    da_src = xr.DataArray(data, coords=[levels], dims=["level"])

    target_levels = np.array([400.0, 600.0])
    result = interpolate_vertical(da_src, target_levels)

    assert result.shape == (2,)
    assert not np.any(np.isnan(result.values))
    assert 230 < result.values[0] < 255
    assert 255 < result.values[1] < 268


def test_interpolate_vertical_duplicate_levels():
    # Test with duplicate levels (should result in NaNs or handled gracefully)
    levels = np.array([300.0, 300.0, 700.0])
    data = np.array([230.0, 230.0, 268.0])
    da_src = xr.DataArray(data, coords=[levels], dims=["level"])

    target_levels = np.array([400.0, 600.0])
    result = interpolate_vertical(da_src, target_levels)

    # _tspack_interp1d should return NaNs because np.diff(x_sorted) <= 0
    assert np.all(np.isnan(result.values))
