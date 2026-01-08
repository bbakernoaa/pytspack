import numpy as np
import xarray as xr
import dask.array as da
import pytest

from renka.vertical import interpolate_vertical


@pytest.fixture
def sample_dataarray():
    """Provides a sample xarray.DataArray for testing."""
    source_levels = np.array([1000, 900, 800, 700, 600, 500])
    lats = np.arange(-90, 91, 90)
    lons = np.arange(-180, 181, 180)
    data = np.arange(len(source_levels) * len(lats) * len(lons)).reshape(
        len(source_levels), len(lats), len(lons)
    )
    return xr.DataArray(
        data,
        dims=["pressure", "lat", "lon"],
        coords={"pressure": source_levels, "lat": lats, "lon": lons},
        attrs={"history": "Initial creation."},
    )


def test_linear_interpolation(sample_dataarray):
    """Tests linear interpolation on a standard DataArray."""
    da = sample_dataarray.rename({"pressure": "height"}).assign_coords(
        {"height": [100, 200, 300, 400, 500, 600]}
    )

    interpolated_da = interpolate_vertical(
        da, [150, 250, 350], level_dim="height", method="linear"
    )

    assert interpolated_da.shape == (3, 3, 3)
    np.testing.assert_array_equal(interpolated_da["height"].values, [150, 250, 350])

    expected_value = 8.5
    actual_value = interpolated_da.sel(lat=0, lon=0, height=150).item()
    assert np.isclose(actual_value, expected_value)


def test_log_interpolation(sample_dataarray):
    """Tests log-pressure interpolation."""
    target_levels = np.array([950, 850, 750])

    interpolated_da = interpolate_vertical(
        sample_dataarray, target_levels, level_dim="pressure", method="log"
    )

    assert interpolated_da.shape == (3, 3, 3)
    np.testing.assert_array_equal(interpolated_da["pressure"].values, target_levels)

    # The expected value is calculated by hand to match the log interpolation
    expected_value = 8.3815242
    actual_value = interpolated_da.sel(lat=0, lon=0, pressure=950).item()
    assert np.isclose(actual_value, expected_value)


def test_dask_integration(sample_dataarray):
    """Ensures the function works with Dask-backed arrays and remains lazy."""
    dask_da = sample_dataarray.chunk({"lat": 1, "lon": 1})
    target_levels = np.array([950, 850])

    interpolated_da = interpolate_vertical(
        dask_da, target_levels, level_dim="pressure", method="log"
    )

    assert isinstance(interpolated_da.data, da.Array)

    computed_result = interpolated_da.compute()
    numpy_result = interpolate_vertical(
        sample_dataarray, target_levels, level_dim="pressure", method="log"
    )

    xr.testing.assert_allclose(computed_result, numpy_result)


def test_dataset_interpolation(sample_dataarray):
    """Tests if the function correctly handles an xarray.Dataset."""
    ds = xr.Dataset(
        {"temperature": sample_dataarray, "humidity": sample_dataarray * 0.5}
    )
    target_levels = [950, 850]

    interpolated_ds = interpolate_vertical(
        ds, target_levels, level_dim="pressure", method="log"
    )

    assert isinstance(interpolated_ds, xr.Dataset)
    assert "temperature" in interpolated_ds
    assert "humidity" in interpolated_ds
    assert interpolated_ds["temperature"].shape == (2, 3, 3)
    assert np.isclose(
        interpolated_ds["humidity"].sel(lat=0, lon=0, pressure=950).item(),
        8.3815242 * 0.5,
    )


def test_provenance(sample_dataarray):
    """Checks that the 'history' attribute is correctly updated."""
    target_levels = [950]
    interpolated_da = interpolate_vertical(
        sample_dataarray, target_levels, level_dim="pressure"
    )

    assert "history" in interpolated_da.attrs
    assert len(interpolated_da.attrs["history"].splitlines()) == 2
    assert "Vertically interpolated" in interpolated_da.attrs["history"]


def test_error_handling(sample_dataarray):
    """Tests for expected errors, like missing dimensions or invalid methods."""
    with pytest.raises(ValueError, match="Dimension 'level' not found"):
        interpolate_vertical(sample_dataarray, [1.5], level_dim="level")

    with pytest.raises(ValueError, match="Unknown interpolation method"):
        interpolate_vertical(
            sample_dataarray, [1.5], level_dim="pressure", method="cubic"
        )
