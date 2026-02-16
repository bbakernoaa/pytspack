import numpy as np
import xarray as xr
import pytest
from pytspack import interpolate_vertical

try:
    import dask  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_aero_eager_lazy_consistency():
    """
    Aero Protocol: Double-Check Test.
    Verify that the logic produces identical results for Eager (NumPy)
    and Lazy (Dask) backends.
    """
    # Create synthetic data
    levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0])
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    data = np.random.rand(len(levels), len(lats), len(lons))

    da_eager = xr.DataArray(
        data,
        coords={"level": levels, "lat": lats, "lon": lons},
        dims=["level", "lat", "lon"],
        name="test_data",
    )

    da_lazy = da_eager.chunk({"lat": 5, "lon": 5})

    target_levels = np.array([925.0, 600.0, 150.0])

    # Run logic
    res_eager = interpolate_vertical(
        da_eager, target_levels, level_dim="level", tension=1.0
    )
    res_lazy = interpolate_vertical(
        da_lazy, target_levels, level_dim="level", tension=1.0
    )

    # Assertions
    # 1. Lazy result should still be lazy (Dask-backed)
    assert hasattr(res_lazy.data, "compute")

    # 2. Values should be identical
    np.testing.assert_allclose(res_eager.values, res_lazy.compute().values, atol=1e-12)

    # 3. Metadata and provenance check
    assert "history" in res_eager.attrs
    assert "history" in res_lazy.attrs
    # Note: timestamps might differ slightly, so we check for the core message
    assert "Vertically interpolated" in res_eager.attrs["history"]
    assert "pytspack" in res_eager.attrs["history"]


def test_aero_dataset_consistency():
    """Verify Dataset handling works correctly."""
    levels = np.array([1000.0, 500.0, 100.0])
    ds = xr.Dataset(
        {"var1": (["level"], [1.0, 2.0, 3.0]), "var2": (["x"], [10.0, 20.0])},
        coords={"level": levels, "x": [0, 1]},
    )

    target = [800.0, 300.0]
    res = interpolate_vertical(ds, target)

    assert "var1" in res.data_vars
    assert "var2" in res.data_vars
    assert res.var1.shape == (2,)
    assert res.var2.shape == (2,)
    assert np.allclose(res.level, target)
