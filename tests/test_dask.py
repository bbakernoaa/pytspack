import numpy as np
import dask.array as da
from renka import SphericalMesh

def get_stable_mesh():
    """Provides a more stable mesh for testing, avoiding ambiguities."""
    # A square of points
    source_lats = np.array([10, 20, 10, 20])
    source_lons = np.array([10, 10, 20, 20])
    source_values = np.array([10.0, 20.0, 30.0, 40.0])
    # (10, 10) -> 10
    # (20, 10) -> 20
    # (10, 20) -> 30
    # (20, 20) -> 40
    return SphericalMesh(lats=source_lats, lons=source_lons), source_values


def test_dask_eager_interpolation():
    """
    Test eager-mode interpolation with Dask.
    """
    # 1. The Logic (Setup)
    mesh, source_values = get_stable_mesh()
    # Grid that includes the source points
    grid_lats = np.linspace(10, 20, 5)
    grid_lons = np.linspace(10, 20, 5)

    # 2. The Proof (Execution & Assertion)
    result = mesh.interpolate(source_values, grid_lats, grid_lons)
    computed_result = result.compute()

    # Assert that the output is a numpy array (eager)
    assert isinstance(computed_result.data, np.ndarray), (
        "Eager result should be a NumPy array."
    )

    # Assert shape and a plausible value
    assert computed_result.shape == (5, 5)
    # Value at a known source point should be very close to the source value
    assert np.isclose(computed_result.sel(lat=20, lon=10).item(), 20.0, atol=1e-5)
    # Check interpolated value in the middle
    middle_val = computed_result.sel(lat=15, lon=15).item()
    assert 10.0 < middle_val < 40.0


def test_dask_lazy_interpolation():
    """
    Test lazy-mode interpolation with Dask.
    """
    # 1. The Logic (Setup)
    mesh, source_values = get_stable_mesh()

    # Use dask arrays for grid coordinates to trigger lazy evaluation
    dask_lats = da.linspace(10, 20, 5, chunks=(3,))
    dask_lons = da.linspace(10, 20, 5, chunks=(3,))

    # 2. The Proof (Execution & Assertion)
    result = mesh.interpolate(source_values, dask_lats, dask_lons)

    # Assert that the output is a dask array (lazy)
    assert isinstance(result.data, da.Array), "Lazy result should be a Dask array."

    # Compute the result and check its properties
    computed_result = result.compute()
    assert computed_result.shape == (5, 5)
    assert np.isclose(computed_result.sel(lat=20, lon=10).item(), 20.0, atol=1e-5)
    middle_val = computed_result.sel(lat=15, lon=15).item()
    assert 10.0 < middle_val < 40.0
