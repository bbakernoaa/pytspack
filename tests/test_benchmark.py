import numpy as np
import pytest
import xarray as xr
from renka.renka import SphericalMesh


@pytest.mark.parametrize("num_points", [10, 100, 1000])
def test_benchmark_regrid_conservative_scaling_source(benchmark, num_points):
    """
    Benchmark the SphericalMesh.regrid_conservative method with varying source mesh sizes.
    """
    # Define sample scattered data points (latitude, longitude, value)
    lats = np.linspace(30.0, 45.0, num_points)
    lons = np.linspace(-90.0, -85.0, num_points)
    values = np.random.rand(num_points) * 10.0

    # Initialize the spherical mesh
    mesh = SphericalMesh(lats, lons)

    # Define a fixed-size grid for regridding
    grid_lats = np.linspace(30, 45, 16)
    grid_lons = np.linspace(-95, -85, 11)

    # Benchmark the regridding function
    benchmark(mesh.regrid_conservative, values, grid_lats, grid_lons)


@pytest.mark.parametrize("grid_size", [10, 50, 100])
def test_benchmark_regrid_conservative_scaling_grid(benchmark, grid_size):
    """
    Benchmark the SphericalMesh.regrid_conservative method with varying destination grid sizes.
    """
    # Define a fixed-size set of scattered data points
    lats = np.array([30.0, 45.0, 35.0, 40.0])
    lons = np.array([-90.0, -85.0, -95.0, -90.0])
    values = np.array([10.0, 20.0, 15.0, 18.0])

    # Initialize the spherical mesh
    mesh = SphericalMesh(lats, lons)

    # Define the grid for regridding with varying size
    grid_lats = np.linspace(30, 45, grid_size)
    grid_lons = np.linspace(-95, -85, grid_size)

    # Benchmark the regridding function
    benchmark(mesh.regrid_conservative, values, grid_lats, grid_lons)


@pytest.mark.parametrize("num_points", [10, 100, 1000])
def test_benchmark_interpolate_numpy_scaling_source(benchmark, num_points):
    """
    Benchmark the SphericalMesh.interpolate_to_numpy_grid method with varying source mesh sizes.
    """
    lats = np.linspace(30.0, 45.0, num_points)
    lons = np.linspace(-90.0, -85.0, num_points)
    values = np.random.rand(num_points) * 10.0
    mesh = SphericalMesh(lats, lons)
    grid_lats = np.linspace(30, 45, 16)
    grid_lons = np.linspace(-95, -85, 11)
    benchmark(mesh.interpolate_to_numpy_grid, values, grid_lats, grid_lons)


@pytest.mark.parametrize("grid_size", [10, 50, 100])
def test_benchmark_interpolate_numpy_scaling_grid(benchmark, grid_size):
    """
    Benchmark the SphericalMesh.interpolate_to_numpy_grid method with varying destination grid sizes.
    """
    lats = np.array([30.0, 45.0, 35.0, 40.0])
    lons = np.array([-90.0, -85.0, -95.0, -90.0])
    values = np.array([10.0, 20.0, 15.0, 18.0])
    mesh = SphericalMesh(lats, lons)
    grid_lats = np.linspace(30, 45, grid_size)
    grid_lons = np.linspace(-95, -85, grid_size)
    benchmark(mesh.interpolate_to_numpy_grid, values, grid_lats, grid_lons)


@pytest.mark.parametrize("num_points", [10, 100, 1000])
def test_benchmark_interpolate_xarray_scaling_source(benchmark, num_points):
    """
    Benchmark the SphericalMesh.interpolate (xarray) method with varying source mesh sizes.
    """
    lats = np.linspace(30.0, 45.0, num_points)
    lons = np.linspace(-90.0, -85.0, num_points)
    values = np.random.rand(num_points) * 10.0
    mesh = SphericalMesh(lats, lons)
    grid_lats = np.linspace(30, 45, 16)
    grid_lons = np.linspace(-95, -85, 11)
    source_da = xr.DataArray(values, dims=["points"]).chunk({"points": -1})

    def run_interpolation():
        mesh.interpolate(source_da, grid_lats, grid_lons).compute()

    benchmark(run_interpolation)


@pytest.mark.parametrize("grid_size", [10, 50, 100])
def test_benchmark_interpolate_xarray_scaling_grid(benchmark, grid_size):
    """
    Benchmark the SphericalMesh.interpolate (xarray) method with varying destination grid sizes.
    """
    lats = np.array([30.0, 45.0, 35.0, 40.0])
    lons = np.array([-90.0, -85.0, -95.0, -90.0])
    values = np.array([10.0, 20.0, 15.0, 18.0])
    mesh = SphericalMesh(lats, lons)
    grid_lats = np.linspace(30, 45, grid_size)
    grid_lons = np.linspace(-95, -85, grid_size)
    source_da = xr.DataArray(values, dims=["points"]).chunk({"points": -1})

    def run_interpolation():
        mesh.interpolate(source_da, grid_lats, grid_lons).compute()

    benchmark(run_interpolation)
