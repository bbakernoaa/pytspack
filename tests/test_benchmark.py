import pytest
import numpy as np
from renka import SphericalMesh


@pytest.mark.parametrize("source_size", [10, 100, 1000])
def test_benchmark_interpolate_scaling_source(benchmark, source_size):
    """
    Tests the scaling of interpolation with respect to the number of source
    points.
    """
    lats = np.random.uniform(-90, 90, source_size)
    lons = np.random.uniform(0, 360, source_size)
    values = np.random.rand(source_size)
    mesh = SphericalMesh(lats, lons)

    grid_lats = np.linspace(-90, 90, 100)
    grid_lons = np.linspace(0, 360, 100)
    result = mesh.interpolate(values, grid_lats, grid_lons)
    benchmark(result.compute)


@pytest.mark.parametrize("grid_size", [10, 50, 100])
def test_benchmark_interpolate_scaling_grid(benchmark, grid_size):
    """
    Tests the scaling of interpolation with respect to the grid size.
    """
    lats = np.array([0, 10, 20, 30])
    lons = np.array([0, 10, 20, 30])
    values = np.array([1, 2, 3, 4])
    mesh = SphericalMesh(lats, lons)

    grid_lats = np.linspace(-90, 90, grid_size)
    grid_lons = np.linspace(0, 360, grid_size)
    result = mesh.interpolate(values, grid_lats, grid_lons)
    benchmark(result.compute)


@pytest.mark.parametrize("source_size", [10, 100, 1000])
def test_benchmark_regrid_conservative_scaling_source(benchmark, source_size):
    """
    Tests the scaling of conservative regridding with respect to the number of
    source points.
    """
    lats = np.random.uniform(-90, 90, source_size)
    lons = np.random.uniform(0, 360, source_size)
    values = np.random.rand(source_size)
    mesh = SphericalMesh(lats, lons)

    grid_lats = np.linspace(-90, 90, 100)
    grid_lons = np.linspace(0, 360, 100)
    benchmark(mesh.regrid_conservative, values, grid_lats, grid_lons)


@pytest.mark.parametrize("grid_size", [10, 50, 100])
def test_benchmark_regrid_conservative_scaling_grid(benchmark, grid_size):
    """
    Tests the scaling of conservative regridding with respect to the grid size.
    """
    lats = np.array([0, 10, 20, 30])
    lons = np.array([0, 10, 20, 30])
    values = np.array([1, 2, 3, 4])
    mesh = SphericalMesh(lats, lons)

    grid_lats = np.linspace(-90, 90, grid_size)
    grid_lons = np.linspace(0, 360, grid_size)
    benchmark(mesh.regrid_conservative, values, grid_lats, grid_lons)
