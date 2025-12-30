import numpy as np
import pytest
from renka import SphericalMesh


def test_spherical_mesh_interpolate_points():
    """
    Test the core point-wise interpolation functionality.
    """
    # 1. The Logic (Setup)
    # Create a simple source mesh (e.g., 4 points on a sphere)
    source_lats = np.array([0, 0, 90, -90])
    source_lons = np.array([0, 90, 0, 0])
    source_values = np.array([10.0, 20.0, 30.0, 40.0])

    mesh = SphericalMesh(lats=source_lats, lons=source_lons)

    # Define target points for interpolation
    # Point 1: Midway between the first two source points
    # Point 2: Close to the north pole
    target_lats = np.array([0, 85])
    target_lons = np.array([45, 0])

    # 2. The Proof (Execution & Assertion)
    interpolated_values = mesh.interpolate_points(
        values=source_values,
        point_lats=target_lats,
        point_lons=target_lons,
    )

    # Assert the output shape is correct
    assert interpolated_values.shape == (2,), "Output array shape is incorrect."

    # Assert the values are plausible.
    # The value at (0, 45) should be between 10 and 20.
    assert 10.0 < interpolated_values[0] < 20.0, (
        "Interpolation at equator is out of range."
    )
    # The value at (85, 0) should be close to 30 (the north pole value).
    assert np.isclose(interpolated_values[1], 30.0, atol=2.0), (
        "Interpolation near the pole is inaccurate."
    )
