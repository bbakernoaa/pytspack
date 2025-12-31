import numpy as np
import pytest
import ctypes
from renka import SphericalMesh
from renka.renka import _lib


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


def test_spherical_mesh_interpolate_points_vec():
    """
    Test the vectorized interpolation vs the original.
    """
    source_lats = np.array([0, 0, 90, -90])
    source_lons = np.array([0, 90, 0, 0])
    source_values = np.array([10.0, 20.0, 30.0, 40.0])
    mesh = SphericalMesh(lats=source_lats, lons=source_lons)
    target_lats = np.array([0, 85])
    target_lons = np.array([45, 0])

    # The `interpolate_points` method now calls the vectorized version by default
    vectorized_values = mesh.interpolate_points(
        values=source_values,
        point_lats=target_lats,
        point_lons=target_lons,
    )

    # To test the original, non-vectorized implementation, we can call it directly
    # by creating a temporary, non-vectorized version of the interpolation method
    import types

    original_interpolate = types.MethodType(_original_interpolate_points, mesh)
    original_values = original_interpolate(
        values=source_values,
        point_lats=target_lats,
        point_lons=target_lons,
    )

    np.testing.assert_allclose(
        vectorized_values,
        original_values,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Vectorized interpolation does not match original.",
    )


def _original_interpolate_points(
    self, values: np.ndarray, point_lats: np.ndarray, point_lons: np.ndarray
) -> np.ndarray:
    """
    This is a copy of the original, non-vectorized interpolate_points method,
    to be used for comparison testing.
    """
    vals = np.ascontiguousarray(values, dtype=np.float64)
    p_lat = self._check_and_convert(point_lats, is_lat=True)
    p_lon = self._check_and_convert(point_lons, is_lat=False)
    n_pts = len(p_lat)
    if n_pts != len(p_lon):
        raise ValueError("point_lats and point_lons must have the same size.")

    sigma = np.zeros(self.n, dtype=np.float64)
    grad = np.zeros(3 * self.n, dtype=np.float64)
    ier = ctypes.c_int()
    nit = ctypes.c_int(20)
    dgmax = ctypes.c_double(0.0)

    _lib.ssrf_gradg(
        self.n,
        self.x,
        self.y,
        self.z,
        vals,
        self.list,
        self.lptr,
        self.lend,
        0,
        sigma,
        ctypes.byref(nit),
        ctypes.byref(dgmax),
        grad,
        ctypes.byref(ier),
    )
    if ier.value < 0:
        raise RuntimeError(f"ssrf_gradg failed with error code {ier.value}")

    fp = np.zeros(n_pts, dtype=np.float64)
    ist = ctypes.c_int(1)

    for i in range(n_pts):
        fp_i = ctypes.c_double(0.0)
        _lib.ssrf_intrc1(
            self.n,
            p_lat[i],
            p_lon[i],
            self.x,
            self.y,
            self.z,
            vals,
            self.list,
            self.lptr,
            self.lend,
            0,
            sigma,
            1,
            grad,
            ctypes.byref(ist),
            ctypes.byref(fp_i),
            ctypes.byref(ier),
        )
        if ier.value < 0:
            raise RuntimeError(
                f"ssrf_intrc1 failed at point {i} with error {ier.value}"
            )
        fp[i] = fp_i.value

    return fp
