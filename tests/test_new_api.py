import unittest
import numpy as np
from renka import TsPack, SphericalMesh


class TestNewApi(unittest.TestCase):
    def test_tspack_interpolation(self):
        """Test the TsPack class."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.0, 1.0])

        tspack = TsPack()
        interpolator = tspack.interpolate(x, y)

        test_points = np.array([0.5, 1.5, 2.5])
        results = interpolator(test_points)

        self.assertEqual(results.shape, (3,))
        self.assertTrue(np.all(np.isfinite(results)))

    def test_spherical_mesh_interpolation(self):
        """Test the SphericalMesh class."""
        lats = np.array([0.0, 10.0, 20.0, 5.0])
        lons = np.array([0.0, 5.0, 15.0, 10.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        mesh = SphericalMesh(lats, lons)

        grid_lats = np.array([5.0, 15.0])
        grid_lons = np.array([5.0, 15.0])

        results = mesh.interpolate(values, grid_lats, grid_lons)

        self.assertEqual(results.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(results)))

    def test_spherical_mesh_points_interpolation(self):
        """Test SphericalMesh point interpolation."""
        lats = np.array([-80.0, -45.0, 0.0, 45.0, 80.0])
        lons = np.array([0.0, 90.0, 180.0, 270.0, 0.0])
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mesh = SphericalMesh(lats, lons)

        point_lats = np.array([-50.0, -40.0, -30.0])
        point_lons = np.array([100.0, 110.0, 120.0])
        output = mesh.interpolate_points(values, point_lats, point_lons)

        self.assertEqual(output.shape, (3,))
        min_val, max_val = np.min(values), np.max(values)
        # Allow for slight extrapolation outside the range
        self.assertTrue(np.all(output >= min_val - 1) and np.all(output <= max_val + 1))


if __name__ == "__main__":
    unittest.main()
