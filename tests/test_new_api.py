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
        lats = np.array([0.0, 10.0, 20.0, 0.0])
        lons = np.array([0.0, 0.0, 0.0, 10.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])

        mesh = SphericalMesh(lats, lons)

        grid_lats = np.array([5.0, 15.0])
        grid_lons = np.array([5.0, 15.0])

        results = mesh.interpolate(values, grid_lats, grid_lons)

        self.assertEqual(results.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(results)))


if __name__ == "__main__":
    unittest.main()
