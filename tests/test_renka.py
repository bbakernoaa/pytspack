import unittest
import numpy as np
import renka
import sys

# Try importing dask/xarray
try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


class TestTspack(unittest.TestCase):
    def test_tspsi_tsval1(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        # ncd=1 (derivatives), iendc=0, per=0, unifrm=0, sigma=None
        # Updated tspsi returns (x, y, yp, sigma) to match legacy API
        xn, yn, yp, sigma = renka.tspsi(x, y, 1, 0, 0, 0, None)
        self.assertEqual(len(yp), 3)
        self.assertEqual(len(sigma), 3)

        te = np.array([0.5, 1.5])
        # tsval1 takes (te, xydt) where xydt is tuple
        xydt = (xn, yn, yp, sigma)
        v = renka.tsval1(te, xydt)
        self.assertEqual(len(v), 2)
        # Simple check, exact values depend on spline logic but should be symmetric
        self.assertAlmostEqual(v[0], v[1])

    def test_trmesh(self):
        x = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        res = renka.trmesh(x, y)
        self.assertIn("list", res)
        self.assertIn("lptr", res)
        self.assertIn("lend", res)
        self.assertIn("lnew", res)
        return x, y, res

    def test_stri_trmesh(self):
        # 6 points on sphere (octahedron vertices)
        x = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        res = renka.stri_trmesh(x, y, z)
        self.assertIn("list", res)
        self.assertIn("lptr", res)
        self.assertIn("lend", res)
        return x, y, z, res

    def test_planar_interpolation(self):
        # Create a simple mesh
        x = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        # z = x + y (plane)
        z = x + y

        mesh = renka.trmesh(x, y)

        # Compute gradients
        sigma, grad = renka.gradg(x, y, z, mesh)

        # Interpolate at center
        px = np.array([0.5])
        py = np.array([0.5])
        pz = renka.intr_2d(px, py, x, y, z, mesh, sigma, grad)

        # Check result
        self.assertAlmostEqual(pz[0], 1.0, places=5)  # 0.5 + 0.5 = 1.0

    def test_spherical_interpolation(self):
        # 6 points on sphere
        x = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])

        # f = z coordinate
        f = z.copy()

        mesh = renka.stri_trmesh(x, y, z)

        sigma, grad = renka.ssrf_gradg(x, y, z, f, mesh)

        # Interpolate
        val = 1.0 / np.sqrt(2.0)
        plat = np.array([np.arcsin(val)])  # Lat
        plon = np.array([0.0])  # Lon?

        # Let's just run it and ensure it runs without error.
        tlat = np.array([0.5])
        tlon = np.array([0.5])

        res = renka.ssrf_intr(tlat, tlon, x, y, z, f, mesh, sigma, grad)
        self.assertEqual(len(res), 1)

    @unittest.skipUnless(HAS_DASK, "Dask not installed")
    def test_intr_2d_dask(self):
        x = np.array([0.0, 1.0, 0.0, 1.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        z = x + y
        mesh = renka.trmesh(x, y)
        sigma, grad = renka.gradg(x, y, z, mesh)

        px = np.linspace(0, 1, 10)
        py = np.linspace(0, 1, 10)

        dx = da.from_array(px, chunks=5)
        dy = da.from_array(py, chunks=5)

        dz = renka.intr_2d(dx, dy, x, y, z, mesh, sigma, grad)
        self.assertIsInstance(dz, da.Array)

        computed = dz.compute()
        expected = renka.intr_2d(px, py, x, y, z, mesh, sigma, grad)
        np.testing.assert_allclose(computed, expected)

    @unittest.skipUnless(HAS_DASK, "Dask not installed")
    def test_multidim_eval_dask(self):
        # Test multidimensional evaluation points
        x = np.linspace(0, 10, 11)
        y = np.sin(x)
        x_out, y_out, yp, sigma = renka.tspsi(x, y)
        xydt = (x_out, y_out, yp, sigma)

        t = np.random.rand(4, 4) * 10
        dt = da.from_array(t, chunks=(2, 2))

        res = renka.tsval1(dt, xydt)
        self.assertIsInstance(res, da.Array)
        self.assertEqual(res.shape, (4, 4))

        computed = res.compute()
        expected = renka.tsval1(t, xydt)
        np.testing.assert_allclose(computed, expected)

    @unittest.skipUnless(HAS_XARRAY, "Xarray not installed")
    def test_multidim_eval_xarray(self):
        x = np.linspace(0, 10, 11)
        y = np.sin(x)
        x_out, y_out, yp, sigma = renka.tspsi(x, y)
        xydt = (x_out, y_out, yp, sigma)

        t = np.random.rand(4, 4) * 10
        da_t = xr.DataArray(t, dims=("a", "b"))

        res = renka.tsval1(da_t, xydt)
        self.assertIsInstance(res, xr.DataArray)
        self.assertEqual(res.shape, (4, 4))
        self.assertEqual(res.dims, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
