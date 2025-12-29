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

class TestDaskXarraySupport(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 10, 11)
        self.y = np.sin(self.x)
        self.t = np.linspace(0, 10, 21)

    @unittest.skipUnless(HAS_DASK, "Dask not installed")
    def test_tspsi_dask(self):
        dx = da.from_array(self.x, chunks=5)
        dy = da.from_array(self.y, chunks=5)

        # tspsi should accept dask arrays but return computed numpy arrays
        x_out, y_out, yp, sigma = renka.tspsi(dx, dy)

        # x_out, y_out should be numpy arrays
        self.assertIsInstance(x_out, np.ndarray)
        self.assertIsInstance(y_out, np.ndarray)
        self.assertIsInstance(yp, np.ndarray)
        self.assertIsInstance(sigma, np.ndarray)

    @unittest.skipUnless(HAS_XARRAY, "Xarray not installed")
    def test_tspsi_xarray(self):
        dx = xr.DataArray(self.x, dims='x')
        dy = xr.DataArray(self.y, dims='x')

        x_out, y_out, yp, sigma = renka.tspsi(dx, dy)

        self.assertIsInstance(x_out, np.ndarray)
        self.assertIsInstance(y_out, np.ndarray)
        self.assertIsInstance(yp, np.ndarray)
        self.assertIsInstance(sigma, np.ndarray)

    @unittest.skipUnless(HAS_DASK, "Dask not installed")
    def test_tsval1_dask(self):
        # Prepare spline params (using numpy for simplicity)
        x_out, y_out, yp, sigma = renka.tspsi(self.x, self.y)
        xydt = (x_out, y_out, yp, sigma)

        dt = da.from_array(self.t, chunks=5)

        # Evaluate
        res = renka.tsval1(dt, xydt)

        # Result should be dask array
        self.assertIsInstance(res, da.Array)

        # Compute and check values
        computed = res.compute()
        expected = renka.tsval1(self.t, xydt)
        np.testing.assert_allclose(computed, expected)

    @unittest.skipUnless(HAS_XARRAY, "Xarray not installed")
    def test_tsval1_xarray(self):
        x_out, y_out, yp, sigma = renka.tspsi(self.x, self.y)
        xydt = (x_out, y_out, yp, sigma)

        dt = xr.DataArray(self.t, dims='t', coords={'t': self.t})

        # Evaluate
        res = renka.tsval1(dt, xydt)

        # Result should be xarray DataArray
        self.assertIsInstance(res, xr.DataArray)
        self.assertEqual(res.dims, ('t',))

        # Check values
        np.testing.assert_allclose(res.values, renka.tsval1(self.t, xydt))

    @unittest.skipUnless(HAS_DASK, "Dask not installed")
    def test_hval_dask(self):
        # hval is wrapper around tsval1
        x_out, y_out, yp, sigma = renka.tspsi(self.x, self.y)
        dt = da.from_array(self.t, chunks=5)

        res = renka.hval(dt, x_out, y_out, yp, sigma)
        self.assertIsInstance(res, da.Array)

        computed = res.compute()
        expected = renka.hval(self.t, x_out, y_out, yp, sigma)
        np.testing.assert_allclose(computed, expected)

if __name__ == '__main__':
    unittest.main()
