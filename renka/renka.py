import ctypes
import numpy as np
import os
import importlib.util
from ctypes import c_int, c_double, POINTER, byref


# Helper to load the C extension
def _load_library():
    spec = importlib.util.find_spec("renka._librenka")
    if spec and spec.origin:
        return ctypes.CDLL(spec.origin)

    # Fallback for local dev
    import glob

    local_libs = glob.glob(os.path.join(os.path.dirname(__file__), "..", "_librenka*"))
    if local_libs:
        return ctypes.CDLL(local_libs[0])
    raise ImportError("Could not find _librenka extension.")


_lib = _load_library()

c_double_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


class TsPack:
    def __init__(self):
        try:
            _lib.ypc1.argtypes = [
                c_int,
                c_double_p,
                c_double_p,
                c_double_p,
                POINTER(c_int),
            ]
            _lib.tsval1.argtypes = [
                c_int,
                c_double_p,
                c_double_p,
                c_double_p,
                c_double_p,
                c_int,
                c_int,
                c_double_p,
                c_double_p,
                POINTER(c_int),
            ]
        except AttributeError:
            raise RuntimeError("TSPACK symbols missing.")

    def interpolate(self, x, y, tension=0.0):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(x)
        yp = np.zeros(n, dtype=np.float64)
        ier = c_int()

        _lib.ypc1(n, x, y, yp, byref(ier))
        if ier.value != 0:
            raise ValueError(f"TSPACK Error: {ier.value}")

        sigma = np.full(n, tension, dtype=np.float64)

        def predict(t):
            t = np.ascontiguousarray(t, dtype=np.float64)
            res = np.zeros(len(t), dtype=np.float64)
            _lib.tsval1(n, x, y, yp, sigma, 0, len(t), t, res, byref(c_int()))
            return res

        return predict


class SphericalMesh:
    def __init__(self, lats, lons):
        self.n = len(lats)
        self.lats = self._check_and_convert(lats, is_lat=True)
        self.lons = self._check_and_convert(lons, is_lat=False)
        np.clip(self.lats, -np.pi / 2, np.pi / 2, out=self.lats)

        self.x = np.zeros(self.n, dtype=np.float64)
        self.y = np.zeros(self.n, dtype=np.float64)
        self.z = np.zeros(self.n, dtype=np.float64)

        _lib.trans.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
        ]
        _lib.trans(self.n, self.lats, self.lons, self.x, self.y, self.z)

        list_len = 6 * self.n - 12
        if list_len < 100:
            list_len = 100
        self.list = np.zeros(list_len, dtype=np.int32)
        self.lptr = np.zeros(list_len, dtype=np.int32)
        self.lend = np.zeros(self.n, dtype=np.int32)

        try:
            _lib.stri_trmesh.argtypes = [
                c_int,
                c_double_p,
                c_double_p,
                c_double_p,
                c_int_p,
                c_int_p,
                c_int_p,
                POINTER(c_int),
            ]
        except AttributeError:
            raise ImportError("stri_trmesh function not found in library.")

        ier = c_int()
        _lib.stri_trmesh(
            self.n,
            self.x,
            self.y,
            self.z,
            self.list,
            self.lptr,
            self.lend,
            byref(ier),
        )

        if ier.value != 0:
            raise RuntimeError(f"STRIPACK TRMESH Error: {ier.value}")
        self._bind()

    def _check_and_convert(self, arr, is_lat=False):
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        max_val = np.nanmax(np.abs(arr))
        if (is_lat and max_val > 1.6) or (not is_lat and max_val > 6.3):
            return np.deg2rad(arr)
        return arr

    def _bind(self):
        try:
            _lib.unif.argtypes = [
                c_int,
                c_int_p,
                c_int,
                c_double_p,
                c_double_p,
                c_double_p,
                c_double_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int,
                c_double_p,
                c_int,
                c_int,
                c_int,
                c_double_p,
                c_double_p,
                c_int,
                c_double,
                c_double_p,
                POINTER(c_int),
            ]
        except AttributeError:
            pass

        try:
            _lib.ssrf_gradg.argtypes = [
                c_int,
                c_double_p,
                c_double_p,
                c_double_p,
                c_double_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int,
                c_double_p,
                POINTER(c_int),
                POINTER(c_double),
                c_double_p,
                POINTER(c_int),
            ]
            _lib.ssrf_intrc1.argtypes = [
                c_int,
                c_double,
                c_double,
                c_double_p,
                c_double_p,
                c_double_p,
                c_double_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int,
                c_double_p,
                c_int,
                c_double_p,
                POINTER(c_int),
                POINTER(c_double),
                POINTER(c_int),
            ]
        except AttributeError:
            # Silently fail if symbols are not available
            pass

    def interpolate(
        self,
        values: np.ndarray,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> np.ndarray:
        """Interpolates data onto a rectilinear grid.

        This method is a convenience wrapper around `interpolate_points`. It
        constructs a grid, flattens it, performs the point-wise
        interpolation, and then reshapes the result back into a 2D grid.

        .. warning::
           This function can be slow for large grids due to the underlying
           loop in `interpolate_points`.

        Parameters
        ----------
        values : np.ndarray
            A 1D NumPy array of data values corresponding to the input `lats`
            and `lons` used to initialize the `SphericalMesh`. The array
            length must be equal to `n`.
        grid_lats : np.ndarray
            A 1D NumPy array specifying the latitude coordinates of the
            output grid. Values can be in degrees or radians.
        grid_lons : np.ndarray
            A 1D NumPy array specifying the longitude coordinates of the
            output grid. Values can be in degrees or radians.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of shape `(len(grid_lats), len(grid_lons))`
            containing the interpolated values. Points outside the convex
            hull of the data will be `np.nan`.
        """
        # Create a meshgrid and flatten it for point-wise interpolation
        lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
        flat_lats = lat_grid.flatten()
        flat_lons = lon_grid.flatten()

        # Interpolate at the flattened points
        interpolated_values = self.interpolate_points(values, flat_lats, flat_lons)

        # Reshape the 1D result back to the 2D grid shape
        return interpolated_values.reshape(len(grid_lats), len(grid_lons))

    def interpolate_points(self, values, point_lats, point_lons):
        """Curvilinear (point-based) Grid Interpolation"""
        vals = np.ascontiguousarray(values, dtype=np.float64)
        p_lat = self._check_and_convert(point_lats, True)
        p_lon = self._check_and_convert(point_lons, False)
        n_pts = len(p_lat)
        if n_pts != len(p_lon):
            raise ValueError("point_lats and point_lons must have the same size.")

        # 1. Compute gradients
        sigma = np.zeros(self.n, dtype=np.float64)
        grad = np.zeros(3 * self.n, dtype=np.float64)
        ier = c_int()
        nit = c_int(20)
        dgmax = c_double(0.0)

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
            byref(nit),
            byref(dgmax),
            grad,
            byref(ier),
        )
        if ier.value < 0:
            raise RuntimeError(f"ssrf_gradg failed with error code {ier.value}")

        # 2. Interpolate at each point
        fp = np.zeros(n_pts, dtype=np.float64)
        ist = c_int(1)

        for i in range(n_pts):
            fp_i = c_double(0.0)
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
                byref(ist),
                byref(fp_i),
                byref(ier),
            )
            if ier.value < 0:
                # Non-fatal errors (e.g., extrapolation) are positive
                raise RuntimeError(
                    f"ssrf_intrc1 failed at point {i} with error code {ier.value}"
                )
            fp[i] = fp_i.value

        return fp
