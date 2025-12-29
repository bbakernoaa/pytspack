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

    def interpolate(self, values, grid_lats, grid_lons):
        """Rectilinear Grid Interpolation"""
        vals = np.ascontiguousarray(values, dtype=np.float64)
        g_lat = self._check_and_convert(grid_lats, True)
        g_lon = self._check_and_convert(grid_lons, False)
        ni, nj = len(g_lat), len(g_lon)
        ff = np.zeros(ni * nj, dtype=np.float64)
        ier = c_int()

        _lib.unif(
            0,
            np.zeros(1, dtype=np.int32),
            self.n,
            self.x,
            self.y,
            self.z,
            vals,
            self.list,
            self.lptr,
            self.lend,
            0,
            np.zeros(1, dtype=np.float64),
            ni,
            ni,
            nj,
            g_lat,
            g_lon,
            0,
            0.0,
            ff,
            byref(ier),
        )
        return ff.reshape((nj, ni)).T

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
