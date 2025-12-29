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
        self.lnew = c_int(0)

        _lib.trmesh.argtypes = [
            c_int,
            c_double_p,
            c_double_p,
            c_double_p,
            c_int_p,
            c_int_p,
            c_int_p,
            POINTER(c_int),
            c_int_p,
            c_int_p,
            c_double_p,
            POINTER(c_int),
        ]

        ier = c_int()
        _lib.trmesh(
            self.n,
            self.x,
            self.y,
            self.z,
            self.list,
            self.lptr,
            self.lend,
            byref(self.lnew),
            np.zeros(self.n, dtype=np.int32),
            np.zeros(self.n, dtype=np.int32),
            np.zeros(self.n, dtype=np.float64),
            byref(ier),
        )

        if ier.value < 0:
            raise RuntimeError(f"TRMESH Error: {ier.value}")
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
