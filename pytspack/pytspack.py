import ctypes
import os
import importlib.util
from ctypes import c_int, POINTER, byref

import numpy as np


# Helper to load the C extension
def _load_library():
    spec = importlib.util.find_spec("pytspack._libpytspack")
    if spec and spec.origin:
        return ctypes.CDLL(spec.origin)

    # Fallback for local dev
    import glob

    local_libs = glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "_libpytspack*")
    )
    if local_libs:
        return ctypes.CDLL(local_libs[0])
    raise ImportError("Could not find _libpytspack extension.")


_lib = _load_library()

c_double_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


class TsPack:
    """
    Python wrapper for the TSPACK (Tension Spline Curve Fitting Package) library.

    TSPACK is a collection of C functions for constructing a smooth curve that
    interpolates a set of data points in the plane. The interpolant is a
    tension spline, which allows for control over the 'tightness' of the curve.
    """

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
            raise RuntimeError("TSPACK symbols missing in the library.")

    def interpolate(self, x, y, tension=0.0):
        """
        Creates an interpolating tension spline for the given data points.

        Parameters
        ----------
        x : array-like
            The x-coordinates of the data points. Must be strictly increasing.
        y : array-like
            The y-coordinates of the data points.
        tension : float or array-like, optional
            The tension factor(s) for the spline. A value of 0 results in a
            standard cubic spline. Higher values make the curve 'tighter'.
            Default is 0.0.

        Returns
        -------
        callable
            A function that can be used to evaluate the spline at new points.
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D arrays.")

        n = len(x)
        if n < 2:
            raise ValueError("x and y must have at least 2 points.")

        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing.")

        yp = np.zeros(n, dtype=np.float64)
        ier = c_int()

        _lib.ypc1(n, x, y, yp, byref(ier))
        if ier.value != 0:
            raise ValueError(f"TSPACK Error (ypc1): {ier.value}")

        if np.isscalar(tension):
            sigma = np.full(n, tension, dtype=np.float64)
        else:
            sigma = np.ascontiguousarray(tension, dtype=np.float64)

        def predict(t):
            is_scalar = np.isscalar(t)
            t_arr = np.atleast_1d(np.ascontiguousarray(t, dtype=np.float64))
            res = np.zeros(len(t_arr), dtype=np.float64)
            ier = c_int()
            _lib.tsval1(n, x, y, yp, sigma, 0, len(t_arr), t_arr, res, byref(ier))
            if ier.value < 0:
                raise ValueError(f"TSPACK Error (tsval1): {ier.value}")
            return res[0] if is_scalar else res

        return predict
