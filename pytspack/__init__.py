"""
Python wrapper for the TSPACK C library.
"""
import ctypes
import glob
import logging
import os
import platform
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Load the shared library
lib_pattern = "tspack*.so"
if platform.system() == "Windows":
    lib_pattern = "tspack*.dll"
elif platform.system() == "Darwin":
    lib_pattern = "tspack*.dylib"

try:
    # Look for the library in the same directory as this file
    lib_dir = os.path.dirname(__file__)
    lib_paths = glob.glob(os.path.join(lib_dir, lib_pattern))
    if not lib_paths:
        raise OSError("Cannot find tspack shared library")
    lib_path = lib_paths[0]
    _tspack = ctypes.CDLL(lib_path)
except OSError as e:
    raise OSError(f"Failed to load tspack shared library: {e}") from e

# Define a common type for arrays of doubles
c_double_p = ctypes.POINTER(ctypes.c_double)

# Set up argument and return types for the C functions
_tspack.hval.argtypes = [
    ctypes.c_double,
    ctypes.c_int,
    c_double_p,
    c_double_p,
    c_double_p,
    c_double_p,
    ctypes.POINTER(ctypes.c_int),
]
_tspack.hval.restype = ctypes.c_double

_tspack.hpval.argtypes = [
    ctypes.c_double,
    ctypes.c_int,
    c_double_p,
    c_double_p,
    c_double_p,
    c_double_p,
    ctypes.POINTER(ctypes.c_int),
]
_tspack.hpval.restype = ctypes.c_double

_tspack.tsval1.argtypes = [
    ctypes.c_int,
    c_double_p,
    c_double_p,
    c_double_p,
    c_double_p,
    ctypes.c_int,
    ctypes.c_int,
    c_double_p,
    c_double_p,
    ctypes.POINTER(ctypes.c_int),
]
_tspack.tsval1.restype = None

_tspack.tspsi.argtypes = [
    ctypes.c_int,
    c_double_p,
    c_double_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    c_double_p,
    c_double_p,
    c_double_p,
    ctypes.POINTER(ctypes.c_int),
]
_tspack.tspsi.restype = None

_tspack.tspss.argtypes = [
    ctypes.c_int,
    c_double_p,
    c_double_p,
    ctypes.c_int,
    ctypes.c_int,
    c_double_p,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    c_double_p,
    c_double_p,
    c_double_p,
    c_double_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_tspack.tspss.restype = None


def hval(
    xp: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a Hermite interpolatory tension spline at given points.

    Parameters
    ----------
    xp : np.ndarray
        New X points at which the spline is to be evaluated.
    x : np.ndarray
        Original X points (abscissae). Must be strictly increasing.
    y : np.ndarray
        Data values at the original X points.
    yp : np.ndarray
        First derivatives at original X points.
    sigma : np.ndarray
        Tension factors for each interval in the original X points.

    Returns
    -------
    np.ndarray
        Spline values at the new X points `xp`.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    yp = np.array(yp, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)
    n = len(x)

    def func(val):
        ier = ctypes.c_int(0)
        return _tspack.hval(
            val,
            n,
            x.ctypes.data_as(c_double_p),
            y.ctypes.data_as(c_double_p),
            yp.ctypes.data_as(c_double_p),
            sigma.ctypes.data_as(c_double_p),
            ctypes.byref(ier),
        )

    vectorized_func = np.vectorize(func)
    return vectorized_func(xp)


def hpval(
    xp: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the first derivative of a Hermite interpolatory tension spline.

    Parameters
    ----------
    xp : np.ndarray
        New X points at which the derivative is to be evaluated.
    x : np.ndarray
        Original X points (abscissae). Must be strictly increasing.
    y : np.ndarray
        Data values at the original X points.
    yp : np.ndarray
        First derivatives at original X points.
    sigma : np.ndarray
        Tension factors for each interval in the original X points.

    Returns
    -------
    np.ndarray
        Derivative values at the new X points `xp`.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    yp = np.array(yp, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)
    n = len(x)

    def func(val):
        ier = ctypes.c_int(0)
        return _tspack.hpval(
            val,
            n,
            x.ctypes.data_as(c_double_p),
            y.ctypes.data_as(c_double_p),
            yp.ctypes.data_as(c_double_p),
            sigma.ctypes.data_as(c_double_p),
            ctypes.byref(ier),
        )

    vectorized_func = np.vectorize(func)
    return vectorized_func(xp)


def tspsi(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    ncd: int = 1,
    slopes: Optional[List[float]] = None,
    curvs: Optional[List[float]] = None,
    per: int = 0,
    tension: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a shape-preserving or unconstrained interpolatory function.

    Parameters
    ----------
    x : Union[List[float], np.ndarray]
        The original x-values of the data points.
    y : Union[List[float], np.ndarray]
        The original y-values of the data points.
    ncd : int, optional
        Number of constraints, by default 1.
    slopes : Optional[List[float]], optional
        Endpoint slope constraints `[slope0, slopeN]`, by default None.
    curvs : Optional[List[float]], optional
        Endpoint curvature constraints `[curv0, curvN]`, by default None.
    per : int, optional
        Flag for periodic boundary conditions, by default 0.
    tension : Optional[float], optional
        Uniform tension factor, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing (x, y, yp, sigma) where `yp` are the computed
        derivatives and `sigma` are the tension factors.

    Raises
    ------
    ValueError
        If both `slopes` and `curvs` are specified.
    TypeError
        If `slopes` or `curvs` are not lists.
    RuntimeError
        If the C library returns an error.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    n = len(x)
    yp = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)

    if slopes is not None and curvs is not None:
        raise ValueError("Cannot constrain both slopes and curvatures")

    iendc = 0
    if slopes is not None:
        if not isinstance(slopes, list):
            raise TypeError("slopes must be a list: [slope0, slope1]")
        iendc = 1
        yp[0], yp[-1] = slopes[0], slopes[1]
    elif curvs is not None:
        if not isinstance(curvs, list):
            raise TypeError("curvs must be a list: [curv1, curv2]")
        iendc = 2
        yp[0], yp[-1] = curvs[0], curvs[1]

    unifrm = 1 if tension is not None else 0
    if unifrm:
        sigma[0] = tension

    lwk = 3 * n
    wk = np.zeros(lwk, dtype=np.float64)

    ier = ctypes.c_int(0)
    _tspack.tspsi(
        n,
        x.ctypes.data_as(c_double_p),
        y.ctypes.data_as(c_double_p),
        ncd,
        iendc,
        per,
        unifrm,
        lwk,
        wk.ctypes.data_as(c_double_p),
        yp.ctypes.data_as(c_double_p),
        sigma.ctypes.data_as(c_double_p),
        ctypes.byref(ier),
    )

    if ier.value >= 0:
        return (x, y, yp, sigma)
    elif ier.value == -1:
        raise RuntimeError("N, NCD or IENDC outside valid range")
    elif ier.value == -2:
        raise RuntimeError("Workspace allocated too small")
    elif ier.value == -3:
        raise RuntimeError("Tension outside its valid range")
    elif ier.value == -4:
        raise RuntimeError("X-values are not strictly increasing")
    else:
        raise RuntimeError(f"Unknown error in tspsi: {ier.value}")


def tspss(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    w: Union[List[float], np.ndarray],
    per: int = 0,
    tension: Optional[float] = None,
    s: Optional[float] = None,
    stol: Optional[float] = None,
    full_output: int = 0,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int, str
    ],
]:
    """
    Construct a smoothing spline.

    Parameters
    ----------
    x : Union[List[float], np.ndarray]
        The original x-values.
    y : Union[List[float], np.ndarray]
        The original y-values.
    w : Union[List[float], np.ndarray]
        Weights for the data points.
    per : int, optional
        Flag for periodic boundary conditions, by default 0.
    tension : Optional[float], optional
        Uniform tension factor, by default None.
    s : Optional[float], optional
        Smoothing factor, by default `len(x)`.
    stol : Optional[float], optional
        Tolerance for the smoothing factor, by default `sqrt(2/n)`.
    full_output : int, optional
        If non-zero, return a tuple with extra information, by default 0.

    Returns
    -------
    Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int, str
        ],
    ]
        If `full_output` is 0, returns `(x, ys, yp, sigma)`.
        If `full_output` is non-zero, returns `((x, ys, yp, sigma), nit, mesg)`.

    Raises
    ------
    RuntimeError
        If the C library returns an error.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    w = np.array(w, dtype=np.float64)

    n = len(x)
    ys = np.zeros(n, dtype=np.float64)
    yp = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)

    unifrm = 1 if tension is not None else 0
    if unifrm:
        sigma[0] = tension

    lwk = 11 * n
    wk = np.zeros(lwk, dtype=np.float64)

    sm = float(n) if s is None else s
    smtol = np.sqrt(2.0 / n) if stol is None else stol

    ier = ctypes.c_int(0)
    nit = ctypes.c_int(0)
    _tspack.tspss(
        n,
        x.ctypes.data_as(c_double_p),
        y.ctypes.data_as(c_double_p),
        per,
        unifrm,
        w.ctypes.data_as(c_double_p),
        sm,
        smtol,
        lwk,
        wk.ctypes.data_as(c_double_p),
        sigma.ctypes.data_as(c_double_p),
        ys.ctypes.data_as(c_double_p),
        yp.ctypes.data_as(c_double_p),
        ctypes.byref(nit),
        ctypes.byref(ier),
    )

    if ier.value in (0, 1):
        mesg = (
            "No errors and constraint is satisfied."
            if ier.value == 0
            else "No errors, but constraint not satisfied."
        )
        xyds = (x, ys, yp, sigma)
        return (xyds, nit.value, mesg) if full_output else xyds
    elif ier.value == -1:
        raise RuntimeError("N, W, SM or SMTOL outside valid range")
    elif ier.value == -2:
        raise RuntimeError("Workspace allocated too small")
    elif ier.value == -3:
        raise RuntimeError("Tension outside its valid range")
    elif ier.value == -4:
        raise RuntimeError("X-values are not strictly increasing")
    else:
        raise RuntimeError(f"Unknown error in tspss: {ier.value}")


def tsval1(
    x: Union[List[float], np.ndarray],
    xydt: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    degree: int = 0,
    verbose: int = 0,
) -> np.ndarray:
    """
    Evaluate an interpolatory function or its derivatives.

    Parameters
    ----------
    x : Union[List[float], np.ndarray]
        Points at which to evaluate the function.
    xydt : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple `(x, y, yp, sigma)` from `tspsi` or `tspss`.
    degree : int, optional
        Derivative to evaluate (0 for function, 1 for first, 2 for second),
        by default 0.
    verbose : int, optional
        If non-zero, print warning about extrapolation, by default 0.

    Returns
    -------
    np.ndarray
        The evaluated values of the function or its derivative.

    Raises
    ------
    TypeError
        If `xydt` is not a 4-tuple.
    RuntimeError
        If the C library returns an error.
    ValueError
        If the x-values in `xydt` are not strictly increasing.
    """
    if not isinstance(xydt, tuple) or len(xydt) != 4:
        raise TypeError("xydt must be a 4-tuple: (x, y, yp, sigma)")

    xx, yy, yp, sigma = xydt
    xx = np.array(xx, dtype=np.float64)
    yy = np.array(yy, dtype=np.float64)
    yp = np.array(yp, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)
    x = np.array(x, dtype=np.float64)

    n = len(xx)
    ne = len(x)
    v = np.zeros(ne, dtype=np.float64)
    ier = ctypes.c_int(0)

    _tspack.tsval1(
        n,
        xx.ctypes.data_as(c_double_p),
        yy.ctypes.data_as(c_double_p),
        yp.ctypes.data_as(c_double_p),
        sigma.ctypes.data_as(c_double_p),
        degree,
        ne,
        x.ctypes.data_as(c_double_p),
        v.ctypes.data_as(c_double_p),
        ctypes.byref(ier),
    )

    if ier.value >= 0:
        if ier.value > 0 and verbose:
            print(f"Warning: extrapolation required for {ier.value} points")
        return v
    elif ier.value == -1:
        raise RuntimeError("Degree is not valid")
    elif ier.value == -2:
        raise ValueError("X-values are not strictly increasing")
    else:
        raise RuntimeError(f"Unknown error in tsval1: {ier.value}")
