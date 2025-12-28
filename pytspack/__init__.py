import logging
import ctypes
import numpy as np
import os
import platform
import glob

logger = logging.getLogger(__name__)

# Load the shared library
lib_pattern = 'tspack*.so'
if platform.system() == 'Windows':
    lib_pattern = 'tspack*.dll'
elif platform.system() == 'Darwin':
    lib_pattern = 'tspack*.dylib'

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
_tspack.hval.argtypes = [ctypes.c_double, ctypes.c_int, c_double_p, c_double_p, c_double_p, c_double_p, ctypes.POINTER(ctypes.c_int)]
_tspack.hval.restype = ctypes.c_double

_tspack.hpval.argtypes = [ctypes.c_double, ctypes.c_int, c_double_p, c_double_p, c_double_p, c_double_p, ctypes.POINTER(ctypes.c_int)]
_tspack.hpval.restype = ctypes.c_double

_tspack.tsval1.argtypes = [ctypes.c_int, c_double_p, c_double_p, c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, c_double_p, c_double_p, ctypes.POINTER(ctypes.c_int)]
_tspack.tsval1.restype = None

_tspack.tspsi.argtypes = [ctypes.c_int, c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, c_double_p, c_double_p, c_double_p, ctypes.POINTER(ctypes.c_int)]
_tspack.tspsi.restype = None

_tspack.tspss.argtypes = [ctypes.c_int, c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, c_double_p, ctypes.c_double, ctypes.c_double, ctypes.c_int, c_double_p, c_double_p, c_double_p, c_double_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_tspack.tspss.restype = None

def hval(xp, x, y, yp, sigma):
    """Function which evaluates a
    Hermite interpolatory tension spline (H)
    at points `xp`.

    Parameters
    ----------
    xp : array_like
        New X points, at which H is to be evaluated.
    x : array_like
        Original X points (abscissae). Must be strictly increasing.
    y : array_like
        Data values at the original X points.
    yp : array_like
        First derivatives at original X points. HP(X(I)) = YP(I),
        where HP is the derivative of H.
    sigma : array
        Tension factors, for each interval in the original X points
        (element I corresponds to the interval (I,I+1);
        the last value in the array is not used).

    Returns
    -------
    list of float
        H values (estimates of Y(X) at new X points `xp`).
    """
    xp = np.array(xp, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    yp = np.array(yp, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    n = len(x)
    y_out = np.zeros_like(xp)

    for i, val in enumerate(xp):
        ier = ctypes.c_int(0)
        y_out[i] = _tspack.hval(val, n,
                                x.ctypes.data_as(c_double_p),
                                y.ctypes.data_as(c_double_p),
                                yp.ctypes.data_as(c_double_p),
                                sigma.ctypes.data_as(c_double_p),
                                ctypes.byref(ier))
    return y_out.tolist()


def hpval(xp, x, y, yp, sigma):
    """Function which evaluates the first derivative of a
    Hermite interpolatory tension spline (HP)
    at points `xp`.

    Parameters
    ----------
    xp : array_like
        New X points, at which HP is to be evaluated.
    x : array_like
        Original X points (abscissae). Must be strictly increasing.
    y : array_like
        Data values at the original X points.
    yp : array_like
        First derivatives at original X points. HP(X(I)) = YP(I).
    sigma : array_like
        Tension factors, for each interval in the original X points
        (element I corresponds to the interval (I,I+1);
        the last value in the array is not used).

    Returns
    -------
    list of float
        HP values (estimates of dY/dX at new X points `xp`).
    """
    xp = np.array(xp, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    yp = np.array(yp, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    n = len(x)
    yp_out = np.zeros_like(xp)

    for i, val in enumerate(xp):
        ier = ctypes.c_int(0)
        yp_out[i] = _tspack.hpval(val, n,
                                 x.ctypes.data_as(c_double_p),
                                 y.ctypes.data_as(c_double_p),
                                 yp.ctypes.data_as(c_double_p),
                                 sigma.ctypes.data_as(c_double_p),
                                 ctypes.byref(ier))
    return yp_out.tolist()


def tspsi(x, y, ncd=1, slopes=None, curvs=None, per=0, tension=None):
    """Subroutine which constructs a shape-preserving or
      unconstrained interpolatory function.  Refer to
      TSVAL1.

    Parameters
    ----------
    x : numpy array or list
        x is the original values of the array.
    y : type
        Description of parameter `y`.
    ncd : type
        Description of parameter `ncd`.
    slopes : type
        Description of parameter `slopes`.
    curvs : type
        Description of parameter `curvs`.
    per : type
        Description of parameter `per`.
    tension : type
        Description of parameter `tension`.

    Returns
    -------
    type
        Description of returned object.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    n = len(x)
    yp = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)

    if slopes is not None and curvs is not None:
        raise ValueError("You can't constrain both the slopes and curvs at the endpoints")

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
    else:
        iendc = 0

    unifrm = 1 if tension is not None else 0
    if unifrm:
        sigma[0] = tension

    if ncd == 1:
        lwk = 1
    else:
        if per:
            if unifrm:
                lwk = 2 * n
            else:
                lwk = 3 * n
        else:
            if unifrm:
                lwk = n
            else:
                lwk = 2 * n
    wk = np.zeros(lwk, dtype=np.float64)

    ier = ctypes.c_int(0)
    _tspack.tspsi(n, x.ctypes.data_as(c_double_p), y.ctypes.data_as(c_double_p), ncd, iendc,
                  per, unifrm, lwk, wk.ctypes.data_as(c_double_p),
                  yp.ctypes.data_as(c_double_p), sigma.ctypes.data_as(c_double_p), ctypes.byref(ier))

    if ier.value >= 0:
        return (x, y, yp, sigma)
    elif ier.value == -1:
        raise RuntimeError("Error, N, NCD or IENDC outside valid range")
    elif ier.value == -2:
        raise RuntimeError("Error, workspace allocated too small")
    elif ier.value == -3:
        raise RuntimeError("Error, tension outside its valid range")
    elif ier.value == -4:
        raise RuntimeError("Error, x-values are not strictly increasing")
    else:
        raise RuntimeError(f"Unknown error in tspsi: {ier.value}")


def tspss(x, y, w, per=0, tension=None, s=None, stol=None, full_output=0):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    w = np.array(w, dtype=np.float64)

    n = len(x)
    yp = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)

    unifrm = 1 if tension is not None else 0
    if unifrm:
        sigma[0] = tension

    if per:
        lwk = 11 * n if not unifrm else 10 * n
    else:
        lwk = 7 * n if not unifrm else 6 * n
    wk = np.zeros(lwk, dtype=np.float64)

    sm = float(n) if s is None else s
    smtol = np.sqrt(2.0/n) if stol is None else stol

    ier = ctypes.c_int(0)
    nit = ctypes.c_int(0)
    _tspack.tspss(n, x.ctypes.data_as(c_double_p), y.ctypes.data_as(c_double_p), per, unifrm,
                  w.ctypes.data_as(c_double_p), sm, smtol, lwk, wk.ctypes.data_as(c_double_p),
                  sigma.ctypes.data_as(c_double_p), ys.ctypes.data_as(c_double_p),
                  yp.ctypes.data_as(c_double_p), ctypes.byref(nit), ctypes.byref(ier))

    if ier.value == 0:
        mesg = "No errors and constraint is satisfied:  chisquare ~ s"
        xyds = (x, ys, yp, sigma)
    elif ier.value == 1:
        mesg = "No errors, but constraint not satisfied:  chisquare !~ s"
        xyds = (x, ys, yp, sigma)
    elif ier.value == -1:
        raise RuntimeError("Error, N, W, SM or SMTOL outside valid range")
    elif ier.value == -2:
        raise RuntimeError("Error, workspace allocated too small")
    elif ier.value == -3:
        raise RuntimeError("Error, tension outside its valid range")
    elif ier.value == -4:
        raise RuntimeError("Error, x-values are not strictly increasing")
    else:
        raise RuntimeError(f"Unknown error in tspss: {ier.value}")

    if full_output:
        return (xyds, nit.value, mesg)
    else:
        return xyds

def tsval1(x, xydt, degree=0, verbose=0):
    if not isinstance(xydt, tuple) or len(xydt) != 4:
        raise TypeError("xydt must be a 4-tuple: x, y, yp, sigma")

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

    _tspack.tsval1(n, xx.ctypes.data_as(c_double_p), yy.ctypes.data_as(c_double_p),
                   yp.ctypes.data_as(c_double_p), sigma.ctypes.data_as(c_double_p),
                   degree, ne, x.ctypes.data_as(c_double_p), v.ctypes.data_as(c_double_p),
                   ctypes.byref(ier))

    if ier.value == 0:
        return v
    elif ier.value > 0 and verbose:
        print(f"Warning: extrapolation required for {ier.value} points")
        return v
    elif ier.value > 0:
        return v
    elif ier.value == -1:
        raise RuntimeError("degree is not valid")
    elif ier.value == -2:
        raise ValueError("x values are not strictly increasing")
    else:
        raise RuntimeError(f"Unknown error in tsval1: {ier.value}")
