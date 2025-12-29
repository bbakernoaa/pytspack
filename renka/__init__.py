import sys
import numpy as np
from .renka import (
    tspsi as _tspsi,
    tsval1 as _tsval1,
    trmesh as _trmesh,
    stri_trmesh as _stri_trmesh,
    tspss as _tspss,
    gradg as _gradg,
    intr_2d as _intr_2d,
    ssrf_gradg as _ssrf_gradg,
    ssrf_intr as _ssrf_intr,
)


def _flatten_and_eval(func, t_block, *args):
    """
    Helper to flatten input block, run function, and reshape result.
    func: The evaluation function (e.g., _tsval1 or _intr_2d wrapper)
    t_block: The main input array block (to be iterated over)
    args: other arguments to func
    """
    t_flat = t_block.ravel()
    # If the function takes more than one variable array (e.g. intr_2d takes px, py),
    # this helper assumes t_block is the one that needs flattening.
    # But intr_2d takes px AND py. They must be same shape.
    # This helper is for single array case (tsval1).
    res_flat = func(t_flat, *args)
    return res_flat.reshape(t_block.shape)


def _evaluate(xn, yn, yp, sigma, t, iflag):
    """
    Helper to evaluate spline at t with dask/xarray support.
    """

    # Wrapper for C function to fix arguments
    def wrapper(t_arr):
        # Flatten t_arr because C extension iterates on dim(0) only
        t_flat = np.ascontiguousarray(t_arr.ravel())
        res = _tsval1(xn, yn, yp, sigma, t_flat, iflag)
        return res.reshape(t_arr.shape)

    # Check for xarray
    if "xarray" in sys.modules:
        import xarray as xr

        if isinstance(t, xr.DataArray):
            return xr.apply_ufunc(
                wrapper,
                t,
                dask="parallelized",
                output_dtypes=[float],
            )

    # Check for dask
    if "dask" in sys.modules:
        import dask.array as da

        if isinstance(t, da.Array):
            return t.map_blocks(wrapper, dtype=float)

    # Fallback to direct call (handles numpy array)
    return wrapper(t)


def _evaluate_surf(px, py, func, *args):
    """
    Helper for surface interpolation (2 args: px, py).
    func: underlying C wrapper (e.g. _intr_2d)
    args: extra args (mesh, sigma, grad...)
    """

    def wrapper(px_arr, py_arr):
        # Flatten
        px_flat = np.ascontiguousarray(px_arr.ravel())
        py_flat = np.ascontiguousarray(py_arr.ravel())
        res = func(px_flat, py_flat, *args)
        return res.reshape(px_arr.shape)

    # Check for xarray
    if "xarray" in sys.modules:
        import xarray as xr

        if isinstance(px, xr.DataArray) or isinstance(py, xr.DataArray):
            return xr.apply_ufunc(
                wrapper,
                px,
                py,
                dask="parallelized",
                output_dtypes=[float],
            )

    # Check for dask
    if "dask" in sys.modules:
        import dask.array as da

        if isinstance(px, da.Array) or isinstance(py, da.Array):
            # Ensure both are dask arrays or compatible
            if not isinstance(px, da.Array):
                px = da.from_array(px, chunks=py.chunks)
            if not isinstance(py, da.Array):
                py = da.from_array(py, chunks=px.chunks)

            return da.map_blocks(wrapper, px, py, dtype=float)

    # Fallback
    return wrapper(px, py)


def tspsi(x, y, ncd=1, iendc=0, per=0, unifrm=0, yp=None, sigma=None):
    """
    Compute derivatives and tension factors for a curve.
    Returns (x, y, yp, sigma).
    """
    # Explicitly convert inputs to numpy arrays for the C extension.
    x_in = np.ascontiguousarray(x, dtype=float)
    y_in = np.ascontiguousarray(y, dtype=float)

    try:
        yp_out, sigma_out = _tspsi(
            x_in, y_in, ncd=ncd, iendc=iendc, per=per, unifrm=unifrm, yp=yp, sigma=sigma
        )
    except RuntimeError as e:
        msg = str(e)
        if "error code -4" in msg:
            raise RuntimeError("x-values are not strictly increasing") from e
        raise e

    # Return the computed numpy arrays to ensure consistency
    return x_in, y_in, yp_out, sigma_out


def tsval1(x, xydt):
    """
    Evaluate spline at points x (te).
    xydt is a tuple (x_data, y_data, yp, sigma).
    """
    if len(xydt) != 4:
        raise ValueError("xydt must contain (x, y, yp, sigma)")

    xn, yn, yp, sigma = xydt
    return _evaluate(xn, yn, yp, sigma, x, 0)


def hval(t, x, y, yp, sigma):
    """
    Evaluate spline at points t.
    """
    return _evaluate(x, y, yp, sigma, t, 0)


def hpval(t, x, y, yp, sigma):
    """
    Evaluate spline derivative at points t.
    """
    return _evaluate(x, y, yp, sigma, t, 1)


def tspss(x, y, w, per=0, unifrm=0, s=0.0, smtol=0.01):
    """
    Smooth curve.
    s maps to sm (smoothing parameter).
    """
    if smtol <= 0.0:
        smtol = 0.01

    res = _tspss(x, y, per, unifrm, w, s, smtol)
    return x, res["ys"], res["yp"], res["sigma"]


def trmesh(x, y):
    """
    Create a triangulation for 2D points.
    Returns dictionary with list, lptr, lend, lnew.
    """
    x_in = np.ascontiguousarray(x, dtype=float)
    y_in = np.ascontiguousarray(y, dtype=float)
    return _trmesh(x_in, y_in)


def stri_trmesh(x, y, z):
    """
    Create a triangulation for points on a sphere.
    Returns dictionary with list, lptr, lend.
    """
    x_in = np.ascontiguousarray(x, dtype=float)
    y_in = np.ascontiguousarray(y, dtype=float)
    z_in = np.ascontiguousarray(z, dtype=float)
    return _stri_trmesh(x_in, y_in, z_in)


# Planar Surface Interpolation


def gradg(x, y, z, mesh):
    """
    Compute gradients for surface interpolation.
    mesh: dict returned by trmesh (must contain list, lptr, lend).
    Returns (sigma, grad).
    """
    # Extract mesh
    list_arr = mesh["list"]
    lptr_arr = mesh["lptr"]
    lend_arr = mesh["lend"]

    return _gradg(x, y, z, list_arr, lptr_arr, lend_arr)


def intr_2d(px, py, x, y, z, mesh, sigma, grad):
    """
    Interpolate 2D surface at points (px, py).
    x, y, z: nodes
    mesh: triangulation
    sigma: tension factors (from gradg)
    grad: gradients (from gradg)
    Returns pz (interpolated values).
    """
    list_arr = mesh["list"]
    lptr_arr = mesh["lptr"]
    lend_arr = mesh["lend"]

    return _evaluate_surf(
        px, py, _intr_2d, x, y, z, list_arr, lptr_arr, lend_arr, sigma, grad
    )


# Spherical Surface Interpolation


def ssrf_gradg(x, y, z, f, mesh):
    """
    Compute gradients for spherical surface interpolation.
    f: function values at nodes.
    mesh: triangulation from stri_trmesh.
    Returns (sigma, grad).
    """
    list_arr = mesh["list"]
    lptr_arr = mesh["lptr"]
    lend_arr = mesh["lend"]

    return _ssrf_gradg(x, y, z, f, list_arr, lptr_arr, lend_arr)


def ssrf_intr(plat, plon, x, y, z, f, mesh, sigma, grad):
    """
    Interpolate spherical surface at points (plat, plon).
    x, y, z: nodes (Cartesian)
    f: function values
    mesh: triangulation
    sigma, grad: from ssrf_gradg
    Returns pval.
    """
    list_arr = mesh["list"]
    lptr_arr = mesh["lptr"]
    lend_arr = mesh["lend"]

    return _evaluate_surf(
        plat, plon, _ssrf_intr, x, y, z, f, list_arr, lptr_arr, lend_arr, sigma, grad
    )
