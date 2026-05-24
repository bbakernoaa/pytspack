import xarray as xr
import numpy as np
import datetime
from typing import Union
from .pytspack import TsPack


def _tspack_interp1d(y, x, x_new, tension):
    """
    Internal wrapper for 1D interpolation using TsPack.
    Designed for use with xarray.apply_ufunc.

    Parameters
    ----------
    y : np.ndarray
        1D array of data values.
    x : np.ndarray
        1D array of source coordinates.
    x_new : np.ndarray
        1D array of target coordinates.
    tension : float
        Tension factor for the spline.

    Returns
    -------
    np.ndarray
        Interpolated values.
    """
    # TsPack requires strictly increasing x.
    # Vertical coordinates like pressure often decrease with index.
    # Check if x is strictly increasing.
    dx = np.diff(x)
    if np.all(dx > 0):
        x_sorted = x
        y_sorted = y
    elif np.all(dx < 0):
        x_sorted = x[::-1]
        y_sorted = y[::-1]
    else:
        # Sort if not monotonic
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]

    # Check for NaNs or non-strictly increasing x which TsPack cannot handle
    if (
        np.any(np.isnan(y_sorted))
        or np.any(np.isnan(x_sorted))
        or np.any(np.diff(x_sorted) <= 0)
    ):
        return np.full(len(x_new), np.nan)

    try:
        tsp = TsPack()
        # Create interpolator
        predict = tsp.interpolate(x_sorted, y_sorted, tension=tension)
        return predict(x_new)
    except Exception:
        # Fallback for errors in interpolation (e.g. non-increasing x)
        return np.full(len(x_new), np.nan)


def interpolate_vertical(
    data: Union[xr.DataArray, xr.Dataset],
    target_levels: Union[np.ndarray, list],
    level_dim: str = "level",
    tension: float = 0.0,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Interpolates data to new vertical levels using tension splines (TSPACK).

    This function is Dask-aware and operates lazily.

    Parameters
    ----------
    data : Union[xr.DataArray, xr.Dataset]
        The input data containing a vertical dimension to be interpolated.
    target_levels : Union[np.ndarray, list]
        A 1D array or list of the target vertical level values.
    level_dim : str, optional
        The name of the vertical dimension in the input data, by default "level".
    tension : float, optional
        The tension factor for the spline interpolation. 0.0 results in a
        standard cubic spline. Higher values make the curve 'tighter'.
        Defaults to 0.0.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        A new xarray object with the data interpolated to the target levels.
    """
    if isinstance(data, xr.Dataset):
        new_ds = xr.Dataset(attrs=data.attrs)
        for var_name, da in data.data_vars.items():
            if level_dim in da.dims:
                new_ds[var_name] = interpolate_vertical(
                    da, target_levels, level_dim, tension
                )
            else:
                new_ds[var_name] = da
        for coord_name, coord in data.coords.items():
            if coord_name not in new_ds.coords:
                new_ds.coords[coord_name] = coord
        return new_ds

    if level_dim not in data.dims:
        raise ValueError(f"Dimension '{level_dim}' not found in DataArray.")

    target_levels = np.asarray(target_levels)

    dask_kwargs = {}
    if data.chunks is not None:
        dask_kwargs["output_sizes"] = {"new_level": len(target_levels)}

    interpolated_data = xr.apply_ufunc(
        _tspack_interp1d,
        data,
        data[level_dim],
        input_core_dims=[[level_dim], [level_dim]],
        output_core_dims=[["new_level"]],
        exclude_dims={level_dim},
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs=dask_kwargs,
        vectorize=True,
        kwargs={"x_new": target_levels, "tension": tension},
    )

    result = interpolated_data.rename({"new_level": level_dim})
    result = result.assign_coords({level_dim: target_levels})

    # Preserve original dimension order
    original_dims = list(data.dims)
    new_dims = [dim for dim in original_dims if dim != level_dim]
    new_dims.insert(data.dims.index(level_dim), level_dim)
    result = result.transpose(*new_dims)

    # Provenance
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history_log = (
        f"{timestamp}: Vertically interpolated from '{level_dim}' "
        f"to {len(target_levels)} levels using pytspack tension spline (tension={tension})."
    )
    if "history" in data.attrs:
        history_log = f"{data.attrs['history']}\n{history_log}"
    result.attrs["history"] = history_log

    return result
