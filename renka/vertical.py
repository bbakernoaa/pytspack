import xarray as xr
import numpy as np
import datetime
from typing import Union

def _interp1d(y_slice, x_coords, x_new_coords, method):
    """
    Wrapper for 1D interpolation that handles linear and log methods.
    This function is designed to be used with `xarray.apply_ufunc`.

    Parameters
    ----------
    y_slice : np.ndarray
        1D array of data values to interpolate.
    x_coords : np.ndarray
        1D array of the source coordinates corresponding to `y_slice`.
    x_new_coords : np.ndarray
        1D array of the target coordinates.
    method : str
        Interpolation method, either 'linear' or 'log'.

    Returns
    -------
    np.ndarray
        Interpolated data values on the `x_new_coords`.
    """
    # np.interp requires monotonically increasing coordinates.
    # Atmospheric pressure often decreases with height, so we sort.
    sorted_indices = np.argsort(x_coords)
    y_slice_sorted = y_slice[sorted_indices]
    x_coords_sorted = x_coords[sorted_indices]

    if method == 'log':
        if np.any(x_coords <= 0) or np.any(x_new_coords <= 0):
            raise ValueError("Log interpolation requires all coordinate values to be positive.")
        return np.interp(
            np.log(x_new_coords),
            np.log(x_coords_sorted),
            y_slice_sorted,
            left=np.nan,
            right=np.nan,
        )
    elif method == 'linear':
        return np.interp(
            x_new_coords,
            x_coords_sorted,
            y_slice_sorted,
            left=np.nan,
            right=np.nan,
        )
    else:
        raise ValueError(f"Unknown interpolation method: '{method}'")

def interpolate_vertical(
    data: Union[xr.DataArray, xr.Dataset],
    target_levels: Union[np.ndarray, list],
    level_dim: str = "level",
    method: str = "linear",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Interpolates data to new vertical levels.

    This function is Dask-aware and operates lazily. It can perform
    linear interpolation on any vertical coordinate or log-linear
    interpolation for pressure levels.

    Parameters
    ----------
    data : Union[xr.DataArray, xr.Dataset]
        The input data containing a vertical dimension to be interpolated.
    target_levels : Union[np.ndarray, list]
        A 1D array or list of the target vertical level values.
    level_dim : str, optional
        The name of the vertical dimension in the input data, by default "level".
    method : str, optional
        The interpolation method. Can be 'linear' for linear interpolation in
        height/pressure, or 'log' for interpolation in log-pressure space.
        Defaults to "linear".

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        A new xarray object with the data interpolated to the target levels.
        The result is a Dask-backed array if the input was Dask-backed.

    Raises
    ------
    ValueError
        If an unknown `method` is specified or if `level_dim` is not in the data.
    """
    if isinstance(data, xr.Dataset):
        new_ds = xr.Dataset(attrs=data.attrs)
        for var_name, da in data.data_vars.items():
            if level_dim in da.dims:
                new_ds[var_name] = interpolate_vertical(
                    da, target_levels, level_dim, method
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
        _interp1d,
        data,
        data[level_dim],
        input_core_dims=[[level_dim], [level_dim]],
        output_core_dims=[["new_level"]],
        exclude_dims={level_dim},
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs=dask_kwargs,
        vectorize=True,
        kwargs={"x_new_coords": target_levels, "method": method},
    )

    result = interpolated_data.rename({"new_level": level_dim})
    result = result.assign_coords({level_dim: target_levels})

    # Preserve original dimension order
    original_dims = list(data.dims)
    new_dims = [dim for dim in original_dims if dim != level_dim]
    new_dims.insert(data.dims.index(level_dim), level_dim)
    result = result.transpose(*new_dims)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history_log = (
        f"{timestamp}: Vertically interpolated from '{level_dim}' "
        f"to {len(target_levels)} levels using '{method}' method."
    )
    if "history" in data.attrs:
        history_log = f"{data.attrs['history']}\n{history_log}"
    result.attrs["history"] = history_log

    return result
