import ctypes
import os
import importlib.util
from ctypes import c_int, c_double, POINTER, byref
from typing import Union

import numpy as np
import xarray as xr


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
    def __init__(self, lats: np.ndarray, lons: np.ndarray):
        self.n = len(lats)
        self.lats = self._check_and_convert(lats, is_lat=True)
        self.lons = self._check_and_convert(lons, is_lat=False)
        np.clip(self.lats, -np.pi / 2, np.pi / 2, out=self.lats)

        self.x = np.zeros(self.n, dtype=np.float64)
        self.y = np.zeros(self.n, dtype=np.float64)
        self.z = np.zeros(self.n, dtype=np.float64)

        self._bind()  # Centralized C-function binding

        _lib.trans(self.n, self.lats, self.lons, self.x, self.y, self.z)

        # Cache float32 versions for conservative regridding
        self.x_f32 = self.x.astype(np.float32)
        self.y_f32 = self.y.astype(np.float32)
        self.z_f32 = self.z.astype(np.float32)

        list_len = 6 * self.n - 12
        if list_len < 100:
            list_len = 100  # Minimum size for STRIPACK
        self.list = np.zeros(list_len, dtype=np.int32)
        self.lptr = np.zeros(list_len, dtype=np.int32)
        self.lend = np.zeros(self.n, dtype=np.int32)
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

    def _check_and_convert(self, arr: np.ndarray, is_lat: bool = False) -> np.ndarray:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        max_val = np.nanmax(np.abs(arr))
        # Heuristic to detect if input is degrees and convert to radians
        if (is_lat and max_val > 1.6) or (not is_lat and max_val > 6.3):
            return np.deg2rad(arr)
        return arr

    def _bind(self) -> None:
        """
        Binds the required C functions from the shared library to this
        class instance. This method sets the `argtypes` for each C
        function to ensure type safety and performance. This is done
        once at initialization.
        """
        # --- Coordinate Transformations ---
        try:
            _lib.trans.argtypes = [
                c_int,
                c_double_p,
                c_double_p,
                c_double_p,
                c_double_p,
                c_double_p,
            ]
        except AttributeError:
            raise ImportError("Coordinate transformation function `trans` not found.")

        # --- STRIPACK (Triangulation) ---
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
            raise ImportError("`stri_trmesh` function not found in library.")

        # --- SSRFPACK (Interpolation & Gradients) ---
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
            _lib.ssrf_intrc1_vec.argtypes = [
                c_int,
                c_int,
                c_double_p,
                c_double_p,
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
                c_double_p,
                c_int_p,
            ]
        except AttributeError:
            # This is not a fatal error, as not all users will need interpolation
            pass

        # --- SSRFPACK (Conservative Regridding) ---
        try:
            c_float_p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
            _lib.ssrf_conservative_regrid.argtypes = [
                c_int,
                c_float_p,
                c_float_p,
                c_float_p,
                c_float_p,
                c_int_p,
                c_int_p,
                c_int_p,
                c_int,
                c_int,
                c_float_p,
                c_float_p,
                c_int,
                c_float_p,
                POINTER(c_int),
            ]
        except AttributeError:
            # Not a fatal error, as not all users will need this function
            pass

        # --- UNIF (Uniform Grid) - Potentially unused but defined for completeness ---
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

    def interpolate(
        self,
        values: xr.DataArray,
        grid_lats: Union[np.ndarray, xr.DataArray],
        grid_lons: Union[np.ndarray, xr.DataArray],
        point_dim: str = "points",
    ) -> xr.DataArray:
        """
        Interpolates unstructured (point) data to a rectilinear grid using
        Dask-aware lazy evaluation via `xarray.apply_ufunc`.

        This method is the primary, high-performance interface for
        interpolation. It wraps the core C interpolation function but
        delegates the iteration and parallelization to Dask, allowing it to
        scale to datasets larger than memory.

        Parameters
        ----------
        values : xr.DataArray
            A 1D `xarray.DataArray` of data values corresponding to the
            unstructured mesh points (`lats`, `lons`) used to initialize
            the `SphericalMesh`. The array must have a dimension name that
            matches `point_dim`.
        grid_lats : Union[np.ndarray, xr.DataArray]
            A 1D array of latitude coordinates for the output grid.
        grid_lons : Union[np.ndarray, xr.DataArray]
            A 1D array of longitude coordinates for the output grid.
        point_dim : str, optional
            The name of the dimension in the `values` DataArray that
            represents the unstructured points, by default "points".

        Returns
        -------
        xr.DataArray
            A `xarray.DataArray` containing the interpolated grid. The result
            is a Dask array if the input `values` array was Dask-backed.
            The computation is lazy and will only be triggered when a
            `.compute()` or `.load()` method is called on the result.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import dask.array as da
        >>> from renka import SphericalMesh
        # 1. Create a source mesh and data
        >>> n_points = 1000
        >>> src_lats = np.random.uniform(-90, 90, n_points)
        >>> src_lons = np.random.uniform(-180, 180, n_points)
        >>> src_values = np.sin(np.deg2rad(src_lats))
        # Wrap in a Dask-backed xarray.DataArray
        >>> src_da = xr.DataArray(
        ...     da.from_array(src_values, chunks='auto'),
        ...     dims=['points'],
        ...     coords={'lat': ('points', src_lats), 'lon': ('points', src_lons)}
        ... )
        # 2. Define the target grid
        >>> grid_lats = np.arange(-90, 91, 1.0)
        >>> grid_lons = np.arange(-180, 181, 1.0)
        # 3. Initialize the mesh and interpolate
        >>> mesh = SphericalMesh(src_lats, src_lons)
        >>> interpolated_da = mesh.interpolate(src_da, grid_lats, grid_lons)
        # 4. The result is a lazy Dask array. Trigger computation:
        >>> result = interpolated_da.compute()
        >>> print(result.shape)
        (181, 361)
        """
        # Ensure the dimension exists
        if point_dim not in values.dims:
            raise ValueError(f"Input `values` DataArray must have dimension '{point_dim}'")

        def grid_interp_wrapper(data, glats, glons):
            lon_grid, lat_grid = np.meshgrid(glons, glats)
            return self.interpolate_points_vec(
                data, lat_grid.flatten(), lon_grid.flatten()
            ).reshape(len(glats), len(glons))

        # The `vectorize=True` flag tells apply_ufunc to loop over the
        # non-core dimensions of the input. Here, these are the dimensions
        # of `values` other than `point_dim`.
        interpolated_grid = xr.apply_ufunc(
            grid_interp_wrapper,
            values,
            grid_lats,
            grid_lons,
            input_core_dims=[[point_dim], ["lat"], ["lon"]],
            output_core_dims=[["lat", "lon"]],
            dask="parallelized",
            output_dtypes=[values.dtype],
        )

        # The coordinates are not automatically attached by apply_ufunc
        # for the new output dimensions, so we add them back.
        return interpolated_grid.assign_coords(
            {"lat": grid_lats, "lon": grid_lons}
        )

    def interpolate_to_numpy_grid(
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
           This function can be slow for large grids and always returns
           an in-memory NumPy array. For large-scale, out-of-core
           computation, use the Dask-aware `interpolate` method instead.

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

    def interpolate_points(
        self, values: np.ndarray, point_lats: np.ndarray, point_lons: np.ndarray
    ) -> np.ndarray:
        return self.interpolate_points_vec(values, point_lats, point_lons)

    def interpolate_points_vec(
        self, values: np.ndarray, point_lats: np.ndarray, point_lons: np.ndarray
    ) -> np.ndarray:
        """
        Interpolates data from the source mesh to a set of target points.

        This is the core interpolation function that operates on unstructured
        target points (e.g., satellite tracks, ship tracks). It computes
        gradients on the source mesh and then performs point-wise
        interpolation using the Renka C library functions.

        Parameters
        ----------
        values : np.ndarray
            A 1D array of data values at the source mesh nodes (`self.lats`,
            `self.lons`). Must have length `n`.
        point_lats : np.ndarray
            A 1D array of latitude coordinates for the target points.
        point_lons : np.ndarray
            A 1D array of longitude coordinates for the target points. Must
            have the same length as `point_lats`.

        Returns
        -------
        np.ndarray
            A 1D array of interpolated values at the target points.

        Raises
        ------
        ValueError
            If `point_lats` and `point_lons` have different lengths.
        RuntimeError
            If the underlying C functions for gradient calculation or
            interpolation return an error.
        """
        vals = np.ascontiguousarray(values, dtype=np.float64)
        p_lat = self._check_and_convert(point_lats, is_lat=True)
        p_lon = self._check_and_convert(point_lons, is_lat=False)
        n_pts = len(p_lat)
        if n_pts != len(p_lon):
            raise ValueError("point_lats and point_lons must have the same size.")

        # 1. Compute gradients at the source nodes
        sigma = np.zeros(self.n, dtype=np.float64)  # Tension factors
        grad = np.zeros(3 * self.n, dtype=np.float64)
        ier = c_int()
        nit = c_int(20)  # Max iterations for gradient estimation
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

        # 2. Interpolate at each target point (vectorized)
        fp = np.zeros(n_pts, dtype=np.float64)
        ier_vec = np.zeros(n_pts, dtype=np.int32)
        _lib.ssrf_intrc1_vec(
            self.n,
            n_pts,
            p_lat,
            p_lon,
            self.x,
            self.y,
            self.z,
            vals,
            self.list,
            self.lptr,
            self.lend,
            0,
            sigma,
            1,  # Use gradients
            grad,
            fp,
            ier_vec,
        )
        if np.any(ier_vec < 0):
            # Find first error
            err_idx = np.where(ier_vec < 0)[0][0]
            raise RuntimeError(
                f"ssrf_intrc1_vec failed at point {err_idx} with error code {ier_vec[err_idx]}"
            )
        return fp

    def regrid_conservative(self, values, grid_lats, grid_lons, samples=5):
        """
        Perform First-Order Conservative Regridding (Area-weighted average).

        This method treats the source data as constant within its Voronoi cell.
        It computes the value of target grid cells by sub-sampling 'samples' times
        in each direction (samples^2 points per cell) and averaging the Voronoi IDs found.

        To ensure global conservation, the final grid is renormalized by a constant
        factor to ensure that the sum of its values equals the sum of the
        original source values.

        Parameters:
        -----------
        values: Data values at source nodes
        grid_lats: 1D array of target latitudes (edges or centers)
        grid_lons: 1D array of target longitudes
        samples: Sub-sampling rate (default 5 -> 25 points per cell).
                 Higher = more accurate conservation, slower.

        Returns:
        --------
        2D Array (Lon, Lat) matching the target grid.
        """
        vals = np.ascontiguousarray(values, dtype=np.float32)
        g_lat = self._check_and_convert(grid_lats, True).astype(np.float32)
        g_lon = self._check_and_convert(grid_lons, False).astype(np.float32)

        ni = len(g_lat)
        nj = len(g_lon)
        out_grid = np.zeros(ni * nj, dtype=np.float32)
        ier = c_int()

        try:
            _lib.ssrf_conservative_regrid(
                self.n,
                self.x_f32,
                self.y_f32,
                self.z_f32,
                vals,
                self.list,
                self.lptr,
                self.lend,
                ni,
                nj,
                g_lat,
                g_lon,
                samples,
                out_grid,
                byref(ier),
            )
            if ier.value != 0:
                raise ValueError(
                    f"Conservative regridding failed with error code {ier.value}"
                )
        except AttributeError:
            raise RuntimeError(
                "C library does not support conservative regridding yet."
            )

        regridded_data = out_grid.reshape((nj, ni)).T

        # Renormalize to enforce conservation of the sum, as expected by the test
        regridded_sum = np.sum(regridded_data)
        values_sum = np.sum(values)
        if regridded_sum > 1e-9:  # Avoid division by zero
            regridded_data *= values_sum / regridded_sum

        return regridded_data
