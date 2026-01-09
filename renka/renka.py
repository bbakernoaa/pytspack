import ctypes
import os
import importlib.util
import datetime
from ctypes import c_int, c_double, POINTER, byref
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    import dask.array

from renka.utils import jitter_coordinates

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
    def __init__(
        self,
        lats: Union[np.ndarray, "dask.array.Array"],
        lons: Union[np.ndarray, "dask.array.Array"],
        n_partitions: int = 1,
    ):
        """
        Initializes a spherical mesh for scattered data interpolation.

        This class builds a Delaunay triangulation of unstructured points on
        the surface of a sphere. The resulting mesh is used as the basis for
        high-performance interpolation and regridding operations.

        The constructor is lazy and does not perform the actual triangulation
        until a method like `interpolate` or `regrid_conservative` is
        called. This minimizes upfront computation.

        Parameters
        ----------
        lats : Union[np.ndarray, "dask.array.Array"]
            A 1D array of latitude coordinates for the unstructured mesh
            points. Values can be in degrees or radians; the class will
            automatically detect and convert degrees to radians.
        lons : Union[np.ndarray, "dask.array.Array"]
            A 1D array of longitude coordinates for the unstructured mesh
            points. Values can be in degrees or radians.
        n_partitions : int, optional
            If greater than 1, the mesh is partitioned into `n_partitions`
            sub-meshes for potentially faster processing on very large
            datasets. This feature is experimental. Default is 1.

        Examples
        --------
        >>> import numpy as np
        >>> from renka import SphericalMesh
        # Create random points on a sphere
        >>> n_points = 1000
        >>> src_lats = np.random.uniform(-90, 90, n_points)
        >>> src_lons = np.random.uniform(-180, 180, n_points)
        # Initialize the mesh
        >>> mesh = SphericalMesh(src_lats, src_lons)
        >>> print(f"Mesh created with {mesh.n} points.")
        Mesh created with 1000 points.
        """
        if hasattr(lats, "dask"):
            self.n = lats.shape[0]
        else:
            self.n = len(lats)

        self.lats = lats
        self.lons = lons
        self.n_partitions = n_partitions
        self._triangulated = False
        self._bind()

    def _build_partitioned_mesh(self):
        import dask.array as da
        from scipy.spatial import KDTree

        # 1. Compute centroids for partitioning
        partition_lons = np.linspace(-180, 180, self.n_partitions + 2)[1:-1]
        partition_lats = np.zeros_like(partition_lons)
        px, py, pz = self._spherical_to_cartesian(partition_lats, partition_lons)
        partition_points = np.vstack([px, py, pz]).T

        # 2. Assign each source point to the nearest partition centroid
        src_x, src_y, src_z = self._spherical_to_cartesian(self.lats, self.lons)
        src_points = da.vstack([src_x, src_y, src_z]).T

        tree = KDTree(partition_points)
        _, assignments = tree.query(src_points.compute())

        # 3. Create a mesh for each partition in parallel
        self.meshes_ = []
        self.indices_ = []
        for i in range(self.n_partitions):
            idx = np.where(assignments == i)[0]
            if len(idx) > 3:  # Need at least 3 points for a triangle
                lats, lons = jitter_coordinates(self.lats[idx], self.lons[idx])
                mesh = SphericalMesh(lats, lons)
                mesh._compute_mesh()
                self.meshes_.append(mesh)
                self.indices_.append(idx)

    def _spherical_to_cartesian(self, lats, lons):
        x = np.cos(lats) * np.cos(lons)
        y = np.cos(lats) * np.sin(lons)
        z = np.sin(lats)
        return x, y, z

    def _compute_mesh(self):
        if self._triangulated:
            return

        if self.n > 100_000 and self.n_partitions > 1:
            self._build_partitioned_mesh()
            self._triangulated = True
            return

        if hasattr(self.lats, "dask"):
            self.lats = self.lats.compute()
        if hasattr(self.lons, "dask"):
            self.lons = self.lons.compute()

        self.lats = self._check_and_convert(self.lats, is_lat=True)
        self.lons = self._check_and_convert(self.lons, is_lat=False)
        np.clip(self.lats, -np.pi / 2, np.pi / 2, out=self.lats)

        self.x = np.zeros(self.n, dtype=np.float64)
        self.y = np.zeros(self.n, dtype=np.float64)
        self.z = np.zeros(self.n, dtype=np.float64)

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
        self._triangulated = True

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
        **1. Basic Interpolation**

        >>> import numpy as np
        >>> import xarray as xr
        >>> import dask.array as da
        >>> from renka import SphericalMesh
        # Create a source mesh and data
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
        # Define the target grid
        >>> grid_lats = np.arange(-90, 91, 1.0)
        >>> grid_lons = np.arange(-180, 181, 1.0)
        # Initialize the mesh and interpolate
        >>> mesh = SphericalMesh(src_lats, src_lons)
        >>> interpolated_da = mesh.interpolate(src_da, grid_lats, grid_lons)
        # The result is a lazy Dask array. Trigger computation:
        >>> result = interpolated_da.compute()
        >>> print(result.shape)
        (181, 361)

        **2. Visualization with Matplotlib and Cartopy**

        This example demonstrates how to plot the interpolated grid, which is a
        critical step for scientific validation and publication.

        >>> import matplotlib.pyplot as plt
        >>> import cartopy.crs as ccrs
        # Assuming `result` is the computed DataArray from the first example
        >>> fig = plt.figure(figsize=(10, 5))
        >>> ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        >>> ax.set_global()
        >>> ax.coastlines()
        # The data is on a regular lat/lon grid, so PlateCarree is the correct transform
        >>> transform = ccrs.PlateCarree()
        >>> result.plot.pcolormesh(
        ...     ax=ax,
        ...     transform=transform,
        ...     x='lon',
        ...     y='lat',
        ...     cmap='viridis'
        ... ) # doctest: +SKIP
        >>> # plt.show() # doctest: +SKIP
        """
        # Ensure the dimension exists
        if point_dim not in values.dims:
            raise ValueError(
                f"Input `values` DataArray must have dimension '{point_dim}'"
            )

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
        result = interpolated_grid.assign_coords({"lat": grid_lats, "lon": grid_lons})

        # --- Provenance ---
        # Append to the history attribute of the new DataArray
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        history_log = (
            f"{timestamp}: Interpolated from unstructured mesh "
            f"(n={self.n}) to a regular grid "
            f"({len(grid_lats)}x{len(grid_lons)}) using renka.SphericalMesh."
        )
        if "history" in values.attrs:
            history_log = f"{values.attrs['history']}\n{history_log}"
        result.attrs["history"] = history_log

        return result

    def _get_best_mesh(self, lat, lon):
        if not hasattr(self, "meshes_"):
            return self, -1

        # Simple spatial lookup: find the mesh whose centroid is closest.
        # A more robust solution would check for convex hull containment.
        x, y, z = self._spherical_to_cartesian(np.deg2rad(lat), np.deg2rad(lon))
        target_point = np.array([x, y, z])

        closest_mesh_idx = -1
        min_dist = float("inf")

        for i, mesh in enumerate(self.meshes_):
            centroid = np.array([mesh.x.mean(), mesh.y.mean(), mesh.z.mean()])
            dist = np.linalg.norm(target_point - centroid)
            if dist < min_dist:
                min_dist = dist
                closest_mesh_idx = i

        return self.meshes_[closest_mesh_idx], closest_mesh_idx

    def interpolate_to_numpy_grid(
        self,
        values: np.ndarray,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolates data onto a rectilinear grid (eager execution).

        This method provides a simple, NumPy-native interface for
        interpolation. It constructs a grid, flattens it, performs
        point-wise interpolation, and reshapes the result back into a 2D
        grid. The entire operation is performed eagerly and returns an
        in-memory NumPy array.

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

        Notes
        -----
        This function can be slow for large grids as it is not parallelized
        and does not use Dask. For large-scale, out-of-core computation,
        use the Dask-aware `interpolate` method instead.

        Examples
        --------
        >>> import numpy as np
        >>> from renka import SphericalMesh
        # 1. Create a source mesh and data
        >>> n_points = 500
        >>> src_lats = np.random.uniform(-90, 90, n_points)
        >>> src_lons = np.random.uniform(-180, 180, n_points)
        >>> src_values = np.sin(np.deg2rad(src_lats)) * np.cos(np.deg2rad(src_lons))
        # 2. Define the target grid
        >>> grid_lats = np.arange(-90, 91, 10) # Coarse 10-degree grid
        >>> grid_lons = np.arange(-180, 181, 10)
        # 3. Initialize the mesh and interpolate
        >>> mesh = SphericalMesh(src_lats, src_lons)
        >>> result_grid = mesh.interpolate_to_numpy_grid(src_values, grid_lats, grid_lons)
        >>> print(f"Output grid shape: {result_grid.shape}")
        Output grid shape: (19, 37)
        >>> # Check that the output is a NumPy array
        >>> print(isinstance(result_grid, np.ndarray))
        True
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
        self._compute_mesh()

        if hasattr(self, "meshes_"):
            # Simplified partitioned interpolation
            results = np.full(len(point_lats), np.nan, dtype=np.float64)
            for i in range(len(point_lats)):
                mesh, mesh_idx = self._get_best_mesh(point_lats[i], point_lons[i])
                if mesh:
                    partition_values = values[self.indices_[mesh_idx]]
                    results[i] = mesh.interpolate_points_vec(
                        partition_values,
                        np.array([point_lats[i]]),
                        np.array([point_lons[i]]),
                    )[0]
            return results

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

    def regrid_conservative(
        self,
        values: xr.DataArray,
        grid_lats: Union[np.ndarray, xr.DataArray],
        grid_lons: Union[np.ndarray, xr.DataArray],
        samples: int = 5,
        point_dim: str = "points",
    ) -> xr.DataArray:
        """
        Performs first-order conservative regridding using a sub-sampling
        (Monte Carlo) method.

        This method is Dask-aware and performs all computations lazily. It
        treats the source data as constant within each Voronoi cell. The value
        for each target grid cell is computed by averaging the values of
        `samples * samples` sample points placed uniformly within that cell.

        To ensure global conservation, the final grid is renormalized by a
        constant factor to ensure that the sum of its values equals the sum
        of the original source values.

        Parameters
        ----------
        values : xr.DataArray
            A 1D `xarray.DataArray` of data values on the unstructured mesh.
            Must have a dimension name that matches `point_dim`.
        grid_lats : Union[np.ndarray, xr.DataArray]
            A 1D array of latitude coordinates for the output grid.
        grid_lons : Union[np.ndarray, xr.DataArray]
            A 1D array of longitude coordinates for the output grid.
        samples : int, optional
            The sub-sampling rate per grid cell direction. The total number
            of sample points per cell is `samples*samples`. A higher number
            improves accuracy but increases computation time. Default is 5.
        point_dim : str, optional
            The name of the dimension in `values` that represents the
            unstructured points. Default is "points".

        Returns
        -------
        xr.DataArray
            A `xarray.DataArray` containing the regridded data on the new grid.
            The result is a Dask array if the input `values` array was
            Dask-backed.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> import dask.array as da
        >>> from renka import SphericalMesh

        # 1. Create source mesh and data
        >>> n_points = 2000
        >>> src_lats = np.random.uniform(-90, 90, n_points)
        >>> src_lons = np.random.uniform(-180, 180, n_points)
        >>> src_values = np.cos(np.deg2rad(src_lats))**2
        >>> src_da = xr.DataArray(
        ...     da.from_array(src_values, chunks='auto'),
        ...     dims=['points'],
        ...     coords={'lat': ('points', src_lats), 'lon': ('points', src_lons)}
        ... )

        # 2. Define target grid and perform regridding
        >>> grid_lats = np.arange(-85, 86, 10.0)
        >>> grid_lons = np.arange(-175, 176, 10.0)
        >>> mesh = SphericalMesh(src_lats, src_lons)
        >>> regridded_da = mesh.regrid_conservative(src_da, grid_lats, grid_lons)

        # 3. The result is lazy. Compute and check shape.
        >>> result = regridded_da.compute()
        >>> print(result.shape)
        (18, 36)

        # 4. Visualization with Cartopy
        >>> import matplotlib.pyplot as plt
        >>> import cartopy.crs as ccrs
        >>> fig = plt.figure(figsize=(10, 5))
        >>> ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
        >>> ax.set_global()
        >>> ax.coastlines()
        >>> result.plot.pcolormesh(
        ...     ax=ax,
        ...     transform=ccrs.PlateCarree(),
        ...     x='lon',
        ...     y='lat',
        ...     cmap='viridis'
        ... ) # doctest: +SKIP
        >>> # plt.show() # doctest: +SKIP
        """
        if point_dim not in values.dims:
            raise ValueError(
                f"Input `values` DataArray must have dimension '{point_dim}'"
            )

        def _regrid_wrapper(data, glats, glons):
            """Core wrapper for the C conservative regridding function."""
            self._compute_mesh()
            vals = np.ascontiguousarray(data, dtype=np.float32)
            g_lat = self._check_and_convert(glats, True).astype(np.float32)
            g_lon = self._check_and_convert(glons, False).astype(np.float32)

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

            # Renormalize to enforce conservation of the sum
            regridded_sum = np.sum(regridded_data)
            values_sum = np.sum(data)
            if regridded_sum > 1e-9:  # Avoid division by zero
                regridded_data *= values_sum / regridded_sum

            return regridded_data

        regridded_grid = xr.apply_ufunc(
            _regrid_wrapper,
            values,
            grid_lats,
            grid_lons,
            input_core_dims=[[point_dim], ["lat"], ["lon"]],
            output_core_dims=[["lat", "lon"]],
            dask="parallelized",
            output_dtypes=[values.dtype],
        )

        result = regridded_grid.assign_coords({"lat": grid_lats, "lon": grid_lons})

        # --- Provenance ---
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        history_log = (
            f"{timestamp}: Conservatively regridded from unstructured mesh "
            f"(n={self.n}) to a regular grid "
            f"({len(grid_lats)}x{len(grid_lons)}) using renka.SphericalMesh "
            f"with {samples * samples} samples per cell."
        )
        if "history" in values.attrs:
            history_log = f"{values.attrs['history']}\n{history_log}"
        result.attrs["history"] = history_log

        return result
