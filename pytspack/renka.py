import ctypes
import numpy as np
import os
import sys
from ctypes import c_int, c_float, c_double, c_bool, POINTER, byref

# ==============================================================================
# 1. Library Loading
# ==============================================================================
# Helper to locate the compiled shared library.
# Ensure 'librenka.so' (Linux/Mac) or 'librenka.dll' (Windows) is in the 
# same directory or system path.

def _load_library():
    lib_name = "librenka.so"
    if os.name == 'nt':
        lib_name = "librenka.dll"
    
    # Check current directory first
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)
    if os.path.exists(local_path):
        return ctypes.CDLL(local_path)
    
    # Fallback to system load
    try:
        return ctypes.CDLL(lib_name)
    except OSError:
        raise OSError(f"Could not load {lib_name}. Ensure the C code is compiled and the library is in the path.")

_lib = _load_library()

# ==============================================================================
# 2. Type Definitions for C-API
# ==============================================================================
# TSPACK uses doubles (float64)
c_double_p = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
# STRIPACK/SSRFPACK use floats (float32)
c_float_p = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')

# ==============================================================================
# 3. TSPACK: 1D Tension Spline Interpolation
# ==============================================================================
class TsPack:
    """
    Wrapper for ACM Algorithm 716 (TSPACK).
    Provides Shape-Preserving Interpolation for 1D curves.
    """
    def __init__(self):
        # Bind C functions
        try:
            _lib.ypc1.argtypes = [c_int, c_double_p, c_double_p, c_double_p, POINTER(c_int)]
            _lib.tsval1.argtypes = [c_int, c_double_p, c_double_p, c_double_p, c_double_p,
                                    c_int, c_int, c_double_p, c_double_p, POINTER(c_int)]
        except AttributeError as e:
            raise RuntimeError("TSPACK symbols missing in library. Check compilation.") from e

    def interpolate(self, x, y, tension=0.0):
        """
        Create an interpolator function for the data (x, y).
        
        Parameters:
        -----------
        x : array-like (float64), strictly increasing
        y : array-like (float64)
        tension : float (default 0.0)
                  0.0  = Cubic Spline (Loose)
                  >20.0 = Linear Interpolation (Tight)
                  
        Returns:
        --------
        A callable function `predict(t)` that takes an array of new coordinates
        and returns interpolated values.
        """
        x_in = np.ascontiguousarray(x, dtype=np.float64)
        y_in = np.ascontiguousarray(y, dtype=np.float64)
        n = len(x_in)
        
        if len(y_in) != n:
            raise ValueError("x and y must have same length")

        # 1. Compute Derivatives (YP) using local quadratic fit (YPC1)
        yp = np.zeros(n, dtype=np.float64)
        ier = c_int()
        
        _lib.ypc1(n, x_in, y_in, yp, byref(ier))
        
        if ier.value > 0:
            raise ValueError(f"TSPACK Error: x not strictly increasing at index {ier.value}")
        if ier.value < 0:
            raise RuntimeError(f"TSPACK Internal Error: {ier.value}")

        # 2. Setup Tension (Sigma)
        # We use uniform tension for simplicity here.
        sigma = np.full(n, tension, dtype=np.float64)
        
        # 3. Create Closure
        def predict(t_targets):
            t_targets = np.ascontiguousarray(t_targets, dtype=np.float64)
            ne = len(t_targets)
            results = np.zeros(ne, dtype=np.float64)
            ier_eval = c_int()
            
            # iflag=0: Function values
            # iflag=1: First derivatives
            _lib.tsval1(n, x_in, y_in, yp, sigma, 0, ne, t_targets, results, byref(ier_eval))
            
            return results
            
        return predict


# ==============================================================================
# 4. STRIPACK/SSRFPACK: Spherical Triangulation & Interpolation
# ==============================================================================
class SphericalMesh:
    """
    Wrapper for ACM Algorithm 772 (STRIPACK/SSRFPACK).
    Handles construction of Delaunay triangulations on a sphere and 
    C1 interpolation (smooth) or C0 interpolation (linear).
    """
    
    def __init__(self, lats, lons):
        """
        Construct the global mesh from scattered observations.
        
        Parameters:
        -----------
        lats : array-like (radians), Latitude
        lons : array-like (radians), Longitude
        
        Note: Data must fit in memory.
        """
        self.n = len(lats)
        # Ensure contiguous float32 arrays (Fortran REAL compatible)
        self.lats = np.ascontiguousarray(lats, dtype=np.float32)
        self.lons = np.ascontiguousarray(lons, dtype=np.float32)
        
        if len(self.lons) != self.n:
            raise ValueError("Latitudes and Longitudes must have equal length")

        # 1. Coordinate Conversion (Spherical -> Cartesian Unit Sphere)
        self.x = np.zeros(self.n, dtype=np.float32)
        self.y = np.zeros(self.n, dtype=np.float32)
        self.z = np.zeros(self.n, dtype=np.float32)
        
        _lib.trans.argtypes = [c_int, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p]
        _lib.trans(self.n, self.lats, self.lons, self.x, self.y, self.z)
        
        # 2. Allocate Adjacency Lists (The Mesh Topology)
        # Renka specifies length 6N-12 for a full triangulation
        list_len = 6 * self.n - 12
        if list_len < 100: list_len = 100 # Safety buffer
        
        self.list = np.zeros(list_len, dtype=np.int32)
        self.lptr = np.zeros(list_len, dtype=np.int32)
        self.lend = np.zeros(self.n, dtype=np.int32)
        self.lnew = c_int(0)
        
        # Workspace arrays for TRMESH
        near = np.zeros(self.n, dtype=np.int32)
        next_arr = np.zeros(self.n, dtype=np.int32)
        dist = np.zeros(self.n, dtype=np.float32)
        ier = c_int()
        
        # 3. Build Triangulation
        _lib.trmesh.argtypes = [c_int, c_float_p, c_float_p, c_float_p, 
                                c_int_p, c_int_p, c_int_p, POINTER(c_int),
                                c_int_p, c_int_p, c_float_p, POINTER(c_int)]
                                
        _lib.trmesh(self.n, self.x, self.y, self.z, 
                    self.list, self.lptr, self.lend, byref(self.lnew),
                    near, next_arr, dist, byref(ier))
                    
        if ier.value == -2:
            raise RuntimeError("TRMESH Error: First three nodes are collinear.")
        if ier.value > 0:
            # Not necessarily fatal, implies duplicate nodes found
            # print(f"Warning: Duplicate nodes detected (Index {ier.value})")
            pass
        if ier.value < 0:
             raise RuntimeError(f"TRMESH Fatal Error: {ier.value}")
             
        # Bind interpolation functions once
        self._bind_functions()

    def _bind_functions(self):
        """Bind C-types for interpolation routines."""
        # Rectilinear Grid (UNIF)
        try:
            _lib.unif.argtypes = [
                c_int, c_float_p, c_float_p, c_float_p, c_float_p,
                c_int_p, c_int_p, c_int_p,
                c_int, c_float_p,
                c_int, c_int, c_int, c_float_p, c_float_p,
                c_int, c_float_p, c_float_p, POINTER(c_int)
            ]
        except AttributeError:
            pass # unif not available?

        # Curvilinear/Scattered (SSRF_INTERP_POINTS)
        try:
            _lib.ssrf_interp_points.argtypes = [
                c_int, c_float_p, c_float_p, c_float_p, c_float_p,
                c_int_p, c_int_p, c_int_p,
                c_int, c_float_p,
                c_int, c_float_p, c_float_p, # n_targets, lats, lons
                c_int, c_float_p, c_float_p, POINTER(c_int)
            ]
        except AttributeError:
            print("Warning: 'ssrf_interp_points' not found in library. Curvilinear interpolation will fail.")

    def interpolate(self, values, grid_lats, grid_lons, method='C1'):
        """
        Interpolate onto a RECTILINEAR (orthogonal) grid.
        Use this if grid_lats and grid_lons are 1D vectors defining axes.
        
        Returns: 2D Array (Lon, Lat)
        """
        vals = np.ascontiguousarray(values, dtype=np.float32)
        ni = len(grid_lats)
        nj = len(grid_lons)
        
        g_lat = np.ascontiguousarray(grid_lats, dtype=np.float32)
        g_lon = np.ascontiguousarray(grid_lons, dtype=np.float32)
        
        # Output buffer (Flattened)
        ff = np.zeros(ni * nj, dtype=np.float32)
        
        # Gradient buffer
        grad = np.zeros((3, self.n), dtype=np.float32)
        grad_flat = grad.ravel()
        
        sigma = np.zeros(1, dtype=np.float32) # Uniform zero tension
        ier = c_int()
        
        # iflgg=0: Compute gradients internally on the fly
        # iflgg=1: User supplied gradients (we don't have them)
        # iflgg=2: Compute gradients globally once and save (Faster for reuse)
        iflgg = 0 
        
        _lib.unif(self.n, self.x, self.y, self.z, vals,
                  self.list, self.lptr, self.lend,
                  0, sigma, # iflgs=0 (uniform)
                  ni, ni, nj, g_lat, g_lon,
                  iflgg, grad_flat, ff, byref(ier))
                  
        if ier.value < 0:
             raise RuntimeError(f"Interpolation Error: {ier.value}")
             
        # Renka's UNIF returns array indexed by (Lat_i, Lon_j)
        # We usually want (Lat, Lon) for plotting
        return ff.reshape((nj, ni)).T

    def interpolate_points(self, values, target_lats, target_lons, method='C1'):
        """
        Interpolate onto CURVILINEAR or SCATTERED points.
        Use this if target_lats/lons are 2D arrays (swath data) or 1D scattered points.
        
        Parameters:
        -----------
        values: Data values at mesh nodes (1D array)
        target_lats: N-D array of target latitudes
        target_lons: N-D array of target longitudes (must match shape of target_lats)
        
        Returns:
        --------
        Array of interpolated values with same shape as target_lats.
        """
        vals = np.ascontiguousarray(values, dtype=np.float32)
        
        # Ensure inputs are float32 and contiguous
        t_lats = np.ascontiguousarray(target_lats, dtype=np.float32)
        t_lons = np.ascontiguousarray(target_lons, dtype=np.float32)
        
        if t_lats.shape != t_lons.shape:
            raise ValueError("Target Latitude and Longitude shapes must match")
            
        original_shape = t_lats.shape
        
        # Flatten for C-API
        n_targets = t_lats.size
        flat_lats = t_lats.ravel()
        flat_lons = t_lons.ravel()
        
        out_vals = np.zeros(n_targets, dtype=np.float32)
        
        # Gradient buffer
        grad = np.zeros((3, self.n), dtype=np.float32)
        grad_flat = grad.ravel()
        
        sigma = np.zeros(1, dtype=np.float32)
        ier = c_int()
        
        # Call the scatter interpolator
        # iflgs=0 (Uniform Tension), iflgg=0 (Compute Grad Internal)
        _lib.ssrf_interp_points(
            self.n, self.x, self.y, self.z, vals,
            self.list, self.lptr, self.lend,
            0, sigma, 
            n_targets, flat_lats, flat_lons,
            0, grad_flat, 
            out_vals, byref(ier)
        )
        
        if ier.value < 0:
             raise RuntimeError(f"Interpolation Error Code: {ier.value}")

        return out_vals.reshape(original_shape)
