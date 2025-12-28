import ctypes
import numpy as np
import os
import sys
from ctypes import c_int, c_float, c_double, c_bool, POINTER, byref

# ==============================================================================
# 1. Library Loading
# ==============================================================================
def _load_library():
    lib_name = "librenka.so"
    if os.name == 'nt':
        lib_name = "librenka.dll"
    
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)
    if os.path.exists(local_path):
        return ctypes.CDLL(local_path)
    
    try:
        return ctypes.CDLL(lib_name)
    except OSError:
        raise OSError(f"Could not load {lib_name}. Ensure the C code is compiled and the library is in the path.")

_lib = _load_library()

# ==============================================================================
# 2. Type Definitions
# ==============================================================================
c_double_p = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
c_float_p = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')

# ==============================================================================
# 3. TSPACK: 1D Tension Spline Interpolation
# ==============================================================================
class TsPack:
    """Wrapper for ACM Algorithm 716 (TSPACK)."""
    def __init__(self):
        try:
            _lib.ypc1.argtypes = [c_int, c_double_p, c_double_p, c_double_p, POINTER(c_int)]
            _lib.tsval1.argtypes = [c_int, c_double_p, c_double_p, c_double_p, c_double_p,
                                    c_int, c_int, c_double_p, c_double_p, POINTER(c_int)]
        except AttributeError as e:
            raise RuntimeError("TSPACK symbols missing. Check library compilation.") from e

    def interpolate(self, x, y, tension=0.0):
        """
        x, y: float64 arrays. x must be strictly increasing.
        tension: 0.0 (Cubic) to >20.0 (Linear).
        """
        x_in = np.ascontiguousarray(x, dtype=np.float64)
        y_in = np.ascontiguousarray(y, dtype=np.float64)
        n = len(x_in)
        
        yp = np.zeros(n, dtype=np.float64)
        ier = c_int()
        
        _lib.ypc1(n, x_in, y_in, yp, byref(ier))
        
        if ier.value != 0:
            raise ValueError(f"TSPACK Error {ier.value}: x might not be strictly increasing.")

        sigma = np.full(n, tension, dtype=np.float64)
        
        def predict(t_targets):
            t_targets = np.ascontiguousarray(t_targets, dtype=np.float64)
            ne = len(t_targets)
            results = np.zeros(ne, dtype=np.float64)
            ier_eval = c_int()
            _lib.tsval1(n, x_in, y_in, yp, sigma, 0, ne, t_targets, results, byref(ier_eval))
            return results
            
        return predict

# ==============================================================================
# 4. STRIPACK/SSRFPACK: Spherical Triangulation
# ==============================================================================
class SphericalMesh:
    """
    Wrapper for ACM Algorithm 772.
    Handles spherical triangulation and interpolation.
    """
    
    def __init__(self, lats, lons):
        """
        Construct global mesh.
        Auto-detects Degrees vs Radians based on range.
        """
        self.n = len(lats)
        
        # --- Unit Detection and Conversion ---
        # We process input copies to avoid modifying user data in place unexpectedly
        self.lats = self._check_and_convert(lats, is_lat=True)
        self.lons = self._check_and_convert(lons, is_lat=False)
        
        # --- Safety Clipping ---
        # Ensure Lats are strictly within [-pi/2, pi/2] to prevent NaN in sqrt()
        # This handles float32 precision errors at the poles.
        half_pi = np.pi / 2.0
        np.clip(self.lats, -half_pi, half_pi, out=self.lats)

        if len(self.lons) != self.n:
            raise ValueError("Latitudes and Longitudes must have equal length")

        # --- Cartesian Conversion ---
        self.x = np.zeros(self.n, dtype=np.float32)
        self.y = np.zeros(self.n, dtype=np.float32)
        self.z = np.zeros(self.n, dtype=np.float32)
        
        _lib.trans.argtypes = [c_int, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p]
        _lib.trans(self.n, self.lats, self.lons, self.x, self.y, self.z)
        
        # --- Triangulation ---
        list_len = 6 * self.n - 12
        if list_len < 100: list_len = 100
        
        self.list = np.zeros(list_len, dtype=np.int32)
        self.lptr = np.zeros(list_len, dtype=np.int32)
        self.lend = np.zeros(self.n, dtype=np.int32)
        self.lnew = c_int(0)
        
        near = np.zeros(self.n, dtype=np.int32)
        next_arr = np.zeros(self.n, dtype=np.int32)
        dist = np.zeros(self.n, dtype=np.float32)
        ier = c_int()
        
        _lib.trmesh.argtypes = [c_int, c_float_p, c_float_p, c_float_p, 
                                c_int_p, c_int_p, c_int_p, POINTER(c_int),
                                c_int_p, c_int_p, c_float_p, POINTER(c_int)]
                                
        _lib.trmesh(self.n, self.x, self.y, self.z, 
                    self.list, self.lptr, self.lend, byref(self.lnew),
                    near, next_arr, dist, byref(ier))
                    
        if ier.value < 0:
             raise RuntimeError(f"TRMESH Fatal Error: {ier.value} (Check for collinear points)")
             
        self._bind_functions()

    def _check_and_convert(self, arr, is_lat=False):
        """
        Heuristic: 
        If max(abs(lat)) > 1.6 (approx pi/2), assume degrees.
        If max(abs(lon)) > 6.3 (approx 2*pi), assume degrees.
        """
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        
        # Check range
        max_val = np.nanmax(np.abs(arr))
        
        is_degrees = False
        if is_lat and max_val > 1.6: # Slightly more than pi/2
            is_degrees = True
        elif not is_lat and max_val > 6.3: # Slightly more than 2*pi
            is_degrees = True
            
        if is_degrees:
            # print(f"Auto-detected {'Lat' if is_lat else 'Lon'} in Degrees. Converting to Radians.")
            return np.deg2rad(arr)
        
        return arr

    def _bind_functions(self):
        try:
            _lib.unif.argtypes = [c_int, c_float_p, c_float_p, c_float_p, c_float_p,
                                  c_int_p, c_int_p, c_int_p, c_int, c_float_p,
                                  c_int, c_int, c_int, c_float_p, c_float_p,
                                  c_int, c_float_p, c_float_p, POINTER(c_int)]
            _lib.ssrf_interp_points.argtypes = [c_int, c_float_p, c_float_p, c_float_p, c_float_p,
                                                c_int_p, c_int_p, c_int_p, c_int, c_float_p,
                                                c_int, c_float_p, c_float_p, 
                                                c_int, c_float_p, c_float_p, POINTER(c_int)]
        except AttributeError:
            pass

    def interpolate(self, values, grid_lats, grid_lons):
        """
        Interpolate onto a RECTILINEAR (orthogonal) grid.
        Inputs can be Deg or Rad; auto-detected.
        Returns: 2D Array (Lat_rows, Lon_cols)
        """
        vals = np.ascontiguousarray(values, dtype=np.float32)
        
        # Convert grid axes
        g_lat = self._check_and_convert(grid_lats, is_lat=True)
        g_lon = self._check_and_convert(grid_lons, is_lat=False)
        
        ni = len(g_lat)
        nj = len(g_lon)
        ff = np.zeros(ni * nj, dtype=np.float32)
        
        # Gradient buffer
        grad = np.zeros((3, self.n), dtype=np.float32)
        grad_flat = grad.ravel()
        sigma = np.zeros(1, dtype=np.float32)
        ier = c_int()
        
        # iflgg=0: Compute gradients internally on the fly
        _lib.unif(self.n, self.x, self.y, self.z, vals,
                  self.list, self.lptr, self.lend,
                  0, sigma, 
                  ni, ni, nj, g_lat, g_lon,
                  0, grad_flat, ff, byref(ier))
                  
        if ier.value < 0:
             raise RuntimeError(f"Interpolation Error: {ier.value}")
             
        # UNIF returns (Lat_i, Lon_j), usually mapped to (Row, Col)
        return ff.reshape((nj, ni)).T

    def interpolate_points(self, values, target_lats, target_lons):
        """
        Interpolate onto CURVILINEAR or SCATTERED points.
        Inputs can be Deg or Rad; auto-detected.
        Returns: Array of same shape as target_lats.
        """
        vals = np.ascontiguousarray(values, dtype=np.float32)
        
        # Convert targets
        t_lats = self._check_and_convert(target_lats, is_lat=True)
        t_lons = self._check_and_convert(target_lons, is_lat=False)
        
        if t_lats.shape != t_lons.shape:
            raise ValueError("Target Lat/Lon shapes must match")
            
        original_shape = t_lats.shape
        n_targets = t_lats.size
        flat_lats = t_lats.ravel()
        flat_lons = t_lons.ravel()
        
        out_vals = np.zeros(n_targets, dtype=np.float32)
        grad = np.zeros((3, self.n), dtype=np.float32)
        grad_flat = grad.ravel()
        sigma = np.zeros(1, dtype=np.float32)
        ier = c_int()
        
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
