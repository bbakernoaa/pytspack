import ctypes
import numpy as np
import os
from ctypes import c_int, c_float, c_double, c_bool, POINTER, byref

# 1. Load the Shared Library
# --------------------------
# Try to find the library in the current directory
lib_path = os.path.abspath("librenka.so")
if os.name == 'nt':
    lib_path = os.path.abspath("librenka.dll")

try:
    _lib = ctypes.CDLL(lib_path)
except OSError:
    raise OSError(f"Could not load library at {lib_path}. Did you compile the C code?")

# 2. Type Definitions
# -------------------
# TSPACK uses doubles (float64)
c_double_p = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
# STRIPACK/SSRFPACK use floats (float32) to match original Fortran REAL
c_float_p = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')

# 3. TSPACK Wrapper (1D Tension Splines)
# --------------------------------------
class TsPack:
    """Wrapper for 1D Interpolation/Smoothing (ACM 716)"""
    
    def __init__(self):
        # Setup argument types for C functions
        _lib.arcl2d.argtypes = [c_int, c_double_p, c_double_p, c_double_p, POINTER(c_int)]
        _lib.ypc1.argtypes = [c_int, c_double_p, c_double_p, c_double_p, POINTER(c_int)]
        _lib.tsval1.argtypes = [c_int, c_double_p, c_double_p, c_double_p, c_double_p,
                                c_int, c_int, c_double_p, c_double_p, POINTER(c_int)]
        _lib.sigs.argtypes = [c_int, c_double_p, c_double_p, c_double_p, c_double,
                              c_double_p, POINTER(c_double), POINTER(c_int)]

    def interpolate(self, x, y, tension=0.0):
        """
        Perform 1D Hermite Interpolation.
        x, y: Input arrays (float64)
        tension: Tension factor (0.0 = Cubic Spline, >20.0 ~= Linear)
        Returns: A function `predict(t)`
        """
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = len(x)
        
        # 1. Compute Derivatives (YP) locally
        yp = np.zeros(n, dtype=np.float64)
        ier = c_int()
        _lib.ypc1(n, x, y, yp, byref(ier))
        if ier.value != 0:
            raise RuntimeError(f"YPC1 Error: {ier.value}")

        # 2. Setup Tension (Sigma)
        # If tension is scalar, broadcast it.
        # Original library allows optimizing sigma per interval, 
        # but uniform tension is standard for simple usage.
        sigma = np.full(n, tension, dtype=np.float64)
        
        # Return a closure for evaluation
        def predict(te):
            te = np.ascontiguousarray(te, dtype=np.float64)
            ne = len(te)
            vals = np.zeros(ne, dtype=np.float64)
            ier_eval = c_int()
            
            # iflag=0 for values, 1 for 1st deriv, 2 for 2nd deriv
            _lib.tsval1(n, x, y, yp, sigma, 0, ne, te, vals, byref(ier_eval))
            return vals
            
        return predict

# 4. STRIPACK/SSRFPACK Wrapper (Spherical)
# ----------------------------------------
class SphericalMesh:
    """Wrapper for Spherical Triangulation (ACM 772)"""
    
    def __init__(self, lats, lons):
        """
        Input: Latitudes and Longitudes in Radians.
        Points must be distinct and N >= 3.
        """
        self.n = len(lats)
        self.lats = np.ascontiguousarray(lats, dtype=np.float32)
        self.lons = np.ascontiguousarray(lons, dtype=np.float32)
        
        # Allocate Cartesian coordinates
        self.x = np.zeros(self.n, dtype=np.float32)
        self.y = np.zeros(self.n, dtype=np.float32)
        self.z = np.zeros(self.n, dtype=np.float32)
        
        # Convert to Cartesian
        _lib.trans.argtypes = [c_int, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p]
        _lib.trans(self.n, self.lats, self.lons, self.x, self.y, self.z)
        
        # Adjacency Lists (The Mesh)
        # Renka's lists size requirement: 6N-12
        list_len = 6 * self.n - 12
        if list_len < 100: list_len = 100 # Safety buffer for small N
        
        self.list = np.zeros(list_len, dtype=np.int32)
        self.lptr = np.zeros(list_len, dtype=np.int32)
        self.lend = np.zeros(self.n, dtype=np.int32)
        self.lnew = c_int(0)
        
        # Workspace for construction
        near = np.zeros(self.n, dtype=np.int32)
        next_arr = np.zeros(self.n, dtype=np.int32)
        dist = np.zeros(self.n, dtype=np.float32)
        ier = c_int()
        
        # Build Mesh
        _lib.trmesh.argtypes = [c_int, c_float_p, c_float_p, c_float_p, 
                                c_int_p, c_int_p, c_int_p, POINTER(c_int),
                                c_int_p, c_int_p, c_float_p, POINTER(c_int)]
                                
        _lib.trmesh(self.n, self.x, self.y, self.z, 
                    self.list, self.lptr, self.lend, byref(self.lnew),
                    near, next_arr, dist, byref(ier))
                    
        if ier.value < 0:
            raise RuntimeError(f"TRMESH Error: {ier.value} (Collinear nodes or N<3)")

    def interpolate(self, values, grid_lats, grid_lons, method='C1'):
        """
        Interpolate 'values' (defined at mesh nodes) onto a grid.
        method: 'C1' (Smooth) or 'Linear' (Barycentric)
        """
        vals = np.ascontiguousarray(values, dtype=np.float32)
        if len(vals) != self.n:
            raise ValueError("Values length must match number of mesh nodes")
            
        ni = len(grid_lats)
        nj = len(grid_lons)
        g_lat = np.ascontiguousarray(grid_lats, dtype=np.float32)
        g_lon = np.ascontiguousarray(grid_lons, dtype=np.float32)
        
        # Output Grid
        ff = np.zeros(ni * nj, dtype=np.float32)
        
        # Gradients (Required for C1, optional for C0 but good to have allocation)
        grad = np.zeros((3, self.n), dtype=np.float32) # Stored column-major in C usually, but flatten handles it
        grad_flat = grad.ravel() 
        
        ier = c_int()
        
        if method == 'C1':
            # 1. Compute Gradients at nodes (Global method usually preferred, simplified to local here)
            # We will use iflgg=0 in UNIF which computes gradients internally on the fly
            # or iflgg=2 to precompute. Let's use internal (0) for simplicity.
            iflgg = 0 
            
            # 2. Interpolate
            _lib.unif.argtypes = [c_int, c_float_p, c_float_p, c_float_p, c_float_p,
                                  c_int_p, c_int_p, c_int_p,
                                  c_int, c_float_p,
                                  c_int, c_int, c_int, c_float_p, c_float_p,
                                  c_int, c_float_p, c_float_p, POINTER(c_int)]
            
            # Dummy sigma (uniform tension = 0 for cubic)
            sigma = np.zeros(1, dtype=np.float32)
            iflgs = 0 # Uniform tension
            
            _lib.unif(self.n, self.x, self.y, self.z, vals,
                      self.list, self.lptr, self.lend,
                      iflgs, sigma,
                      ni, ni, nj, g_lat, g_lon,
                      iflgg, grad_flat, ff, byref(ier))
                      
        if ier.value < 0:
             raise RuntimeError(f"Interpolation Error: {ier.value}")
             
        return ff.reshape((nj, ni)).T # Reshape to (Lat, Lon)
