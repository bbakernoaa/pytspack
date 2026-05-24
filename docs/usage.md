# Usage Guide

This guide provides examples of how to use `pytspack` for various interpolation tasks.

## 1D Interpolation

The `TsPack` class provides the core 1D interpolation functionality.

```python
import numpy as np
from pytspack import TsPack

# Define data points
x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([0.0, 1.0, 0.0, 1.0])

# Create interpolator
tsp = TsPack()
interpolator = tsp.interpolate(x, y, tension=2.0)

# Evaluate at new points
test_points = np.linspace(0, 3, 50)
values = interpolator(test_points)
```

## Vertical Interpolation with xarray

For multi-dimensional data, `interpolate_vertical` offers a convenient wrapper for xarray objects.

```python
import xarray as xr
from pytspack import interpolate_vertical

# Load or create a DataArray
# da.dims = ('time', 'level', 'lat', 'lon')

# Interpolate to new levels
target_levels = [1000.0, 925.0, 850.0, 700.0, 500.0]
new_da = interpolate_vertical(da, target_levels, level_dim="level", tension=1.5)
```

## Tension Factor

The tension factor $\sigma$ controls the smoothness of the curve:
- $\sigma = 0$: Standard cubic spline.
- Large $\sigma$: Approaches linear interpolation between points.
- $\sigma$ can be a scalar (constant tension) or an array (varying tension between segments).
