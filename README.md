# pytspack

A high-performance Python wrapper for Robert J. Renka's C library for tension spline curve fitting (TSPACK).

## Installation

This package requires a C compiler and NumPy to be installed.

### Standard Installation

For most users, a simple `pip` install from the repository root will work:

```bash
pip install .
```

This command compiles the C extension and installs the `pytspack` package into your active Python environment.

### Editable Mode

For development, install the package in editable mode with test dependencies:

```bash
pip install -e .[test]
```

To run the examples, you might need `matplotlib`:

```bash
pip install .[examples]
```

## Usage

```python
import numpy as np
from pytspack import TsPack

x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([0.0, 1.0, 0.0, 1.0])

tspack = TsPack()
interpolator = tspack.interpolate(x, y, tension=2.0)

test_points = np.linspace(0, 3, 100)
results = interpolator(test_points)
```

### Vertical Interpolation (xarray)

```python
import xarray as xr
from pytspack import interpolate_vertical

# Create a sample dataset with a 'level' dimension
# ...
new_ds = interpolate_vertical(ds, target_levels=[1000, 925, 850, 700], level_dim="level")
```

## API Reference

### `TsPack`

The main class for 1D tension spline interpolation.

- `interpolate(x, y, tension=0.0)`: Returns a callable that evaluates the spline.
    - `x`: 1D array of strictly increasing coordinates.
    - `y`: 1D array of values.
    - `tension`: Scalar or 1D array of tension factors.

### `interpolate_vertical`

A high-level wrapper for interpolating xarray objects.

- `interpolate_vertical(data, target_levels, level_dim="level", tension=0.0)`:
    - `data`: `xarray.DataArray` or `xarray.Dataset`.
    - `target_levels`: 1D array of target coordinates for `level_dim`.
    - `level_dim`: Name of the vertical dimension.
    - `tension`: Tension factor.

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
