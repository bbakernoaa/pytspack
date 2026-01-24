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

For development, install the package in editable mode:

```bash
pip install -e .
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

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
