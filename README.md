# renka

A high-performance Python wrapper for Robert J. Renka's C libraries for triangulation and interpolation, including:

*   **TSPACK:** Tension Spline Curve Fitting
*   **STRIPACK:** Delaunay Triangulation on a Sphere
*   **SSRFPACK:** Scattered Data Interpolation on a Sphere
*   **SRFPACK:** Scattered Data Interpolation on a Plane
*   **TRIPACK:** Planar Triangulation

This package provides direct access to the C functions, enabling high-performance geospatial and scientific computing in Python.

## Installation

This package requires a C compiler and NumPy to be installed.

### Standard Installation

For most users, a simple `pip` install from the repository root will work:

```bash
pip install .
```

This command compiles the C extension and installs the `renka` package into your active Python environment.

### Editable Mode

For development, install the package in editable mode:

```bash
pip install -e .
```

This allows you to modify the Python wrapper code and have the changes immediately reflected without reinstalling.

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
