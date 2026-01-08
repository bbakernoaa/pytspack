"""
==============================
1D Interpolation with TsPack
==============================

This example demonstrates how to use the `renka.TsPack` class to perform
one-dimensional spline interpolation. This class is a Python wrapper around
the tensioned spline interpolation routines from the TSPACK Fortran library.

The script will:
1.  Generate a set of sparse, noisy data points based on a sine wave.
2.  Use `TsPack` to create an interpolation function from these points.
3.  Evaluate the interpolation function over a dense grid.
4.  Visualize the original data and the smooth, interpolated curve using
    Matplotlib and save it to a file.
"""

import numpy as np
import matplotlib.pyplot as plt
from renka import TsPack

# 1. Generate some sparse, noisy sample data
np.random.seed(0)
x_source = np.linspace(0, 10, 15)
y_source = np.sin(x_source) + (np.random.rand(len(x_source)) - 0.5) * 0.5

# 2. Initialize the TsPack interpolator
# The `interpolate` method takes the source x and y points and returns
# a callable prediction function. You can optionally set a tension parameter.
tspack_interpolator = TsPack()
predict_function = tspack_interpolator.interpolate(x_source, y_source, tension=0.0)

# 3. Define a dense grid of points to evaluate the interpolation
x_target = np.linspace(0, 10, 200)

# The returned function can be called with the target points
y_target = predict_function(x_target)

# 4. Plot the results using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(x_source, y_source, "o", label="Source Points", markersize=8, color="blue")
plt.plot(x_target, y_target, "-", label="Interpolated Curve (Tension=0.0)", color="red")

# Demonstrate the effect of tension
predict_tensioned = tspack_interpolator.interpolate(x_source, y_source, tension=2.5)
y_target_tensioned = predict_tensioned(x_target)
plt.plot(
    x_target,
    y_target_tensioned,
    "--",
    label="Interpolated Curve (Tension=2.5)",
    color="green",
)


plt.title("1D Spline Interpolation with renka.TsPack")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("tspack_interpolation.png")
