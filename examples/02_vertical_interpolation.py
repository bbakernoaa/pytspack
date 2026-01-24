"""
====================================
Vertical Interpolation with pytspack
====================================

This example demonstrates how to use the `pytspack.interpolate_vertical`
function to regrid a multi-dimensional xarray.DataArray along a vertical
dimension.

We will:
1. Create a synthetic 4D dataset (time, level, lat, lon).
2. Use tension spline interpolation to regrid from a sparse set of
   pressure levels to a denser set.
3. Compare the results with different tension values.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pytspack import interpolate_vertical

# 1. Create a synthetic 4D dataset
# Coordinates
times = [np.datetime64("2024-01-01")]
lats = np.linspace(-90, 90, 10)
lons = np.linspace(-180, 180, 20)
# Standard atmospheric pressure levels (hPa)
src_levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0])

# Generate a temperature-like profile: T decreases with height (lower pressure)
# T ~ log(P)
data = np.zeros((1, len(src_levels), len(lats), len(lons)))
for i, level in enumerate(src_levels):
    # Base temperature with some latitudinal variation
    # Adding a small "bump" at 500hPa to see how splines handle it
    bump = 5.0 if level == 500.0 else 0.0
    data[0, i, :, :] = (
        250 + 50 * np.log10(level) + 10 * np.cos(np.deg2rad(lats[:, None])) + bump
    )

da = xr.DataArray(
    data,
    coords={"time": times, "level": src_levels, "lat": lats, "lon": lons},
    dims=["time", "level", "lat", "lon"],
    name="temperature",
    attrs={"units": "K"},
)

# 2. Define target levels
target_levels = np.linspace(1000, 100, 50)

# 3. Perform vertical interpolation
# No tension (standard cubic spline)
da_interp_0 = interpolate_vertical(da, target_levels, tension=0.0)

# High tension (approaches linear interpolation)
da_interp_5 = interpolate_vertical(da, target_levels, tension=5.0)

# 4. Visualize a single profile at the equator
lat_idx, lon_idx = 5, 10
profile_src = da.isel(time=0, lat=lat_idx, lon=lon_idx)
profile_0 = da_interp_0.isel(time=0, lat=lat_idx, lon=lon_idx)
profile_5 = da_interp_5.isel(time=0, lat=lat_idx, lon=lon_idx)

plt.figure(figsize=(8, 10))
plt.plot(profile_src, profile_src.level, "ko", label="Source Points (Standard Levels)")
plt.plot(profile_0, profile_0.level, "r-", label="Tension = 0.0 (Cubic Spline)")
plt.plot(profile_5, profile_5.level, "b--", label="Tension = 5.0 (Higher Tension)")

plt.gca().invert_yaxis()  # Pressure decreases with height
plt.title(
    f"Vertical Temperature Profile Interpolation\n"
    f"(Lat: {lats[lat_idx]:.1f}, Lon: {lons[lon_idx]:.1f})"
)
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure Level (hPa)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("vertical_interpolation_comparison.png")
print("Saved comparison plot to vertical_interpolation_comparison.png")
