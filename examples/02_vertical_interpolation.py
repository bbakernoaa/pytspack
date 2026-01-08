"""
======================================
Vertical Interpolation of Atmospheric Data
======================================

This example demonstrates how to use the `renka.interpolate_vertical`
function to interpolate atmospheric data to different vertical levels.
This function is Dask-aware, allowing for lazy, out-of-core computations.

The script will:
1.  Create a sample `xarray.DataArray` representing temperature on multiple
    pressure levels.
2.  Define a set of target pressure levels.
3.  Perform log-pressure interpolation to the target levels.
4.  Define a set of target height levels.
5.  Perform linear-height interpolation to the target levels.
6.  Print the results to show the new vertical coordinates.
"""

import numpy as np
import xarray as xr
from renka import interpolate_vertical

# 1. Create a sample xarray.DataArray with pressure levels
source_pressure_levels = np.array([1000, 950, 900, 850, 800, 750, 700])
n_lats = 5
n_lons = 10
temperature_data = np.random.uniform(
    250, 300, size=(len(source_pressure_levels), n_lats, n_lons)
)

temp_da = xr.DataArray(
    temperature_data,
    dims=["pressure", "lat", "lon"],
    coords={
        "pressure": source_pressure_levels,
        "lat": np.linspace(-90, 90, n_lats),
        "lon": np.linspace(-180, 180, n_lons),
    },
    attrs={
        "long_name": "Temperature",
        "units": "K",
    },
)
print("Original DataArray:")
print(temp_da)
print("\n" + "=" * 50 + "\n")

# 2. Perform log-pressure interpolation
target_pressure_levels = np.array([975, 925, 875, 825, 725])
print(f"Target pressure levels (hPa): {target_pressure_levels}\n")

interpolated_log_p = interpolate_vertical(
    temp_da, target_levels=target_pressure_levels, level_dim="pressure", method="log"
)

print("Result of Log-Pressure Interpolation:")
print(interpolated_log_p)
print("\n" + "=" * 50 + "\n")


# 3. Perform linear interpolation on a height coordinate
# Create a corresponding height coordinate for the sample data
source_height_levels = (1 - (source_pressure_levels / 1013.25)) * 44330
temp_da_height = temp_da.assign_coords(
    pressure=("pressure", source_height_levels)
).rename({"pressure": "height"})
temp_da_height.coords["height"].attrs["units"] = "m"

print("DataArray with Height Coordinate:")
print(temp_da_height)
print("\n" + "=" * 50 + "\n")

target_height_levels = np.array([500, 1000, 1500, 2000, 2500])
print(f"Target height levels (m): {target_height_levels}\n")

interpolated_linear_h = interpolate_vertical(
    temp_da_height,
    target_levels=target_height_levels,
    level_dim="height",
    method="linear",
)

print("Result of Linear-Height Interpolation:")
print(interpolated_linear_h)
