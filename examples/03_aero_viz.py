"""
=========================================
Aero Protocol: Two-Track Visualization
=========================================

This example demonstrates the Aero Protocol's visualization requirements:
1. Track A (Publication): Static plots using Matplotlib and Cartopy.
2. Track B (Exploration): Interactive plots using HvPlot.

It also showcases backend-agnostic interpolation on a geospatial grid.
"""

import numpy as np
import xarray as xr
from pytspack import interpolate_vertical

# 1. Setup Synthetic Geospatial Data
lats = np.linspace(-90, 90, 180)
lons = np.linspace(-180, 180, 360)
src_levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0])

# T(p, lat) = T0 + dT * log10(p) * cos(lat)
data = np.zeros((len(src_levels), len(lats), len(lons)))
for i, p in enumerate(src_levels):
    data[i, :, :] = 250 + 50 * np.log10(p) * np.cos(np.deg2rad(lats[:, None]))

da = xr.DataArray(
    data,
    coords={"level": src_levels, "lat": lats, "lon": lons},
    dims=["level", "lat", "lon"],
    name="temperature",
    attrs={"units": "K", "long_name": "Air Temperature"},
)

# Interpolate to a specific pressure level (e.g., 600 hPa)
target_level = [600.0]
da_600 = interpolate_vertical(da, target_level).sel(level=600.0)

# ---------------------------------------------------------
# TRACK A: Publication Quality (Matplotlib + Cartopy)
# ---------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    print("Generating Track A (Static Publication Plot)...")
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Mandatory: transform=ccrs.PlateCarree() for data coordinates
    im = da_600.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        add_colorbar=True,
        cbar_kwargs={"label": "Temperature (K)"},
    )

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.title("Temperature at 600hPa (Tension Spline Interpolation)")

    plt.savefig("aero_publication_plot.png", bbox_inches="tight", dpi=150)
    print("Saved Track A plot to aero_publication_plot.png")

except ImportError:
    print("Matplotlib or Cartopy not installed. Skipping Track A.")

# ---------------------------------------------------------
# TRACK B: Exploration (HvPlot / GeoViews)
# ---------------------------------------------------------
try:
    import hvplot.xarray  # noqa: F401
    import holoviews as hv

    print("Generating Track B (Interactive Exploration Plot)...")
    # Mandatory: rasterize=True for large grids
    interactive_plot = da_600.hvplot.quadmesh(
        x="lon",
        y="lat",
        projection=ccrs.PlateCarree(),
        cmap="RdBu_r",
        rasterize=True,
        title="Interactive Temperature at 600hPa",
    )

    # In a real environment, we would use hv.save or show
    # For this example, we'll just confirm it's created.
    hv.save(interactive_plot, "aero_exploration_plot.html")
    print("Saved Track B plot to aero_exploration_plot.html")

except (ImportError, Exception) as e:
    print(f"HvPlot/GeoViews not installed or failed. Skipping Track B. Error: {e}")

print("\nAero Protocol Visualization Complete.")
