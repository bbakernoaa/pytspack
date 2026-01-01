import numpy as np
import xarray as xr
import dask.array as da
import hvplot.xarray  # noqa
import renka

# --- 1. The Logic (Implementation) ---

# Define a sample unstructured dataset
np.random.seed(0)
n_points = 500
lats = np.random.uniform(20, 50, n_points)
lons = np.random.uniform(-130, -70, n_points)
# Add a pattern to the data for visualization
values = 10 * np.sin(lats / 10) + np.cos(lons / 10)

# Create a Dask-backed xarray.DataArray
source_da = xr.DataArray(
    da.from_array(values, chunks=(n_points // 2,)),
    coords={"lat": (("points",), lats), "lon": (("points",), lons)},
    dims=["points"],
    name="air_temperature",
).chunk({"points": -1})

# Initialize the spherical mesh
mesh = renka.SphericalMesh(source_da["lat"].values, source_da["lon"].values)

# Define the target grid
grid_lats = np.arange(20, 50, 0.5)
grid_lons = np.arange(-130, -70, 0.5)
target_ds = xr.Dataset(coords={"lat": grid_lats, "lon": grid_lons})

# Perform lazy interpolation
interpolated_da = mesh.interpolate(source_da, target_ds)

# Update provenance
interpolated_da.attrs["history"] = f"Interpolated from unstructured data using renka."

# --- 2. The Proof (Validation) ---

# The proof is provided by the pytest suite in `tests/test_renka.py`.
# To run the tests locally:
# python -m pip install -e .
# python -m pip install pytest xarray dask
# python -m pytest

print("--- DataArray after interpolation ---")
print(interpolated_da)
print("\n--- Computation has been lazy. Now computing... ---")
computed_data = interpolated_da.compute()
print("\n--- Computed Data ---")
print(computed_data)


# --- 3. The UI (Visualization) ---

# Interactive visualization with hvplot
# The `rasterize=True` option is essential for large grids.
plot = interpolated_da.hvplot.quadmesh(
    "lon",
    "lat",
    geo=True,
    tiles="CartoLight",
    cmap="viridis",
    rasterize=True,
    title="Interpolated Air Temperature",
)

# To display the plot, you would typically run this in a notebook
# or save it to a file.
if __name__ == "__main__":
    # The hvplot save function requires geoviews and selenium
    try:
        import holoviews as hv

        hv.extension("bokeh")
        hvplot.save(plot, "interpolated_map.html")
        print("\n--- Visualization saved to interpolated_map.html ---")
    except ImportError:
        print(
            "\n--- Please install holoviews, geoviews, and selenium to save the plot ---"
        )
