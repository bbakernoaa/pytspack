import argparse
import datetime
import numpy as np
import xarray as xr
import s3fs
from renka import SphericalMesh
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def fetch_gfs_data(date):
    """
    Downloads GFS data for a specific date and extracts wind components.
    """
    gfs_path = f"s3://noaa-gfs-bdp-pds/gfs.{date.strftime('%Y%m%d')}/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    fs = s3fs.S3FileSystem(anon=True)

    with fs.open(gfs_path) as f:
        ds = xr.open_dataset(
            f,
            engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10}
            },
        )
    return ds


def run_simulation(ds, num_particles=100, time_steps=100):
    """
    Runs the particle trajectory simulation.
    """
    # Extract data from the xarray dataset
    lats = ds.latitude.values.flatten()
    lons = ds.longitude.values.flatten()
    u_wind = ds.u10.values.flatten()
    v_wind = ds.v10.values.flatten()

    # Initialize spherical mesh for wind components
    mesh = SphericalMesh(lats, lons)

    # Initialize particle positions (randomly)
    particle_lats = np.random.uniform(-90, 90, num_particles)
    particle_lons = np.random.uniform(-180, 180, num_particles)

    # Store trajectory history
    lat_history = [particle_lats.copy()]
    lon_history = [particle_lons.copy()]

    # Advection loop
    dt = 3600  # Time step in seconds (1 hour)
    R = 6371e3  # Earth radius in meters

    for _ in range(time_steps):
        # Interpolate wind components at particle locations
        u_interp = mesh.interpolate_points(u_wind, particle_lats, particle_lons)
        v_interp = mesh.interpolate_points(v_wind, particle_lats, particle_lons)

        # Update particle positions
        dlat = (v_interp * dt) / R
        dlon = (u_interp * dt) / (R * np.cos(np.deg2rad(particle_lats)))

        particle_lats += np.rad2deg(dlat)
        particle_lons += np.rad2deg(dlon)

        # Handle particles crossing poles or date line
        particle_lats = np.clip(particle_lats, -90, 90)
        particle_lons = (particle_lons + 180) % 360 - 180

        lat_history.append(particle_lats.copy())
        lon_history.append(particle_lons.copy())

    return np.array(lat_history), np.array(lon_history)


def plot_trajectories(
    lat_history, lon_history, output_file="particle_trajectories.png"
):
    """
    Plots the particle trajectories on a map.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()

    for i in range(lon_history.shape[1]):
        ax.plot(lon_history[:, i], lat_history[:, i], transform=ccrs.Geodetic())

    plt.savefig(output_file)
    print(f"Trajectory plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a particle trajectory simulation using GFS data."
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2024-07-26",
        help="The date for the GFS data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=100,
        help="The number of particles to simulate.",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="The number of time steps to simulate."
    )
    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, "%Y-%m-%d")
    ds = fetch_gfs_data(date)
    lat_history, lon_history = run_simulation(
        ds, num_particles=args.particles, time_steps=args.steps
    )
    plot_trajectories(lat_history, lon_history)
