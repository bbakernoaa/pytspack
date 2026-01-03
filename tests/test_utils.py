import numpy as np
from renka.utils import jitter_coordinates


def test_jitter_coordinates():
    """
    Tests that the jitter_coordinates function adds a small amount of noise
    and that the jittered coordinates are not the same as the originals.
    """
    lats = np.array([0.0, 10.0, 20.0])
    lons = np.array([0.0, 5.0, 15.0])

    amount = 1e-6

    jittered_lats, jittered_lons = jitter_coordinates(lats, lons, amount=amount)

    # Check that the shapes are the same
    assert lats.shape == jittered_lats.shape
    assert lons.shape == jittered_lons.shape

    # Check that the values are not identical (probabilistically very unlikely)
    assert not np.array_equal(lats, jittered_lats)
    assert not np.array_equal(lons, jittered_lons)

    # Check that the jitter is within the expected range
    lat_diff = np.abs(lats - jittered_lats)
    lon_diff = np.abs(lons - jittered_lons)

    assert np.all(lat_diff < amount / 2)
    assert np.all(lon_diff < amount / 2)
