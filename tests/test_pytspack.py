import numpy as np
from pytspack import TsPack


def test_tspack_interpolation():
    """Test the TsPack class."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])

    tspack = TsPack()
    interpolator = tspack.interpolate(x, y)

    test_points = np.array([0.5, 1.5, 2.5])
    results = interpolator(test_points)

    assert results.shape == (3,)
    assert np.all(np.isfinite(results))
    # Check some values
    assert results[0] > 0
    assert results[1] < 1
