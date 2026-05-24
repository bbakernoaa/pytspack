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


def test_tspack_scalar_input():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    tspack = TsPack()
    interpolator = tspack.interpolate(x, y)

    res = interpolator(0.5)
    assert isinstance(res, (float, np.float64))
    assert res > 0


def test_tspack_errors():
    tspack = TsPack()

    # Non-increasing x
    x = np.array([0.0, 2.0, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    try:
        tspack.interpolate(x, y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "strictly increasing" in str(e)

    # Short arrays
    x = np.array([0.0])
    y = np.array([0.0])
    try:
        tspack.interpolate(x, y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least 2 points" in str(e)

    # 2D arrays
    x = np.random.rand(2, 2)
    y = np.random.rand(2, 2)
    try:
        tspack.interpolate(x, y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "1D arrays" in str(e)
