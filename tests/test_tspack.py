import numpy as np
from renka import TsPack


def test_tspack_interpolation():
    """
    Tests the TsPack.interpolate method.
    This test verifies that the interpolation function produces a predictable
    result for a simple linear dataset.
    """
    # Define sample data points
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])

    # Initialize the tspack interpolator
    tspack = TsPack()
    predict = tspack.interpolate(x, y)

    # Define the points for interpolation
    t = np.array([0.5, 1.5, 2.5, 3.5])
    expected_values = np.array([1.0, 3.0, 5.0, 7.0])

    # Perform interpolation
    interpolated_values = predict(t)

    # Check that the interpolated values are close to the expected values
    np.testing.assert_allclose(
        interpolated_values,
        expected_values,
        atol=1e-9,
        err_msg="TsPack interpolation result does not match expected values.",
    )

def test_tspack_tension():
    """
    Tests the tension parameter of the TsPack.interpolate method.
    A higher tension value should pull the interpolated curve closer to the
    straight line segments connecting the data points.
    """
    x = np.array([0., 1., 2., 3., 4.])
    y = np.sin(x)

    tspack = TsPack()
    # With tension, the curve should be "tighter"
    predict_tension = tspack.interpolate(x, y, tension=1.0)
    # Without tension, it's a standard cubic spline
    predict_no_tension = tspack.interpolate(x, y, tension=0.0)

    t = np.array([0.5, 1.5, 2.5, 3.5])

    interpolated_tension = predict_tension(t)
    interpolated_no_tension = predict_no_tension(t)

    # For comparison, calculate the simple linear interpolation
    linear_interpolation = np.interp(t, x, y)

    # The absolute difference from the linear interpolation should be smaller
    # with tension than without.
    diff_tension = np.abs(interpolated_tension - linear_interpolation)
    diff_no_tension = np.abs(interpolated_no_tension - linear_interpolation)

    assert np.all(diff_tension < diff_no_tension)
