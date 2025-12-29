import numpy as np
import pytest
from renka import tspsi, tsval1, hval, hpval, tspss


def test_tspsi_basic():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    x_out, y_out, yp, sigma = tspsi(x, y)

    assert np.array_equal(x, x_out)
    assert np.array_equal(y, y_out)
    assert len(yp) == len(x)
    assert len(sigma) == len(x)


def test_tsval1_evaluation():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

    xydt = tspsi(x, y)

    y_eval = tsval1(x, xydt)

    assert np.allclose(y, y_eval)


def test_hval_evaluation():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

    x_out, y_out, yp, sigma = tspsi(x, y)

    y_eval = hval(x, x, y, yp, sigma)

    assert np.allclose(y, y_eval)


def test_hpval_evaluation():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

    x_out, y_out, yp, sigma = tspsi(x, y)

    yp_eval = hpval(x, x, y, yp, sigma)

    assert np.allclose(yp_eval, yp)


def test_tspss_smoothing():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.1, 1.2, 3.9, 9.3, 15.8])
    w = np.ones_like(x)
    s = float(len(x))

    x_out, ys, yp, sigma = tspss(x, y, w, s=s)

    assert len(ys) == len(x)
    assert len(yp) == len(x)
    assert len(sigma) == len(x)
    assert not np.allclose(y, ys)


def test_tspsi_periodic():
    x = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    x = np.append(x, 2 * np.pi)
    y = np.sin(x)
    y[-1] = y[0]

    x_out, y_out, yp, sigma = tspsi(x, y, per=1)

    assert np.array_equal(x, x_out)
    assert np.array_equal(y, y_out)
    assert len(yp) == len(x)
    assert len(sigma) == len(x)
    assert np.isclose(yp[0], yp[-1])


def test_tspsi_non_monotonic_x():
    x = np.array([0.0, 2.0, 1.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    with pytest.raises(RuntimeError) as excinfo:
        tspsi(x, y)
    assert "x-values are not strictly increasing" in str(excinfo.value)
