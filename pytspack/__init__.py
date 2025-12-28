from .tspack import (
    tspsi as _tspsi,
    tsval1 as _tsval1,
    trmesh as _trmesh,
    stri_trmesh as _stri_trmesh,
    tspss as _tspss,
)


def tspsi(x, y, ncd=1, iendc=0, per=0, unifrm=0, yp=None, sigma=None):
    """
    Compute derivatives and tension factors for a curve.
    Returns (x, y, yp, sigma).
    """
    try:
        yp_out, sigma_out = _tspsi(
            x, y, ncd=ncd, iendc=iendc, per=per, unifrm=unifrm, yp=yp, sigma=sigma
        )
    except RuntimeError as e:
        msg = str(e)
        if "error code -4" in msg:
            raise RuntimeError("x-values are not strictly increasing") from e
        raise e

    return x, y, yp_out, sigma_out


def tsval1(x, xydt):
    """
    Evaluate spline at points x (te).
    xydt is a tuple (x_data, y_data, yp, sigma).
    """
    if len(xydt) != 4:
        raise ValueError("xydt must contain (x, y, yp, sigma)")

    xn, yn, yp, sigma = xydt
    return _tsval1(xn, yn, yp, sigma, x, 0)


def hval(t, x, y, yp, sigma):
    """
    Evaluate spline at points t.
    """
    # Use tsval1 with iflag=0 (function value)
    return _tsval1(x, y, yp, sigma, t, 0)


def hpval(t, x, y, yp, sigma):
    """
    Evaluate spline derivative at points t.
    """
    # Use tsval1 with iflag=1 (first derivative)
    return _tsval1(x, y, yp, sigma, t, 1)


def tspss(x, y, w, per=0, unifrm=0, s=0.0, smtol=0.01):
    """
    Smooth curve.
    s maps to sm (smoothing parameter).
    """
    # _tspss signature: (x, y, per, unifrm, w, sm, smtol) -> (sigma, ys, yp, nit)
    # The default smtol=0.0 in my C code causes error -1.
    # The C code checks: if (smtol <= 0.0 || smtol >= 1.0)
    # So I must ensure smtol is positive.
    if smtol <= 0.0:
        smtol = 0.01  # Default to something small but positive if 0 is passed?

    res = _tspss(x, y, per, unifrm, w, s, smtol)
    # The test expects: x_out, ys, yp, sigma
    # res is dict: {'sigma': ..., 'ys': ..., 'yp': ..., 'nit': ...}
    return x, res["ys"], res["yp"], res["sigma"]


def trmesh(x, y):
    return _trmesh(x, y)


def stri_trmesh(x, y, z):
    return _stri_trmesh(x, y, z)
