import unittest
import numpy as np
import pytspack


class TestTspack(unittest.TestCase):
    def test_tspsi_tsval1(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        # ncd=1 (derivatives), iendc=0, per=0, unifrm=0, sigma=None
        # Updated tspsi returns (x, y, yp, sigma) to match legacy API
        xn, yn, yp, sigma = pytspack.tspsi(x, y, 1, 0, 0, 0, None)
        self.assertEqual(len(yp), 3)
        self.assertEqual(len(sigma), 3)

        te = np.array([0.5, 1.5])
        # tsval1 takes (te, xydt) where xydt is tuple
        xydt = (xn, yn, yp, sigma)
        v = pytspack.tsval1(te, xydt)
        self.assertEqual(len(v), 2)
        # Simple check, exact values depend on spline logic but should be symmetric
        self.assertAlmostEqual(v[0], v[1])

    def test_trmesh(self):
        x = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        res = pytspack.trmesh(x, y)
        self.assertIn("list", res)
        self.assertIn("lptr", res)
        self.assertIn("lend", res)
        self.assertIn("lnew", res)

    def test_stri_trmesh(self):
        # 6 points on sphere (octahedron vertices)
        x = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        res = pytspack.stri_trmesh(x, y, z)
        self.assertIn("list", res)
        self.assertIn("lptr", res)
        self.assertIn("lend", res)


if __name__ == "__main__":
    unittest.main()
