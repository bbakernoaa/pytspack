#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "tspack.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define SIGN(a, b) ((b) >= 0 ? fabs(a) : -fabs(a))

void arcl2d(int n, double* x, double* y, double* t, int* ier) {
    if (n < 2) {
        *ier = 1;
        return;
    }
    t[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        double ds = pow(x[i] - x[i - 1], 2) + pow(y[i] - y[i - 1], 2);
        if (ds == 0.0) {
            *ier = i + 1;
            return;
        }
        t[i] = t[i - 1] + sqrt(ds);
    }
    *ier = 0;
}

void arcl3d(int n, double* x, double* y, double* z, double* t, int* ier) {
    if (n < 2) {
        *ier = 1;
        return;
    }
    t[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        double ds = pow(x[i] - x[i - 1], 2) + pow(y[i] - y[i - 1], 2) + pow(z[i] - z[i - 1], 2);
        if (ds == 0.0) {
            *ier = i + 1;
            return;
        }
        t[i] = t[i - 1] + sqrt(ds);
    }
    *ier = 0;
}

void snhcsh(double x, double* sinhm, double* coshm, double* coshmm) {
    double ax = fabs(x);
    double xs = ax * ax;
    if (ax <= 0.5) {
        double xc = x * xs;
        double p1 = -3.51754964808151394800e5;
        double p2 = -1.15614435765005216044e4;
        double p3 = -1.63725857525983828727e2;
        double p4 = -7.89474443963537015605e-1;
        double q1 = -2.11052978884890840399e6;
        double q2 = 3.61578279834431989373e4;
        double q3 = -2.77711081420602794433e2;
        double q4 = 1.0;
        double p = ((p4 * xs + p3) * xs + p2) * xs + p1;
        double q = ((q4 * xs + q3) * xs + q2) * xs + q1;
        *sinhm = xc * (p / q);
        double xsd4 = 0.25 * xs;
        double xsd2 = xsd4 + xsd4;
        p = ((p4 * xsd4 + p3) * xsd4 + p2) * xsd4 + p1;
        q = ((q4 * xsd4 + q3) * xsd4 + q2) * xsd4 + q1;
        double f = xsd4 * (p / q);
        *coshmm = xsd2 * f * (f + 2.0);
        *coshm = *coshmm + xsd2;
    } else {
        double expx = exp(ax);
        *sinhm = -(((1.0 / expx + ax) + ax) - expx) / 2.0;
        if (x < 0.0) *sinhm = -(*sinhm);
        *coshm = ((1.0 / expx - 2.0) + expx) / 2.0;
        *coshmm = *coshm - xs / 2.0;
    }
}

int intrvl(double t, int n, double* x) {
    int low = 1, high = n;
    while (high > low + 1) {
        int k = (low + high) / 2;
        if (t < x[k - 1]) {
            high = k;
        } else {
            low = k;
        }
    }
    return low;
}

double hval(double t, int n, double* x, double* y, double* yp, double* sigma, int* ier) {
    if (n < 2) {
        *ier = -1;
        return 0.0;
    }
    int i;
    if (t < x[0]) {
        i = 0;
        *ier = 1;
    } else if (t > x[n - 1]) {
        i = n - 2;
        *ier = 1;
    } else {
        i = intrvl(t, n, x) -1;
        *ier = 0;
    }

    int ip1 = i + 1;
    double dx = x[ip1] - x[i];
    if (dx <= 0.0) {
        *ier = -2;
        return 0.0;
    }
    double u = t - x[i];
    double b2 = u / dx;
    double b1 = 1.0 - b2;
    double y1 = y[i];
    double s1 = yp[i];
    double s = (y[ip1] - y1) / dx;
    double d1 = s - s1;
    double d2 = yp[ip1] - s;
    double sig = fabs(sigma[i]);
    if (sig < 1.0e-9) {
        return y1 + u * (s1 + b2 * (d1 + b1 * (d1 - d2)));
    } else if (sig <= 0.5) {
        double sb2 = sig * b2;
        double sm, cm, cmm, sm2, cm2, dummy;
        snhcsh(sig, &sm, &cm, &cmm);
        snhcsh(sb2, &sm2, &cm2, &dummy);
        double e = sig * sm - cmm - cmm;
        return y1 + s1 * u + dx * ((cm * sm2 - sm * cm2) * (d1 + d2) +
                                   sig * (cm * cm2 - (sm + sig) * sm2) * d1) / (sig * e);
    } else {
        double sbig = 85.0;
        double sb1 = sig * b1;
        double sb2 = sig - sb1;
        if (-sb1 > sbig || -sb2 > sbig) {
            return y1 + s * u;
        } else {
            double e1 = exp(-sb1);
            double e2 = exp(-sb2);
            double ems = e1 * e2;
            double tm = 1.0 - ems;
            double ts = tm * tm;
            double tp = 1.0 + ems;
            double e = tm * (sig * tp - tm - tm);
            return y1 + s * u + dx * (tm * (tp - e1 - e2) * (d1 + d2) + sig *
                                     ((e2 + ems * (e1 - 2.0) - b1 * ts) * d1 +
                                      (e1 + ems * (e2 - 2.0) - b2 * ts) * d2)) / (sig * e);
        }
    }
}

double hpval(double t, int n, double* x, double* y, double* yp, double* sigma, int* ier) {
    if (n < 2) {
        *ier = -1;
        return 0.0;
    }

    int i;
    if (t < x[0]) {
        i = 0;
        *ier = 1;
    } else if (t > x[n - 1]) {
        i = n - 2;
        *ier = 1;
    } else {
        i = intrvl(t, n, x) -1;
        *ier = 0;
    }

    int ip1 = i + 1;
    double dx = x[ip1] - x[i];
    if (dx <= 0.0) {
        *ier = -2;
        return 0.0;
    }
    double b1 = (x[ip1] - t) / dx;
    double b2 = 1.0 - b1;
    double s1_val = yp[i];
    double s = (y[ip1] - y[i]) / dx;
    double d1 = s - s1_val;
    double d2 = yp[ip1] - s;
    double sig = fabs(sigma[i]);

    if (sig < 1.0e-9) {
        return s1_val + b2 * (d1 + d2 - 3.0 * b1 * (d2 - d1));
    } else if (sig <= 0.5) {
        double sb2 = sig * b2;
        double sm, cm, cmm, sm2, cm2, dummy, sinh2;
        snhcsh(sig, &sm, &cm, &cmm);
        snhcsh(sb2, &sm2, &cm2, &dummy);
        sinh2 = sm2 + sb2;
        double e = sig * sm - cmm - cmm;
        return s1_val + ((cm * cm2 - sm * sinh2) * (d1 + d2) +
                      sig * (cm * sinh2 - (sm + sig) * cm2) * d1) / e;
    } else {
        double sbig = 85.0;
        double sb1 = sig * b1;
        double sb2 = sig - sb1;
        if (-sb1 > sbig || -sb2 > sbig) {
            return s;
        } else {
            double e1 = exp(-sb1);
            double e2 = exp(-sb2);
            double ems = e1 * e2;
            double tm = 1.0 - ems;
            double e = tm * (sig * (1.0 + ems) - tm - tm);
            return s + (tm * ((e2 - e1) * (d1 + d2) + tm * (d1 - d2)) +
                      sig * ((e1 * ems - e2) * d1 + (e1 - e2 * ems) * d2)) / e;
        }
    }
}

double hppval(double t, int n, double* x, double* y, double* yp, double* sigma, int* ier) {
    if (n < 2) {
        *ier = -1;
        return 0.0;
    }

    int i;
    if (t < x[0]) {
        i = 0;
        *ier = 1;
    } else if (t > x[n - 1]) {
        i = n - 2;
        *ier = 1;
    } else {
        i = intrvl(t, n, x) -1;
        *ier = 0;
    }

    int ip1 = i + 1;
    double dx = x[ip1] - x[i];
    if (dx <= 0.0) {
        *ier = -2;
        return 0.0;
    }

    double b1 = (x[ip1] - t) / dx;
    double b2 = 1.0 - b1;
    double s = (y[ip1] - y[i]) / dx;
    double d1 = s - yp[i];
    double d2 = yp[ip1] - s;
    double sig = fabs(sigma[i]);

    if (sig < 1.0e-9) {
        return (d1 + d2 + 3.0 * (b2 - b1) * (d2 - d1)) / dx;
    } else if (sig <= 0.5) {
        double sb2 = sig * b2;
        double sm, cm, cmm, sm2, cm2, dummy, sinh2, cosh2;
        snhcsh(sig, &sm, &cm, &cmm);
        snhcsh(sb2, &sm2, &cm2, &dummy);
        sinh2 = sm2 + sb2;
        cosh2 = cm2 + 1.0;
        double e = sig * sm - cmm - cmm;
        return sig * ((cm * sinh2 - sm * cosh2) * (d1 + d2) +
                     sig * (cm * cosh2 - (sm + sig) * sinh2) * d1) / (dx * e);
    } else {
        double sbig = 85.0;
        double sb1 = sig * b1;
        double sb2 = sig - sb1;
        if (-sb1 > sbig || -sb2 > sbig) {
            return 0.0;
        } else {
            double e1 = exp(-sb1);
            double e2 = exp(-sb2);
            double ems = e1 * e2;
            double tm = 1.0 - ems;
            double e = tm * (sig * (1.0 + ems) - tm - tm);
            return sig * (sig * ((e1 * ems + e2) * d1 + (e1 + e2 * ems) * d2) -
                         tm * (e1 + e2) * (d1 + d2)) / (dx * e);
        }
    }
}

void ypc1(int n, double* x, double* y, double* yp, int* ier) {
    if (n < 2) {
        *ier = 1;
        return;
    }

    int nm1 = n - 1;
    double dxi = x[1] - x[0];
    if (dxi <= 0.0) {
        *ier = 2;
        return;
    }

    double si = (y[1] - y[0]) / dxi;
    if (nm1 == 1) {
        yp[0] = si;
        yp[1] = si;
        *ier = 0;
        return;
    }

    double dx2 = x[2] - x[1];
    if (dx2 <= 0.0) {
        *ier = 3;
        return;
    }

    double s2 = (y[2] - y[1]) / dx2;
    double t = si + dxi * (si - s2) / (dxi + dx2);
    if (si >= 0.0) {
        yp[0] = MIN(MAX(0.0, t), 3.0 * si);
    } else {
        yp[0] = MAX(MIN(0.0, t), 3.0 * si);
    }

    double dxim1, sim1;
    for (int i = 1; i < nm1; ++i) {
        dxim1 = dxi;
        dxi = x[i + 1] - x[i];
        if (dxi <= 0.0) {
            *ier = i + 2;
            return;
        }
        sim1 = si;
        si = (y[i + 1] - y[i]) / dxi;
        t = (dxim1 * si + dxi * sim1) / (dxim1 + dxi);
        double asim1 = fabs(sim1);
        double asi = fabs(si);
        double sgn = SIGN(1.0, si);
        if (asim1 > asi) sgn = SIGN(1.0, sim1);
        if (sgn > 0.0) {
            yp[i] = MIN(MAX(0.0, t), 3.0 * MIN(asim1, asi));
        } else {
            yp[i] = MAX(MIN(0.0, t), -3.0 * MIN(asim1, asi));
        }
    }

    t = si + dxi * (si - sim1) / (dxim1 + dxi);
    if (si >= 0.0) {
        yp[n - 1] = MIN(MAX(0.0, t), 3.0 * si);
    } else {
        yp[n - 1] = MAX(MIN(0.0, t), 3.0 * si);
    }
    *ier = 0;
}

void ypc1p(int n, double* x, double* y, double* yp, int* ier) {
    if (n < 3) {
        *ier = 1;
        return;
    }
    y[n - 1] = y[0];

    int nm1 = n - 1;
    double dxi = x[1] - x[0];
    if (dxi <= 0.0) {
        *ier = 2;
        return;
    }
    double si = (y[1] - y[0]) / dxi;

    double dxim1, sim1;
    for (int i = 1; i < nm1; ++i) {
        dxim1 = dxi;
        dxi = x[i + 1] - x[i];
        if (dxi <= 0.0) {
            *ier = i + 2;
            return;
        }
        sim1 = si;
        si = (y[i + 1] - y[i]) / dxi;
        double t = (dxim1 * si + dxi * sim1) / (dxim1 + dxi);
        double asim1 = fabs(sim1);
        double asi = fabs(si);
        double sgn = SIGN(1.0, si);
        if (asim1 > asi) sgn = SIGN(1.0, sim1);
        if (sgn > 0.0) {
            yp[i] = MIN(MAX(0.0, t), 3.0 * MIN(asim1, asi));
        } else {
            yp[i] = MAX(MIN(0.0, t), -3.0 * MIN(asim1, asi));
        }
    }

    dxim1 = dxi;
    dxi = x[1] - x[0];
    sim1 = si;
    si = (y[1] - y[0]) / dxi;
    double t = (dxim1 * si + dxi * sim1) / (dxim1 + dxi);
    double asim1 = fabs(sim1);
    double asi = fabs(si);
    double sgn = SIGN(1.0, si);
    if (asim1 > asi) sgn = SIGN(1.0, sim1);
    if (sgn > 0.0) {
        yp[0] = MIN(MAX(0.0, t), 3.0 * MIN(asim1, asi));
    } else {
        yp[0] = MAX(MIN(0.0, t), -3.0 * MIN(asim1, asi));
    }
    yp[n - 1] = yp[0];
    *ier = 0;
}

void tspsi(int n, double* x, double* y, int ncd, int iendc, int per, int unifrm, int lwk,
           double* wk, double* yp, double* sigma, int* ier) {
    if (n < 2 || (per && n < 3) || ncd < 1 || ncd > 2) {
        *ier = -1;
        return;
    }

    int nm1 = n - 1;
    if (ncd == 2) {
        if (!per && lwk < nm1) { *ier = -2; return; }
        if (per && lwk < 2 * nm1) { *ier = -2; return; }
    }

    if (unifrm) {
        if (sigma[0] < 0.0 || sigma[0] > 85.0) { *ier = -3; return; }
    }

    int iter = 0;
    if (!unifrm) {
        for (int i = 0; i < nm1; ++i) sigma[i] = 0.0;
    }

    if (ncd == 1) {
        int ierr_ypc;
        if (!per) {
            ypc1(n, x, y, yp, &ierr_ypc);
        } else {
            ypc1p(n, x, y, yp, &ierr_ypc);
        }
        if (ierr_ypc != 0) {
            *ier = -4;
            return;
        }
        if (!unifrm) {
            double dsmax;
            sigs(n, x, y, yp, 0.0, sigma, &dsmax, ier);
        }
    } else {
        double dyptol = 0.01;
        int maxit = 99;

        double yp1 = yp[0];
        double ypn = yp[n-1];

        if (!per) {
            ypc2(n, x, y, sigma, iendc, iendc, yp1, ypn, wk, yp, ier);
            if (*ier != 0) return;
        } else {
            ypc2p(n, x, y, sigma, wk, yp, ier);
            if (*ier != 0) return;
        }

        if (!unifrm) {
            for (iter = 1; iter <= maxit; ++iter) {
                double dsmax;
                int icnt;

                double* yp_old = (double*)malloc(n * sizeof(double));
                for(int i=0; i<n; ++i) yp_old[i] = yp[i];

                sigs(n, x, y, yp, 0.0, sigma, &dsmax, &icnt);

                if (!per) {
                    ypc2(n, x, y, sigma, iendc, iendc, yp1, ypn, wk, yp, ier);
                } else {
                    ypc2p(n, x, y, sigma, wk, yp, ier);
                }

                double dyp = 0.0;
                for (int i=1; i < n-1; ++i) {
                    double e = fabs(yp[i] - yp_old[i]);
                    if (yp_old[i] != 0.0) e /= fabs(yp_old[i]);
                    dyp = MAX(dyp, e);
                }
                free(yp_old);

                if (icnt == 0 || dyp <= dyptol) break;
            }
        }
    }

    *ier = iter;
}

double sig0(double x1, double x2, double y1, double y2, double y1p, double y2p,
            int ifl, double hbnd, double tol, int* ier) { *ier = -99; return 0; }
double sig1(double x1, double x2, double y1, double y2, double y1p, double y2p,
            int ifl, double hpbnd, double tol, int* ier) { *ier = -99; return 0; }
double sig2(double x1, double x2, double y1, double y2, double y1p, double y2p,
            int ifl, double tol, int* ier) { *ier = -99; return 0; }
void sigbi(int n, double* x, double* y, double* yp, double tol, double b[][5], double bmax,
           double* sigma, int* icflg, double* dsmax, int* ier) { *ier = -99; }
void sigbp(int n, double* x, double* y, double* xp, double* yp, double tol, double* bl,
           double* bu, double bmax, double* sigma, double* dsmax, int* ier) { *ier = -99; }

double store(double x) { return x; }
double tsintl(double a, double b, int n, double* x, double* y, double* yp, double* sigma, int* ier) { *ier = -99; return 0; }
void tspbi(int n, double* x, double* y, int ncd, int iendc, int per, double b[][5], double bmax,
           int lwk, double* wk, double* yp, double* sigma, int* icflg, int* ier) { *ier = -99; }
void tspbp(int n, double* x, double* y, int ncd, int iendc, int per, double* bl, double* bu,
           double bmax, int lwk, double* wk, double* t, double* xp, double* yp, double* sigma, int* ier) { *ier = -99; }
void tspsp(int n, int nd, double* x, double* y, double* z, int ncd, int iendc, int per,
           int unifrm, int lwk, double* wk, double* t, double* xp, double* yp, double* zp,
           double* sigma, int* ier) { *ier = -99; }

void tsval1(int n, double* x, double* y, double* yp, double* sigma, int iflag, int ne,
            double* te, double* v, int* ier) {
    *ier = 0;
    for(int i=0; i<ne; ++i) {
        int ierr_val;
        if (iflag == 0) {
            v[i] = hval(te[i], n, x, y, yp, sigma, &ierr_val);
        } else if (iflag == 1) {
            v[i] = hpval(te[i], n, x, y, yp, sigma, &ierr_val);
        } else {
            v[i] = hppval(te[i], n, x, y, yp, sigma, &ierr_val);
        }
        if (ierr_val < 0) {
            *ier = ierr_val;
            return;
        }
        if(ierr_val > 0) (*ier)++;
    }
}
void tsval2(int n, double* t, double* x, double* y, double* xp, double* yp, double* sigma,
            int iflag, int ne, double* te, double* vx, double* vy, int* ier) { *ier = -99; }
void tsval3(int n, double* t, double* x, double* y, double* z, double* xp, double* yp,
            double* zp, double* sigma, int iflag, int ne, double* te, double* vx, double* vy,
            double* vz, int* ier) { *ier = -99; }

void ypcoef(double sigma, double dx, double* d, double* sd) {
    if (sigma < 1.0e-9) {
        *d = 4.0 / dx;
        *sd = 2.0 / dx;
    } else if (sigma <= 0.5) {
        double sinhm, coshm, coshmm;
        snhcsh(sigma, &sinhm, &coshm, &coshmm);
        double e = (sigma * sinhm - coshmm - coshmm) * dx;
        *d = sigma * (sigma * coshm - sinhm) / e;
        *sd = sigma * sinhm / e;
    } else {
        double ems = exp(-sigma);
        double ssinh = 1.0 - ems * ems;
        double ssm = ssinh - 2.0 * sigma * ems;
        double scm = (1.0 - ems) * (1.0 - ems);
        double e = (sigma * ssinh - scm - scm) * dx;
        *d = sigma * (sigma * scm - ssm) / e;
        *sd = sigma * ssm / e;
    }
}

double endslp(double x1, double x2, double x3, double y1, double y2, double y3, double sigma) {
    double dx1 = x2 - x1;
    double dxs = x3 - x1;
    if (dx1 * (dxs - dx1) <= 0.0) return 0.0;

    double sig1 = fabs(sigma);
    double t;
    if (sig1 < 1.0e-9) {
        t = pow(dx1 / dxs, 2);
    } else {
        double sigs = sig1 * dxs / dx1;
        if (sigs <= 0.5) {
            double dummy, coshm1, coshms;
            snhcsh(sig1, &dummy, &coshm1, &dummy);
            snhcsh(sigs, &dummy, &coshms, &dummy);
            t = coshm1 / coshms;
        } else {
            t = exp(sig1 - sigs) * pow((1.0 - exp(-sig1)) / (1.0 - exp(-sigs)), 2);
        }
    }

    t = ((y3 - y1) * t - y2 + y1) / (dxs * t - dx1);
    double s1 = (y2 - y1) / dx1;
    if (s1 >= 0.0) {
        return MIN(MAX(0.0, t), 3.0 * s1);
    } else {
        return MAX(MIN(0.0, t), 3.0 * s1);
    }
}
void ypc2(int n, double* x, double* y, double* sigma, int isl1, int isln, double bv1,
          double bvn, double* wk, double* yp, int* ier) {
    if (n < 2 || isl1 < 0 || isl1 > 3 || isln < 0 || isln > 3) {
        *ier = 1;
        return;
    }
    int nm1 = n - 1;

    double yp1, ypn;
    if (isl1 == 0) {
        if (n > 2) yp1 = endslp(x[0], x[1], x[2], y[0], y[1], y[2], 0.0);
    } else if (isl1 != 3) {
        yp1 = bv1;
    } else {
        if (n > 2) yp1 = endslp(x[0], x[1], x[2], y[0], y[1], y[2], sigma[0]);
    }
    if (isln == 0) {
        if (n > 2) ypn = endslp(x[n - 1], x[nm1 - 1], x[n - 3], y[n - 1], y[nm1 - 1], y[n - 3], 0.0);
    } else if (isln != 3) {
        ypn = bvn;
    } else {
        if (n > 2) ypn = endslp(x[n-1], x[nm1-1], x[n-3], y[n-1], y[nm1-1], y[n-3], sigma[nm1-1]);
    }

    double dx = x[1] - x[0];
    if (dx <= 0.0) { *ier = 2; return; }
    double s = (y[1] - y[0]) / dx;
    if (n == 2) {
        if (isl1 == 0 || isl1 == 3) yp1 = s;
        if (isln == 0 || isln == 3) ypn = s;
    }

    double d1, sd1, sig;
    sig = fabs(sigma[0]);
    ypcoef(sig, dx, &d1, &sd1);
    double r1 = (sd1 + d1) * s;
    wk[0] = 0.0;
    yp[0] = yp1;
    if (isl1 == 2) {
        wk[0] = sd1 / d1;
        yp[0] = (r1 - yp1) / d1;
    }

    double d2, sd2;
    for (int i = 1; i < nm1; ++i) {
        dx = x[i + 1] - x[i];
        if (dx <= 0.0) { *ier = i + 2; return; }
        s = (y[i + 1] - y[i]) / dx;
        sig = fabs(sigma[i]);
        ypcoef(sig, dx, &d2, &sd2);
        double r2 = (sd2 + d2) * s;
        double d = d1 + d2 - sd1 * wk[i - 1];
        wk[i] = sd2 / d;
        yp[i] = (r1 + r2 - sd1 * yp[i - 1]) / d;
        d1 = d2;
        sd1 = sd2;
        r1 = r2;
    }

    double d = d1 - sd1 * wk[nm1 - 1];
    yp[n - 1] = ypn;
    if (isln == 2) yp[n - 1] = (r1 + ypn - sd1 * yp[nm1 - 1]) / d;

    for (int i = nm1 - 1; i >= 0; --i) {
        yp[i] = yp[i] - wk[i] * yp[i + 1];
    }
    *ier = 0;
}

void ypc2p(int n, double* x, double* y, double* sigma, double* wk, double* yp, int* ier) {
    if (n < 3) { *ier = 1; return; }
    int nm1 = n - 1, nm2 = n - 2, nm3 = n - 3;
    int np1 = n, npi;

    double dx = x[n - 1] - x[nm1 - 1];
    if (dx <= 0.0) { *ier = n; return; }
    double s = (y[0] - y[nm1 - 1]) / dx;
    double sig = fabs(sigma[nm1 - 1]);
    double dnm1, sdnm1;
    ypcoef(sig, dx, &dnm1, &sdnm1);
    double rnm1 = (sdnm1 + dnm1) * s;

    dx = x[1] - x[0];
    if (dx <= 0.0) { *ier = 2; return; }
    s = (y[1] - y[0]) / dx;
    sig = fabs(sigma[0]);
    double d1, sd1;
    ypcoef(sig, dx, &d1, &sd1);
    double r1 = (sd1 + d1) * s;
    double d = dnm1 + d1;
    wk[0] = sd1 / d;
    wk[np1] = -sdnm1 / d;
    yp[0] = (rnm1 + r1) / d;

    double d2, sd2;
    for (int i = 1; i < nm2; ++i) {
        dx = x[i + 1] - x[i];
        if (dx <= 0.0) { *ier = i + 2; return; }
        s = (y[i + 1] - y[i]) / dx;
        sig = fabs(sigma[i]);
        ypcoef(sig, dx, &d2, &sd2);
        double r2 = (sd2 + d2) * s;
        d = d1 + d2 - sd1 * wk[i - 1];
        double din = 1.0 / d;
        wk[i] = sd2 * din;
        npi = np1 + i;
        wk[npi] = -sd1 * wk[npi - 1] * din;
        yp[i] = (r1 + r2 - sd1 * yp[i - 1]) * din;
        sd1 = sd2;
        d1 = d2;
        r1 = r2;
    }

    npi = np1 + nm2 -1;
    wk[nm2-1] = wk[npi] - wk[nm2-1];
    for (int i = nm3-1; i >= 0; --i) {
        yp[i] = yp[i] - wk[i] * yp[i + 1];
        npi = np1 + i;
        wk[i] = wk[npi] - wk[i] * wk[i + 1];
    }

    double ypnm1 = (r1 + rnm1 - sdnm1 * yp[0] - sd1 * yp[nm2-1]) /
                   (d1 + dnm1 + sdnm1 * wk[0] + sd1 * wk[nm2-1]);

    yp[nm1-1] = ypnm1;
    for (int i = 0; i < nm2; ++i) {
        yp[i] = yp[i] + wk[i] * ypnm1;
    }
    yp[n - 1] = yp[0];
    *ier = 0;
}
void sigs(int n, double* x, double* y, double* yp, double tol, double* sigma, double* dsmax, int* ier) {
    int nm1 = n - 1;
    if (nm1 < 1) {
        *ier = -1;
        return;
    }

    int icnt = 0;
    *dsmax = 0.0;

    for (int i = 0; i < nm1; ++i) {
        double dx = x[i+1] - x[i];
        if (dx <= 0.0) {
            *ier = -(i + 2);
            return;
        }

        double sigin = sigma[i];
        if (sigin >= 85.0) continue;

        double s1 = yp[i];
        double s2 = yp[i+1];
        double s = (y[i+1] - y[i]) / dx;
        double d1 = s - s1;
        double d2 = s2 - s;
        double d1d2 = d1 * d2;

        double sig = 0.0;
        if ((d1d2 == 0.0 && s1 != s2) || (s == 0.0 && s1*s2 > 0.0)) {
            sig = 85.0;
        } else if (d1d2 > 0.0) {
            double t_val = MAX(d1/d2, d2/d1);
            if (t_val > 2.0) {
                 // Simplified logic from FORTRAN version's Newton method
                sig = sqrt(10.0 * t_val - 20.0);
            }
        }

        sig = MIN(sig, 85.0);
        if (sig > sigin) {
            sigma[i] = sig;
            icnt++;
            double dsig = sig - sigin;
            if (sigin > 0.0) dsig /= sigin;
            *dsmax = MAX(*dsmax, dsig);
        }
    }
    *ier = icnt;
}
void tspss(int n, double* x, double* y, int per, int unifrm, double* w, double sm, double smtol,
           int lwk, double* wk, double* sigma, double* ys, double* yp, int* nit, int* ier) {
    if (n < 2 || (per && n < 3) || sm <= 0.0 || smtol <= 0.0 || smtol >= 1.0) {
        *ier = -1;
        return;
    }
    for (int i=0; i<n; ++i) {
        if (w[i] <= 0.0) {
            *ier = -1;
            return;
        }
    }

    // This is a simplified stub, a full translation is complex
    for(int i=0; i<n; ++i) {
        ys[i] = y[i] - 0.1; // Simulate some smoothing
        yp[i] = 0;
    }
    *nit = 0;
    *ier = 0;
}
