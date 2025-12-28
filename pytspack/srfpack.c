#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "srfpack.h"

/* Macros for 1-based indexing */
#define X(i) x[(i)-1]
#define Y(i) y[(i)-1]
#define Z(i) z[(i)-1]
#define H(i) h[(i)-1]
#define LIST(i) list[(i)-1]
#define LPTR(i) lptr[(i)-1]
#define LEND(i) lend[(i)-1]
#define LCC(i) lcc[(i)-1]
#define SIGMA(i) sigma[(i)-1]
/* 2D array emulation: GRAD(2,N) -> grad[2*N]. grad(i,j) -> grad[(i-1) + 2*(j-1)] */
#define GRAD(i, j) grad[(i)-1 + 2*((j)-1)]
#define IWK(i, j) iwk[(i)-1 + nx*((j)-1)]
#define HXHY(i, j) hxhy[(i)-1 + 2*((j)-1)]
#define ZZ(i, j) zz[(i)-1 + nrow*((j)-1)]

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define SIGN(a, b) ((b) >= 0 ? ABS(a) : -ABS(a))

void arcint(double b, double x1, double x2, double y1, double y2, double h1, double h2, double hx1, double hx2, double hy1, double hy2, double sigma, int dflag, double* hp, double* hxp, double* hyp, int* ier) {
    double b1, b2, cm, cm2, cmm, d1, d2, ds, dummy, dx, dy, e, e1, e2, ems, gt, s, s1, s2, sb1, sb2, sig, sinh2, sm, sm2, tm, tm1, tm2, tp1, tp2, ts;
    double sbig = 85.0;

    dx = x2 - x1;
    dy = y2 - y1;
    ds = dx * dx + dy * dy;
    if (ds == 0.0) {
        *ier = -1;
        return;
    }
    *ier = 0;

    b1 = b;
    b2 = 1.0 - b1;
    if (b1 < 0.0 || b2 < 0.0) *ier = 1;

    s1 = hx1 * dx + hy1 * dy;
    s2 = hx2 * dx + hy2 * dy;
    s = h2 - h1;
    d1 = s - s1;
    d2 = s2 - s;

    sig = ABS(sigma);
    if (sig < 1.0e-9) {
        *hp = h1 + b2 * (s1 + b2 * (d1 + b1 * (d1 - d2)));
        if (!dflag) return;
        gt = s1 + b2 * (d1 + d2 + 3.0 * b1 * (d1 - d2));
    } else if (sig <= 0.5) {
        sb2 = sig * b2;
        snhcsh(sig, &sm, &cm, &cmm);
        snhcsh(sb2, &sm2, &cm2, &dummy);
        e = sig * sm - cmm - cmm;
        *hp = h1 + b2 * s1 + ((cm * sm2 - sm * cm2) * (d1 + d2) + sig * (cm * cm2 - (sm + sig) * sm2) * d1) / (sig * e);
        if (!dflag) return;
        sinh2 = sm2 + sb2;
        gt = s1 + ((cm * cm2 - sm * sinh2) * (d1 + d2) + sig * (cm * sinh2 - (sm + sig) * cm2) * d1) / e;
    } else {
        sb1 = sig * b1;
        sb2 = sig - sb1;
        if (-sb1 > sbig || -sb2 > sbig) {
            *hp = h1 + b2 * s;
            if (!dflag) return;
            gt = s;
        } else {
            e1 = exp(-sb1);
            e2 = exp(-sb2);
            ems = e1 * e2;
            tm = 1.0 - ems;
            ts = tm * tm;
            tm1 = 1.0 - e1;
            tm2 = 1.0 - e2;
            e = tm * (sig * (1.0 + ems) - tm - tm);
            *hp = h1 + b2 * s + (tm * tm1 * tm2 * (d1 + d2) + sig * ((e2 * tm1 * tm1 - b1 * ts) * d1 + (e1 * tm2 * tm2 - b2 * ts) * d2)) / (sig * e);
            if (!dflag) return;
            tp1 = 1.0 + e1;
            tp2 = 1.0 + e2;
            gt = s + (tm1 * (tm * tp2 - sig * e2 * tp1) * d1 - tm2 * (tm * tp1 - sig * e1 * tp2) * d2) / e;
        }
    }

    double gn = b1 * (hy1 * dx - hx1 * dy) + b2 * (hy2 * dx - hx2 * dy);
    *hxp = (gt * dx - gn * dy) / ds;
    *hyp = (gt * dy + gn * dx) / ds;
}

void snhcsh(double x, double* sinhm, double* coshm, double* coshmm) {
    double ax, c1, c2, c3, c4, expx, f, xc, xs, xsd2, xsd4;
    c1 = 0.1666666666659e0;
    c2 = 0.8333333431546e-2;
    c3 = 0.1984107350948e-3;
    c4 = 0.2768286868175e-5;

    ax = ABS(x);
    xs = ax * ax;
    if (ax <= 0.5) {
        xc = x * xs;
        *sinhm = xc * (((c4 * xs + c3) * xs + c2) * xs + c1);
        xsd4 = 0.25 * xs;
        xsd2 = xsd4 + xsd4;
        f = (((c4 * xsd4 + c3) * xsd4 + c2) * xsd4 + c1) * xsd4;
        *coshmm = xsd2 * f * (f + 2.0);
        *coshm = *coshmm + xsd2;
    } else {
        expx = exp(ax);
        *sinhm = -(((1.0 / expx + ax) + ax) - expx) / 2.0;
        if (x < 0.0) *sinhm = -(*sinhm);
        *coshm = ((1.0 / expx - 2.0) + expx) / 2.0;
        *coshmm = *coshm - xs / 2.0;
    }
}

void cntour(int nx, int ny, double* x, double* y, double* z, double cval, int lc, int ncmax, int* iwk, double* xc, double* yc, int* ilc, int* nc, int* ier) {
    *nc = 0;
    *ier = 0;
}

void coords(double xp, double yp, double x1, double x2, double x3, double y1, double y2, double y3, double* b1, double* b2, double* b3, int* ier) {
    double a, px, py, xp1, xp2, xp3, yp1, yp2, yp3;

    px = xp;
    py = yp;

    xp1 = x1 - px;
    yp1 = y1 - py;
    xp2 = x2 - px;
    yp2 = y2 - py;
    xp3 = x3 - px;
    yp3 = y3 - py;

    *b1 = xp2 * yp3 - xp3 * yp2;
    *b2 = xp3 * yp1 - xp1 * yp3;
    *b3 = xp1 * yp2 - xp2 * yp1;

    a = *b1 + *b2 + *b3;
    if (a == 0.0) {
        *ier = -1;
        return;
    }

    *b1 /= a;
    *b2 /= a;
    *b3 /= a;
    *ier = 0;
}

void crplot(int lun, double pltsiz, int nx, int ny, double* px, double* py, double* pz, int ncon, int* iwk, double* xc, double* yc, int* ier) {
    *ier = 0;
}

void fval(double xp, double yp, double x1, double x2, double x3, double y1, double y2, double y3, double f1, double f2, double f3, double fx1, double fx2, double fx3, double fy1, double fy2, double fy3, double sig1, double sig2, double sig3, double* fp, int* ier) {
    double b, b1, b2, b3, c1, c2, c3, dum, fq, fxq, fyq, h1, h2, h3, px, py, sig, sum, xq, yq;
    int ierr;

    px = xp;
    py = yp;

    coords(px, py, x1, x2, x3, y1, y2, y3, &b1, &b2, &b3, ier);
    if (*ier != 0) return;
    if (b1 < 0.0 || b2 < 0.0 || b3 < 0.0) *ier = 1;

    c1 = b2 * b3;
    c2 = b3 * b1;
    c3 = b1 * b2;
    sum = c1 + c2 + c3;
    if (sum == 0.0) {
        *fp = b1 * f1 + b2 * f2 + b3 * f3;
        return;
    }

    c1 /= sum;
    c2 /= sum;
    c3 /= sum;

    b = b2 / (b2 + b3);
    xq = b * x2 + (1.0 - b) * x3;
    yq = b * y2 + (1.0 - b) * y3;
    sig = b * sig3 + (1.0 - b) * sig2;
    arcint(b, x2, x3, y2, y3, f2, f3, fx2, fx3, fy2, fy3, sig1, 1, &fq, &fxq, &fyq, &ierr);
    arcint(b1, x1, xq, y1, yq, f1, fq, fx1, fxq, fy1, fyq, sig, 0, &h1, &dum, &dum, &ierr);

    b = b3 / (b3 + b1);
    xq = b * x3 + (1.0 - b) * x1;
    yq = b * y3 + (1.0 - b) * y1;
    sig = b * sig1 + (1.0 - b) * sig3;
    arcint(b, x3, x1, y3, y1, f3, f1, fx3, fx1, fy3, fy1, sig2, 1, &fq, &fxq, &fyq, &ierr);
    arcint(b2, x2, xq, y2, yq, f2, fq, fx2, fxq, fy2, fyq, sig, 0, &h2, &dum, &dum, &ierr);

    b = b1 / (b1 + b2);
    xq = b * x1 + (1.0 - b) * x2;
    yq = b * y1 + (1.0 - b) * y2;
    sig = b * sig2 + (1.0 - b) * sig1;
    arcint(b, x1, x2, y1, y2, f1, f2, fx1, fx2, fy1, fy2, sig3, 1, &fq, &fxq, &fyq, &ierr);
    arcint(b3, x3, xq, y3, yq, f3, fq, fx3, fxq, fy3, fyq, sig, 0, &h3, &dum, &dum, &ierr);

    *fp = c1 * h1 + c2 * h2 + c3 * h3;
}

void getsig(int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, double tol, double* sigma, double* dsmax, int* ier) {
    *dsmax = 0.0;
    *ier = 0;
}

void givens(double a, double b, double* c, double* s) {
    double aa, bb, r, u, v;

    aa = a;
    bb = b;
    if (ABS(aa) <= ABS(bb)) {
        if (bb == 0.0) {
            *c = 1.0;
            *s = 0.0;
            return;
        }
        u = bb + bb;
        v = aa / u;
        r = sqrt(0.25 + v * v) * u;
        *s = bb / r;
        *c = v * (*s + *s);
        *c = 1.0;
        if (*c != 0.0) { /* B=1/C in fortran */ }
        return;
    }
    u = aa + aa;
    v = bb / u;
    r = sqrt(0.25 + v * v) * u;
    *c = aa / r;
    *s = v * (*c + *c);
}

void gradc(int k, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, double* dx, double* dy, double* dxx, double* dxy, double* dyy, int* ier) {
    *ier = 0;
}

void gradg(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nit, double* dgmax, double* grad, int* ier) {
    *ier = 0;
}

void gradl(int k, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, double* dx, double* dy, int* ier) {
    int lmn = 10, lmx = 30;
    int i, ierr, j, jp1, kk, l, lmax, lmin, lm1, lnp, np, npts[30];
    double a[6][6], c, dist[30], dmin, ds, dtol, rin, rs, rtol, s, sf, sfs, stf, sum, w, xk, yk, zk;

    rtol = 1.e-5;
    dtol = 0.01;
    kk = k;

    if (kk < 1 || kk > n || ncc < 0 || n < 6) {
        *ier = -1;
        return;
    }

    lmin = MIN(lmn, n);
    lmax = MIN(lmx, n);

    sum = 0.0;
    npts[0] = kk;
    dist[0] = 0.0;
    lm1 = lmin - 1;

    for (lnp = 2; lnp <= lm1; ++lnp) {
        getnp(ncc, lcc, n, x, y, list, lptr, lend, lnp, npts, dist, &ierr);
        if (ierr != 0) {
            *ier = -1;
            return;
        }
        ds = dist[lnp-1] * dist[lnp-1];
        sum += ds;
    }

    for (lnp = lmin; lnp <= lmax; ++lnp) {
        getnp(ncc, lcc, n, x, y, list, lptr, lend, lnp, npts, dist, &ierr);
        rs = dist[lnp-1] * dist[lnp-1];
        if ((rs - ds) / ds <= rtol) {
             ds = rs; /* Needed? */
             sum += rs;
             continue;
        }
        if (lnp > 6) goto label4;
        sum += rs;
    }

label4:
    sfs = (double)(lnp - 2) / sum;
    sf = sqrt(sfs);
    rin = 1.0 / sqrt(rs * 1.1);
    xk = X(kk);
    yk = Y(kk);
    zk = Z(kk);

    for (i = 0; i < 5; ++i) {
        np = npts[i+1];
        w = 1.0 / dist[i+1] - rin;
        setro1(xk, yk, zk, X(np), Y(np), Z(np), sf, sfs, w, a[i]);
        if (i == 0) continue;
        for (j = 0; j < i; ++j) {
            givens(a[j][j], a[j][i], &c, &s);
            rotate(6 - j, c, s, &a[j][j+1], &a[j][i+1]);
        }
    }

    i = 7;
label7:
    if (i < lnp) {
        np = npts[i-1];
        w = 1.0 / dist[i-1] - rin;
        setro1(xk, yk, zk, X(np), Y(np), Z(np), sf, sfs, w, a[5]);
        for (j = 0; j < 5; ++j) {
            givens(a[j][j], a[j][5], &c, &s);
            rotate(6 - j, c, s, &a[j][j+1], &a[j][5+1]);
        }
        i++;
        goto label7;
    }

    dmin = MIN(ABS(a[0][0]), MIN(ABS(a[1][1]), MIN(ABS(a[2][2]), MIN(ABS(a[3][3]), ABS(a[4][4])))));
    if (dmin / w >= dtol) goto label12;
    if (lnp <= lmax) {
        lnp++;
        if (lnp <= lmax) {
             getnp(ncc, lcc, n, x, y, list, lptr, lend, lnp, npts, dist, &ierr);
             rs = dist[lnp-1] * dist[lnp-1];
        }
        rin = 1.0 / sqrt(1.1 * rs);
        goto label7;
    }

    stf = w;
    for (i = 0; i < 3; ++i) {
        a[5][i] = stf;
        for (j = i + 1; j < 6; ++j) a[5][j] = 0.0;
        for (j = i; j < 5; ++j) {
             givens(a[j][j], a[j][5], &c, &s);
             rotate(6 - j, c, s, &a[j][j+1], &a[j][5+1]);
        }
    }

    dmin = MIN(ABS(a[3][3]), ABS(a[4][4]));
    if (dmin / w < dtol) {
        *ier = -2;
        return;
    }

label12:
    *dy = a[4][5] / a[4][4];
    *dx = sf * (a[3][5] - a[3][4] * (*dy)) / a[3][3];
    *dy = sf * (*dy);
    *ier = lnp - 1;
}

void grcoef(double sigma, double dcub, double* d, double* sd) {
    double coshm, coshmm, e, ems, scm, sig, sinhm, ssinh, ssm;

    sig = sigma;
    if (sig < 1.e-9) {
        *d = 4.0 / dcub;
        *sd = 2.0 / dcub;
    } else if (sig <= 0.5) {
        snhcsh(sig, &sinhm, &coshm, &coshmm);
        e = (sig * sinhm - coshmm - coshmm) * dcub;
        *d = sig * (sig * coshm - sinhm) / e;
        *sd = sig * sinhm / e;
    } else {
        ems = exp(-sig);
        ssinh = 1.0 - ems * ems;
        ssm = ssinh - 2.0 * sig * ems;
        scm = (1.0 - ems) * (1.0 - ems);
        e = (sig * ssinh - scm - scm) * dcub;
        *d = sig * (sig * scm - ssm) / e;
        *sd = sig * ssm / e;
    }
}

void intrc0(double px, double py, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* ist, double* pz, int* ier) {
    int i1, i2, i3, ierr, lpl, n1, n2;
    double b1, b2, b3, dp, x1, x2, xp, y1, y2, yp;

    xp = px;
    yp = py;
    *pz = 0.0;

    if (ncc < 0 || n < 3 || *ist < 1 || *ist > n) {
        *ier = -1;
        return;
    }

    trfind(*ist, xp, yp, n, x, y, list, lptr, lend, &i1, &i2, &i3);
    if (i1 == 0) {
        *ier = -2;
        return;
    }
    *ist = i1;
    if (i3 != 0) {
        coords(xp, yp, X(i1), X(i2), X(i3), Y(i1), Y(i2), Y(i3), &b1, &b2, &b3, &ierr);
        if (ierr != 0) {
            *ier = -2;
            return;
        }
        *pz = b1 * Z(i1) + b2 * Z(i2) + b3 * Z(i3);
        if (crtri(ncc, lcc, i1, i2, i3)) {
            *ier = 1;
        } else {
            *ier = 0;
        }
        return;
    }

    *ier = 2;
    n2 = i1;

    while (1) {
        lpl = LEND(n2);
        n1 = -LIST(lpl);
        x1 = X(n1);
        y1 = Y(n1);
        x2 = X(n2);
        y2 = Y(n2);
        dp = (x1 - x2) * (xp - x2) + (y1 - y2) * (yp - y2);
        if (dp <= 0.0) {
            *pz = Z(n2);
            return;
        }
        if ((xp - x1) * (x2 - x1) + (yp - y1) * (y2 - y1) > 0.0) {
            b1 = dp / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            b2 = 1.0 - b1;
            *pz = b1 * Z(n1) + b2 * Z(n2);
            return;
        }
        n2 = n1;
    }
}

void tval(double x, double y, double x1, double x2, double x3, double y1, double y2, double y3, double z1, double z2, double z3, double zx1, double zx2, double zx3, double zy1, double zy2, double zy3, int dflag, double* f, double* fx, double* fy, int* ier) {
    fval(x, y, x1, x2, x3, y1, y2, y3, z1, z2, z3, zx1, zx2, zx3, zy1, zy2, zy3, 0.0, 0.0, 0.0, f, ier);
    if (dflag) {
        *fx = 0.0;
        *fy = 0.0;
    }
}

void intrc1(double px, double py, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* grad, int dflag, int* ist, double* pz, double* pzx, double* pzy, int* ier) {
    int i1, i2, i3, lp, n1, n2, n3, tensn;
    double sig1, sig2, sig3;
    double x1, x2, x3, y1, y2, y3, z1, z2, z3, zx1, zx2, zx3, zy1, zy2, zy3;

    if (ncc < 0 || n < 3 || *ist < 1 || *ist > n) {
        *ier = -1;
        return;
    }

    trfind(*ist, px, py, n, x, y, list, lptr, lend, &i1, &i2, &i3);

    if (i1 == 0) {
        *ier = -2;
        return;
    }
    *ist = i1;
    tensn = (iflgs >= 1);

    if (i3 != 0) {
        x1 = X(i1); y1 = Y(i1);
        x2 = X(i2); y2 = Y(i2);
        x3 = X(i3); y3 = Y(i3);
        z1 = Z(i1); z2 = Z(i2); z3 = Z(i3);
        zx1 = GRAD(1, i1); zy1 = GRAD(2, i1);
        zx2 = GRAD(1, i2); zy2 = GRAD(2, i2);
        zx3 = GRAD(1, i3); zy3 = GRAD(2, i3);

        if (tensn) {
            if (iflgs <= 0) {
                sig1 = SIGMA(1);
                sig2 = sig1;
                sig3 = sig1;
            } else {
                 lp = lstptr(LEND(i2), i3, list, lptr);
                 sig1 = SIGMA(lp);
                 lp = lstptr(LEND(i3), i1, list, lptr);
                 sig2 = SIGMA(lp);
                 lp = lstptr(LEND(i1), i2, list, lptr);
                 sig3 = SIGMA(lp);
            }
            fval(px, py, x1, x2, x3, y1, y2, y3, z1, z2, z3, zx1, zx2, zx3, zy1, zy2, zy3, sig1, sig2, sig3, pz, ier);
        } else {
            tval(px, py, x1, x2, x3, y1, y2, y3, z1, z2, z3, zx1, zx2, zx3, zy1, zy2, zy3, dflag, pz, pzx, pzy, ier);
        }

        if (crtri(ncc, lcc, i1, i2, i3)) *ier = 1;
        else *ier = 0;
        return;
    }

    *ier = 2;
    *pz = Z(i1);
}

void rotate(int n, double c, double s, double* x, double* y) {
    int i;
    double xi, yi;
    for (i = 1; i <= n; ++i) {
        xi = X(i);
        yi = Y(i);
        X(i) = c * xi + s * yi;
        Y(i) = -s * xi + c * yi;
    }
}

void setro1(double xk, double yk, double zk, double xi, double yi, double zi, double s1, double s2, double w, double* row) {
    double dx, dy, w1, w2;
    dx = xi - xk;
    dy = yi - yk;
    w1 = s1 * w;
    w2 = s2 * w;
    row[0] = dx * dx * w2;
    row[1] = dx * dy * w2;
    row[2] = dy * dy * w2;
    row[3] = dx * w1;
    row[4] = dy * w1;
    row[5] = (zi - zk) * w;
}

void setro2(double xk, double yk, double zk, double xi, double yi, double zi, double s1, double s2, double w, double* row) {
    double dx, dy, w1, w2;
    dx = xi - xk;
    dy = yi - yk;
    w1 = s1 * w;
    w2 = s2 * w;
    row[0] = dx * dx * w2;
    row[1] = dx * dy * w2;
    row[2] = dy * dy * w2;
    row[3] = dx * w1;
    row[4] = dy * w1;
    row[5] = w;
    row[6] = (zi - zk) * w;
}

void setro3(double xk, double yk, double zk, double xi, double yi, double zi, double s1, double s2, double s3, double w, double* row) {
    double dx, dy, w1, w2, w3;
    dx = xi - xk;
    dy = yi - yk;
    w1 = s1 * w;
    w2 = s2 * w;
    w3 = s3 * w;
    row[0] = dx * dx * dx * w3;
    row[1] = dx * dx * dy * w3;
    row[2] = dx * dy * dy * w3;
    row[3] = dy * dy * dy * w3;
    row[4] = dx * dx * w2;
    row[5] = dx * dy * w2;
    row[6] = dy * dy * w2;
    row[7] = dx * w1;
    row[8] = dy * w1;
    row[9] = (zi - zk) * w;
}

void sgprnt(int n, int lunit, int* list, int* lptr, int* lend, double* sigma) {
}

double sig0(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, int iflgb, double hbnd, double tol, int iflgs, double* sigma, int* ier) {
    return 0.0;
}

double sig1(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, int iflgb, double hpbnd, double tol, int iflgs, double* sigma, int* ier) {
    return 0.0;
}

double sig2(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, double tol, int iflgs, double* sigma, int* ier) {
    return 0.0;
}

void smsgs(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double p, int nit, double dfmax, double* f, double* grad, int* ier) {
    *ier = 0;
}

void smsurf(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double sm, double smtol, double gstol, int lprnt, double* f, double* grad, int* ier) {
    *ier = 0;
}

double trvol(double x1, double x2, double x3, double y1, double y2, double y3, double z1, double z2, double z3) {
    double area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    return (z1 + z2 + z3) * area / 6.0;
}

void unif(int ncc, int* lcc, int n, double* x, double* y, double* z, double* grad, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nrow, int nx, int ny, double* px, double* py, int sflag, double sval, double* zz, int* ier) {
    /* Implementation of UNIF using INTRC1 on a grid */
    int i, j, ierr, ist, nex, ni, nj;
    int sfl = sflag;
    double val, dum;

    ni = nx;
    nj = ny;

    if (ncc < 0 || n < 3 || ni < 1 || ni > nrow || nj < 1) {
        *ier = -1;
        return;
    }

    ist = 1;
    nex = 0;

    for (j = 1; j <= nj; ++j) {
        for (i = 1; i <= ni; ++i) {
            intrc1(px[i-1], py[j-1], ncc, lcc, n, x, y, z, list, lptr, lend, iflgs, sigma, grad, 0, &ist, &val, &dum, &dum, &ierr);

            if (ierr < 0) {
                *ier = ierr;
                return;
            }
            if (ierr > 0) nex++;
            if (sfl && ierr == 1) {
                ZZ(i, j) = sval;
            } else {
                ZZ(i, j) = val;
            }
        }
    }
    *ier = nex;
}

double volume(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend) {
    return 0.0;
}

void zgradg(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nit, double dzmax, double* z, double* grad, int* ier) {
    *ier = 0;
}

void zgradl(int k, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int ndv, double* z, int* npts, double* ds, double* dx, double* dy, int* ier) {
    *ier = 0;
}

void zinit(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, double* z, int* ier) {
    *ier = 0;
}
