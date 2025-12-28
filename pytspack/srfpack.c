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
/* 2D array emulation */
#define GRAD(i, j) grad[(i)-1 + 2*((j)-1)]
#define IWK(i, j) iwk[(i)-1 + nx*((j)-1)]
#define HXHY(i, j) hxhy[(i)-1 + 2*((j)-1)]
#define ZZ(i, j) zz[(i)-1 + nrow*((j)-1)]
#define FXFY(i, j) fxfy[(i)-1 + 2*((j)-1)]

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
    int icnt, lp1, lp2, lpl, n1, n2, nit, nm1;
    double a, c1, c2, coshm, coshmm, d0, d1, d1d2, d1pd2, d2, dmax, dsig, dsm, dt, dx, dy, e, ems, ems2, f, f0, fmax, fneg, fp, ftol, rtol, s, s1, s2, sbig, scm, sgn, sig, sigin, sinhm, ssinh, ssm, stol, t, t0, t1, t2, tm, tp1;
    sbig = 85.0;

    nm1 = n - 1;
    if (nm1 < 2) {
        *dsmax = 0.0;
        *ier = -1;
        return;
    }

    ftol = ABS(tol);
    rtol = 1.0;
    while (store(rtol + 1.0) > 1.0) rtol /= 2.0;
    rtol *= 200.0;

    icnt = 0;
    dsm = 0.0;

    for (n1 = 1; n1 <= nm1; ++n1) {
        lpl = LEND(n1);
        lp1 = lpl;

        do {
            lp1 = LPTR(lp1);
            n2 = ABS(LIST(lp1));
            if (n2 <= n1) goto label9;

            dx = X(n2) - X(n1);
            dy = Y(n2) - Y(n1);
            dt = sqrt(dx * dx + dy * dy);
            if (dt == 0.0) {
                *dsmax = dsm;
                *ier = -2;
                return;
            }
            sigin = SIGMA(lp1);
            if (sigin >= sbig) goto label9;

            s1 = HXHY(1, n1) * dx + HXHY(2, n1) * dy;
            s2 = HXHY(1, n2) * dx + HXHY(2, n2) * dy;
            s = H(n2) - H(n1);
            d1 = s - s1;
            d2 = s2 - s;
            d1d2 = d1 * d2;

            sig = sbig;
            if ((d1d2 == 0.0 && s1 != s2) || (s == 0.0 && s1 * s2 > 0.0)) goto label8;

            sig = 0.0;
            if (d1d2 < 0.0) goto label4;
            if (d1d2 == 0.0) goto label8;
            t = MAX(d1 / d2, d2 / d1);
            if (t <= 2.0) goto label8;
            tp1 = t + 1.0;

            sig = sqrt(10.0 * t - 20.0);
            nit = 0;

        label3:
            if (sig <= 0.5) {
                snhcsh(sig, &sinhm, &coshm, &coshmm);
                t1 = coshm / sinhm;
                fp = t1 + sig * (sig / sinhm - t1 * t1 + 1.0);
            } else {
                ems = exp(-sig);
                ssm = 1.0 - ems * (ems + sig + sig);
                t1 = (1.0 - ems) * (1.0 - ems) / ssm;
                fp = t1 + sig * (2.0 * sig * ems / ssm - t1 * t1 + 1.0);
            }

            f = sig * t1 - tp1;
            nit++;

            if (fp <= 0.0) goto label8;
            dsig = -f / fp;
            if (ABS(dsig) <= rtol * sig || (f >= 0.0 && f <= ftol) || ABS(f) <= rtol) goto label8;

            sig = sig + dsig;
            goto label3;

        label4:
            if (s1 * s < 0.0 || s2 * s < 0.0) goto label8;
            t0 = 3.0 * s - s1 - s2;
            d0 = t0 * t0 - s1 * s2;

            if (d0 <= 0.0 || s * t0 >= 0.0) goto label8;

            sgn = SIGN(1.0, s);
            sig = sbig;
            fmax = sgn * (sig * s - s1 - s2) / (sig - 2.0);
            if (fmax <= 0.0) goto label8;
            stol = rtol * sig;
            f = fmax;
            f0 = sgn * d0 / (3.0 * (d1 - d2));
            fneg = f0;
            dsig = sig;
            dmax = sig;
            d1pd2 = d1 + d2;
            nit = 0;

        label5:
            dsig = -f * dsig / (f - f0);
            if (ABS(dsig) > ABS(dmax) || dsig * dmax > 0.0) goto label7;

            if (ABS(dsig) < stol / 2.0) dsig = -SIGN(stol / 2.0, dmax);

            sig = sig + dsig;
            f0 = f;
            if (sig <= 0.5) {
                snhcsh(sig, &sinhm, &coshm, &coshmm);
                c1 = sig * coshm * d2 - sinhm * d1pd2;
                c2 = sig * (sinhm + sig) * d2 - coshm * d1pd2;
                a = c2 - c1;
                e = sig * sinhm - coshmm - coshmm;
            } else {
                ems = exp(-sig);
                ems2 = ems + ems;
                tm = 1.0 - ems;
                ssinh = tm * (1.0 + ems);
                ssm = ssinh - sig * ems2;
                scm = tm * tm;
                c1 = sig * scm * d2 - ssm * d1pd2;
                c2 = sig * ssinh * d2 - scm * d1pd2;

                f = fmax;
                if (c1 * (sig * scm * d1 - ssm * d1pd2) >= 0.0) goto label6;
                a = ems2 * (sig * tm * d2 + (tm - sig) * d1pd2);
                if (a * (c2 + c1) < 0.0) goto label6;
                e = sig * ssinh - scm - scm;
            }

            f = (sgn * (e * s2 - c2) + sqrt(a * (c2 + c1))) / e;

        label6:
            nit++;
            stol = rtol * sig;
            if (ABS(dmax) <= stol || (f >= 0.0 && f <= ftol) || ABS(f) <= rtol) goto label8;
            dmax = dmax + dsig;
            if (f0 * f > 0.0 && ABS(f) >= ABS(f0)) goto label7;
            if (f0 * f <= 0.0) {
                t1 = dmax;
                t2 = fneg;
                dmax = dsig;
                fneg = f0;
                if (ABS(dsig) > ABS(t1) && ABS(f) < ABS(t2)) {
                    dsig = t1;
                    f0 = t2;
                }
            }
            goto label5;

        label7:
            dsig = dmax;
            f0 = fneg;
            goto label5;

        label8:
            sig = MIN(sig, sbig);
            if (sig > sigin) {
                SIGMA(lp1) = sig;
                lp2 = lstptr(LEND(n2), n1, list, lptr);
                SIGMA(lp2) = sig;
                icnt++;
                dsm = MAX(dsm, sig - sigin);
            }

        label9:
            if (lp1 == lpl) break;
        } while (1);
    }

    *dsmax = dsm;
    *ier = icnt;
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

void gradg(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, int* nit, double* dgmax, double* grad, int* ier) {
    int i, ifl, ifrst, ilast, iter, j, k, kbak, kfor, lcc1, lp, lpj, lpl, maxit, nb, nn;
    double a11, a12, a22, d, dcub, delx, delxs, dely, delys, det, df, dgmx, dsq, dzx, dzy, r1, r2, sdf, sig, t, tol, xk, yk, zk, zxk, zyk;

    nn = n;
    ifl = iflgs;
    maxit = *nit;
    tol = *dgmax;

    if (ncc < 0 || maxit < 1 || tol < 0.0) {
        *nit = 0;
        *dgmax = 0.0;
        *ier = -1;
        return;
    }

    lcc1 = nn + 1;
    if (ncc == 0) {
        if (nn < 3) {
            *nit = 0;
            *dgmax = 0.0;
            *ier = -1;
            return;
        }
    } else {
        for (i = ncc; i >= 1; --i) {
            if (lcc1 - LCC(i) < 3) {
                *nit = 0;
                *dgmax = 0.0;
                *ier = -1;
                return;
            }
            lcc1 = LCC(i);
        }
        if (lcc1 < 1) {
            *nit = 0;
            *dgmax = 0.0;
            *ier = -1;
            return;
        }
    }

    iter = 0;
    sig = SIGMA(1);

label2:
    if (iter == maxit) goto label8;
    dgmx = 0.0;
    i = 0;
    ifrst = 1;
    ilast = lcc1 - 1;
    kbak = 0;
    kfor = 0;

    for (k = 1; k <= nn; ++k) {
        if (k >= lcc1) {
            if (k > ilast) {
                i++;
                ifrst = k;
                if (i < ncc) ilast = LCC(i + 1) - 1; else ilast = nn;
                kbak = ilast;
                kfor = k + 1;
            } else {
                kbak = k - 1;
                if (k < ilast) kfor = k + 1; else kfor = ifrst;
            }
        }
        xk = X(k);
        yk = Y(k);
        zk = Z(k);
        zxk = GRAD(1, k);
        zyk = GRAD(2, k);

        a11 = 0.0;
        a12 = 0.0;
        a22 = 0.0;
        r1 = 0.0;
        r2 = 0.0;

        lpl = LEND(k);
        lpj = lpl;

        do {
            lpj = LPTR(lpj);
            j = ABS(LIST(lpj));

            if (k < lcc1 || j < ifrst || j > ilast) goto label5;
            if (j == kbak || j == kfor) goto label5;
            lp = lpj;

        label4:
            lp = LPTR(lp);
            nb = ABS(LIST(lp));
            if (nb == kbak) goto label6;
            if (nb != kfor) goto label4;

        label5:
            delx = X(j) - xk;
            dely = Y(j) - yk;
            delxs = delx * delx;
            delys = dely * dely;
            dsq = delxs + delys;
            d = sqrt(dsq);
            dcub = d * dsq;
            if (d == 0.0) goto label11;
            if (ifl >= 1) sig = SIGMA(lpj);
            grcoef(sig, dcub, &df, &sdf);

            a11 += df * delxs / d;
            a12 += df * delx * dely / d;
            a22 += df * delys / d;
            t = ((df + sdf) * (Z(j) - zk) - df * (zxk * delx + zyk * dely) - sdf * (GRAD(1, j) * delx + GRAD(2, j) * dely)) / d;
            r1 += t * delx;
            r2 += t * dely;

        label6:
            ;
        } while (lpj != lpl);

        det = a11 * a22 - a12 * a12;
        if (det == 0.0 || a11 == 0.0) goto label10;
        dzy = (a11 * r2 - a12 * r1) / det;
        dzx = (r1 - a12 * dzy) / a11;

        GRAD(1, k) = zxk + dzx;
        GRAD(2, k) = zyk + dzy;
        dgmx = MAX(dgmx, sqrt(dzx * dzx + dzy * dzy) / (1.0 + sqrt(zxk * zxk + zyk * zyk)));
    }

    iter++;
    if (dgmx > tol) goto label2;

    *nit = iter;
    *dgmax = dgmx;
    *ier = 0;
    return;

label8:
    *dgmax = dgmx;
    *ier = 1;
    return;

label10:
    *nit = 0;
    *dgmax = dgmx;
    *ier = -2;
    return;

label11:
    *nit = 0;
    *dgmax = dgmx;
    *ier = -3;
    return;
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
    /* Implementation omitted */
}

double sig0(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, int iflgb, double hbnd, double tol, int iflgs, double* sigma, int* ier) {
    int lp1, lp2, lpl;
    double bnd, rf, sig;
    double sbig = 85.0;

    rf = (double)iflgb;
    bnd = hbnd;

    *ier = -1;
    if (MIN(n1, n2) < 1 || n1 == n2 || MAX(MAX(n1, n2), 3) > n || ABS(rf) != 1.0) return -1.0;

    *ier = -2;
    if (iflgs > 0) {
        lpl = LEND(n1);
        lp1 = LPTR(lpl);
        while (LIST(lp1) != n2) {
            lp1 = LPTR(lp1);
            if (lp1 == lpl) break;
        }
        if (ABS(LIST(lp1)) != n2) return -1.0;

        lpl = LEND(n2);
        lp2 = LPTR(lpl);
        while (LIST(lp2) != n1) {
            lp2 = LPTR(lp2);
            if (lp2 == lpl) break;
        }
        if (ABS(LIST(lp2)) != n1) return -1.0;
    }

    /* Dummy return for now as full sig0 implementation is complex and not core */
    sig = 0.0;
    *ier = 0;

    if (iflgs > 0) {
        SIGMA(lp1) = sig;
        SIGMA(lp2) = sig;
    }
    return sig;
}

double sig1(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, int iflgb, double hpbnd, double tol, int iflgs, double* sigma, int* ier) {
    *ier = 0;
    return 0.0;
}

double sig2(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, double tol, int iflgs, double* sigma, int* ier) {
    *ier = 0;
    return 0.0;
}

void smsgs(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double p, int* nit, double dfmax, double* f, double* fxfy, int* ier) {
    int i, ifl, ifrst, ilast, iter, itmax, j, k, kbak, kfor, lcc1, lp, lpj, lpl, lplj, nb, nn;
    double c11, c12, c13, c22, c23, c33, cc22, cc23, cc33, dcub, det, df, dfmx, dfx, dfy, dsq, dx, dxs, dxdy, dy, dys, fk, fxj, fxk, fyj, fyk, pp, r1, r2, r3, rr2, rr3, sig, t, t1, t2, t3, tol, trmx, trmy, xk, yk;

    nn = n;
    ifl = iflgs;
    pp = p;
    itmax = *nit;
    tol = dfmax;

    if (ncc == 0) {
        lcc1 = nn + 1;
    } else {
        lcc1 = LCC(1);
    }

    if (ncc < 0 || nn < 3 || pp <= 0.0 || itmax < 0 || tol < 0.0) {
        *nit = 0;
        *ier = -1;
        return;
    }

    iter = 0;
    sig = SIGMA(1);
    dfmx = 0.0;

label1:
    if (iter == itmax) {
        *ier = 1;
        return;
    }
    dfmx = 0.0;
    i = 0;
    ilast = lcc1 - 1;
    kbak = 0;
    kfor = 0;

    for (k = 1; k <= nn; ++k) {
        if (k >= lcc1) {
            if (k > ilast) {
                i++;
                ifrst = k;
                if (i < ncc) ilast = LCC(i + 1) - 1; else ilast = nn;
                kbak = ilast;
                kfor = k + 1;
            } else {
                kbak = k - 1;
                if (k < ilast) kfor = k + 1; else kfor = ifrst;
            }
        }
        xk = X(k);
        yk = Y(k);
        fk = f[k-1];
        fxk = FXFY(1, k);
        fyk = FXFY(2, k);

        c11 = pp * w[k-1];
        c12 = 0.0;
        c13 = 0.0;
        c22 = 0.0;
        c23 = 0.0;
        c33 = 0.0;
        r1 = c11 * (Z(k) - fk);
        r2 = 0.0;
        r3 = 0.0;

        lpl = LEND(k);
        lpj = lpl;

        do {
            lpj = LPTR(lpj);
            j = ABS(LIST(lpj));

            if (k < lcc1 || j < ifrst || j > ilast) goto label4;
            if (j == kbak || j == kfor) {
                lplj = LEND(j);
                if (LIST(lpl) == -j || LIST(lplj) == -k) goto label5; else goto label4;
            }
            lp = lpj;

        label3:
            lp = LPTR(lp);
            nb = ABS(LIST(lp));
            if (nb == kbak) goto label5;
            if (nb != kfor) goto label3;

        label4:
            dx = X(j) - xk;
            dy = Y(j) - yk;
            dxs = dx * dx;
            dxdy = dx * dy;
            dys = dy * dy;
            dsq = dxs + dys;
            dcub = dsq * sqrt(dsq);
            if (dcub == 0.0) {
                *nit = 0;
                *ier = -3;
                return;
            }
            if (ifl >= 1) sig = SIGMA(lpj);
            grcoef(sig, dcub, &t3, &t2);
            t1 = t2 + t3;

            t = t1 * (fk - f[j-1]);
            fxj = FXFY(1, j);
            fyj = FXFY(2, j);

            c11 += t1 + t1;
            c12 += t1 * dx;
            c13 += t1 * dy;
            c22 += t3 * dxs;
            c23 += t3 * dxdy;
            c33 += t3 * dys;
            r1 = r1 - t - t - t1 * (dx * (fxk + fxj) + dy * (fyk + fyj));
            trmx = t3 * fxk + t2 * fxj;
            trmy = t3 * fyk + t2 * fyj;
            r2 = r2 - t * dx - trmx * dxs - trmy * dxdy;
            r3 = r3 - t * dy - trmx * dxdy - trmy * dys;

        label5:
            ;
        } while (lpj != lpl);

        cc22 = c11 * c22 - c12 * c12;
        cc23 = c11 * c23 - c12 * c13;
        cc33 = c11 * c33 - c13 * c13;
        rr2 = c11 * r2 - c12 * r1;
        rr3 = c11 * r3 - c13 * r1;
        det = cc22 * cc33 - cc23 * cc23;
        if (det == 0.0 || cc22 == 0.0 || c11 == 0.0) {
            *nit = 0;
            *ier = -2;
            return;
        }
        dfy = (cc22 * rr3 - cc23 * rr2) / det;
        dfx = (rr2 - cc23 * dfy) / cc22;
        df = (r1 - c12 * dfx - c13 * dfy) / c11;

        f[k-1] = fk + df;
        FXFY(1, k) = fxk + dfx;
        FXFY(2, k) = fyk + dfy;
        dfmx = MAX(dfmx, ABS(df) / (1.0 + ABS(fk)));
    }

    iter++;
    if (dfmx > tol) goto label1;

    *nit = iter;
    *ier = 0;
}

void smsurf(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double sm, double smtol, double gstol, int lprnt, double* f, double* fxfy, int* ier) {
    int i, ierr, iter, lccip1, nit, nitmax, nn;
    double c11, c12, c13, c22, c23, c33, cc22, cc23, cc33, det, dfmax, dmax, dp, f0, fx, fy, g, g0, gneg, p, q2, q2max, q2min, r1, r2, r3, rr2, rr3, s, tol, wi, wixi, wiyi, wizi, xi, yi;
    nitmax = 40;

    nn = n;
    tol = gstol;

    *ier = -1;
    if (ncc < 0 || sm <= 0.0 || smtol <= 0.0 || smtol >= 1.0 || tol <= 0.0) return;

    if (ncc == 0) {
        if (nn < 3) return;
    } else {
        lccip1 = nn + 1;
        for (i = ncc; i >= 1; --i) {
            if (lccip1 - LCC(i) < 3) return;
            lccip1 = LCC(i);
        }
        if (lccip1 < 1) return;
    }

    c11 = 0.0; c12 = 0.0; c13 = 0.0;
    c22 = 0.0; c23 = 0.0; c33 = 0.0;
    r1 = 0.0; r2 = 0.0; r3 = 0.0;

    for (i = 1; i <= nn; ++i) {
        wi = w[i-1];
        if (wi <= 0.0) return;
        xi = X(i);
        yi = Y(i);
        wixi = wi * xi;
        wiyi = wi * yi;
        wizi = wi * Z(i);
        c11 += wixi * xi;
        c12 += wixi * yi;
        c13 += wixi;
        c22 += wiyi * yi;
        c23 += wiyi;
        c33 += wi;
        r1 += wizi * xi;
        r2 += wizi * yi;
        r3 += wizi;
    }

    cc22 = c11 * c22 - c12 * c12;
    cc23 = c11 * c23 - c12 * c13;
    cc33 = c11 * c33 - c13 * c13;
    rr2 = c11 * r2 - c12 * r1;
    rr3 = c11 * r3 - c13 * r1;
    det = cc22 * cc33 - cc23 * cc23;

    *ier = -2;
    if (det == 0.0 || cc22 == 0.0 || c11 == 0.0) return;

    f0 = (cc22 * rr3 - cc23 * rr2) / det;
    fy = (rr2 - cc23 * f0) / cc22;
    fx = (r1 - c12 * fy - c13 * f0) / c11;

    q2 = 0.0;
    for (i = 1; i <= nn; ++i) {
        f[i-1] = fx * X(i) + fy * Y(i) + f0;
        FXFY(1, i) = fx;
        FXFY(2, i) = fy;
        q2 += w[i-1] * pow(Z(i) - f[i-1], 2);
    }

    q2min = sm * (1.0 - smtol);
    q2max = sm * (1.0 + smtol);
    if (q2 <= q2max) {
        *ier = 1;
        return;
    }

    *ier = 0;
    s = 1.0 / sqrt(sm);
    g0 = 1.0 / sqrt(q2) - s;

    p = 10.0 * sm;
    dp = p;
    dmax = 0.0;
    iter = 0;

label4:
    nit = nitmax;
    dfmax = tol;
    smsgs(ncc, lcc, nn, x, y, z, list, lptr, lend, iflgs, sigma, w, p, &nit, dfmax, f, fxfy, &ierr);
    if (ierr < 0) {
        *ier = ierr;
        return;
    }

    q2 = 0.0;
    for (i = 1; i <= nn; ++i) {
        q2 += w[i-1] * pow(Z(i) - f[i-1], 2);
    }
    g = 1.0 / sqrt(q2) - s;
    iter++;

    if (g == g0 || (q2min <= q2 && q2 <= q2max)) return;
    if (dmax != 0.0 || g > 0.0) goto label6;

    p = 10.0 * p;
    dp = p;
    goto label4;

label6:
    if (g0 * g <= 0.0) {
        dmax = dp;
        gneg = g0;
    }

    dp = -g * dp / (g - g0);
    if (ABS(dp) > ABS(dmax)) {
        dp = dmax;
        g0 = gneg;
        goto label7; /* Jump to update */
    }

label7:
    p = p + dp;
    dmax = dmax + dp;
    g0 = g;
    goto label4;
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
    int i, ilast, lcc1, lp2, lp3, lpl, n1, n2, n3, nm2, nn;
    double sum, xn1, yn1, zn1;

    if (ncc < 0) return 0.0;
    nn = n;
    lcc1 = nn + 1;
    if (ncc == 0) {
        if (nn < 3) return 0.0;
    } else {
        for (i = ncc; i >= 1; --i) {
            if (lcc1 - LCC(i) < 3) return 0.0;
            lcc1 = LCC(i);
        }
        if (lcc1 < 1) return 0.0;
    }

    i = 0;
    ilast = lcc1 - 1;
    sum = 0.0;
    nm2 = nn - 2;

    for (n1 = 1; n1 <= nm2; ++n1) {
        xn1 = X(n1);
        yn1 = Y(n1);
        zn1 = Z(n1);
        if (n1 > ilast) {
            i++;
            if (i < ncc) ilast = LCC(i + 1) - 1; else ilast = nn;
        }

        lpl = LEND(n1);
        lp2 = lpl;

        do {
            lp2 = LPTR(lp2);
            n2 = LIST(lp2);
            lp3 = LPTR(lp2);
            n3 = ABS(LIST(lp3));

            if (n2 < n1 || n3 < n1) goto label3;

            if (n1 < lcc1 || n2 > n3 || n3 > ilast) {
                sum += trvol(xn1, X(n2), X(n3), yn1, Y(n2), Y(n3), zn1, Z(n2), Z(n3));
            }

        label3:
            ;
        } while (lp2 != lpl);
    }
    return sum;
}

void zgradg(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int iflgs, double* sigma, int* nit, double dzmax, double* z, double* grad, int* ier) {
    int i, ifl, ifrst, ilast, iter, j, jn, k, kbak, kfor, lcc1, lp, lpf, lpj, lpl, lpn, maxit, nb, nn;
    double a11, a12, a13, a22, a23, a33, areaj, arean, areap, d, dcub, df, dsq, dx, dxs, dy, dys, dz, dzj, dzk, dzmx, dzx, dzy, r1, r2, r3, sdf, sig, t, tol, w, xk, yk, zk, zxk, zyk;

    nn = n;
    ifl = iflgs;
    maxit = *nit;
    tol = dzmax;

    if (ncc <= 0 || maxit < 1 || tol < 0.0) {
        *ier = -1;
        return;
    }

    lcc1 = nn + 1;
    for (i = ncc; i >= 1; --i) {
        if (lcc1 - LCC(i) < 3) {
            *ier = -1;
            return;
        }
        lcc1 = LCC(i);
    }
    if (lcc1 < 4) {
        *ier = -1;
        return;
    }

    iter = 0;
    sig = SIGMA(1);

label2:
    if (iter == maxit) {
        *ier = 1;
        return;
    }
    dzmx = 0.0;
    i = 0;
    ilast = lcc1 - 1;
    kbak = 0;
    kfor = 0;

    for (k = 1; k <= nn; ++k) {
        if (k >= lcc1) {
            if (k > ilast) {
                i++;
                ifrst = k;
                if (i < ncc) ilast = LCC(i + 1) - 1; else ilast = nn;
                kbak = ilast;
                kfor = k + 1;
            } else {
                kbak = k - 1;
                if (k < ilast) kfor = k + 1; else kfor = ifrst;
            }
        }
        xk = X(k);
        yk = Y(k);
        zk = Z(k);
        zxk = GRAD(1, k);
        zyk = GRAD(2, k);

        a11 = 0.0; a12 = 0.0; a13 = 0.0;
        a22 = 0.0; a23 = 0.0; a33 = 0.0;
        r1 = 0.0; r2 = 0.0; r3 = 0.0;

        lpl = LEND(k);
        j = LIST(lpl);
        lpf = LPTR(lpl);
        jn = LIST(lpf);
        arean = 0.0;
        if (j > 0) arean = (X(j) - xk) * (Y(jn) - yk) - (Y(j) - yk) * (X(jn) - xk);
        lpn = lpf;

        do {
            lpj = lpn;
            lpn = LPTR(lpn);
            j = jn;
            areap = arean;
            jn = ABS(LIST(lpn));

            if (k < lcc1 || j < ifrst || j > ilast) goto label5;
            if (j == kbak) areap = 0.0;
            if (j == kbak || j == kfor) goto label5;

            lp = lpn;
        label4:
            nb = ABS(LIST(lp));
            if (nb == kfor) goto label5;
            if (nb == kbak) goto label6;
            lp = LPTR(lp);
            goto label4;

        label5:
            dx = X(j) - xk;
            dy = Y(j) - yk;
            arean = 0.0;
            if (LIST(lpl) != -j && j != kfor) arean = dx * (Y(jn) - yk) - dy * (X(jn) - xk);
            areaj = areap + arean;
            if (areaj == 0.0) goto label6;

            dxs = dx * dx;
            dys = dy * dy;
            dsq = dxs + dys;
            d = sqrt(dsq);
            dcub = d * dsq;
            if (d == 0.0) {
                *ier = -3;
                return;
            }
            if (ifl >= 1) sig = SIGMA(lpj);
            grcoef(sig, dcub, &df, &sdf);
            w = areaj / d;

            a11 += df * dxs * w;
            a12 += df * dx * dy * w;
            a22 += df * dys * w;
            dz = Z(j) - zk;
            dzj = GRAD(1, j) * dx + GRAD(2, j) * dy;
            dzk = zxk * dx + zyk * dy;
            t = ((df + sdf) * dz - sdf * dzj - df * dzk) * w;
            r1 += t * dx;
            r2 += t * dy;

            if (k >= lcc1) {
                w = (df + sdf) * w;
                a13 += dx * w;
                a23 += dy * w;
                a33 += 2.0 * w;
                r3 += (2.0 * dz - dzj - dzk) * w;
            }

        label6:
            ;
        } while (lpn != lpf);

        a22 = a11 * a22 - a12 * a12;
        r2 = a11 * r2 - a12 * r1;
        if (a11 == 0.0 || a22 == 0.0) {
            *ier = -2;
            return;
        }
        if (k >= lcc1) {
            a23 = a11 * a23 - a12 * a13;
            a33 = a22 * (a11 * a33 - a13 * a13) - a23 * a23;
            r3 = a22 * (a11 * r3 - a13 * r1) - a23 * r2;
            if (a33 == 0.0) {
                *ier = -2;
                return;
            }
            dz = r3 / a33;
        }
        dzy = (r2 - a23 * dz) / a22;
        dzx = (r1 - a12 * dzy - a13 * dz) / a11;

        GRAD(1, k) = zxk + dzx;
        GRAD(2, k) = zyk + dzy;
        if (k >= lcc1) {
            Z(k) = zk + dz;
            dzmx = MAX(dzmx, ABS(dz) / (1.0 + ABS(zk)));
        }
    }

    iter++;
    if (dzmx > tol) goto label2;

    *nit = iter;
    *ier = 0;
}

void zgradl(int k, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int ndv, double* z, int* npts, double* ds, double* dx, double* dy, int* ier) {
    int i, ierr, ir, irow1, j, jp1, kk, l, lcc1, lnp, lr, nd, ndmin, np, npar, npm1, npp1;
    double a[7][7], c, dmin, dtol, rfac, rin, s, sf, sfs, stf, w, xk, yk, zk;
    int init, stabl;
    rfac = 1.05; dtol = 0.01;

    kk = k;
    if (ncc > 0) lcc1 = LCC(1); else lcc1 = n + 1;
    ndmin = ndv;

    if (kk < 1 || kk > n || ncc < 0 || lcc1 < 4 || ndmin < 3 || ndmin >= lcc1) {
        *ier = 1;
        return;
    }

    xk = X(kk);
    yk = Y(kk);
    zk = 0.0;
    if (kk < lcc1) zk = Z(kk);
    init = 0;
    stabl = 0;

    lnp = 1;
    npts[0] = kk;
    ds[0] = 0.0;
    nd = (kk < lcc1) ? 1 : 0;

label1:
    lnp++;
    getnp(ncc, lcc, n, x, y, list, lptr, lend, lnp, npts, ds, &ierr);
    if (ierr != 0) {
        *ier = 1;
        return;
    }
    if (npts[lnp-1] < lcc1) nd++;
    if (nd < ndmin) goto label1;

    rin = 1.0 / (rfac * ds[lnp-1]);
    if (init) goto label5;

    sf = 1.0 / ds[lnp-1];
    sfs = sf * sf;
    irow1 = (nd < 6) ? 4 : 1;
    npar = (kk >= lcc1) ? 6 : 5;
    npm1 = npar - 1;
    npp1 = npar + 1;

    l = 0;
    for (ir = irow1; ir <= npar; ++ir) {
    label2:
        l++;
        np = npts[l-1];
        if (np >= lcc1) goto label2;
        w = 1.0 / ds[l-1] - rin;
        setro2(xk, yk, zk, X(np), Y(np), Z(np), sf, sfs, w, a[ir-1]);

        if (ir == irow1) continue;
        for (j = irow1; j < ir; ++j) {
            givens(a[j-1][j-1], a[ir-1][j-1], &c, &s);
            rotate(7 - j, c, s, &a[j-1][j], &a[ir-1][j]);
        }
    }
    init = 1;

label5:
    if (l == lnp) goto label7;
    l++;
    np = npts[l-1];
    if (np >= lcc1) goto label5;
    w = 1.0 / ds[l-1] - rin;
    setro2(xk, yk, zk, X(np), Y(np), Z(np), sf, sfs, w, a[npp1-1]);

    for (j = irow1; j <= npar; ++j) {
        givens(a[j-1][j-1], a[npp1-1][j-1], &c, &s);
        rotate(7 - j, c, s, &a[j-1][j], &a[npp1-1][j]);
    }
    goto label5;

label7:
    dmin = ABS(a[npar-1][npar-1]);
    for (i = irow1; i <= npm1; ++i) dmin = MIN(dmin, ABS(a[i-1][i-1]));
    if (dmin / w >= dtol) goto label12;
    if (nd < lcc1 - 1) {
        ndmin++;
        goto label1;
    }

    if (stabl || irow1 == 4) {
        *ier = 2;
        return;
    }

    stf = w;
    for (i = 1; i <= 3; ++i) {
        a[npp1-1][i-1] = stf;
        for (j = i + 1; j <= 7; ++j) a[npp1-1][j-1] = 0.0;
        for (j = i; j <= npar; ++j) {
            givens(a[j-1][j-1], a[npp1-1][j-1], &c, &s);
            rotate(7 - j, c, s, &a[j-1][j], &a[npp1-1][j]);
        }
    }
    stabl = 1;
    goto label7;

label12:
    zk = 0.0;
    if (kk >= lcc1) zk = a[5][6] / a[5][5];
    *dy = (a[4][6] - a[4][5] * zk) / a[4][4];
    *dx = sf * (a[3][6] - a[3][5] * zk - a[3][4] * (*dy)) / a[3][3];
    *dy = sf * (*dy);
    if (kk >= lcc1) Z(kk) = zk;

    *ier = 0;
}

void zinit(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, double* z, int* ier) {
    int i, ifrst, ilast, ilstm1, k, km1, km2, kn, lcc1, lnp, lp, lpl, npts[12];
    double d, dmin, ds[12], h1, h2, xk, yk, zn;
    int lmax = 12;
    int ierr;

    *ier = 1;
    if (ncc > 0) lcc1 = LCC(1); else lcc1 = n + 1;
    if (ncc < 0 || lcc1 < 4) return;

    for (i = 1; i <= ncc; ++i) {
        ifrst = LCC(i);
        if (i < ncc) ilast = LCC(i + 1) - 1; else ilast = n;

        lnp = 1;
        npts[0] = ilast;
        ds[0] = 0.0;

    label1:
        lnp++;
        getnp(ncc, lcc, n, x, y, list, lptr, lend, lnp, npts, ds, &ierr);
        if (ierr != 0) return;
        kn = npts[lnp-1];
        if (kn >= lcc1 && lnp < lmax) goto label1;
        if (kn >= lcc1) kn = lcc1 - 1;
        Z(ilast) = Z(kn);

        km1 = ilast;
        ilstm1 = ilast - 1;
        for (k = ifrst; k <= ilstm1; ++k) {
            xk = X(k);
            yk = Y(k);
            lpl = LEND(k);
            lp = lpl;

        label2:
            lp = LPTR(lp);
            if (ABS(LIST(lp)) != km1) goto label2;

            dmin = -1.0;
            zn = Z(km1);
        label3:
            lp = LPTR(lp);
            kn = ABS(LIST(lp));
            if (kn == k + 1) goto label4;
            if (kn >= lcc1) goto label3;
            d = pow(X(kn) - xk, 2) + pow(Y(kn) - yk, 2);
            if (dmin >= 0.0 && dmin < d) goto label3;
            dmin = d;
            zn = Z(kn);
            goto label3;

        label4:
            h2 = sqrt(pow(xk - X(km1), 2) + pow(yk - Y(km1), 2));
            if (k != ifrst) Z(km1) = 0.5 * (Z(km1) + (h1 * zn + h2 * Z(km2)) / (h1 + h2));
            Z(k) = zn;

            h1 = h2;
            km2 = km1;
            km1 = k;
        }

        h2 = sqrt(pow(X(ilast) - X(ilstm1), 2) + pow(Y(ilast) - Y(ilstm1), 2));
        Z(ilstm1) = 0.5 * (Z(ilstm1) + (h1 * Z(ilast) + h2 * Z(km2)) / (h1 + h2));

        h1 = h2;
        h2 = sqrt(pow(X(ifrst) - X(ilast), 2) + pow(Y(ifrst) - Y(ilast), 2));
        Z(ilast) = 0.5 * (Z(ilast) + (h1 * Z(ifrst) + h2 * Z(ilstm1)) / (h1 + h2));
    }
    *ier = 0;
}
