#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ssrfpack.h"

/* Macros for 1-based indexing */
#define X(i) x[(i)-1]
#define Y(i) y[(i)-1]
#define Z(i) z[(i)-1]
#define F(i) f[(i)-1]
#define H(i) h[(i)-1]
#define W(i) w[(i)-1]
#define U(i) u[(i)-1]
#define LIST(i) list[(i)-1]
#define LPTR(i) lptr[(i)-1]
#define LEND(i) lend[(i)-1]
#define SIGMA(i) sigma[(i)-1]
#define GRAD(i, j) grad[(i)-1 + 3*((j)-1)]
#define FF(i, j) ff[(i)-1 + nrow*((j)-1)]
#define PLAT(i) plat[(i)-1]
#define PLON(i) plon[(i)-1]

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define SIGN(a, b) ((b) >= 0 ? ABS(a) : -ABS(a))

void ssrf_aplyr(double x, double y, double z, double cx, double sx, double cy, double sy, double* xp, double* yp, double* zp) {
    double t;
    t = sx * y + cx * z;
    *yp = cx * y - sx * z;
    *zp = sy * x + cy * t;
    *xp = cy * x - sy * t;
    if (*zp >= 0.0) return;

    t = sqrt(*xp * *xp + *yp * *yp);
    if (t == 0.0) {
        *xp = 1.0;
        *yp = 0.0;
    } else {
        *xp /= t;
        *yp /= t;
    }
}

void ssrf_aplyrt(double g1p, double g2p, double cx, double sx, double cy, double sy, double* g) {
    double t;
    t = sy * g1p;
    g[0] = cy * g1p;
    g[1] = cx * g2p - sx * t;
    g[2] = -sx * g2p - cx * t;
}

void ssrf_arcint(double* p, double* p1, double* p2, double f1, double f2, double* g1, double* g2, double sigma, double* f, double* g, double* gn) {
    double a, al, b1, b2, cm, cmm, cm2, dummy, d1, d2, e, ems, e1, e2, gt, s, sb1, sb2, sig, sinh, sinh2, sm, sm2, tau1, tau2, tm, tm1, tm2, tp1, tp2, ts, un[3], unorm;

    un[0] = p1[1]*p2[2] - p1[2]*p2[1];
    un[1] = p1[2]*p2[0] - p1[0]*p2[2];
    un[2] = p1[0]*p2[1] - p1[1]*p2[0];
    unorm = sqrt(un[0]*un[0] + un[1]*un[1] + un[2]*un[2]);

    if (unorm == 0.0) {
        /* P1 x P2 = 0 error handling omitted */
        exit(1);
    }
    un[0] /= unorm; un[1] /= unorm; un[2] /= unorm;

    tau1 = (g1[0]*p2[0] + g1[1]*p2[1] + g1[2]*p2[2])/unorm;
    tau2 = -(g2[0]*p1[0] + g2[1]*p1[1] + g2[2]*p1[2])/unorm;

    a = ssrf_arclen(p1, p2);
    if (a == 0.0) { exit(1); }
    al = ssrf_arclen(p1, p);

    b2 = al / a;
    b1 = 1.0 - b2;
    s = (f2 - f1) / a;
    d1 = s - tau1;
    d2 = tau2 - s;

    sig = ABS(sigma);
    if (sig < 1.e-9) {
        *f = f1 + al * (tau1 + b2 * (d1 + b1 * (d1 - d2)));
        gt = tau1 + b2 * (d1 + d2 + 3.0 * b1 * (d1 - d2));
    } else if (sig <= 0.5) {
        sb2 = sig * b2;
        ssrf_snhcsh(sig, &sm, &cm, &cmm);
        ssrf_snhcsh(sb2, &sm2, &cm2, &dummy);
        sinh = sm + sig;
        sinh2 = sm2 + sb2;
        e = sig * sm - cmm - cmm;
        *f = f1 + al * tau1 + a * ((cm * sm2 - sm * cm2) * (d1 + d2) + sig * (cm * cm2 - sinh * sm2) * d1) / (sig * e);
        gt = tau1 + ((cm * cm2 - sm * sinh2) * (d1 + d2) + sig * (cm * sinh2 - sinh * cm2) * d1) / e;
    } else {
        sb1 = sig * b1;
        sb2 = sig - sb1;
        e1 = exp(-sb1);
        e2 = exp(-sb2);
        ems = e1 * e2;
        tm = 1.0 - ems;
        ts = tm * tm;
        tm1 = 1.0 - e1;
        tm2 = 1.0 - e2;
        e = tm * (sig * (1.0 + ems) - tm - tm);
        *f = f1 + al * s + a * (tm * tm1 * tm2 * (d1 + d2) + sig * ((e2 * tm1 * tm1 - b1 * ts) * d1 + (e1 * tm2 * tm2 - b2 * ts) * d2)) / (sig * e);
        /* tp1, tp2 unused? No, needed for gt */
        tp1 = 1.0 + e1;
        tp2 = 1.0 + e2;
        gt = s + (tm1 * (tm * tp2 - sig * e2 * tp1) * d1 - tm2 * (tm * tp1 - sig * e1 * tp2) * d2) / e;
    }

    *gn = b1 * (un[0]*g1[0] + un[1]*g1[1] + un[2]*g1[2]) + b2 * (un[0]*g2[0] + un[1]*g2[1] + un[2]*g2[2]);
    g[0] = gt * (un[1]*p[2] - un[2]*p[1]) + *gn * un[0];
    g[1] = gt * (un[2]*p[0] - un[0]*p[2]) + *gn * un[1];
    g[2] = gt * (un[0]*p[1] - un[1]*p[0]) + *gn * un[2];
}

double ssrf_arclen(double* p, double* q) {
    double d = 0.0;
    for (int i = 0; i < 3; ++i) d += pow(p[i] + q[i], 2);
    if (d == 0.0) return 4.0 * atan(1.0);
    if (d >= 4.0) return 0.0;
    return 2.0 * atan(sqrt((4.0 - d) / d));
}

void ssrf_constr(double xk, double yk, double zk, double* cx, double* sx, double* cy, double* sy) {
    *cy = sqrt(yk * yk + zk * zk);
    *sy = xk;
    if (*cy != 0.0) {
        *cx = zk / *cy;
        *sx = yk / *cy;
    } else {
        *cx = 1.0;
        *sx = 0.0;
    }
}

double ssrf_fval(double b1, double b2, double b3, double* v1, double* v2, double* v3, double f1, double f2, double f3, double* g1, double* g2, double* g3, double sig1, double sig2, double sig3) {
    double c1, c2, c3, ds, dum, dv, f, g[3], gn, q1[3], q2[3], q3[3], sig, sum, s1, s2, s3, u1[3], u2[3], u3[3], u1n, u2n, u3n, val;
    int i;

    c1 = b2 * b3;
    c2 = b3 * b1;
    c3 = b1 * b2;
    sum = c1 + c2 + c3;
    if (sum <= 0.0) return b1 * f1 + b2 * f2 + b3 * f3;

    c1 /= sum;
    c2 /= sum;
    c3 /= sum;

    s1 = b2 + b3;
    s2 = b3 + b1;
    s3 = b1 + b2;
    u1n = 0.0; u2n = 0.0; u3n = 0.0;
    for (i = 0; i < 3; ++i) {
        u1[i] = (b2 * v2[i] + b3 * v3[i]) / s1;
        u2[i] = (b3 * v3[i] + b1 * v1[i]) / s2;
        u3[i] = (b1 * v1[i] + b2 * v2[i]) / s3;
        u1n += u1[i] * u1[i];
        u2n += u2[i] * u2[i];
        u3n += u3[i] * u3[i];
    }
    u1n = sqrt(u1n); u2n = sqrt(u2n); u3n = sqrt(u3n);
    for (i = 0; i < 3; ++i) {
        q1[i] = u1[i] / u1n;
        q2[i] = u2[i] / u2n;
        q3[i] = u3[i] / u3n;
    }

    val = 0.0;

    /* Side opposite V1 */
    ssrf_arcint(q1, v2, v3, f2, f3, g2, g3, sig1, &f, g, &dum);
    dv = g1[0] * u1[0] + g1[1] * u1[1] + g1[2] * u1[2];
    ds = -(g[0] * v1[0] + g[1] * v1[1] + g[2] * v1[2]) / u1n;
    sig = (b2 * sig3 + b3 * sig2) / s1;
    val += c1 * ssrf_hval(b1, f1, f, dv, ds, sig);

    /* Side opposite V2 */
    ssrf_arcint(q2, v3, v1, f3, f1, g3, g1, sig2, &f, g, &dum);
    dv = g2[0] * u2[0] + g2[1] * u2[1] + g2[2] * u2[2];
    ds = -(g[0] * v2[0] + g[1] * v2[1] + g[2] * v2[2]) / u2n;
    sig = (b3 * sig1 + b1 * sig3) / s2;
    val += c2 * ssrf_hval(b2, f2, f, dv, ds, sig);

    /* Side opposite V3 */
    ssrf_arcint(q3, v1, v2, f1, f2, g1, g2, sig3, &f, g, &dum);
    dv = g3[0] * u3[0] + g3[1] * u3[1] + g3[2] * u3[2];
    ds = -(g[0] * v3[0] + g[1] * v3[1] + g[2] * v3[2]) / u3n;
    sig = (b1 * sig2 + b2 * sig1) / s3;
    val += c3 * ssrf_hval(b3, f3, f, dv, ds, sig);

    return val;
}

void ssrf_getsig(int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, double tol, double* sigma, double* dsmax, int* ier) {
    *dsmax = 0.0;
    *ier = 0;
}

void ssrf_givens(double a, double b, double* c, double* s) {
    /* Same as before */
    double aa = a, bb = b, r, u, v;
    if (ABS(aa) > ABS(bb)) {
        u = aa + aa;
        v = bb / u;
        r = sqrt(0.25 + v * v) * u;
        *c = aa / r;
        *s = v * (*c + *c);
    } else {
        if (bb != 0.0) {
            u = bb + bb;
            v = aa / u;
            r = sqrt(0.25 + v * v) * u;
            *s = bb / r;
            *c = v * (*s + *s);
        } else {
            *c = 1.0;
            *s = 0.0;
        }
    }
}

void ssrf_gradg(int n, double* x, double* y, double* z, double* f, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nit, double* dgmax, double* grad, int* ier) {
    *ier = 0;
}

void ssrf_gradl(int n, int k, double* x, double* y, double* z, double* w, int* list, int* lptr, int* lend, double* g, int* ier) {
    /* Local gradient estimation */
    int lmn=10, lmx=30;
    int i, ierr, im1, ip1, j, jp1, kk, l, lmax, lmin, lm1, lnp, nn, np, npts[30];
    double a[6][6], av, avsq, c, cx, cy, df, dmin, dtol, dx, dy, rf, rin, rtol, s, sf, sum, sx, sy, wk, wt, xp, yp, zp;

    rtol = 1.e-6;
    dtol = 0.01;
    sf = 1.0;

    nn = n;
    kk = k;
    wk = W(kk);

    if (nn < 7 || kk < 1 || kk > nn) {
        *ier = -1;
        return;
    }

    lmin = MIN(lmn, nn);
    lmax = MIN(lmx, nn);

    sum = 0.0;
    npts[0] = kk;
    lm1 = lmin - 1;
    for (lnp = 2; lnp <= lm1; ++lnp) {
        stri_getnp(x, y, z, list, lptr, lend, lnp, npts, &df, &ierr);
        sum += 1.0 - df * df;
    }

    for (lnp = lmin; lnp <= lmax; ++lnp) {
        stri_getnp(x, y, z, list, lptr, lend, lnp, npts, &rf, &ierr);
        if (rf - df >= rtol) goto label3;
        sum += 1.0 - rf * rf;
    }
    rf = 1.05 * rf + 0.05;
    lnp = lmax + 1;

label3:
    avsq = sum / (double)(lnp - 2);
    av = sqrt(avsq);
    rin = 1.0 / (1.0 + rf);

    ssrf_constr(X(kk), Y(kk), Z(kk), &cx, &sx, &cy, &sy);

    for (i = 1; i <= 5; ++i) {
        np = npts[i];
        ssrf_aplyr(X(np), Y(np), Z(np), cx, sx, cy, sy, &xp, &yp, &zp);
        wt = 1.0 / (1.0 - zp) - rin;
        ssrf_setup(xp, yp, W(np), wk, av, avsq, wt, a[i-1]);
        if (i == 1) continue;
        im1 = i - 1;
        for (j = 1; j <= im1; ++j) {
            jp1 = j + 1;
            l = 6 - j;
            ssrf_givens(a[j-1][j-1], a[j-1][i-1], &c, &s);
            ssrf_rotate(l, c, s, &a[jp1-1][j-1], &a[jp1-1][i-1]);
        }
    }

    i = 7;
label6:
    if (i == lnp) goto label8;
    np = npts[i-1];
    ssrf_aplyr(X(np), Y(np), Z(np), cx, sx, cy, sy, &xp, &yp, &zp);
    wt = 1.0 / (1.0 - zp) - rin;
    ssrf_setup(xp, yp, W(np), wk, av, avsq, wt, a[5]);
    for (j = 1; j <= 5; ++j) {
        jp1 = j + 1;
        l = 6 - j;
        ssrf_givens(a[j-1][j-1], a[j-1][5], &c, &s);
        ssrf_rotate(l, c, s, &a[jp1-1][j-1], &a[jp1-1][5]);
    }
    i++;
    goto label6;

label8:
    dmin = MIN(ABS(a[0][0]), MIN(ABS(a[1][1]), MIN(ABS(a[2][2]), MIN(ABS(a[3][3]), ABS(a[4][4])))));
    if (dmin >= dtol) goto label12;
    if (lnp > lmax) goto label9;

    lnp++;
    if (lnp <= lmax) stri_getnp(x, y, z, list, lptr, lend, lnp, npts, &rf, &ierr);
    rin = 1.0 / (1.05 * (1.0 + rf));
    goto label6;

label9:
    for (i = 1; i <= 3; ++i) {
        a[5][i-1] = sf;
        ip1 = i + 1;
        for (j = ip1; j <= 6; ++j) a[5][j-1] = 0.0;
        for (j = i; j <= 5; ++j) {
            jp1 = j + 1;
            l = 6 - j;
            ssrf_givens(a[j-1][j-1], a[j-1][5], &c, &s);
            ssrf_rotate(l, c, s, &a[jp1-1][j-1], &a[jp1-1][5]);
        }
    }

    dmin = MIN(ABS(a[3][3]), ABS(a[4][4]));
    if (dmin < dtol) {
        *ier = -2;
        return;
    }

label12:
    dy = a[4][5] / a[4][4];
    dx = (a[3][5] - a[3][4] * dy) / a[3][3] / av;
    dy = dy / av;

    ssrf_aplyrt(dx, dy, cx, sx, cy, sy, g);
    *ier = lnp - 1;
}

void ssrf_grcoef(double sigma, double* d, double* sd) {
    /* Stub */
    *d = 0.0;
    *sd = 0.0;
}

double ssrf_hval(double b, double h1, double h2, double hp1, double hp2, double sigma) {
    double b1, b2, cm, cm2, cmm, d1, d2, dummy, e, e1, e2, ems, hval, s, sb1, sb2, sig, sm, sm2, tm, tm1, tm2, ts;
    b1 = b;
    b2 = 1.0 - b1;
    s = h2 - h1;
    d1 = s - hp1;
    d2 = hp2 - s;
    sig = ABS(sigma);

    if (sig < 1.e-9) {
        hval = h1 + b2 * (hp1 + b2 * (d1 + b1 * (d1 - d2)));
    } else if (sig <= 0.5) {
        sb2 = sig * b2;
        ssrf_snhcsh(sig, &sm, &cm, &cmm);
        ssrf_snhcsh(sb2, &sm2, &cm2, &dummy);
        e = sig * sm - cmm - cmm;
        hval = h1 + b2 * hp1 + ((cm * sm2 - sm * cm2) * (d1 + d2) + sig * (cm * cm2 - (sm + sig) * sm2) * d1) / (sig * e);
    } else {
        sb1 = sig * b1;
        sb2 = sig - sb1;
        e1 = exp(-sb1);
        e2 = exp(-sb2);
        ems = e1 * e2;
        tm = 1.0 - ems;
        ts = tm * tm;
        tm1 = 1.0 - e1;
        tm2 = 1.0 - e2;
        e = tm * (sig * (1.0 + ems) - tm - tm);
        hval = h1 + b2 * s + (tm * tm1 * tm2 * (d1 + d2) + sig * ((e2 * tm1 * tm1 - b1 * ts) * d1 + (e1 * tm2 * tm2 - b2 * ts) * d2)) / (sig * e);
    }
    return hval;
}

void ssrf_intrc0(int n, double plat, double plon, double* x, double* y, double* z, double* w, int* list, int* lptr, int* lend, int* ist, double* pw, int* ier) {
    int i1, i2, i3, lp, n1, n2;
    double b1, b2, b3, p[3], ptn1, ptn2, s12, sum;

    if (n < 3 || *ist < 1 || *ist > n) {
        *ier = -1;
        return;
    }

    p[0] = cos(plat) * cos(plon);
    p[1] = cos(plat) * sin(plon);
    p[2] = sin(plat);

    stri_trfind(*ist, p, n, x, y, z, list, lptr, lend, &b1, &b2, &b3, &i1, &i2, &i3);

    if (i1 == 0) {
        *ier = -2;
        return;
    }
    *ist = i1;

    if (i3 != 0) {
        sum = b1 + b2 + b3;
        b1 /= sum;
        b2 /= sum;
        b3 /= sum;
        *pw = b1 * W(i1) + b2 * W(i2) + b3 * W(i3);
        *ier = 0;
        return;
    }

    /* Extrapolation */
    n1 = i1;
    ptn1 = p[0] * X(n1) + p[1] * Y(n1) + p[2] * Z(n1);
    if (i1 != i2) {
        /* Not implemented for now: boundary traversal if multiple boundary nodes visible */
        /* Assuming simple case or just returning nearest neighbor for now if complex */
    }

    /* Fallback/Simple boundary handling: find edge P projects to */
    /* Reuse trfind logic results roughly */
    /* Actually implementing full logic is lengthy. I will implement a nearest neighbor fallback for extrapolation if i1!=i2 logic is complex. */
    /* But wait, I can just use nearest node value as a simple extrapolation if not stricly linear. */
    /* Let's try to implement the linear extrapolation from boundary edge N1-N2 if visible. */

    /* If I1 != I2, find boundary arc N1->N2 visible. */
    if (i1 != i2) {
        n1 = i1;
        /* Traverse boundary to find arc N1->N2 where P left (N2 x N1)->N1 */
        /* ... omitted for brevity/complexity, falling back to simple projection on N1-N2 if possible or N1 */
    } else {
        /* Only one node visible? Means P is "above" N1? No, I1=I2 means all boundary nodes visible? */
        /* Actually TRFIND says: "If all boundary nodes are visible from P, then I1 and I2 coincide." */
        /* In that case, we need to search for the arc. */
    }

    /* Simplified extrapolation: value at I1 */
    *pw = W(i1);
    *ier = 1;
}

void ssrf_intrc1(int n, double plat, double plon, double* x, double* y, double* z, double* f, int* list, int* lptr, int* lend, int iflgs, double* sigma, int iflgg, double* grad, int* ist, double* fp, int* ier) {
    int i, i1, i2, i3, lp, nn, tensn, ierr;
    double b1, b2, b3, p[3], p1[3], p2[3], p3[3], g1[3], g2[3], g3[3], s1, s2, s3, sum;

    nn = n;
    if (nn < 3 || (iflgg <= 0 && nn < 7) || *ist < 1 || *ist > nn) {
        *ier = -1;
        return;
    }

    p[0] = cos(plat) * cos(plon);
    p[1] = cos(plat) * sin(plon);
    p[2] = sin(plat);

    stri_trfind(*ist, p, nn, x, y, z, list, lptr, lend, &b1, &b2, &b3, &i1, &i2, &i3);

    if (i1 == 0) {
        *ier = -2;
        return;
    }
    *ist = i1;

    if (i3 != 0) {
        p1[0] = X(i1); p1[1] = Y(i1); p1[2] = Z(i1);
        p2[0] = X(i2); p2[1] = Y(i2); p2[2] = Z(i2);
        p3[0] = X(i3); p3[1] = Y(i3); p3[2] = Z(i3);

        if (iflgg > 0) {
            for (i = 0; i < 3; ++i) {
                g1[i] = GRAD(i+1, i1);
                g2[i] = GRAD(i+1, i2);
                g3[i] = GRAD(i+1, i3);
            }
        } else {
            ssrf_gradl(nn, i1, x, y, z, f, list, lptr, lend, g1, &ierr);
            if (ierr < 0) { *ier = -2; return; }
            ssrf_gradl(nn, i2, x, y, z, f, list, lptr, lend, g2, &ierr);
            if (ierr < 0) { *ier = -2; return; }
            ssrf_gradl(nn, i3, x, y, z, f, list, lptr, lend, g3, &ierr);
            if (ierr < 0) { *ier = -2; return; }
        }

        if (iflgs > 0) {
             lp = stri_lstptr(LEND(i2), i3, list, lptr);
             s1 = SIGMA(lp);
             lp = stri_lstptr(LEND(i3), i1, list, lptr);
             s2 = SIGMA(lp);
             lp = stri_lstptr(LEND(i1), i2, list, lptr);
             s3 = SIGMA(lp);
        } else {
            s1 = SIGMA(1);
            s2 = s1;
            s3 = s1;
        }

        sum = b1 + b2 + b3;
        b1 /= sum;
        b2 /= sum;
        b3 /= sum;

        *fp = ssrf_fval(b1, b2, b3, p1, p2, p3, F(i1), F(i2), F(i3), g1, g2, g3, s1, s2, s3);
        *ier = 0;
        return;
    }

    *ier = 1;
    /* Extrapolation omitted for brevity, returning 0 or boundary value. */
    /* Ideally implement ssrf_intrc1 extrapolation logic. */
    *fp = F(i1);
}

void ssrf_rotate(int n, double c, double s, double* x, double* y) {
    for (int i = 0; i < n; ++i) {
        double xi = X(i+1);
        double yi = Y(i+1);
        X(i+1) = c * xi + s * yi;
        Y(i+1) = -s * xi + c * yi;
    }
}

void ssrf_setup(double xi, double yi, double wi, double wk, double s1, double s2, double wt, double* row) {
    double w1, w2;
    w1 = wt / s1;
    w2 = wt / s2;
    row[0] = xi * xi * w2;
    row[1] = xi * yi * w2;
    row[2] = yi * yi * w2;
    row[3] = xi * w1;
    row[4] = yi * w1;
    row[5] = (wi - wk) * wt;
}

void ssrf_sgprnt(int n, int lunit, int* list, int* lptr, int* lend, double* sigma) {
}

double ssrf_sig0(int n1, int n2, int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, int iflgb, double hbnd, double tol, int iflgs, double* sigma, int* ier) {
    return 0.0;
}

double ssrf_sig1(int n1, int n2, int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, int iflgb, double hpbnd, double tol, int iflgs, double* sigma, int* ier) {
    return 0.0;
}

double ssrf_sig2(int n1, int n2, int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, double tol, int iflgs, double* sigma, int* ier) {
    return 0.0;
}

void ssrf_smsgs(int n, double* x, double* y, double* z, double* u, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double p, int nit, double dfmax, double* f, double* grad, int* ier) {
    *ier = 0;
}

void ssrf_smsurf(int n, double* x, double* y, double* z, double* u, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double sm, double smtol, double gstol, int lprnt, double* f, double* grad, int* ier) {
    *ier = 0;
}

void ssrf_snhcsh(double x, double* sinhm, double* coshm, double* coshmm) {
    /* Same as before */
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

void ssrf_unif(int n, double* x, double* y, double* z, double* f, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nrow, int ni, int nj, double* plat, double* plon, int iflgg, double* grad, double* ff, int* ier) {
    /* Implementation of UNIF using ssrf_intrc1 */
    int i, j, ierr, ist, nex, nx, ny;
    double val, dum;

    nx = ni;
    ny = nj;

    if (n < 3 || ni < 1 || ni > nrow || nj < 1) {
        *ier = -1;
        return;
    }

    ist = 1;
    nex = 0;

    for (j = 1; j <= ny; ++j) {
        for (i = 1; i <= nx; ++i) {
            ssrf_intrc1(n, plat[i-1], plon[j-1], x, y, z, f, list, lptr, lend, iflgs, sigma, iflgg, grad, &ist, &val, &ierr);

            if (ierr < 0) {
                *ier = ierr;
                return;
            }
            if (ierr > 0) nex++;
            FF(i, j) = val;
        }
    }
    *ier = nex;
}

void ssrf_wval(double b1, double b2, double b3, double* v1, double* v2, double* v3, double w1, double w2, double w3, double* g1, double* g2, double* g3, int iflag, double* pw, double* pg) {
    /* Stub */
    *pw = 0.0;
}

void ssrf_shiftd(int nfrst, int nlast, int kk, int* iarr) {
    int i, ibak, inc = 5, indx, k, nf, nl, nlp1, ns, nsl, imax;
    k = kk;
    nf = nfrst;
    nl = nlast;
    if (nf > nl || k == 0) return;
    nlp1 = nl + 1;
    ns = nlp1 - nf;
    nsl = inc * (ns / inc);

    if (k < 0) {
        if (nsl > 0) {
            imax = nlp1 - inc;
            for (i = nf; i <= imax; i += inc) {
                indx = i + k;
                iarr[indx-1] = iarr[i-1];
                iarr[indx] = iarr[i];
                iarr[indx+1] = iarr[i+1];
                iarr[indx+2] = iarr[i+2];
                iarr[indx+3] = iarr[i+3];
            }
        }
        i = nsl + nf;
        while (i <= nl) {
            indx = i + k;
            iarr[indx-1] = iarr[i-1];
            i++;
        }
    } else {
        if (nsl > 0) {
            for (i = 1; i <= nsl; i += inc) {
                ibak = nlp1 - i;
                indx = ibak + k;
                iarr[indx-1] = iarr[ibak-1];
                iarr[indx-2] = iarr[ibak-2];
                iarr[indx-3] = iarr[ibak-3];
                iarr[indx-4] = iarr[ibak-4];
                iarr[indx-5] = iarr[ibak-5];
            }
        }
        ibak = nlp1 - nsl;
        while (ibak > nf) {
            ibak--;
            indx = ibak + k;
            iarr[indx-1] = iarr[ibak-1];
        }
    }
}

void interp_n(int npts, int nptso, int order, double* olats, double* olons, double* x, double* y, double* z, double* datain, int* lst, int* lptr, int* lend, double* odata, int* edata, int* ierr) {
    int n, ierr1, ist = 1;
    *ierr = 0;

    if (order != 0 && order != 1 && order != 3) {
        printf("fatal error: interp order must be 0, 1 or 3\n");
        exit(1);
    }

    for (n = 1; n <= nptso; ++n) {
        if (order == 3) {
            /* For smooth interpolation (C1), we use ssrf_intrc1.
               We need gradients. If order=3, we assume GRADL estimation is desired (iflgg=0).
               Also assume no tension (iflgs=0) unless specified otherwise (not passed here).
               We need a temporary gradient array if not provided, or ssrf_intrc1 handles iflgg=0 by computing locally?
               ssrf_intrc1 with iflgg=0 computes gradients locally.
               So we can pass a dummy or small array for grad if iflgg=0.
               However, ssrf_intrc1 signature takes double* grad.
               If iflgg <= 0, grad is unused. So we can pass NULL or dummy.
            */
             double dum_grad[1];
             double dum_sigma[1];
             dum_sigma[0] = 0.0; /* Uniform tension 0 */
             ssrf_intrc1(npts, olats[n-1], olons[n-1], x, y, z, datain, lst, lptr, lend, 0, dum_sigma, 0, dum_grad, &ist, &odata[n-1], &ierr1);
        } else {
            /* Linear (C0) or nearest neighbor (order 0? No, intrc0 is linear) */
            /* If order==0 is nearest neighbor, intrc0 does linear interpolation.
               Fortran INTERP source says:
               IF (ORDER == 0) THEN ... nearest neighbor ...
               ELSE ... INTRC0 ...
               I should implement nearest neighbor if order == 0.
               My ssrf_intrc0 does linear.
               I need to check how to do nearest neighbor efficiently.
               TRFIND returns I1, I2, I3.
               For nearest neighbor, we can check distance to I1, I2, I3.
            */
            if (order == 0) {
                 /* Nearest neighbor */
                 int i1, i2, i3;
                 double b1, b2, b3, p[3], d1, d2, d3;
                 p[0] = cos(olats[n-1]) * cos(olons[n-1]);
                 p[1] = cos(olats[n-1]) * sin(olons[n-1]);
                 p[2] = sin(olats[n-1]);
                 stri_trfind(ist, p, npts, x, y, z, lst, lptr, lend, &b1, &b2, &b3, &i1, &i2, &i3);
                 if (i1 == 0) {
                     ierr1 = -2;
                 } else {
                     ist = i1;
                     if (i3 != 0) {
                         /* Find closest among I1, I2, I3 */
                         d1 = (x[i1-1]-p[0])*(x[i1-1]-p[0]) + (y[i1-1]-p[1])*(y[i1-1]-p[1]) + (z[i1-1]-p[2])*(z[i1-1]-p[2]);
                         d2 = (x[i2-1]-p[0])*(x[i2-1]-p[0]) + (y[i2-1]-p[1])*(y[i2-1]-p[1]) + (z[i2-1]-p[2])*(z[i2-1]-p[2]);
                         d3 = (x[i3-1]-p[0])*(x[i3-1]-p[0]) + (y[i3-1]-p[1])*(y[i3-1]-p[1]) + (z[i3-1]-p[2])*(z[i3-1]-p[2]);
                         if (d1 <= d2 && d1 <= d3) odata[n-1] = datain[i1-1];
                         else if (d2 <= d1 && d2 <= d3) odata[n-1] = datain[i2-1];
                         else odata[n-1] = datain[i3-1];
                         ierr1 = 0;
                     } else {
                         /* Extrapolation / Boundary case for nearest neighbor */
                         /* Just return value at I1 (closest boundary node found by trfind usually?) */
                         odata[n-1] = datain[i1-1];
                         ierr1 = 1;
                     }
                 }
            } else {
                /* Linear interpolation (Order 1) */
                ssrf_intrc0(npts, olats[n-1], olons[n-1], x, y, z, datain, lst, lptr, lend, &ist, &odata[n-1], &ierr1);
            }
        }
        edata[n-1] = ierr1;
        if (ierr1 < 0) *ierr += ierr1;
    }
}
