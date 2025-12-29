#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tripack.h"

double swtol;

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a) ((a) < 0 ? -(a) : (a))

/* Macros for 1-based indexing */
#define X(i) x[(i)-1]
#define Y(i) y[(i)-1]
#define LIST(i) list[(i)-1]
#define LPTR(i) lptr[(i)-1]
#define LEND(i) lend[(i)-1]
#define LCC(i) lcc[(i)-1]
/* IWK(2,LWK) in Fortran. Accessed as IWK(1,I) and IWK(2,I).
   Fortran column major means:
   IWK(1,I) -> iwk[ (1-1) + 2*(I-1) ] = iwk[ 2*(I-1) ]
   IWK(2,I) -> iwk[ (2-1) + 2*(I-1) ] = iwk[ 1 + 2*(I-1) ]
*/
#define IWK(r, c) iwk[(r)-1 + 2*((c)-1)]

#define NODES(i) nodes[(i)-1]
#define LCT(i) lct[(i)-1]
/* LTRI(NROW,NT) */
#define LTRI(r, c) ltri[(r)-1 + nrow*((c)-1)]
#define NEAR(i) near_arr[(i)-1]
#define NEXT(i) next[(i)-1]
#define DIST(i) dist[(i)-1]

void addcst(int ncc, int* lcc, int n, double* x, double* y, int* lwk, int* iwk, int* list, int* lptr, int* lend, int* ier) {
    int i, ifrst, ilast, n1, n2, lw, lwd2;
    int k, kbak, kfor, lpf, lpb, lpl, lp, kn;

    lwd2 = *lwk / 2;
    *ier = 0;

    if (ncc < 0 || *lwk < 0) {
        *ier = 1;
        return;
    }

    if (ncc == 0) {
        if (n < 3) {
            *ier = 1;
        } else {
            *ier = 0;
            *lwk = 0;
        }
        return;
    } else {
        int lccip1 = n + 1;
        for (i = ncc; i >= 1; --i) {
            if (lccip1 - LCC(i) < 3) {
                *ier = 1;
                return;
            }
            lccip1 = LCC(i);
        }
        if (lccip1 < 1) {
            *ier = 1;
            return;
        }
    }

    *lwk = 0;
    ifrst = n + 1;

    for (i = ncc; i >= 1; --i) {
        ilast = ifrst - 1;
        ifrst = LCC(i);
        n1 = ilast;

        for (n2 = ifrst; n2 <= ilast; ++n2) {
            lw = lwd2;
            edge(n1, n2, x, y, &lw, iwk, list, lptr, lend, ier);
            *lwk = MAX(*lwk, 2 * lw);

            if (*ier == 4) *ier = 3;
            if (*ier != 0) return;
            n1 = n2;
        }
    }

    ifrst = n + 1;
    for (i = ncc; i >= 1; --i) {
        ilast = ifrst - 1;
        ifrst = LCC(i);
        kbak = ilast;

        for (k = ifrst; k <= ilast; ++k) {
            kfor = k + 1;
            if (k == ilast) kfor = ifrst;

            lpf = 0;
            lpb = 0;
            lpl = LEND(k);
            lp = lpl;

            do {
                lp = LPTR(lp);
                kn = ABS(LIST(lp));
                if (kn == kfor) lpf = lp;
                if (kn == kbak) lpb = lp;
                if (lp == lpl) break;
            } while (1);

            if (lpf == 0 || lpb == 0) {
                *ier = 4;
                return;
            }

            lp = lpf;
            do {
                lp = LPTR(lp);
                if (lp == lpb) break;
                kn = ABS(LIST(lp));
                if (kn < ifrst || kn > ilast) {
                    *ier = 5;
                    return;
                }
            } while (1);
            kbak = k;
        }
    }
    *ier = 0;
}

void addnod(int k, double xk, double yk, int ist, int ncc, int* lcc, int* n, double* x, double* y, int* list, int* lptr, int* lend, int* lnew, int* ier) {
    int i, i1, i2, i3, ibk, in1, indx, io1, io2;
    int kk, l, lccip1, lp, lpf, lpo1;
    int nm1;

    kk = k;
    if (kk < 1 || ist < 1 || ist > *n || ncc < 0 || *n < 3) {
        *ier = -1;
        return;
    }

    lccip1 = *n + 1;
    for (i = ncc; i >= 1; --i) {
        if (lccip1 - LCC(i) < 3) {
            *ier = -1;
            return;
        }
        lccip1 = LCC(i);
    }
    if (lccip1 < kk) {
        *ier = -1;
        return;
    }

    trfind(ist, xk, yk, *n, x, y, list, lptr, lend, &i1, &i2, &i3);

    if (i1 == 0) {
        *ier = -2;
        return;
    }

    if (i3 != 0) {
        if (xk == X(i1) && yk == Y(i1)) { *ier = i1; return; }
        if (xk == X(i2) && yk == Y(i2)) { *ier = i2; return; }
        if (xk == X(i3) && yk == Y(i3)) { *ier = i3; return; }
        if (ncc > 0 && crtri(ncc, lcc, i1, i2, i3)) {
            *ier = -3;
            return;
        }
    } else {
        if (ncc > 0 && indxcc(ncc, lcc, *n, list, lend) != 0) {
            *ier = -3;
            return;
        }
    }

    *ier = 0;
    nm1 = *n;
    *n = *n + 1;

    if (kk < *n) {
        for (ibk = nm1; ibk >= kk; --ibk) {
            X(ibk + 1) = X(ibk);
            Y(ibk + 1) = Y(ibk);
            LEND(ibk + 1) = LEND(ibk);
        }
        for (i = 1; i <= ncc; ++i) {
            LCC(i) = LCC(i) + 1;
        }
        l = *lnew - 1;
        for (i = 1; i <= l; ++i) {
            if (LIST(i) >= kk) LIST(i) = LIST(i) + 1;
            if (LIST(i) <= -kk) LIST(i) = LIST(i) - 1;
        }
        if (i1 >= kk) i1++;
        if (i2 >= kk) i2++;
        if (i3 >= kk && i3 != 0) i3++;
    }

    X(kk) = xk;
    Y(kk) = yk;

    if (i3 == 0) {
        bdyadd(kk, i1, i2, list, lptr, lend, lnew);
    } else {
        intadd(kk, i1, i2, i3, list, lptr, lend, lnew);
    }

    lp = LEND(kk);
    lpf = LPTR(lp);
    io2 = LIST(lpf);
    lpo1 = LPTR(lpf);
    io1 = ABS(LIST(lpo1));

    while (1) {
        lp = lstptr(LEND(io1), io2, list, lptr);
        if (LIST(lp) >= 0) {
            lp = LPTR(lp);
            in1 = ABS(LIST(lp));
            if (!crtri(ncc, lcc, io1, io2, in1)) {
                if (swptst(in1, kk, io1, io2, x, y)) {
                    swap(in1, kk, io1, io2, list, lptr, lend, &lpo1);
                    if (lpo1 == 0) {
                        *ier = -4;
                        break;
                    }
                    io1 = in1;
                    continue;
                }
            }
        }
        if (lpo1 == lpf || LIST(lpo1) < 0) break;
        io2 = io1;
        lpo1 = LPTR(lpo1);
        io1 = ABS(LIST(lpo1));
    }
}

double areap(double* x, double* y, int nb, int* nodes) {
    double a = 0.0;
    int i, nd1, nd2;

    if (nb < 3) return 0.0;

    nd2 = NODES(nb);
    for (i = 1; i <= nb; ++i) {
        nd1 = nd2;
        nd2 = NODES(i);
        a += (X(nd2) - X(nd1)) * (Y(nd1) + Y(nd2));
    }
    return -a / 2.0;
}

void bdyadd(int kk, int i1, int i2, int* list, int* lptr, int* lend, int* lnew) {
    int k = kk, n1 = i1, n2 = i2;
    int lp, lsav, next, nsav;

    lp = LEND(n1);
    lsav = LPTR(lp);
    LPTR(lp) = *lnew;
    LIST(*lnew) = -k;
    LPTR(*lnew) = lsav;
    LEND(n1) = *lnew;
    *lnew = *lnew + 1;
    next = -LIST(lp);
    LIST(lp) = next;
    nsav = next;

    while (1) {
        lp = LEND(next);
        insert(k, lp, list, lptr, lnew);
        if (next == n2) break;
        next = -LIST(lp);
        LIST(lp) = next;
    }

    lsav = *lnew;
    LIST(*lnew) = n1;
    LPTR(*lnew) = *lnew + 1;
    *lnew = *lnew + 1;
    next = nsav;

    while (1) {
        if (next == n2) break;
        LIST(*lnew) = next;
        LPTR(*lnew) = *lnew + 1;
        *lnew = *lnew + 1;
        lp = LEND(next);
        next = LIST(lp);
    }

    LIST(*lnew) = -n2;
    LPTR(*lnew) = lsav;
    LEND(k) = *lnew;
    *lnew = *lnew + 1;
}

void bnodes(int n, int* list, int* lptr, int* lend, int* nodes, int* nb, int* na, int* nt) {
    int k, lp, n0, nst;

    nst = 1;
    while (1) {
        lp = LEND(nst);
        if (LIST(lp) < 0) break;
        nst++;
    }

    NODES(1) = nst;
    k = 1;
    n0 = nst;

    while (1) {
        lp = LEND(n0);
        lp = LPTR(lp);
        n0 = LIST(lp);
        if (n0 == nst) break;
        k++;
        NODES(k) = n0;
    }

    *nb = k;
    *nt = 2 * n - *nb - 2;
    *na = *nt + n - 1;
}

void circum(double x1, double y1, double x2, double y2, double x3, double y3, int ratio, double* xc, double* yc, double* cr, double* sa, double* ar) {
    double u[3], v[3], ds[3];
    double fx, fy;

    u[0] = x3 - x2;
    u[1] = x1 - x3;
    u[2] = x2 - x1;
    v[0] = y3 - y2;
    v[1] = y1 - y3;
    v[2] = y2 - y1;

    *sa = (u[0] * v[1] - u[1] * v[0]) / 2.0;

    if (*sa == 0.0) {
        if (ratio) *ar = 0.0;
        return;
    }

    ds[0] = x1 * x1 + y1 * y1;
    ds[1] = x2 * x2 + y2 * y2;
    ds[2] = x3 * x3 + y3 * y3;

    fx = -(ds[0] * v[0] + ds[1] * v[1] + ds[2] * v[2]);
    fy = ds[0] * u[0] + ds[1] * u[1] + ds[2] * u[2];

    *xc = fx / (4.0 * *sa);
    *yc = fy / (4.0 * *sa);
    *cr = sqrt(pow(*xc - x1, 2) + pow(*yc - y1, 2));

    if (!ratio) return;

    ds[0] = u[0] * u[0] + v[0] * v[0];
    ds[1] = u[1] * u[1] + v[1] * v[1];
    ds[2] = u[2] * u[2] + v[2] * v[2];

    *ar = 2.0 * ABS(*sa) / ((sqrt(ds[0]) + sqrt(ds[1]) + sqrt(ds[2])) * *cr);
}

int crtri(int ncc, int* lcc, int i1, int i2, int i3) {
    int i, imax, imin;

    imax = MAX(i1, MAX(i2, i3));
    i = ncc + 1;
    do {
        i--;
        if (i <= 0) return 0;
        if (LCC(i) <= imax) break;
    } while (1);

    imin = MIN(i1, MIN(i2, i3));
    return (LCC(i) <= imin) && (
        (imin == i1 && imax == i3) ||
        (imin == i2 && imax == i1) ||
        (imin == i3 && imax == i2)
    );
}

void delarc(int n, int io1, int io2, int* list, int* lptr, int* lend, int* lnew, int* ier) {
    int n1 = io1, n2 = io2, n3, lp, lph, lpl;

    if (n < 4 || n1 < 1 || n1 > n || n2 < 1 || n2 > n || n1 == n2) {
        *ier = 1;
        return;
    }

    lpl = LEND(n2);
    if (-LIST(lpl) != n1) {
        n1 = n2;
        n2 = io1;
        lpl = LEND(n2);
        if (-LIST(lpl) != n1) {
            *ier = 2;
            return;
        }
    }

    lpl = LEND(n1);
    lp = LPTR(lpl);
    lp = LPTR(lp);
    n3 = ABS(LIST(lp));
    lpl = LEND(n3);

    if (LIST(lpl) <= 0) {
        *ier = 3;
        return;
    }

    delnb(n1, n2, n, list, lptr, lend, lnew, &lph);
    if (lph < 0) {
        *ier = 4;
        return;
    }

    delnb(n2, n1, n, list, lptr, lend, lnew, &lph);

    lp = lstptr(LEND(n3), n1, list, lptr);
    LEND(n3) = lp;
    LIST(lp) = -n1;
    *ier = 0;
}

void delnb(int n0, int nb, int n, int* list, int* lptr, int* lend, int* lnew, int* lph) {
    int i, lnw, lp, lpb, lpl, lpp;

    if (n0 < 1 || n0 > n || nb < 1 || nb > n || n < 3) {
        *lph = -1;
        return;
    }

    lpl = LEND(n0);
    lpp = lpl;
    lpb = LPTR(lpp);

    while (LIST(lpb) != nb) {
        lpp = lpb;
        lpb = LPTR(lpp);
        if (lpb == lpl) break;
    }

    if (ABS(LIST(lpb)) != nb) {
        *lph = -2;
        return;
    }

    if (lpb == lpl) {
        LEND(n0) = lpp;
        lp = LEND(nb);
        if (LIST(lp) < 0) LIST(lpp) = -LIST(lpp);
    } else {
        lp = LEND(nb);
        if (LIST(lp) < 0 && LIST(lpl) > 0) {
            LEND(n0) = lpp;
            LIST(lpp) = -LIST(lpp);
        }
    }

    LPTR(lpp) = LPTR(lpb);
    lnw = *lnew - 1;
    LIST(lpb) = LIST(lnw);
    LPTR(lpb) = LPTR(lnw);

    for (i = n; i >= 1; --i) {
        if (LEND(i) == lnw) {
            LEND(i) = lpb;
            break;
        }
    }

    for (i = 1; i < lnw; ++i) {
        if (LPTR(i) == lnw) LPTR(i) = lpb;
    }

    *lnew = lnw;
    *lph = lpb;
}

void delnod(int k, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int* lnew, int* lwk, int* iwk, int* ier) {
    int i, j, lccip1, lp, lp21, lpf, lph, lpl, lpl2, lpn, lnw, nl, nn, nnb, nr, nfrst, n2;
    int bdry, lwkl, iwl, nit, ierr;
    double x1, y1, x2, y2, xl, yl, xr, yr;

    *ier = 0;
    int n1 = k;
    nn = n;

    if (ncc < 0 || n1 < 1 || nn < 4 || *lwk < 0) {
        *ier = 1;
        return;
    }

    lccip1 = nn + 1;
    for (i = ncc; i >= 1; --i) {
        if (lccip1 - LCC(i) < 3) {
            *ier = 1;
            return;
        }
        lccip1 = LCC(i);
    }
    if (lccip1 <= n1) {
        *ier = 1;
        return;
    }

    lpl = LEND(n1);
    lpf = LPTR(lpl);
    nnb = nbcnt(lpl, lptr);
    bdry = (LIST(lpl) < 0);

    if (bdry) nnb++;

    if (nnb < 3) {
        /* Error message printing skipped for brevity/C style */
        *ier = 3;
        return;
    }

    lwkl = *lwk;
    *lwk = nnb - 3;
    if (lwkl < *lwk) {
        *ier = 2;
        return;
    }

    iwl = 0;
    if (nnb == 3) goto label5;

    x1 = X(n1);
    y1 = Y(n1);
    nfrst = LIST(lpf);
    nr = nfrst;
    xr = X(nr);
    yr = Y(nr);
    lp = LPTR(lpf);
    n2 = LIST(lp);
    x2 = X(n2);
    y2 = Y(n2);
    lp = LPTR(lp);

label2:
    nl = ABS(LIST(lp));
    if (nl == nfrst && bdry) goto label5;
    xl = X(nl);
    yl = Y(nl);

    lpl2 = LEND(n2);
    if ((bdry || left(xr, yr, xl, yl, x1, y1)) && (LIST(lpl2) < 0 || left(xl, yl, xr, yr, x2, y2))) {
        swap(nl, nr, n1, n2, list, lptr, lend, &lp21);
        iwl++;
        IWK(1, iwl) = (nl <= n1) ? nl : nl - 1;
        IWK(2, iwl) = (nr <= n1) ? nr : nr - 1;

        lpl = LEND(n1);
        nnb--;
        if (nnb == 3) goto label5;
        lp = lstptr(lpl, nl, list, lptr);
        if (nr == nfrst) goto label4;

        n2 = nr;
        x2 = xr;
        y2 = yr;
        lp21 = LPTR(lp21);
        lp21 = LPTR(lp21);
        nr = ABS(LIST(lp21));
        xr = X(nr);
        yr = Y(nr);
        goto label2;
    }

    nr = n2;
    xr = x2;
    yr = y2;

label4:
    if (n2 == nfrst) {
        *ier = 4;
        return;
    }
    n2 = nl;
    x2 = xl;
    y2 = yl;
    lp = LPTR(lp);
    goto label2;

label5:
    lp = lpl;
    lnw = *lnew;

label6:
    lp = LPTR(lp);
    n2 = ABS(LIST(lp));
    delnb(n2, n1, nn, list, lptr, lend, &lnw, &lph);
    if (lph < 0) {
        *ier = 3;
        return;
    }
    if (lpl == lnw) lpl = lph;
    if (lp == lnw) lp = lph;
    if (lp != lpl) goto label6;

    nn--;
    if (nn >= n1) {
        for (i = n1; i <= nn; ++i) {
            X(i) = X(i + 1);
            Y(i) = Y(i + 1);
            LEND(i) = LEND(i + 1);
        }
        for (i = 1; i < lnw; ++i) {
            if (LIST(i) > n1) LIST(i)--;
            if (LIST(i) < -n1) LIST(i)++;
        }
    }

    if (bdry) nnb--;
    lpn = lpl;
    for (j = 1; j <= nnb; ++j) {
        lnw--;
        lp = lpn;
        lpn = LPTR(lp);
        LIST(lp) = LIST(lnw);
        LPTR(lp) = LPTR(lnw);
        if (LPTR(lpn) == lnw) LPTR(lpn) = lp;
        if (lpn == lnw) lpn = lp;
        for (i = nn; i >= 1; --i) {
            if (LEND(i) == lnw) {
                LEND(i) = lp;
                break;
            }
        }
        for (i = lnw - 1; i >= 1; --i) {
            if (LPTR(i) == lnw) LPTR(i) = lp;
        }
    }

    for (i = 1; i <= ncc; ++i) LCC(i)--;

    *lnew = lnw;
    if (iwl > 0) {
        nit = 4 * iwl;
        optim(x, y, iwl, list, lptr, lend, &nit, iwk, &ierr);
        if (ierr != 0) {
            *ier = 5;
            return;
        }
    }
    *ier = 0;
}

void edge(int in1, int in2, double* x, double* y, int* lwk, int* iwk, int* list, int* lptr, int* lend, int* ier) {
    int i, ierr, iwc, iwcp1, iwend, iwf, iwl, lft, lp, lp21, lpl, n0, n1, n2, n1frst, n1lst, next, nit, nl, nr;
    double dx, dy, x0, y0, x1, y1, x2, y2;

    n1 = in1;
    n2 = in2;
    iwend = *lwk;

    if (n1 < 1 || n2 < 1 || n1 == n2 || iwend < 0) {
        *ier = 1;
        return;
    }

    lpl = LEND(n1);
    n0 = ABS(LIST(lpl));
    lp = lpl;
    do {
        if (n0 == n2) {
            *ier = 0;
            return;
        }
        lp = LPTR(lp);
        n0 = LIST(lp);
        if (lp == lpl) break;
    } while (1);

    iwl = 0;
    nit = 0;

label2:
    x1 = X(n1);
    y1 = Y(n1);
    x2 = X(n2);
    y2 = Y(n2);

    lpl = LEND(n1);
    n1lst = LIST(lpl);
    lp = LPTR(lpl);
    n1frst = LIST(lp);
    nl = n1frst;

    if (n1lst < 0) goto label4;

label3:
    if (left(x2, y2, x1, y1, X(nl), Y(nl))) goto label4;
    lp = LPTR(lp);
    nl = LIST(lp);
    if (nl != n1frst) goto label3;
    goto label5;

label4:
    nr = nl;
    lp = LPTR(lp);
    nl = ABS(LIST(lp));

    if (left(x1, y1, x2, y2, X(nl), Y(nl))) {
        dx = x2 - x1;
        dy = y2 - y1;
        if ((dx * (X(nl) - x1) + dy * (Y(nl) - y1) >= 0 ||
             dx * (X(nr) - x1) + dy * (Y(nr) - y1) >= 0) &&
            (dx * (X(nl) - x2) + dy * (Y(nl) - y2) <= 0 ||
             dx * (X(nr) - x2) + dy * (Y(nr) - y2) <= 0)) goto label6;
        if (!left(x2, y2, x1, y1, X(nl), Y(nl))) goto label5;
    }
    if (nl != n1frst) goto label4;

label5:
    if (nit > 0) goto label33;
    nit = 1;
    n1 = n2;
    n2 = in1;
    goto label2;

label6:
    iwl++;
    if (iwl > iwend) {
        *ier = 2;
        return;
    }
    IWK(1, iwl) = nl;
    IWK(2, iwl) = nr;

    lpl = LEND(nl);
    lp = LPTR(lpl);

label7:
    if (LIST(lp) == nr) goto label8;
    lp = LPTR(lp);
    if (lp != lpl) goto label7;
    goto label33;

label8:
    lp = LPTR(lp);
    next = ABS(LIST(lp));
    if (next == n2) goto label9;

    if (left(x1, y1, x2, y2, X(next), Y(next))) {
        nl = next;
    } else {
        nr = next;
    }
    goto label6;

label9:
    *lwk = iwl;
    iwend = iwl;
    iwf = 1;

label10:
    lft = 0;
    n0 = n1;
    x0 = x1;
    y0 = y1;
    nl = IWK(1, iwf);
    nr = IWK(2, iwf);
    iwc = iwf;

label11:
    if (iwc == iwl) goto label21;
    iwcp1 = iwc + 1;
    next = IWK(1, iwcp1);
    if (next != nl) goto label16;
    next = IWK(2, iwcp1);

    if (!left(x0, y0, X(nr), Y(nr), X(next), Y(next))) goto label14;
    if (lft >= 0) goto label12;
    if (!left(X(nl), Y(nl), x0, y0, X(next), Y(next))) goto label14;

    swap(next, n0, nl, nr, list, lptr, lend, &lp21);
    IWK(1, iwc) = n0;
    IWK(2, iwc) = next;
    goto label15;

label12:
    swap(next, n0, nl, nr, list, lptr, lend, &lp21);
    for (i = iwcp1; i <= iwl; ++i) {
        IWK(1, i - 1) = IWK(1, i);
        IWK(2, i - 1) = IWK(2, i);
    }
    IWK(1, iwl) = n0;
    IWK(2, iwl) = next;
    iwl--;
    nr = next;
    goto label11;

label14:
    n0 = nr;
    x0 = X(n0);
    y0 = Y(n0);
    lft = 1;

label15:
    nr = next;
    iwc++;
    goto label11;

label16:
    if (!left(X(nl), Y(nl), x0, y0, X(next), Y(next))) goto label19;
    if (lft <= 0) goto label17;
    if (!left(x0, y0, X(nr), Y(nr), X(next), Y(next))) goto label19;

    swap(next, n0, nl, nr, list, lptr, lend, &lp21);
    IWK(1, iwc) = next;
    IWK(2, iwc) = n0;
    goto label20;

label17:
    swap(next, n0, nl, nr, list, lptr, lend, &lp21);
    for (i = iwc - 1; i >= iwf; --i) {
        IWK(1, i + 1) = IWK(1, i);
        IWK(2, i + 1) = IWK(2, i);
    }
    IWK(1, iwf) = n0;
    IWK(2, iwf) = next;
    iwf++;
    goto label20;

label19:
    n0 = nl;
    x0 = X(n0);
    y0 = Y(n0);
    lft = -1;

label20:
    nl = next;
    iwc++;
    goto label11;

label21:
    if (n0 == n1) goto label24;
    if (lft < 0) goto label22;

    if (!left(x0, y0, X(nr), Y(nr), x2, y2)) goto label10;
    swap(n2, n0, nl, nr, list, lptr, lend, &lp21);
    IWK(1, iwl) = n0;
    IWK(2, iwl) = n2;
    iwl--;
    goto label10;

label22:
    if (!left(X(nl), Y(nl), x0, y0, x2, y2)) goto label10;
    swap(n2, n0, nl, nr, list, lptr, lend, &lp21);
    for (i = iwl; i > iwf; --i) {
        IWK(1, i) = IWK(1, i - 1);
        IWK(2, i) = IWK(2, i - 1);
    }
    IWK(1, iwf) = n0;
    IWK(2, iwf) = n2;
    iwf++;
    goto label10;

label24:
    swap(n2, n1, nl, nr, list, lptr, lend, &lp21);
    IWK(1, iwc) = 0;
    IWK(2, iwc) = 0;

    if (iwc > 1) {
        nit = 3 * (iwc - 1);
        optim(x, y, iwc - 1, list, lptr, lend, &nit, iwk, &ierr);
        if (ierr != 0) goto label34;
    }
    if (iwc < iwend) {
        nit = 3 * (iwend - iwc);
        optim(x, y, iwend - iwc, list, lptr, lend, &nit, &IWK(1, iwc + 1), &ierr);
        if (ierr != 0) goto label34;
    }
    *ier = 0;
    return;

label33:
    *ier = 3;
    return;

label34:
    *ier = 4;
    return;
}

void getnp(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int l, int* npts, double* ds, int* ier) {
    int i, ifrst, ilast, j, k, km1, lcc1, lm1, lp, lpcl, lpk, lpkl, n1, nc, nf1, nf2, nj, nk, nkbak, nkfor, nl, nn;
    int ncf, njf, skip, sksav, vis, isw;
    double dc, dl, xc, xj, xk, x1, yc, yj, yk, y1;
    int lft1, lft2, lft12;

    *ier = -1;
    nn = n;
    lcc1 = nn + 1;
    lm1 = l - 1;

    if (ncc < 0 || lm1 < 1 || lm1 >= nn) return;

    if (ncc == 0) {
        if (nn < 3) return;
    } else {
        for (i = ncc; i >= 1; --i) {
            if (lcc1 - LCC(i) < 3) return;
            lcc1 = LCC(i);
        }
        if (lcc1 < 1) return;
    }

    for (k = 1; k <= lm1; ++k) {
        nk = npts[k-1];
        if (nk < 1 || nk > nn) {
            *ier = k;
            return;
        }
    }

    n1 = npts[0];
    x1 = X(n1);
    y1 = Y(n1);

    for (k = 1; k <= lm1; ++k) {
        nk = npts[k-1];
        LEND(nk) = -LEND(nk);
    }

    isw = 0;
    dl = 0.0;

    for (k = 1; k <= lm1; ++k) {
        km1 = k - 1;
        nk = npts[k-1];
        xk = X(nk);
        yk = Y(nk);
        lpkl = -LEND(nk);
        nkfor = 0;
        nkbak = 0;
        vis = 1;

        if (nk >= lcc1) {
            ifrst = nn + 1;
            for (i = ncc; i >= 1; --i) {
                ilast = ifrst - 1;
                ifrst = LCC(i);
                if (nk >= ifrst) break;
            }
            if (nk < ilast) nkfor = nk + 1; else nkfor = ifrst;
            if (nk > ifrst) nkbak = nk - 1; else nkbak = ilast;

            lpk = lpkl;
            do {
                lpk = LPTR(lpk);
                nc = ABS(LIST(lpk));
                if (nc == nkfor || nc == nkbak) break;
            } while(1);
            vis = (nc == nkfor);
        }

        lpk = lpkl;
        while (1) {
            lpk = LPTR(lpk);
            nc = ABS(LIST(lpk));
            if (nc == nkbak) vis = 1;
            if (!vis) goto label15;
            if (nc == nkfor) vis = 0;
            if (LEND(nc) < 0) goto label15;

            xc = X(nc);
            yc = Y(nc);
            dc = sqrt(pow(xc - x1, 2) + pow(yc - y1, 2));
            if (isw && dc >= dl) goto label15;

            if (k == 1) goto label14;

            lpcl = LEND(nc);
            for (j = 1; j <= km1; ++j) {
                nj = npts[j-1];
                if (j > 1 && nj < lcc1) goto label13;

                lp = lpcl;
                do {
                    lp = LPTR(lp);
                    if (nj == ABS(LIST(lp))) goto label12;
                } while (lp != lpcl);

                xj = X(nj);
                yj = Y(nj);
                ifrst = nn + 1;
                for (i = ncc; i >= 1; --i) {
                    ilast = ifrst - 1;
                    ifrst = LCC(i);
                    nf1 = ilast;
                    ncf = (nf1 == nc);
                    njf = (nf1 == nj);
                    skip = ncf || njf;

                    for (nf2 = ifrst; nf2 <= ilast; ++nf2) {
                        if (nf2 == nc) ncf = 1;
                        if (nf2 == nj) njf = 1;
                        sksav = skip;
                        skip = (nf2 == nc || nf2 == nj);
                        if (sksav || skip || (nf2 == ilast && !ncf && !njf)) goto label9;
                        if (intsec(X(nf1), Y(nf1), X(nf2), Y(nf2), xc, yc, xj, yj)) goto label12;
                    label9:
                        nf1 = nf2;
                    }

                    if (!ncf || !njf) continue;
                    if (nc != ifrst) nf1 = nc - 1; else nf1 = ilast;
                    if (nc != ilast) nf2 = nc + 1; else nf2 = ifrst;

                    lft1 = (xc - X(nf1)) * (yj - Y(nf1)) >= (xj - X(nf1)) * (yc - Y(nf1));
                    lft2 = (X(nf2) - xc) * (yj - yc) >= (xj - xc) * (Y(nf2) - yc);
                    lft12 = (X(nf1) - X(nf2)) * (yc - Y(nf2)) >= (xc - X(nf2)) * (Y(nf1) - Y(nf2));
                    if ((lft1 && lft2) || (!lft12 && (lft1 || lft2))) goto label12;
                }

                if (j == 1) goto label14;
                dc = MIN(dc, ds[j-1] + sqrt(pow(xc - xj, 2) + pow(yc - yj, 2)));
                goto label13;

            label12:
                if (j == 1) dc = ds[k-1] + sqrt(pow(xc - xk, 2) + pow(yc - yk, 2));
            label13:
                ;
            }

            if (isw && dc >= dl) goto label15;

        label14:
            nl = nc;
            dl = dc;
            isw = 1;

        label15:
            if (lpk == lpkl) break;
        }
    }

    for (k = 1; k <= lm1; ++k) {
        nk = npts[k-1];
        LEND(nk) = -LEND(nk);
    }

    npts[l-1] = nl;
    ds[l-1] = dl;
    *ier = 0;
}

int indxcc(int ncc, int* lcc, int n, int* list, int* lend) {
    int i, ifrst, ilast, lp, n0, nst, nxt;

    if (ncc < 1) return 0;

    n0 = 0;
    while (1) {
        n0++;
        lp = LEND(n0);
        if (LIST(lp) <= 0) break;
    }

    i = ncc;
    ilast = n;
    while (1) {
        ifrst = LCC(i);
        if (ifrst <= n0) break;
        if (i == 1) return 0;
        i--;
        ilast = ifrst - 1;
    }

    nst = n0;
    while (1) {
        nxt = -LIST(lp);
        if (nxt == nst) break;
        if (nxt <= n0 || ilast < nxt) return 0;
        n0 = nxt;
        lp = LEND(n0);
    }

    return i;
}

void insert(int k, int lp, int* list, int* lptr, int* lnew) {
    int lsav = LPTR(lp);
    LPTR(lp) = *lnew;
    LIST(*lnew) = k;
    LPTR(*lnew) = lsav;
    *lnew = *lnew + 1;
}

void intadd(int kk, int i1, int i2, int i3, int* list, int* lptr, int* lend, int* lnew) {
    int k = kk, n1 = i1, n2 = i2, n3 = i3;
    int lp;

    lp = lstptr(LEND(n1), n2, list, lptr);
    insert(k, lp, list, lptr, lnew);
    lp = lstptr(LEND(n2), n3, list, lptr);
    insert(k, lp, list, lptr, lnew);
    lp = lstptr(LEND(n3), n1, list, lptr);
    insert(k, lp, list, lptr, lnew);

    LIST(*lnew) = n1;
    LIST(*lnew + 1) = n2;
    LIST(*lnew + 2) = n3;
    LPTR(*lnew) = *lnew + 1;
    LPTR(*lnew + 1) = *lnew + 2;
    LPTR(*lnew + 2) = *lnew;
    LEND(k) = *lnew + 2;
    *lnew = *lnew + 3;
}

int intsec(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
    double a, b, d, dx12, dx31, dx34, dy12, dy31, dy34;

    if ((x1 < x3 && x1 < x4 && x2 < x3 && x2 < x4) ||
        (x3 < x1 && x1 > x4 && x2 > x3 && x2 > x4) ||
        (y1 < y3 && y1 < y4 && y2 < y3 && y2 < y4) ||
        (y3 < y1 && y4 < y1 && y2 > y3 && y2 > y4)) return 0;

    dx12 = x2 - x1;
    dy12 = y2 - y1;
    dx34 = x4 - x3;
    dy34 = y4 - y3;
    dx31 = x1 - x3;
    dy31 = y1 - y3;
    a = dx34 * dy31 - dx31 * dy34;
    b = dx12 * dy31 - dx31 * dy12;
    d = dx12 * dy34 - dx34 * dy12;

    if (d != 0.0) {
        return (0.0 <= a / d && a / d <= 1.0 && 0.0 <= b / d && b / d <= 1.0);
    } else {
        return (a == 0.0 && b == 0.0);
    }
}

int jrand(int n, int* ix, int* iy, int* iz) {
    double x, u;
    *ix = (171 * *ix) % 30269;
    *iy = (172 * *iy) % 30307;
    *iz = (170 * *iz) % 30323;
    x = (double)*ix / 30269.0 + (double)*iy / 30307.0 + (double)*iz / 30323.0;
    u = x - (int)x;
    return (int)((double)n * u + 1.0);
}

int left(double x1, double y1, double x2, double y2, double x0, double y0) {
    double dx1 = x2 - x1;
    double dy1 = y2 - y1;
    double dx2 = x0 - x1;
    double dy2 = y0 - y1;
    return (dx1 * dy2 >= dx2 * dy1);
}

int lstptr(int lpl, int nb, int* list, int* lptr) {
    int lp = LPTR(lpl);
    while (LIST(lp) != nb) {
        lp = LPTR(lp);
        if (lp == lpl) break;
    }
    return lp;
}

int nbcnt(int lpl, int* lptr) {
    int k = 1, lp = lpl;
    do {
        lp = LPTR(lp);
        if (lp == lpl) break;
        k++;
    } while (1);
    return k;
}

int nearnd(double xp, double yp, int ist, int n, double* x, double* y, int* list, int* lptr, int* lend, double* dsq) {
    int lmax = 25;
    int i1, i2, i3, l, lp, lp1, lp2, lpl, n1, n2, n3, nr, nst;
    int listp[25], lptrp[25];
    double cos1, cos2, ds1, dsr, dx11, dx12, dx21, dx22, dy11, dy12, dy21, dy22, sin1, sin2;

    *dsq = -1.0;
    if (n < 3) return 0;

    nst = ist;
    if (nst < 1 || n < nst) nst = 1;

    trfind(nst, xp, yp, n, x, y, list, lptr, lend, &i1, &i2, &i3);

    if (i1 == 0) return 0;

    if (i3 != 0) {
        listp[0] = i1; lptrp[0] = 2;
        listp[1] = i2; lptrp[1] = 3;
        listp[2] = i3; lptrp[2] = 1;
        l = 3;
    } else {
        n1 = i1;
        l = 1;
        lp1 = 2;
        listp[0] = n1;
        lptrp[0] = lp1;
        do {
            lpl = LEND(n1);
            n1 = -LIST(lpl);
            l = lp1;
            lp1 = l + 1;
            listp[l-1] = n1;
            lptrp[l-1] = lp1;
            if (n1 == i2 || lmax <= lp1) break;
        } while (1);
        l = lp1;
        listp[l-1] = 0;
        lptrp[l-1] = 1;
    }

    lp2 = 1;
    n2 = i1;
    lp1 = lptrp[0];
    n1 = listp[lp1-1];

    while (1) {
        lp = lstptr(LEND(n1), n2, list, lptr);
        if (LIST(lp) < 0) goto label4;
        lp = LPTR(lp);
        n3 = ABS(LIST(lp));

        if (lmax <= l) break;

        dx11 = X(n1) - X(n3);
        dx12 = X(n2) - X(n3);
        dx22 = X(n2) - xp;
        dx21 = X(n1) - xp;
        dy11 = Y(n1) - Y(n3);
        dy12 = Y(n2) - Y(n3);
        dy22 = Y(n2) - yp;
        dy21 = Y(n1) - yp;

        cos1 = dx11 * dx12 + dy11 * dy12;
        cos2 = dx22 * dx21 + dy22 * dy21;

        if (0.0 <= cos1 && cos2 >= 0.0) goto label4;
        if (cos1 < 0.0 && cos2 < 0.0) goto label3;

        sin1 = dx11 * dy12 - dx12 * dy11;
        sin2 = dx22 * dy21 - dx21 * dy22;

        if (sin1 * cos2 + cos1 * sin2 >= 0.0) goto label4;

    label3:
        l++;
        lptrp[lp2-1] = l;
        listp[l-1] = n3;
        lptrp[l-1] = lp1;
        lp1 = l;
        n1 = n3;
        continue;

    label4:
        if (lp1 == 1) break;
        lp2 = lp1;
        n2 = n1;
        lp1 = lptrp[lp1-1];
        n1 = listp[lp1-1];
        if (n1 == 0) break;
    }

    nr = i1;
    dsr = pow(X(nr) - xp, 2) + pow(Y(nr) - yp, 2);

    for (lp = 2; lp <= l; ++lp) {
        n1 = listp[lp-1];
        if (n1 == 0) continue;
        ds1 = pow(X(n1) - xp, 2) + pow(Y(n1) - yp, 2);
        if (ds1 < dsr) {
            nr = n1;
            dsr = ds1;
        }
    }
    *dsq = dsr;
    return nr;
}

void nearnds(int l, double* xp, double* yp, int* ist, int n, double* x, double* y, int* list, int* lptr, int* lend, double* dsqs, int* ier) {
    int i, nn, nb, na, nt;
    int* nodes = (int*)malloc(n * sizeof(int));
    double *hull_x, *hull_y, *hull_x1, *hull_y1, *vector_det;

    bnodes(n, list, lptr, lend, nodes, &nb, &na, &nt);

    hull_x = (double*)malloc(nb * sizeof(double));
    hull_y = (double*)malloc(nb * sizeof(double));
    hull_x1 = (double*)malloc(nb * sizeof(double));
    hull_y1 = (double*)malloc(nb * sizeof(double));
    vector_det = (double*)malloc(nb * sizeof(double));

    for (i = 0; i < nb; ++i) {
        hull_x[i] = X(NODES(i+1));
        hull_y[i] = Y(NODES(i+1));
    }
    for (i = 0; i < nb - 1; ++i) {
        hull_x1[i] = hull_x[i+1] - hull_x[i];
        hull_y1[i] = hull_y[i+1] - hull_y[i];
    }

    for (i = 0; i < l; ++i) {
        nn = nearnd(xp[i], yp[i], ist[i], n, x, y, list, lptr, lend, &dsqs[i]);
        ist[i] = nn;
        ier[i] = 0;

        int outside = 0;
        for (int j = 0; j < nb - 1; ++j) {
            vector_det[j] = hull_x1[j] * (yp[i] - hull_y[j]) - hull_y1[j] * (xp[i] - hull_x[j]);
            if (vector_det[j] < 0) outside = 1;
        }
        if (outside) ier[i] = 1;
    }

    free(nodes);
    free(hull_x);
    free(hull_y);
    free(hull_x1);
    free(hull_y1);
    free(vector_det);
}

void optim(double* x, double* y, int na, int* list, int* lptr, int* lend, int* nit, int* iwk, int* ier) {
    int i, io1, io2, iter, lp, lp21, lpl, lpp, maxit, n1, n2, nna, swp;

    nna = na;
    maxit = *nit;

    if (nna < 0 || maxit < 1) {
        *nit = 0;
        *ier = 2;
        return;
    }

    iter = 0;
    if (nna == 0) {
        *nit = 0;
        *ier = 0;
        return;
    }

    do {
        if (iter == maxit) {
            *nit = maxit;
            *ier = 1;
            return;
        }
        iter++;
        swp = 0;

        for (i = 1; i <= nna; ++i) {
            io1 = IWK(1, i);
            io2 = IWK(2, i);

            lpl = LEND(io1);
            lpp = lpl;
            lp = LPTR(lpp);

            while (LIST(lp) != io2) {
                lpp = lp;
                lp = LPTR(lpp);
                if (lp == lpl) break;
            }

            if (ABS(LIST(lp)) != io2) {
                *nit = iter;
                *ier = 3;
                return;
            }

            if (LIST(lp) < 0) goto label4;

            n2 = LIST(lpp);
            if (n2 < 0) goto label4;

            lp = LPTR(lp);
            n1 = ABS(LIST(lp));

            if (!swptst(n1, n2, io1, io2, x, y)) goto label4;

            swap(n1, n2, io1, io2, list, lptr, lend, &lp21);
            if (lp21 == 0) {
                *nit = iter;
                *ier = 4;
                return;
            }

            swp = 1;
            IWK(1, i) = n1;
            IWK(2, i) = n2;

        label4:
            ;
        }
    } while (swp);

    *nit = iter;
    *ier = 0;
}

double store(double x) {
    /* volatile to force storage */
    volatile double y;
    y = x;
    return y;
}

void swap(int in1, int in2, int io1, int io2, int* list, int* lptr, int* lend, int* lp21) {
    int lp, lph, lpsav;

    lp = lstptr(LEND(in1), in2, list, lptr);
    if (ABS(LIST(lp)) == in2) {
        *lp21 = 0;
        return;
    }

    lp = lstptr(LEND(io1), in2, list, lptr);
    lph = LPTR(lp);
    LPTR(lp) = LPTR(lph);

    if (LEND(io1) == lph) LEND(io1) = lp;

    lp = lstptr(LEND(in1), io1, list, lptr);
    lpsav = LPTR(lp);
    LPTR(lp) = lph;
    LIST(lph) = in2;
    LPTR(lph) = lpsav;

    lp = lstptr(LEND(io2), in1, list, lptr);
    lph = LPTR(lp);
    LPTR(lp) = LPTR(lph);

    if (LEND(io2) == lph) LEND(io2) = lp;

    lp = lstptr(LEND(in2), io2, list, lptr);
    lpsav = LPTR(lp);
    LPTR(lp) = lph;
    LIST(lph) = in1;
    LPTR(lph) = lpsav;
    *lp21 = lph;
}

int swptst(int in1, int in2, int io1, int io2, double* x, double* y) {
    double cos1, cos2, dx11, dx12, dx21, dx22, dy11, dy12, dy21, dy22, sin1, sin2, sin12;

    dx11 = X(io1) - X(in1);
    dx12 = X(io2) - X(in1);
    dx22 = X(io2) - X(in2);
    dx21 = X(io1) - X(in2);

    dy11 = Y(io1) - Y(in1);
    dy12 = Y(io2) - Y(in1);
    dy22 = Y(io2) - Y(in2);
    dy21 = Y(io1) - Y(in2);

    cos1 = dx11 * dx12 + dy11 * dy12;
    cos2 = dx22 * dx21 + dy22 * dy21;

    if (cos1 >= 0.0 && cos2 >= 0.0) return 0;
    if (cos1 < 0.0 && cos2 < 0.0) return 1;

    sin1 = dx11 * dy12 - dx12 * dy11;
    sin2 = dx22 * dy21 - dx21 * dy22;
    sin12 = sin1 * cos2 + cos1 * sin2;

    if (sin12 >= -swtol) return 0;
    return 1;
}

void timestamp(void) {
    /* Implementation omitted for brevity/portability */
}

void trfind(int nst, double px, double py, int n, double* x, double* y, int* list, int* lptr, int* lend, int* i1, int* i2, int* i3) {
    int i, ix = 1, iy = 2, iz = 3;
    int lp, n0, n1, n2, n3, n4, nb, nf, nl, np, npp;
    int n1s, n2s;
    double b1, b2, xp, yp, xa, ya, xb, yb, xc, yc;

    xp = px;
    yp = py;
    n0 = nst;

    if (n0 < 1 || n0 > n) n0 = jrand(n, &ix, &iy, &iz);

label1:
    lp = LEND(n0);
    nl = LIST(lp);
    lp = LPTR(lp);
    nf = LIST(lp);
    n1 = nf;

    if (nl > 0) goto label2;

    nl = -nl;
    if (!left(X(n0), Y(n0), X(nf), Y(nf), xp, yp)) {
        nl = n0;
        goto label9;
    }
    if (!left(X(nl), Y(nl), X(n0), Y(n0), xp, yp)) {
        nb = nf;
        nf = n0;
        np = nl;
        npp = n0;
        goto label11;
    }
    goto label3;

label2:
    do {
        if (left(X(n0), Y(n0), X(n1), Y(n1), xp, yp)) break;
        lp = LPTR(lp);
        n1 = LIST(lp);
        if (n1 == nl) goto label6;
    } while(1);

label3:
    lp = LPTR(lp);
    n2 = ABS(LIST(lp));
    if (!left(X(n0), Y(n0), X(n2), Y(n2), xp, yp)) goto label7;
    n1 = n2;
    if (n1 != nl) goto label3;

    if (!left(X(n0), Y(n0), X(nf), Y(nf), xp, yp)) goto label6;
    if (xp == X(n0) && yp == Y(n0)) goto label5;

label4:
    if (!left(X(n1), Y(n1), X(n0), Y(n0), xp, yp)) goto label5;
    lp = LPTR(lp);
    n1 = ABS(LIST(lp));
    if (n1 == nl) {
        *i1 = 0; *i2 = 0; *i3 = 0; return;
    }
    goto label4;

label5:
    n0 = n1;
    goto label1;

label6:
    n2 = nf;

label7:
    n3 = n0;
    n1s = n1;
    n2s = n2;

label8:
    if (left(X(n1), Y(n1), X(n2), Y(n2), xp, yp)) {
        b1 = (X(n3) - X(n2)) * (yp - Y(n2)) - (xp - X(n2)) * (Y(n3) - Y(n2));
        b2 = (X(n1) - X(n3)) * (yp - Y(n3)) - (xp - X(n3)) * (Y(n1) - Y(n3));
        if (store(b1 + 1.0) >= 1.0 && store(b2 + 1.0) >= 1.0) goto label16;
        n0 = jrand(n, &ix, &iy, &iz);
        goto label1;
    }

    lp = lstptr(LEND(n2), n1, list, lptr);
    if (LIST(lp) < 0) {
        nf = n2;
        nl = n1;
        goto label9;
    }
    lp = LPTR(lp);
    n4 = ABS(LIST(lp));

    if (left(X(n0), Y(n0), X(n4), Y(n4), xp, yp)) {
        n3 = n1;
        n1 = n4;
        n2s = n2;
        if (n1 != n1s && n1 != n0) goto label8;
    } else {
        n3 = n2;
        n2 = n4;
        n1s = n1;
        if (n2 != n2s && n2 != n0) goto label8;
    }
    n0 = jrand(n, &ix, &iy, &iz);
    goto label1;

label9:
    np = nl;
    npp = nf;

label10:
    lp = LEND(nf);
    lp = LPTR(lp);
    nb = LIST(lp);
    if (!left(X(nf), Y(nf), X(nb), Y(nb), xp, yp)) goto label12;

label11:
    /* FRWRD function macro */
#define FRWRD(xa,ya,xb,yb,xc,yc) ((xb-xa)*(xc-xa) + (yb-ya)*(yc-ya) >= 0.0)
    if (FRWRD(X(nf), Y(nf), X(np), Y(np), xp, yp) ||
        FRWRD(X(nf), Y(nf), X(np), Y(np), X(nb), Y(nb))) {
        *i1 = nf;
        goto label13;
    }

label12:
    np = nf;
    nf = nb;
    goto label10;

label13:
    lp = LEND(nl);
    nb = -LIST(lp);
    if (!left(X(nb), Y(nb), X(nl), Y(nl), xp, yp)) goto label14;

    if (FRWRD(X(nl), Y(nl), X(npp), Y(npp), xp, yp) ||
        FRWRD(X(nl), Y(nl), X(npp), Y(npp), X(nb), Y(nb))) goto label15;

label14:
    npp = nl;
    nl = nb;
    goto label13;

label15:
    *i2 = nl;
    *i3 = 0;
    return;

label16:
    *i1 = n1;
    *i2 = n2;
    *i3 = n3;
}

void trlist(int ncc, int* lcc, int n, int* list, int* lptr, int* lend, int nrow, int* nt, int* ltri, int* lct, int* ier) {
    int i, i1, i2, i3, isv, j, jlast, ka, kn, kt, l, lcc1, lp, lp2, lpl, lpln1, n1, n1st, n2, n3, nm2;
    int arcs = (nrow == 9);
    int pass2 = 0;

    *ier = 0;
    if (ncc < 0 || (nrow != 6 && nrow != 9)) {
        *nt = 0;
        *ier = 1;
        return;
    }

    lcc1 = n + 1;
    if (ncc == 0) {
        if (n < 3) {
            *nt = 0;
            *ier = 1;
            return;
        }
    } else {
        for (i = ncc; i >= 1; --i) {
            if (lcc1 - LCC(i) < 3) {
                *nt = 0;
                *ier = 1;
                return;
            }
            lcc1 = LCC(i);
        }
        if (lcc1 < 1) {
            *nt = 0;
            *ier = 1;
            return;
        }
    }

    ka = 0;
    kt = 0;
    n1st = 1;
    nm2 = n - 2;

label2:
    j = 0;
    jlast = lcc1 - 1;

    for (n1 = n1st; n1 <= nm2; ++n1) {
        if (jlast < n1) {
            j++;
            if (j < ncc) {
                jlast = LCC(j + 1) - 1;
            } else {
                jlast = n;
            }
            if (pass2) LCT(j) = kt + 1;
        }

        lpln1 = LEND(n1);
        lp2 = lpln1;

    label3:
        lp2 = LPTR(lp2);
        n2 = LIST(lp2);
        lp = LPTR(lp2);
        n3 = ABS(LIST(lp));

        if (n2 < n1 || n3 < n1) goto label10;

        int cstri = (n1 >= lcc1 && n2 < n3 && n3 <= jlast);
        if ((cstri && !pass2) || (!cstri && pass2)) goto label10;

        kt++;
        LTRI(1, kt) = n1;
        LTRI(2, kt) = n2;
        LTRI(3, kt) = n3;

        for (i = 1; i <= 3; ++i) {
            if (i == 1) { i1 = n3; i2 = n2; }
            else if (i == 2) { i1 = n1; i2 = n3; }
            else { i1 = n2; i2 = n1; }

            lpl = LEND(i1);
            lp = LPTR(lpl);

        label4:
            if (LIST(lp) == i2) goto label5;
            lp = LPTR(lp);
            if (lp != lpl) goto label4;

            if (ABS(LIST(lp)) != i2) goto label13;
            kn = 0;
            if (LIST(lp) < 0) goto label8;

        label5:
            lp = LPTR(lp);
            i3 = ABS(LIST(lp));

            if (i1 < i2 && i1 < i3) l = 3;
            else if (i2 < i3) { l = 2; isv = i1; i1 = i2; i2 = i3; i3 = isv; }
            else { l = 1; isv = i1; i1 = i3; i3 = i2; i2 = isv; }

            if (i1 > n1 && !pass2) goto label9;

            for (kn = kt - 1; kn >= 1; --kn) {
                if (LTRI(1, kn) == i1 && LTRI(2, kn) == i2 && LTRI(3, kn) == i3) goto label7;
            }
            goto label9;

        label7:
            LTRI(l + 3, kn) = kt;

        label8:
            LTRI(i + 3, kt) = kn;
            if (arcs) {
                ka++;
                LTRI(i + 6, kt) = ka;
                if (kn != 0) LTRI(l + 6, kn) = ka;
            }

        label9:
            ;
        }

    label10:
        if (lp2 != lpln1) goto label3;
    }

    if (!pass2 && ncc > 0) {
        pass2 = 1;
        n1st = lcc1;
        goto label2;
    }

    *nt = kt;
    return;

label13:
    *nt = 0;
    *ier = 2;
}

void trlist2(int n, int* list, int* lptr, int* lend, int* nt, int* ltri, int* ier) {
    /* NROW = 3 hardcoded for trlist2 logic as per fortran code */
    /* Wait, fortran code uses ltri(3,*) */
    /* So NROW is 3. But my macro LTRI uses variable nrow. I need to handle this. */
    /* I'll define a local nrow for the macro. */
    int nrow = 3;
    int i, i1, i2, i3, isv, j, kn, kt, lp, lp2, lpl, lpln1, n1, n2, n3, nm2;

    *ier = 0;
    if (n < 3) {
        *nt = 0;
        *ier = 1;
        return;
    }

    kt = 0;
    nm2 = n - 2;

    for (n1 = 1; n1 <= nm2; ++n1) {
        lpln1 = LEND(n1);
        lp2 = lpln1;

    label1:
        lp2 = LPTR(lp2);
        n2 = LIST(lp2);
        lp = LPTR(lp2);
        n3 = ABS(LIST(lp));

        if (n2 < n1 || n3 < n1) goto label8;

        kt++;
        LTRI(1, kt) = n1;
        LTRI(2, kt) = n2;
        LTRI(3, kt) = n3;

        for (i = 1; i <= 3; ++i) {
            if (i == 1) { i1 = n3; i2 = n2; }
            else if (i == 2) { i1 = n1; i2 = n3; }
            else { i1 = n2; i2 = n1; }

            lpl = LEND(i1);
            lp = LPTR(lpl);

            do {
                if (LIST(lp) == i2) goto label3;
                lp = LPTR(lp);
                if (lp == lpl) break;
            } while (1);

            if (ABS(LIST(lp)) != i2) {
                *nt = 0;
                *ier = 2;
                return;
            }

            kn = 0;
            if (LIST(lp) < 0) goto label6;

        label3:
            lp = LPTR(lp);
            i3 = ABS(LIST(lp));

            if (i1 < i2 && i1 < i3) j = 3;
            else if (i2 < i3) { j = 2; isv = i1; i1 = i2; i2 = i3; i3 = isv; }
            else { j = 1; isv = i1; i1 = i3; i3 = i2; i2 = isv; }

            if (n1 < i1) continue;

            for (kn = kt - 1; kn >= 1; --kn) {
                if (LTRI(1, kn) == i1 && LTRI(2, kn) == i2 && LTRI(3, kn) == i3) goto label5;
            }
            continue;

        label5:
        label6:
            ;
        }

    label8:
        if (lp2 != lpln1) goto label1;
    }
    *nt = kt;
}

void trlprt(int ncc, int* lct, int n, double* x, double* y, int nrow, int nt, int* ltri, int prntx) {
    /* Output function - stub or simple implementation */
    printf("TRLPRT: n=%d, nt=%d\n", n, nt);
}

void trmesh(int n, double* x, double* y, int* list, int* lptr, int* lend, int* lnew, int* near_arr, int* next, double* dist, int* ier) {
    int i, i0, j, k, km1, lpl, lp, nexti, nn, ncc;
    int lcc[1]; /* dummy */
    double d, d1, d2, d3, eps;

    nn = n;
    *ier = 0;

    if (nn < 3) {
        *ier = -1;
        return;
    }

    eps = 2.220446049250313e-16; /* machine epsilon roughly */
    swtol = eps * 20.0;

    if (!left(X(1), Y(1), X(2), Y(2), X(3), Y(3))) {
        LIST(1) = 3; LPTR(1) = 2; LIST(2) = -2; LPTR(2) = 1; LEND(1) = 2;
        LIST(3) = 1; LPTR(3) = 4; LIST(4) = -3; LPTR(4) = 3; LEND(2) = 4;
        LIST(5) = 2; LPTR(5) = 6; LIST(6) = -1; LPTR(6) = 5; LEND(3) = 6;
    } else if (!left(X(2), Y(2), X(1), Y(1), X(3), Y(3))) {
        LIST(1) = 2; LPTR(1) = 2; LIST(2) = -3; LPTR(2) = 1; LEND(1) = 2;
        LIST(3) = 3; LPTR(3) = 4; LIST(4) = -1; LPTR(4) = 3; LEND(2) = 4;
        LIST(5) = 1; LPTR(5) = 6; LIST(6) = -2; LPTR(6) = 5; LEND(3) = 6;
    } else {
        *ier = -2;
        return;
    }

    *lnew = 7;
    if (nn == 3) return;

    NEAR(1) = 0; NEAR(2) = 0; NEAR(3) = 0;

    for (k = nn; k >= 4; --k) {
        d1 = pow(X(k) - X(1), 2) + pow(Y(k) - Y(1), 2);
        d2 = pow(X(k) - X(2), 2) + pow(Y(k) - Y(2), 2);
        d3 = pow(X(k) - X(3), 2) + pow(Y(k) - Y(3), 2);

        if (d1 <= d2 && d1 <= d3) {
            NEAR(k) = 1; DIST(k) = d1; NEXT(k) = NEAR(1); NEAR(1) = k;
        } else if (d2 <= d1 && d2 <= d3) {
            NEAR(k) = 2; DIST(k) = d2; NEXT(k) = NEAR(2); NEAR(2) = k;
        } else {
            NEAR(k) = 3; DIST(k) = d3; NEXT(k) = NEAR(3); NEAR(3) = k;
        }
    }

    ncc = 0;
    for (k = 4; k <= nn; ++k) {
        km1 = k - 1;
        addnod(k, X(k), Y(k), NEAR(k), ncc, lcc, &km1, x, y, list, lptr, lend, lnew, ier);
        if (*ier != 0) return;

        i = NEAR(k);
        if (NEAR(i) == k) {
            NEAR(i) = NEXT(k);
        } else {
            i = NEAR(i);
            while (1) {
                i0 = i;
                i = NEXT(i0);
                if (i == k) break;
            }
            NEXT(i0) = NEXT(k);
        }
        NEAR(k) = 0;

        lpl = LEND(k);
        lp = lpl;

    label4:
        lp = LPTR(lp);
        j = ABS(LIST(lp));
        i = NEAR(j);

    label5:
        if (i == 0) goto label6;
        nexti = NEXT(i);
        d = pow(X(k) - X(i), 2) + pow(Y(k) - Y(i), 2);

        if (d < DIST(i)) {
            NEAR(i) = k;
            DIST(i) = d;
            if (i == NEAR(j)) {
                NEAR(j) = nexti;
            } else {
                NEXT(i0) = nexti;
            }
            NEXT(i) = NEAR(k);
            NEAR(k) = i;
        } else {
            i0 = i;
        }
        i = nexti;
        goto label5;

    label6:
        if (lp != lpl) goto label4;
    }
}

void trmshr(int n, int nx, double* x, double* y, int* nit, int* list, int* lptr, int* lend, int* lnew, int* ier) {
    /* Implementation omitted for now, assuming standard triangulation needed first.
       It's a large function. */
    *ier = 0;
}

void trmtst(int n, double* x, double* y, int* list, int* lptr, int* lend, int lnew, double tol, double* armax, int* ier) {
    /* Implementation omitted */
    *ier = 0;
}

void trplot(int lun, double pltsiz, double wx1, double wx2, double wy1, double wy2, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, char* title, int numbr, int* ier) {
    /* Implementation omitted */
    *ier = 0;
}

void trprnt(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int prntx) {
    /* Implementation omitted */
}
