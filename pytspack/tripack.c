#include "tripack.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* --- Global State (Equivalent to MODULE SWPCOM) --- */
static float swtol_val;

/* --- Helper Function: STORE --- 
   Forces storage to memory to truncate precision (legacy float handling) */
float store(float x) {
    volatile float y = x;
    return y;
}

/* --- Helper Function: JRAND --- 
   Pseudo-random number generator */
int jrand(int n, int *ix, int *iy, int *iz) {
    *ix = (171 * (*ix)) % 30269;
    *iy = (172 * (*iy)) % 30307;
    *iz = (170 * (*iz)) % 30323;
    
    float x_rand = ((float)(*ix) / 30269.0f) + 
                   ((float)(*iy) / 30307.0f) + 
                   ((float)(*iz) / 30323.0f);
    float u = x_rand - (int)x_rand;
    return (int)((float)n * u + 1.0f);
}

/* --- Helper Function: LEFT --- 
   Geometric predicate: Is N0 to the left of line N1->N2? */
bool left(float x1, float y1, float x2, float y2, float x0, float y0) {
    float dx1 = x2 - x1;
    float dy1 = y2 - y1;
    float dx2 = x0 - x1;
    float dy2 = y0 - y1;
    return (dx1 * dy2 >= dx2 * dy1);
}

/* --- Helper Function: LSTPTR --- 
   Find pointer to NB in N0's adjacency list (where lpl = lend[N0]) */
int lstptr(int lpl, int nb, int *list, int *lptr) {
    int lp = lptr[lpl - 1];
    while (abs(list[lp - 1]) != nb) {
        lp = lptr[lp - 1];
        if (lp == lpl) break;
    }
    return lp;
}

/* --- Helper Function: SWPTST --- 
   Circumcircle test for diagonal swapping */
bool swptst(int in1, int in2, int io1, int io2, float *x, float *y) {
    float dx11 = x[io1 - 1] - x[in1 - 1];
    float dx12 = x[io2 - 1] - x[in1 - 1];
    float dx22 = x[io2 - 1] - x[in2 - 1];
    float dx21 = x[io1 - 1] - x[in2 - 1];

    float dy11 = y[io1 - 1] - y[in1 - 1];
    float dy12 = y[io2 - 1] - y[in1 - 1];
    float dy22 = y[io2 - 1] - y[in2 - 1];
    float dy21 = y[io1 - 1] - y[in2 - 1];

    float cos1 = dx11 * dx12 + dy11 * dy12;
    float cos2 = dx22 * dx21 + dy22 * dy21;

    if (cos1 >= 0.0f && cos2 >= 0.0f) return false;
    if (cos1 < 0.0f && cos2 < 0.0f) return true;

    float sin1 = dx11 * dy12 - dx12 * dy11;
    float sin2 = dx22 * dy21 - dx21 * dy22;
    float sin12 = sin1 * cos2 + cos1 * sin2;

    return (sin12 >= -swtol_val);
}

/* --- Helper Function: INSERT --- */
void insert_node(int k, int lp, int *list, int *lptr, int *lnew) {
    int lsav = lptr[lp - 1];
    lptr[lp - 1] = *lnew;
    list[*lnew - 1] = k;
    lptr[*lnew - 1] = lsav;
    (*lnew)++;
}

/* --- Helper Function: SWAP --- 
   Swaps diagonal IO1-IO2 with IN1-IN2 */
void swap_arc(int in1, int in2, int io1, int io2, 
              int *list, int *lptr, int *lend, int *lp21) {
    int lp, lph, lpsav;

    /* Check if IN1 and IN2 are already adjacent */
    lp = lstptr(lend[in1 - 1], in2, list, lptr);
    if (abs(list[lp - 1]) == in2) {
        *lp21 = 0;
        return;
    }

    /* Delete IO2 as neighbor of IO1 */
    lp = lstptr(lend[io1 - 1], in2, list, lptr);
    lph = lptr[lp - 1];
    lptr[lp - 1] = lptr[lph - 1];
    if (lend[io1 - 1] == lph) lend[io1 - 1] = lp;

    /* Insert IN2 as neighbor of IN1 following IO1 */
    lp = lstptr(lend[in1 - 1], io1, list, lptr);
    lpsav = lptr[lp - 1];
    lptr[lp - 1] = lph;
    list[lph - 1] = in2;
    lptr[lph - 1] = lpsav;

    /* Delete IO1 as neighbor of IO2 */
    lp = lstptr(lend[io2 - 1], in1, list, lptr);
    lph = lptr[lp - 1];
    lptr[lp - 1] = lptr[lph - 1];
    if (lend[io2 - 1] == lph) lend[io2 - 1] = lp;

    /* Insert IN1 as neighbor of IN2 following IO2 */
    lp = lstptr(lend[in2 - 1], io2, list, lptr);
    lpsav = lptr[lp - 1];
    lptr[lp - 1] = lph;
    list[lph - 1] = in1;
    lptr[lph - 1] = lpsav;

    *lp21 = lph;
}

/* --- Helper Function: TRFIND --- 
   Locates a point P relative to the triangulation */
void trfind(int nst, float px, float py, int n, float *x, float *y,
            int *list, int *lptr, int *lend, 
            int *i1, int *i2, int *i3) {
    
    static int ix = 1, iy = 2, iz = 3; /* Seeds */
    int n0, n1, n2, n3, n4, nf, nl, lp;
    int n1s, n2s, nb, np, npp;
    float b1, b2, xp, yp;

    xp = px; yp = py;
    n0 = nst;
    if (n0 < 1 || n0 > n) n0 = jrand(n, &ix, &iy, &iz);

    // Loop Start (Label 1)
    while (1) {
        lp = lend[n0 - 1];
        nl = list[lp - 1];
        lp = lptr[lp - 1];
        nf = list[lp - 1];
        n1 = nf;

        /* Check boundaries */
        if (nl > 0) {
            // Interior node logic (Label 2)
             while (left(x[n0-1], y[n0-1], x[n1-1], y[n1-1], xp, yp)) {
                lp = lptr[lp - 1];
                n1 = list[lp - 1];
                if (n1 == nl) goto Label6;
            }
        } else {
            // Boundary node logic
            nl = -nl;
            if (!left(x[n0-1], y[n0-1], x[nf-1], y[nf-1], xp, yp)) {
                nl = n0;
                goto Label9; 
            }
            if (!left(x[nl-1], y[nl-1], x[n0-1], y[n0-1], xp, yp)) {
                nb = nf; nf = n0; np = nl; npp = n0;
                goto Label11;
            }
        }

        // Label 3
        while (1) {
            lp = lptr[lp - 1];
            n2 = abs(list[lp - 1]);
            if (!left(x[n0-1], y[n0-1], x[n2-1], y[n2-1], xp, yp)) {
                n1 = n2;
                if (n1 != nl) continue; // Go to 3 start
            }
            break; 
        }

        if (!left(x[n0-1], y[n0-1], x[nf-1], y[nf-1], xp, yp)) goto Label6;
        if (xp == x[n0-1] && yp == y[n0-1]) goto Label5;

        // Label 4
        while (!left(x[n1-1], y[n1-1], x[n0-1], y[n0-1], xp, yp)) {
            lp = lptr[lp - 1];
            n1 = abs(list[lp - 1]);
            if (n1 == nl) { *i1=0; *i2=0; *i3=0; return; } // Collinear
        }

        Label5:
        n0 = n1;
        continue; // Go to 1

        Label6:
        n2 = nf;
        n3 = n0;
        n1s = n1;
        n2s = n2;

        // Edge hopping loop (Label 8)
        while (1) {
            if (left(x[n1-1], y[n1-1], x[n2-1], y[n2-1], xp, yp)) {
                // P is in (N1, N2, N3)
                b1 = (x[n3-1] - x[n2-1]) * (yp - y[n2-1]) - (xp - x[n2-1]) * (y[n3-1] - y[n2-1]);
                b2 = (x[n1-1] - x[n3-1]) * (yp - y[n3-1]) - (xp - x[n3-1]) * (y[n1-1] - y[n3-1]);
                if (store(b1 + 1.0f) >= 1.0f && store(b2 + 1.0f) >= 1.0f) {
                     *i1 = n1; *i2 = n2; *i3 = n3;
                     return;
                }
                // Restart random
                n0 = jrand(n, &ix, &iy, &iz);
                break; // Break to outer loop (Go to 1)
            }

            lp = lstptr(lend[n2-1], n1, list, lptr);
            if (list[lp-1] < 0) {
                nf = n2; nl = n1;
                goto Label9;
            }
            lp = lptr[lp-1];
            n4 = abs(list[lp-1]);

            if (left(x[n0-1], y[n0-1], x[n4-1], y[n4-1], xp, yp)) {
                n3 = n1; n1 = n4; n2s = n2;
                if (n1 != n1s && n1 != n0) continue;
            } else {
                n3 = n2; n2 = n4; n1s = n1;
                if (n2 != n2s && n2 != n0) continue;
            }
            
            // Cycle detected
            n0 = jrand(n, &ix, &iy, &iz);
            break; // Break to outer loop
        }
    }

    Label9:
    np = nl; npp = nf;
    
    // Label 10
    while(1) {
        lp = lend[nf-1];
        lp = lptr[lp-1];
        nb = list[lp-1];
        if (!left(x[nf-1], y[nf-1], x[nb-1], y[nb-1], xp, yp)) {
            np = nf; nf = nb;
            continue;
        }
        break;
    }
    
    Label11: 
    // Forward check
    {
       float val1 = (x[np-1]-x[nf-1])*(xp-x[nf-1]) + (y[np-1]-y[nf-1])*(yp-y[nf-1]);
       float val2 = (x[np-1]-x[nf-1])*(x[nb-1]-x[nf-1]) + (y[np-1]-y[nf-1])*(y[nb-1]-y[nf-1]);
       if (val1 >= 0.0f || val2 >= 0.0f) { *i1 = nf; goto Label13; }
    }
    np = nf; nf = nb; goto Label10; // Logic of 12 goes back to 10

    Label13:
    while(1) {
        lp = lend[nl-1];
        nb = -list[lp-1];
        if (!left(x[nb-1], y[nb-1], x[nl-1], y[nl-1], xp, yp)) {
             npp = nl; nl = nb;
             continue;
        }
        // Check 15
        float val1 = (x[npp-1]-x[nl-1])*(xp-x[nl-1]) + (y[npp-1]-y[nl-1])*(yp-y[nl-1]);
        float val2 = (x[npp-1]-x[nl-1])*(x[nb-1]-x[nl-1]) + (y[npp-1]-y[nl-1])*(y[nb-1]-y[nl-1]);
        if (val1 >= 0.0f || val2 >= 0.0f) break; 
        
        npp = nl; nl = nb;
    }
    
    *i2 = nl;
    *i3 = 0;
}

/* --- Helper Function: BDYADD --- */
void bdyadd(int kk, int i1, int i2, int *list, int *lptr, int *lend, int *lnew) {
    int lp, lsav, next, nsav;
    int k = kk, n1 = i1, n2 = i2;

    lp = lend[n1 - 1];
    lsav = lptr[lp - 1];
    lptr[lp - 1] = *lnew;
    list[*lnew - 1] = -k;
    lptr[*lnew - 1] = lsav;
    lend[n1 - 1] = *lnew;
    (*lnew)++;
    next = -list[lp - 1];
    list[lp - 1] = next;
    nsav = next;

    while (1) {
        lp = lend[next - 1];
        insert_node(k, lp, list, lptr, lnew);
        if (next == n2) break;
        next = -list[lp - 1];
        list[lp - 1] = next;
    }

    lsav = *lnew;
    list[*lnew - 1] = n1;
    lptr[*lnew - 1] = *lnew + 1;
    (*lnew)++;
    next = nsav;

    while (next != n2) {
        list[*lnew - 1] = next;
        lptr[*lnew - 1] = *lnew + 1;
        (*lnew)++;
        lp = lend[next - 1];
        next = list[lp - 1];
    }

    list[*lnew - 1] = -n2;
    lptr[*lnew - 1] = lsav;
    lend[k - 1] = *lnew;
    (*lnew)++;
}

/* --- Helper Function: INTADD --- */
void intadd(int kk, int i1, int i2, int i3, int *list, int *lptr, int *lend, int *lnew) {
    int k = kk;
    int lp;

    lp = lstptr(lend[i1 - 1], i2, list, lptr);
    insert_node(k, lp, list, lptr, lnew);
    lp = lstptr(lend[i2 - 1], i3, list, lptr);
    insert_node(k, lp, list, lptr, lnew);
    lp = lstptr(lend[i3 - 1], i1, list, lptr);
    insert_node(k, lp, list, lptr, lnew);

    list[*lnew - 1] = i1;
    list[*lnew]     = i2;
    list[*lnew + 1] = i3;
    lptr[*lnew - 1] = *lnew + 1;
    lptr[*lnew]     = *lnew + 2;
    lptr[*lnew + 1] = *lnew;
    lend[k - 1] = *lnew + 2;
    *lnew += 3;
}

/* --- SUBROUTINE ADDNOD --- */
void addnod(int k, float xk, float yk, int ist, int ncc, int *lcc, int *n,
            float *x, float *y, int *list, int *lptr, int *lend, int *lnew, int *ier) {
    int i1, i2, i3, l, kk = k;
    int lp, lpf, lpo1, io1, io2, in1, lp21;

    if (kk < 1 || ist < 1 || ist > *n || ncc < 0 || *n < 3) { *ier = -1; return; }
    
    // Note: NCC/Constraint checks omitted for brevity in conversion logic, 
    // assuming NCC=0 for pure Delaunay logic usually.

    trfind(ist, xk, yk, *n, x, y, list, lptr, lend, &i1, &i2, &i3);

    if (i1 == 0) { *ier = -2; return; } // Collinear
    if (i3 != 0) {
        if (xk == x[i1-1] && yk == y[i1-1]) { *ier = i1; return; }
        if (xk == x[i2-1] && yk == y[i2-1]) { *ier = i2; return; }
        if (xk == x[i3-1] && yk == y[i3-1]) { *ier = i3; return; }
    }

    *ier = 0;
    int nm1 = *n;
    (*n)++;
    
    // Shift data if inserting (rarely used in plain Delaunay construction)
    if (kk < *n) {
        for (int ibk = nm1; ibk >= kk; ibk--) {
            x[ibk] = x[ibk - 1];
            y[ibk] = y[ibk - 1];
            lend[ibk] = lend[ibk - 1];
        }
        // Adjustment of pointers would happen here for full constraints
    }

    x[kk - 1] = xk;
    y[kk - 1] = yk;

    if (i3 == 0) bdyadd(kk, i1, i2, list, lptr, lend, lnew);
    else         intadd(kk, i1, i2, i3, list, lptr, lend, lnew);

    // Optimization (swapping)
    lp = lend[kk - 1];
    lpf = lptr[lp - 1];
    io2 = list[lpf - 1];
    lpo1 = lptr[lpf - 1];
    io1 = abs(list[lpo1 - 1]);

    while (1) {
        lp = lstptr(lend[io1 - 1], io2, list, lptr);
        if (list[lp - 1] < 0) goto Label6;
        lp = lptr[lp - 1];
        in1 = abs(list[lp - 1]);

        if (swptst(in1, kk, io1, io2, x, y)) {
            swap_arc(in1, kk, io1, io2, list, lptr, lend, &lp21);
            if (lp21 == 0) { *ier = -4; return; }
            io1 = in1;
            continue;
        }

        Label6:
        if (lpo1 == lpf || list[lpo1 - 1] < 0) return;
        io2 = io1;
        lpo1 = lptr[lpo1 - 1];
        io1 = abs(list[lpo1 - 1]);
    }
}

/* --- MAIN SUBROUTINE: TRMESH --- */
void trmesh(int n, float *x, float *y, int *list, int *lptr, int *lend, 
            int *lnew, int *near_arr, int *next, float *dist, int *ier) {
    
    float d, d1, d2, d3, eps;
    int i, i0, j, k, km1, lp, lpl, nexti, ncc = 0;
    int lcc_dummy[1];

    if (n < 3) { *ier = -1; return; }

    /* Set Tolerance */
    eps = 1.0f;
    while (store(eps + 1.0f) > 1.0f) eps /= 2.0f;
    swtol_val = eps * 20.0f;

    /* Initialize Triangle */
    if (!left(x[0], y[0], x[1], y[1], x[2], y[2])) {
        // (3, 2, 1) order
        list[0] = 3; lptr[0] = 2;
        list[1] = -2; lptr[1] = 1; lend[0] = 2;

        list[2] = 1; lptr[2] = 4;
        list[3] = -3; lptr[3] = 3; lend[1] = 4;

        list[4] = 2; lptr[4] = 6;
        list[5] = -1; lptr[5] = 5; lend[2] = 6;
    } else if (!left(x[1], y[1], x[0], y[0], x[2], y[2])) {
        // (1, 2, 3) order
        list[0] = 2; lptr[0] = 2;
        list[1] = -3; lptr[1] = 1; lend[0] = 2;

        list[2] = 3; lptr[2] = 4;
        list[3] = -1; lptr[3] = 3; lend[1] = 4;

        list[4] = 1; lptr[4] = 6;
        list[5] = -2; lptr[5] = 5; lend[2] = 6;
    } else {
        *ier = -2; // Collinear first 3
        return;
    }

    *lnew = 7;
    if (n == 3) { *ier = 0; return; }

    /* Initialize NN search structure */
    near_arr[0] = 0; near_arr[1] = 0; near_arr[2] = 0;
    
    for (k = n; k >= 4; k--) {
        d1 = pow(x[k-1]-x[0], 2) + pow(y[k-1]-y[0], 2);
        d2 = pow(x[k-1]-x[1], 2) + pow(y[k-1]-y[1], 2);
        d3 = pow(x[k-1]-x[2], 2) + pow(y[k-1]-y[2], 2);

        if (d1 <= d2 && d1 <= d3) {
            near_arr[k-1] = 1; dist[k-1] = d1;
            next[k-1] = near_arr[0]; near_arr[0] = k;
        } else if (d2 <= d1 && d2 <= d3) {
            near_arr[k-1] = 2; dist[k-1] = d2;
            next[k-1] = near_arr[1]; near_arr[1] = k;
        } else {
            near_arr[k-1] = 3; dist[k-1] = d3;
            next[k-1] = near_arr[2]; near_arr[2] = k;
        }
    }

    /* Add remaining nodes */
    for (k = 4; k <= n; k++) {
        km1 = k - 1;
        addnod(k, x[k-1], y[k-1], near_arr[k-1], ncc, lcc_dummy, 
               &km1, x, y, list, lptr, lend, lnew, ier);
        if (*ier != 0) return;

        // Remove K from NEAR list
        i = near_arr[k-1];
        if (near_arr[i-1] == k) {
            near_arr[i-1] = next[k-1];
        } else {
            i = near_arr[i-1];
            while (1) {
                i0 = i;
                i = next[i0-1];
                if (i == k) break;
            }
            next[i0-1] = next[k-1];
        }
        near_arr[k-1] = 0;

        // Update NN for K's neighbors
        lpl = lend[k-1];
        lp = lpl;
        
        do {
            lp = lptr[lp-1];
            j = abs(list[lp-1]);
            
            i = near_arr[j-1];
            while (i != 0) {
                nexti = next[i-1];
                d = pow(x[k-1]-x[i-1], 2) + pow(y[k-1]-y[i-1], 2);
                if (d < dist[i-1]) {
                    near_arr[i-1] = k;
                    dist[i-1] = d;
                    if (i == near_arr[j-1]) {
                        near_arr[j-1] = nexti;
                    } else {
                        next[i0-1] = nexti;
                    }
                    next[i-1] = near_arr[k-1];
                    near_arr[k-1] = i;
                } else {
                    i0 = i;
                }
                i = nexti;
            }
        } while (lp != lpl);
    }
}
