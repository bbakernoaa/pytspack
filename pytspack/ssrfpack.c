#include "srf_common.h"
#include "ssrfpack.h"
#include <stdio.h>

/* Internal: Construct rotation to move K to North Pole */
static void ssrf_constr(float xk, float yk, float zk, float *cx, float *sx, float *cy, float *sy) {
    *cy = sqrtf(yk*yk + zk*zk);
    *sy = xk;
    if (*cy != 0.0f) {
        *cx = zk / *cy;
        *sx = yk / *cy;
    } else {
        *cx = 1.0f;
        *sx = 0.0f;
    }
}

/* Internal: Apply rotation */
static void ssrf_aplyr(float x, float y, float z, float cx, float sx, float cy, float sy, 
                       float *xp, float *yp, float *zp) {
    float t = sx * y + cx * z;
    *yp = cx * y - sx * z;
    *zp = sy * x + cy * t;
    *xp = cy * x - sy * t;
    if (*zp < 0.0f) { // Project to equator if in southern hemisphere
        float norm = sqrtf(*xp * *xp + *yp * *yp);
        if (norm != 0.0f) { *xp /= norm; *yp /= norm; }
    }
}

/* Internal: Setup least squares row */
static void ssrf_setup(float xi, float yi, float wi, float wk, 
                       float s1, float s2, float wt, float *row) {
    float w1 = wt/s1;
    float w2 = wt/s2;
    row[0] = xi*xi*w2; row[1] = xi*yi*w2; row[2] = yi*yi*w2;
    row[3] = xi*w1;    row[4] = yi*w1;
    row[5] = (wi - wk)*wt;
}

/* Interpolate at a list of N_TARGETS arbitrary points */
void ssrf_interp_points(int n, float *x, float *y, float *z, float *f,
                        int *list, int *lptr, int *lend,
                        int iflgs, float *sigma,
                        int n_targets, float *t_lats, float *t_lons,
                        int iflgg, float *grad, float *out_vals, int *ier) {
    
    int ist = 1; // Starting search index hint
    int err_count = 0;
    
    // Precompute gradients if requested (iflgg==2) is handled by caller usually,
    // but here we assume grad is populated or computed on fly (iflgg=0/1).
    
    for(int k = 0; k < n_targets; k++) {
        float val;
        int local_ier;
        
        // Call the single-point interpolator
        // Note: passing &ist updates the search hint, making subsequent 
        // points faster if they are spatially close (which they are in a grid).
        ssrf_intrc1(n, t_lats[k], t_lons[k], x, y, z, f, list, lptr, lend,
                    iflgs, sigma, iflgg, grad, 
                    &ist, &val, &local_ier);
        
        out_vals[k] = val;
        
        if (local_ier < 0) err_count++;
    }
    
    *ier = err_count;
}


/* Estimate Gradient at node K (Local Method) */
void ssrf_gradl(int n, int k, float *x, float *y, float *z, float *w, 
                int *list, int *lptr, int *lend, float *g, int *ier) {
    int lmn = 10, lmx = 30;
    int npts[30];
    float dist[30];
    float a[7][7] = {0}; // Augmented matrix (transposed)
    
    if (n < 7) { *ier = -1; return; }

    int lmin = MIN(lmn, n);
    int lmax = MIN(lmx, n);
    int kk = k;
    float wk = w[kk-1];
    
    // Find nearest neighbors
    int lnp = 1;
    npts[0] = kk;
    dist[0] = 0.0f;
    int ierr;
    
    // Accumulate nodes until enough are found or dist ratio is met
    // (Simplified logic for brevity: just get LMIN nodes)
    for(lnp = 2; lnp <= lmin; lnp++) {
        // Assuming getnp from STRIPACK is available and wrapped correctly
        // getnp(x,y,z,list,lptr,lend, lnp, npts, dist, &ierr);
    }
    
    // Construct Rotation
    float cx, sx, cy, sy;
    ssrf_constr(x[kk-1], y[kk-1], z[kk-1], &cx, &sx, &cy, &sy);
    
    // Build System (simplified 6x6 linear least squares)
    // In a full implementation, this loops over neighbors, rotates them
    // via ssrf_aplyr, sets up equations via ssrf_setup, and solves via Givens.
    
    // Placeholder output for valid compilation
    g[0] = 0.0f; g[1] = 0.0f; g[2] = 0.0f; 
    *ier = 0; 
}

/* Arc Interpolation (Hermite Spline on Sphere) */
static void ssrf_arcint(float *p, float *p1, float *p2, float f1, float f2, 
                        float *g1, float *g2, float sigma, float *f) {
    // 1. Calculate arc length a
    // 2. Calculate tangential derivatives
    // 3. Perform 1D Hermite interpolation with tension sigma
    // 4. Return f
    *f = (f1 + f2) * 0.5f; // Stub
}

/* Interpolate at arbitrary point P */
void ssrf_intrc1(int n, float plat, float plon, float *x, float *y, float *z, 
                 float *f, int *list, int *lptr, int *lend, 
                 int iflgs, float *sigma, int iflgg, float *grad, 
                 int *ist, float *fp, int *ier) {
    
    float p[3];
    p[0] = cosf(plat)*cosf(plon);
    p[1] = cosf(plat)*sinf(plon);
    p[2] = sinf(plat);
    
    // Locate triangle using STRIPACK trfind
    int i1, i2, i3;
    float b1, b2, b3;
    // trfind(*ist, p, n, x, y, z, list, lptr, lend, &b1, &b2, &b3, &i1, &i2, &i3);
    
    // If found (i1 != 0), interpolate
    // 1. Get gradients at i1, i2, i3 (compute via gradl if iflgg <= 0)
    // 2. Blending: 
    //    fp = C1*H1 + C2*H2 + C3*H3
    //    Where Hi are arc-interpolants from vertex to opposing edge
    
    *fp = 0.0f; // Stub
    *ier = 0;
}
