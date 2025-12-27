#ifndef SRFPACK_H
#define SRFPACK_H

/* Requires TRIPACK for triangulation structures */
// #include "tripack.h" 

// Interpolate at point (px, py)
void srf_intrc1(float px, float py, int ncc, int *lcc, int n, 
                float *x, float *y, float *z, 
                int *list, int *lptr, int *lend,
                int iflgs, float *sigma, float *grad, 
                int dflag, int *ist, float *pz, float *pzx, float *pzy, int *ier);

// Estimate gradients
void srf_gradl(int k, int ncc, int *lcc, int n, float *x, float *y, float *z,
               int *list, int *lptr, int *lend, float *dx, float *dy, int *ier);

#endif
