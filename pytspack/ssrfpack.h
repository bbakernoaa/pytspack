#ifndef SSRFPACK_H
#define SSRFPACK_H

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include "stripack.h" // Requires the previous STRIPACK conversion

#define PI 3.14159265358979323846

/* Core Interpolation & Smoothing Routines */

// Compute gradient estimates (Global method)
void gradg(int n, float *x, float *y, float *z, float *f, 
           int *list, int *lptr, int *lend, 
           int iflgs, float *sigma, 
           int nit, float dgmax, float *grad, int *ier);

// Compute gradient estimate (Local method)
void gradl(int n, int k, float *x, float *y, float *z, float *w,
           int *list, int *lptr, int *lend, float *g, int *ier);

// Interpolate at a specific point
void intrc1(int n, float plat, float plon, float *x, float *y, float *z,
            float *f, int *list, int *lptr, int *lend,
            int iflgs, float *sigma, int iflgg, float *grad, 
            int *ist, float *fp, int *ier);

// Interpolate to a uniform grid
void unif(int n, float *x, float *y, float *z, float *f,
          int *list, int *lptr, int *lend,
          int iflgs, float *sigma,
          int nrow, int ni, int nj, float *plat, float *plon,
          int iflgg, float *grad, float *ff, int *ier);

// Smoothing surface fitting
void smsurf(int n, float *x, float *y, float *z, float *u,
            int *list, int *lptr, int *lend,
            int iflgs, float *sigma, float *w, float sm, 
            float smtol, float gstol, int lprnt, 
            float *f, float *grad, int *ier);

#endif
