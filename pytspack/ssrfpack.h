#ifndef SSRFPACK_H
#define SSRFPACK_H

/* Requires STRIPACK for triangulation structures */
#include "stripack.h" 

void ssrf_gradl(int n, int k, float *x, float *y, float *z, float *w, 
                int *list, int *lptr, int *lend, float *g, int *ier);

void ssrf_intrc1(int n, float plat, float plon, float *x, float *y, float *z, 
                 float *f, int *list, int *lptr, int *lend, 
                 int iflgs, float *sigma, int iflgg, float *grad, 
                 int *ist, float *fp, int *ier);

#endif
