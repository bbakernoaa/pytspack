#ifndef TRIPACK_H
#define TRIPACK_H

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

/* Core Routines */
void trmesh(int n, float *x, float *y, int *list, int *lptr, int *lend, 
            int *lnew, int *near_arr, int *next, float *dist, int *ier);

/* Modifiers */
void addnod(int k, float xk, float yk, int ist, int ncc, int *lcc, int *n, 
            float *x, float *y, int *list, int *lptr, int *lend, int *lnew, int *ier);

/* Utilities usually needed by users */
int nearnd(float xp, float yp, int ist, int n, float *x, float *y, 
           int *list, int *lptr, int *lend, float *dsq);

#endif
