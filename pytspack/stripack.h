#ifndef STRIPACK_H
#define STRIPACK_H

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

/* * STRIPACK: Spherical Triangulation Package
 * Nodes are points on the unit sphere (x^2 + y^2 + z^2 = 1).
 */

/* Coordinate Transformation */
void trans(int n, float *rlat, float *rlon, float *x, float *y, float *z);
void scoord(float px, float py, float pz, float *plat, float *plon, float *pnrm);

/* Core Triangulation Construction */
void trmesh(int n, float *x, float *y, float *z, 
            int *list, int *lptr, int *lend, int *lnew, 
            int *near_arr, int *next, float *dist, int *ier);

/* Updates */
void addnod(int nst, int k, float *x, float *y, float *z, 
            int *list, int *lptr, int *lend, int *lnew, int *ier);

/* Geometric Queries */
void trfind(int nst, float *p, int n, float *x, float *y, float *z,
            int *list, int *lptr, int *lend, 
            float *b1, float *b2, float *b3, 
            int *i1, int *i2, int *i3);

int nearnd(float *p, int ist, int n, float *x, float *y, float *z,
           int *list, int *lptr, int *lend, float *al);

void circum(float *v1, float *v2, float *v3, float *c, int *ier);
float areas(float *v1, float *v2, float *v3);

#endif
