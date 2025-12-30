#ifndef STRIPACK_H
#define STRIPACK_H

#include "tripack.h"

#ifdef __cplusplus
extern "C" {
#endif

void stri_addnod(int nst, int k, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* lnew, int* ier);
double arc_cosine(double c);
double areas(double* v1, double* v2, double* v3);
void triareas(int n, double* x, double* y, double* z, int nt, int* triangles, double* triarea);
void stri_bdyadd(int kk, int i1, int i2, int* list, int* lptr, int* lend, int* lnew);
void stri_bnodes(int n, int* list, int* lptr, int* lend, int* nodes, int* nb, int* na, int* nt);
void stri_circum(double* v1, double* v2, double* v3, double* c, int* ier);
void covsph(int kk, int n0, int* list, int* lptr, int* lend, int* lnew);
void crlist(int n, int ncol, double* x, double* y, double* z, int* list, int* lend, int* lptr, int* lnew, int* ltri, int* listc, int* nb, double* xc, double* yc, double* zc, double* rc, int* ier);
void stri_delarc(int n, int io1, int io2, int* list, int* lptr, int* lend, int* lnew, int* ier);
void stri_delnb(int n0, int nb, int n, int* list, int* lptr, int* lend, int* lnew, int* lph);
void stri_delnod(int k, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* lnew, int lwk, int* iwk, int* ier);
void stri_edge(int in1, int in2, double* x, double* y, double* z, int* lwk, int* iwk, int* list, int* lptr, int* lend, int* ier);
void stri_getnp(double* x, double* y, double* z, int* list, int* lptr, int* lend, int l, int* npts, double* df, int* ier);
void stri_insert(int k, int lp, int* list, int* lptr, int* lnew);
void stri_intadd(int kk, int i1, int i2, int i3, int* list, int* lptr, int* lend, int* lnew);
int stri_jrand(int n, int* ix, int* iy, int* iz);
int stri_left(double x1, double y1, double z1, double x2, double y2, double z2, double x0, double y0, double z0);
int stri_lstptr(int lpl, int nb, int* list, int* lptr);
int stri_nbcnt(int lpl, int* lptr);
int stri_nearnd(float* p, int ist, int n, float* x, float* y, float* z, int* list, int* lptr, int* lend, float* al);
void stri_optim(double* x, double* y, double* z, int na, int* list, int* lptr, int* lend, int* nit, int* iwk, int* ier);
double stri_store(double x);
void stri_swap(int in1, int in2, int io1, int io2, int* list, int* lptr, int* lend, int* lp21);
int stri_swptst(int n1, int n2, int n3, int n4, double* x, double* y, double* z);
void trans(int n, double* rlat, double* rlon, double* x, double* y, double* z);
void stri_trfind(int nst, double* p, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, double* b1, double* b2, double* b3, int* i1, int* i2, int* i3);
void stri_trlist(int n, int* list, int* lptr, int* lend, int nrow, int* nt, int* ltri, int* ier);
void stri_trlist2(int n, int* list, int* lptr, int* lend, int* nt, int* ltri, int* ier);
void stri_trmesh(int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* ier);
void vrplot(int lun, double pltsiz, double wx1, double wx2, double wy1, double wy2, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* lnew, char* title, int numbr, int* ier);
void stri_trplot(int lun, double pltsiz, double wx1, double wx2, double wy1, double wy2, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, char* title, int numbr, int* ier);
void stri_trprnt(int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int prntx);
void inside(double* p, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* ist, int* ltri, int* inside_flag);
void intrsc(double* p1, double* p2, double* p3, double* p4, double* p, int* ier);
void scoord(double* p, double* lat, double* lon);

#ifdef __cplusplus
}
#endif

#endif /* STRIPACK_H */
