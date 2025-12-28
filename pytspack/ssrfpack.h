#ifndef SSRFPACK_H
#define SSRFPACK_H

#include "stripack.h"

#ifdef __cplusplus
extern "C" {
#endif

void ssrf_aplyr(double x, double y, double z, double cx, double sx, double cy, double sy, double* xp, double* yp, double* zp);
void ssrf_aplyrt(double g1p, double g2p, double cx, double sx, double cy, double sy, double* g);
void ssrf_arcint(double* p, double* p1, double* p2, double f1, double f2, double* g1, double* g2, double sigma, double* f, double* g, double* gn);
double ssrf_arclen(double* p, double* q);
void ssrf_constr(double xk, double yk, double zk, double* cx, double* sx, double* cy, double* sy);
double ssrf_fval(double b1, double b2, double b3, double* v1, double* v2, double* v3, double f1, double f2, double f3, double* g1, double* g2, double* g3, double sig1, double sig2, double sig3);
void ssrf_getsig(int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, double tol, double* sigma, double* dsmax, int* ier);
void ssrf_givens(double a, double b, double* c, double* s);
void ssrf_gradg(int n, double* x, double* y, double* z, double* f, int* list, int* lptr, int* lend, int iflgs, double* sigma, int* nit, double* dgmax, double* grad, int* ier);
void ssrf_gradl(int n, int k, double* x, double* y, double* z, double* w, int* list, int* lptr, int* lend, double* g, int* ier);
void ssrf_grcoef(double sigma, double* d, double* sd);
double ssrf_hval(double b, double h1, double h2, double hp1, double hp2, double sigma);
void ssrf_intrc0(int n, double plat, double plon, double* x, double* y, double* z, double* w, int* list, int* lptr, int* lend, int* ist, double* pw, int* ier);
void ssrf_intrc1(int n, double plat, double plon, double* x, double* y, double* z, double* f, int* list, int* lptr, int* lend, int iflgs, double* sigma, int iflgg, double* grad, int* ist, double* fp, int* ier);
void ssrf_rotate(int n, double c, double s, double* x, double* y);
void ssrf_setup(double xi, double yi, double wi, double wk, double s1, double s2, double wt, double* row);
void ssrf_sgprnt(int n, int lunit, int* list, int* lptr, int* lend, double* sigma);
double ssrf_sig0(int n1, int n2, int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, int iflgb, double hbnd, double tol, int iflgs, double* sigma, int* ier);
double ssrf_sig1(int n1, int n2, int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, int iflgb, double hpbnd, double tol, int iflgs, double* sigma, int* ier);
double ssrf_sig2(int n1, int n2, int n, double* x, double* y, double* z, double* h, int* list, int* lptr, int* lend, double* grad, double tol, int iflgs, double* sigma, int* ier);
void ssrf_smsgs(int n, double* x, double* y, double* z, double* u, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double p, int* nit, double dfmax, double* f, double* grad, int* ier);
void ssrf_smsurf(int n, double* x, double* y, double* z, double* u, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double sm, double smtol, double gstol, int lprnt, double* f, double* grad, int* ier);
void ssrf_snhcsh(double x, double* sinhm, double* coshm, double* coshmm);
void ssrf_unif(int n, double* x, double* y, double* z, double* f, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nrow, int ni, int nj, double* plat, double* plon, int iflgg, double* grad, double* ff, int* ier);

#ifdef __cplusplus
}
#endif

#endif /* SSRFPACK_H */
