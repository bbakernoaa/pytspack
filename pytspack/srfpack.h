#ifndef SRFPACK_H
#define SRFPACK_H

#include "tripack.h"

#ifdef __cplusplus
extern "C" {
#endif

void arcint(double b, double x1, double x2, double y1, double y2, double h1, double h2, double hx1, double hx2, double hy1, double hy2, double sigma, int dflag, double* hp, double* hxp, double* hyp, int* ier);
void cntour(int nx, int ny, double* x, double* y, double* z, double cval, int lc, int ncmax, int* iwk, double* xc, double* yc, int* ilc, int* nc, int* ier);
void coords(double xp, double yp, double x1, double x2, double x3, double y1, double y2, double y3, double* b1, double* b2, double* b3, int* ier);
void crplot(int lun, double pltsiz, int nx, int ny, double* px, double* py, double* pz, int ncon, int* iwk, double* xc, double* yc, int* ier);
void fval(double xp, double yp, double x1, double x2, double x3, double y1, double y2, double y3, double f1, double f2, double f3, double fx1, double fx2, double fx3, double fy1, double fy2, double fy3, double sig1, double sig2, double sig3, double* fp, int* ier);
void getsig(int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, double tol, double* sigma, double* dsmax, int* ier);
void givens(double a, double b, double* c, double* s);
void gradc(int k, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, double* dx, double* dy, double* dxx, double* dxy, double* dyy, int* ier);
void gradg(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, int* nit, double* dgmax, double* grad, int* ier);
void gradl(int k, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, double* dx, double* dy, int* ier);
void grcoef(double sigma, double dcub, double* d, double* sd);
void intrc0(double px, double py, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int* ist, double* pz, int* ier);
void intrc1(double px, double py, int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* grad, int dflag, int* ist, double* pz, double* pzx, double* pzy, int* ier);
void rotate(int n, double c, double s, double* x, double* y);
void setro1(double xk, double yk, double zk, double xi, double yi, double zi, double s1, double s2, double w, double* row);
void setro2(double xk, double yk, double zk, double xi, double yi, double zi, double s1, double s2, double w, double* row);
void setro3(double xk, double yk, double zk, double xi, double yi, double zi, double s1, double s2, double s3, double w, double* row);
void sgprnt(int n, int lunit, int* list, int* lptr, int* lend, double* sigma);
double srf_sig0(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, int iflgb, double hbnd, double tol, int iflgs, double* sigma, int* ier);
double srf_sig1(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, int iflgb, double hpbnd, double tol, int iflgs, double* sigma, int* ier);
double srf_sig2(int n1, int n2, int n, double* x, double* y, double* h, int* list, int* lptr, int* lend, double* hxhy, double tol, int iflgs, double* sigma, int* ier);
void smsgs(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double p, int* nit, double dfmax, double* f, double* fxfy, int* ier);
void smsurf(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, double* w, double sm, double smtol, double gstol, int lprnt, double* f, double* fxfy, int* ier);
void srf_snhcsh(double x, double* sinhm, double* coshm, double* coshmm);
double trvol(double x1, double x2, double x3, double y1, double y2, double y3, double z1, double z2, double z3);
void tval(double x, double y, double x1, double x2, double x3, double y1, double y2, double y3, double z1, double z2, double z3, double zx1, double zx2, double zx3, double zy1, double zy2, double zy3, int dflag, double* f, double* fx, double* fy, int* ier);
void unif(int ncc, int* lcc, int n, double* x, double* y, double* z, double* grad, int* list, int* lptr, int* lend, int iflgs, double* sigma, int nrow, int nx, int ny, double* px, double* py, int sflag, double sval, double* zz, int* ier);
double volume(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend);
void zgradg(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int iflgs, double* sigma, int* nit, double dzmax, double* z, double* grad, int* ier);
void zgradl(int k, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int ndv, double* z, int* npts, double* ds, double* dx, double* dy, int* ier);
void zinit(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, double* z, int* ier);

#ifdef __cplusplus
}
#endif

#endif /* SRFPACK_H */
