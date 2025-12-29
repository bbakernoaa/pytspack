#ifndef TSPACK_H
#define TSPACK_H

#ifdef __cplusplus
extern "C" {
#endif

void arcl2d(int n, double* x, double* y, double* t, int* ier);
void arcl3d(int n, double* x, double* y, double* z, double* t, int* ier);
void b2tri(int n, double* x, double* y, double* w, double p, double* d, double* sd,
           double* t11, double* t12, double* t21, double* t22, double* ys, double* yp, int* ier);
void b2trip(int n, double* x, double* y, double* w, double p, double* d, double* sd,
            double* t11, double* t12, double* t21, double* t22, double* u11, double* u12,
            double* u21, double* u22, double* ys, double* yp, int* ier);
double endslp(double x1, double x2, double x3, double y1, double y2, double y3, double sigma);
double hppval(double t, int n, double* x, double* y, double* yp, double* sigma, int* ier);
double hpval(double t, int n, double* x, double* y, double* yp, double* sigma, int* ier);
double hval(double t, int n, double* x, double* y, double* yp, double* sigma, int* ier);
int intrvl(double t, int n, double* x);
double sig0(double x1, double x2, double y1, double y2, double y1p, double y2p,
            int ifl, double hbnd, double tol, int* ier);
double sig1(double x1, double x2, double y1, double y2, double y1p, double y2p,
            int ifl, double hpbnd, double tol, int* ier);
double sig2(double x1, double x2, double y1, double y2, double y1p, double y2p,
            int ifl, double tol, int* ier);
void sigbi(int n, double* x, double* y, double* yp, double tol, double b[][5], double bmax,
           double* sigma, int* icflg, double* dsmax, int* ier);
void sigbp(int n, double* x, double* y, double* xp, double* yp, double tol, double* bl,
           double* bu, double bmax, double* sigma, double* dsmax, int* ier);
void sigs(int n, double* x, double* y, double* yp, double tol, double* sigma, double* dsmax, int* ier);
void smcrv(int n, double* x, double* y, double* sigma, int period, double* w, double sm,
           double smtol, double wk[][10], double* ys, double* yp, int* ier);
void snhcsh(double x, double* sinhm, double* coshm, double* coshmm);
double ts_store(double x);
double tsintl(double a, double b, int n, double* x, double* y, double* yp, double* sigma, int* ier);
void tspbi(int n, double* x, double* y, int ncd, int iendc, int per, double b[][5], double bmax,
           int lwk, double* wk, double* yp, double* sigma, int* icflg, int* ier);
void tspbp(int n, double* x, double* y, int ncd, int iendc, int per, double* bl, double* bu,
           double bmax, int lwk, double* wk, double* t, double* xp, double* yp, double* sigma, int* ier);
void tspsi(int n, double* x, double* y, int ncd, int iendc, int per, int unifrm, int lwk,
           double* wk, double* yp, double* sigma, int* ier);
void tspsp(int n, int nd, double* x, double* y, double* z, int ncd, int iendc, int per,
           int unifrm, int lwk, double* wk, double* t, double* xp, double* yp, double* zp,
           double* sigma, int* ier);
void tspss(int n, double* x, double* y, int per, int unifrm, double* w, double sm, double smtol,
           int lwk, double* wk, double* sigma, double* ys, double* yp, int* nit, int* ier);
void tsval1(int n, double* x, double* y, double* yp, double* sigma, int iflag, int ne,
            double* te, double* v, int* ier);
void tsval2(int n, double* t, double* x, double* y, double* xp, double* yp, double* sigma,
            int iflag, int ne, double* te, double* vx, double* vy, int* ier);
void tsval3(int n, double* t, double* x, double* y, double* z, double* xp, double* yp,
            double* zp, double* sigma, int iflag, int ne, double* te, double* vx, double* vy,
            double* vz, int* ier);
void ypc1(int n, double* x, double* y, double* yp, int* ier);
void ypc1p(int n, double* x, double* y, double* yp, int* ier);
void ypc2(int n, double* x, double* y, double* sigma, int isl1, int isln, double bv1,
          double bvn, double* wk, double* yp, int* ier);
void ypc2p(int n, double* x, double* y, double* sigma, double* wk, double* yp, int* ier);
void ypcoef(double sigma, double dx, double* d, double* sd);

#ifdef __cplusplus
}
#endif

#endif /* TSPACK_H */
