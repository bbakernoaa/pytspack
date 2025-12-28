#ifndef TRIPACK_H
#define TRIPACK_H

#ifdef __cplusplus
extern "C" {
#endif

extern double swtol;

void addcst(int ncc, int* lcc, int n, double* x, double* y, int* lwk, int* iwk, int* list, int* lptr, int* lend, int* ier);
void addnod(int k, double xk, double yk, int ist, int ncc, int* lcc, int* n, double* x, double* y, int* list, int* lptr, int* lend, int* lnew, int* ier);
double areap(double* x, double* y, int nb, int* nodes);
void bdyadd(int kk, int i1, int i2, int* list, int* lptr, int* lend, int* lnew);
void bnodes(int n, int* list, int* lptr, int* lend, int* nodes, int* nb, int* na, int* nt);
void circum(double x1, double y1, double x2, double y2, double x3, double y3, int ratio, double* xc, double* yc, double* cr, double* sa, double* ar);
int crtri(int ncc, int* lcc, int i1, int i2, int i3);
void delarc(int n, int io1, int io2, int* list, int* lptr, int* lend, int* lnew, int* ier);
void delnb(int n0, int nb, int n, int* list, int* lptr, int* lend, int* lnew, int* lph);
void delnod(int k, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int* lnew, int* lwk, int* iwk, int* ier);
void edge(int in1, int in2, double* x, double* y, int* lwk, int* iwk, int* list, int* lptr, int* lend, int* ier);
void getnp(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int l, int* npts, double* ds, int* ier);
int indxcc(int ncc, int* lcc, int n, int* list, int* lend);
void insert(int k, int lp, int* list, int* lptr, int* lnew);
void intadd(int kk, int i1, int i2, int i3, int* list, int* lptr, int* lend, int* lnew);
int intsec(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4);
int jrand(int n, int* ix, int* iy, int* iz);
int left(double x1, double y1, double x2, double y2, double x0, double y0);
int lstptr(int lpl, int nb, int* list, int* lptr);
int nbcnt(int lpl, int* lptr);
int nearnd(double xp, double yp, int ist, int n, double* x, double* y, int* list, int* lptr, int* lend, double* dsq);
void nearnds(int l, double* xp, double* yp, int* ist, int n, double* x, double* y, int* list, int* lptr, int* lend, double* dsqs, int* ier);
void optim(double* x, double* y, int na, int* list, int* lptr, int* lend, int* nit, int* iwk, int* ier);
double store(double x);
void swap(int in1, int in2, int io1, int io2, int* list, int* lptr, int* lend, int* lp21);
int swptst(int in1, int in2, int io1, int io2, double* x, double* y);
void timestamp(void);
void trfind(int nst, double px, double py, int n, double* x, double* y, int* list, int* lptr, int* lend, int* i1, int* i2, int* i3);
void trlist(int ncc, int* lcc, int n, int* list, int* lptr, int* lend, int nrow, int* nt, int* ltri, int* lct, int* ier);
void trlist2(int n, int* list, int* lptr, int* lend, int* nt, int* ltri, int* ier);
void trlprt(int ncc, int* lct, int n, double* x, double* y, int nrow, int nt, int* ltri, int prntx);
void trmesh(int n, double* x, double* y, int* list, int* lptr, int* lend, int* lnew, int* near_arr, int* next, double* dist, int* ier);
void trmshr(int n, int nx, double* x, double* y, int* nit, int* list, int* lptr, int* lend, int* lnew, int* ier);
void trmtst(int n, double* x, double* y, int* list, int* lptr, int* lend, int lnew, double tol, double* armax, int* ier);
void trplot(int lun, double pltsiz, double wx1, double wx2, double wy1, double wy2, int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, char* title, int numbr, int* ier);
void trprnt(int ncc, int* lcc, int n, double* x, double* y, int* list, int* lptr, int* lend, int prntx);

#ifdef __cplusplus
}
#endif

#endif /* TRIPACK_H */
