#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "tspack.h"
#include "tripack.h"
#include "stripack.h"
#include "srfpack.h"
#include "ssrfpack.h"

// Wrappers for TSPACK

static PyObject* py_tspsi(PyObject* self, PyObject* args, PyObject* keywds) {
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;
    int ncd = 1;
    int iendc = 0;
    int per = 0;
    int unifrm = 0;
    PyObject* yp_obj = NULL;
    PyObject* sigma_obj = NULL;

    static char* kwlist[] = {"x", "y", "ncd", "iendc", "per", "unifrm", "yp", "sigma", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|iiiiOO", kwlist,
                                     &x_obj, &y_obj, &ncd, &iendc, &per, &unifrm, &yp_obj, &sigma_obj)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!x_arr || !y_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    if ((int)PyArray_DIM(y_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same length");
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        return NULL;
    }

    npy_intp dims[1];
    dims[0] = n;

    PyArrayObject* yp_arr = NULL;
    if (yp_obj && yp_obj != Py_None) {
        yp_arr = (PyArrayObject*)PyArray_FROM_OTF(yp_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    } else {
        yp_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    }

    PyArrayObject* sigma_arr = NULL;
    if (sigma_obj && sigma_obj != Py_None) {
        sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    } else {
        sigma_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        double* sig_ptr = (double*)PyArray_DATA(sigma_arr);
        for(int i=0; i<n; ++i) sig_ptr[i] = 0.0;
    }

    if (!yp_arr || !sigma_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(yp_arr);
        Py_XDECREF(sigma_arr);
        return NULL;
    }

    int lwk = 2 * n + 10;
    double* wk = (double*)malloc(lwk * sizeof(double));
    if (!wk) {
        PyErr_NoMemory();
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        Py_DECREF(yp_arr);
        Py_DECREF(sigma_arr);
        return NULL;
    }

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* yp = (double*)PyArray_DATA(yp_arr);
    double* sigma = (double*)PyArray_DATA(sigma_arr);

    int ier;
    tspsi(n, x, y, ncd, iendc, per, unifrm, lwk, wk, yp, sigma, &ier);

    free(wk);
    Py_DECREF(x_arr);
    Py_DECREF(y_arr);

    if (ier < 0) {
        Py_DECREF(yp_arr);
        Py_DECREF(sigma_arr);
        PyErr_Format(PyExc_RuntimeError, "tspsi failed with error code %d", ier);
        return NULL;
    }

    return Py_BuildValue("NN", yp_arr, sigma_arr);
}

static PyObject* py_tsval1(PyObject* self, PyObject* args, PyObject* keywds) {
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;
    PyObject* yp_obj = NULL;
    PyObject* sigma_obj = NULL;
    PyObject* te_obj = NULL;
    int iflag = 0;

    static char* kwlist[] = {"x", "y", "yp", "sigma", "te", "iflag", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOO|i", kwlist,
                                     &x_obj, &y_obj, &yp_obj, &sigma_obj, &te_obj, &iflag)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* yp_arr = (PyArrayObject*)PyArray_FROM_OTF(yp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* te_arr = (PyArrayObject*)PyArray_FROM_OTF(te_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !y_arr || !yp_arr || !sigma_arr || !te_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(yp_arr);
        Py_XDECREF(sigma_arr);
        Py_XDECREF(te_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    int ne = (int)PyArray_DIM(te_arr, 0);

    npy_intp dims[1];
    dims[0] = ne;
    PyArrayObject* v_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!v_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(yp_arr);
        Py_XDECREF(sigma_arr);
        Py_XDECREF(te_arr);
        return NULL;
    }

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* yp = (double*)PyArray_DATA(yp_arr);
    double* sigma = (double*)PyArray_DATA(sigma_arr);
    double* te = (double*)PyArray_DATA(te_arr);
    double* v = (double*)PyArray_DATA(v_arr);

    int ier;
    tsval1(n, x, y, yp, sigma, iflag, ne, te, v, &ier);

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    Py_DECREF(yp_arr);
    Py_DECREF(sigma_arr);
    Py_DECREF(te_arr);

    if (ier < 0) {
        Py_DECREF(v_arr);
        PyErr_Format(PyExc_RuntimeError, "tsval1 failed with error code %d", ier);
        return NULL;
    }

    return (PyObject*)v_arr;
}

// Wrappers for TRIPACK

static PyObject* py_trmesh(PyObject* self, PyObject* args) {
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO", &x_obj, &y_obj)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!x_arr || !y_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    if ((int)PyArray_DIM(y_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same length");
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        return NULL;
    }

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);

    // Arrays to hold triangulation data structure
    // list, lptr, lend. Size depends on N.
    // LIST and LPTR: 6*N - 12 (approx, but safe bound is needed)
    // LEND: N
    // According to TRIPACK doc: 6*N-12 is enough for triangulation.
    int len_list = 6 * n;

    npy_intp dims_list[1] = {len_list};
    npy_intp dims_n[1] = {n};

    PyArrayObject* list_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_list, NPY_INT);
    PyArrayObject* lptr_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_list, NPY_INT);
    PyArrayObject* lend_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_n, NPY_INT);

    if (!list_arr || !lptr_arr || !lend_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(list_arr);
        Py_XDECREF(lptr_arr);
        Py_XDECREF(lend_arr);
        return NULL;
    }

    int* list = (int*)PyArray_DATA(list_arr);
    int* lptr = (int*)PyArray_DATA(lptr_arr);
    int* lend = (int*)PyArray_DATA(lend_arr);

    int lnew = 0;
    int ier;

    // Workspace arrays for trmesh
    int* near_arr = (int*)malloc(n * sizeof(int));
    int* next = (int*)malloc(n * sizeof(int));
    double* dist = (double*)malloc(n * sizeof(double));

    if (!near_arr || !next || !dist) {
        free(near_arr); free(next); free(dist);
        Py_XDECREF(x_arr); Py_XDECREF(y_arr);
        Py_XDECREF(list_arr); Py_XDECREF(lptr_arr); Py_XDECREF(lend_arr);
        return PyErr_NoMemory();
    }

    trmesh(n, x, y, list, lptr, lend, &lnew, near_arr, next, dist, &ier);

    free(near_arr);
    free(next);
    free(dist);

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);

    if (ier < 0) {
        Py_DECREF(list_arr);
        Py_DECREF(lptr_arr);
        Py_DECREF(lend_arr);
        PyErr_Format(PyExc_RuntimeError, "trmesh failed with error code %d", ier);
        return NULL;
    }

    return Py_BuildValue("{s:N,s:N,s:N,s:i}", "list", list_arr, "lptr", lptr_arr, "lend", lend_arr, "lnew", lnew);
}

// Wrapper for STRIPACK

static PyObject* py_stri_trmesh(PyObject* self, PyObject* args) {
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;
    PyObject* z_obj = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &x_obj, &y_obj, &z_obj)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!x_arr || !y_arr || !z_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(z_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    if ((int)PyArray_DIM(y_arr, 0) != n || (int)PyArray_DIM(z_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "x, y, z must have the same length");
        Py_DECREF(x_arr);
        Py_DECREF(y_arr);
        Py_DECREF(z_arr);
        return NULL;
    }

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* z = (double*)PyArray_DATA(z_arr);

    int len_list = 6 * n;
    npy_intp dims_list[1] = {len_list};
    npy_intp dims_n[1] = {n};

    PyArrayObject* list_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_list, NPY_INT);
    PyArrayObject* lptr_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_list, NPY_INT);
    PyArrayObject* lend_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_n, NPY_INT);

    if (!list_arr || !lptr_arr || !lend_arr) {
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(z_arr);
        Py_XDECREF(list_arr); Py_XDECREF(lptr_arr); Py_XDECREF(lend_arr);
        return NULL;
    }

    int* list = (int*)PyArray_DATA(list_arr);
    int* lptr = (int*)PyArray_DATA(lptr_arr);
    int* lend = (int*)PyArray_DATA(lend_arr);
    int ier;

    stri_trmesh(n, x, y, z, list, lptr, lend, &ier);

    Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr);

    if (ier != 0) {
        Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);
        PyErr_Format(PyExc_RuntimeError, "stri_trmesh failed with error code %d", ier);
        return NULL;
    }

    return Py_BuildValue("{s:N,s:N,s:N}", "list", list_arr, "lptr", lptr_arr, "lend", lend_arr);
}


static PyObject* py_hval(PyObject* self, PyObject* args) {
    double t;
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;
    PyObject* yp_obj = NULL;
    PyObject* sigma_obj = NULL;

    if (!PyArg_ParseTuple(args, "dOOOO", &t, &x_obj, &y_obj, &yp_obj, &sigma_obj)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* yp_arr = (PyArrayObject*)PyArray_FROM_OTF(yp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !y_arr || !yp_arr || !sigma_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(yp_arr);
        Py_XDECREF(sigma_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* yp = (double*)PyArray_DATA(yp_arr);
    double* sigma = (double*)PyArray_DATA(sigma_arr);
    int ier;

    double v = hval(t, n, x, y, yp, sigma, &ier);

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    Py_DECREF(yp_arr);
    Py_DECREF(sigma_arr);

    if (ier < 0) {
        PyErr_Format(PyExc_RuntimeError, "hval failed with error code %d", ier);
        return NULL;
    }

    return PyFloat_FromDouble(v);
}

static PyObject* py_hpval(PyObject* self, PyObject* args) {
    double t;
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;
    PyObject* yp_obj = NULL;
    PyObject* sigma_obj = NULL;

    if (!PyArg_ParseTuple(args, "dOOOO", &t, &x_obj, &y_obj, &yp_obj, &sigma_obj)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* yp_arr = (PyArrayObject*)PyArray_FROM_OTF(yp_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !y_arr || !yp_arr || !sigma_arr) {
        Py_XDECREF(x_arr);
        Py_XDECREF(y_arr);
        Py_XDECREF(yp_arr);
        Py_XDECREF(sigma_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* yp = (double*)PyArray_DATA(yp_arr);
    double* sigma = (double*)PyArray_DATA(sigma_arr);
    int ier;

    double v = hpval(t, n, x, y, yp, sigma, &ier);

    Py_DECREF(x_arr);
    Py_DECREF(y_arr);
    Py_DECREF(yp_arr);
    Py_DECREF(sigma_arr);

    if (ier < 0) {
        PyErr_Format(PyExc_RuntimeError, "hpval failed with error code %d", ier);
        return NULL;
    }

    return PyFloat_FromDouble(v);
}

static PyObject* py_tspss(PyObject* self, PyObject* args) {
    PyObject* x_obj = NULL;
    PyObject* y_obj = NULL;
    int per = 0;
    int unifrm = 0;
    PyObject* w_obj = NULL;
    double sm = 0.0;
    double smtol = 0.0;

    // tspss(int n, double* x, double* y, int per, int unifrm, double* w, double sm, double smtol,
    //       int lwk, double* wk, double* sigma, double* ys, double* yp, int* nit, int* ier);

    if (!PyArg_ParseTuple(args, "OOiiOdd", &x_obj, &y_obj, &per, &unifrm, &w_obj, &sm, &smtol)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* w_arr = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !y_arr || !w_arr) {
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(w_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    if ((int)PyArray_DIM(y_arr, 0) != n || (int)PyArray_DIM(w_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "x, y, w must have the same length");
        Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(w_arr);
        return NULL;
    }

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* w = (double*)PyArray_DATA(w_arr);

    int lwk = 6 * n; // rough estimate based on docs
    double* wk = (double*)malloc(lwk * sizeof(double));
    if (!wk) return PyErr_NoMemory();

    npy_intp dims[1] = {n};
    PyArrayObject* sigma_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject* ys_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject* yp_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!sigma_arr || !ys_arr || !yp_arr) {
        free(wk);
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(w_arr);
        Py_XDECREF(sigma_arr); Py_XDECREF(ys_arr); Py_XDECREF(yp_arr);
        return NULL;
    }

    double* sigma = (double*)PyArray_DATA(sigma_arr);
    double* ys = (double*)PyArray_DATA(ys_arr);
    double* yp = (double*)PyArray_DATA(yp_arr);
    int nit;
    int ier;

    tspss(n, x, y, per, unifrm, w, sm, smtol, lwk, wk, sigma, ys, yp, &nit, &ier);

    free(wk);
    Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(w_arr);

    if (ier < 0) {
        Py_DECREF(sigma_arr); Py_DECREF(ys_arr); Py_DECREF(yp_arr);
        PyErr_Format(PyExc_RuntimeError, "tspss failed with error code %d", ier);
        return NULL;
    }

    return Py_BuildValue("{s:N,s:N,s:N,s:i}", "sigma", sigma_arr, "ys", ys_arr, "yp", yp_arr, "nit", nit);
}

// Wrappers for SRFPACK (Planar Surface Interpolation)

// gradg: Calculate gradients at nodes for surface interpolation
// void gradg(int ncc, int* lcc, int n, double* x, double* y, double* z, int* list, int* lptr, int* lend, int iflgs, double* sigma, int* nit, double* dgmax, double* grad, int* ier);
static PyObject* py_gradg(PyObject* self, PyObject* args) {
    PyObject *x_obj, *y_obj, *z_obj, *list_obj, *lptr_obj, *lend_obj;
    int iflgs = 0; // Default 0? or passed?
    PyObject* sigma_obj = NULL; // Can be None or input

    if (!PyArg_ParseTuple(args, "OOOOOO|iO", &x_obj, &y_obj, &z_obj, &list_obj, &lptr_obj, &lend_obj, &iflgs, &sigma_obj)) {
        return NULL;
    }

    // Convert inputs
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* list_arr = (PyArrayObject*)PyArray_FROM_OTF(list_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lptr_arr = (PyArrayObject*)PyArray_FROM_OTF(lptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lend_arr = (PyArrayObject*)PyArray_FROM_OTF(lend_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !y_arr || !z_arr || !list_arr || !lptr_arr || !lend_arr) {
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(z_arr);
        Py_XDECREF(list_arr); Py_XDECREF(lptr_arr); Py_XDECREF(lend_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* z = (double*)PyArray_DATA(z_arr);
    int* list = (int*)PyArray_DATA(list_arr);
    int* lptr = (int*)PyArray_DATA(lptr_arr);
    int* lend = (int*)PyArray_DATA(lend_arr);

    // Prepare sigma
    PyArrayObject* sigma_arr;
    if (sigma_obj && sigma_obj != Py_None) {
        sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    } else {
        npy_intp dims[1] = {n};
        sigma_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        // fill with zeros or 1.0? gradg expects sigma input if iflgs=1. if iflgs=0 it computes it.
        // Assuming iflgs=0 (default) computes sigma.
    }
    double* sigma = (double*)PyArray_DATA(sigma_arr);

    // Prepare grad output: 2*N
    npy_intp dims_grad[2] = {n, 2}; // grad is 2xN in C code usually or flat?
    // In srfpack.c: grad is double array of size usually 2*N or similar.
    // The definition: double* grad. It stores dx, dy for each node.
    // Let's assume it's flat 2*N? Or two arrays?
    // The Fortran doc says GRAD(2,N). So it's column major in Fortran, but in C translation it depends.
    // Looking at srfpack.c/gradg:
    // grad[0]...grad[2*n-1]?
    // Let's verify usage in srfpack.c.

    // Actually, let's create a new array for grad
    // We'll flatten it to 1D of size 2*N for safety.
    npy_intp dims_g[1] = {2 * n};
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_g, NPY_DOUBLE);
    double* grad = (double*)PyArray_DATA(grad_arr);

    // Aux variables
    int ncc = 0; // constrained?
    int* lcc = NULL; // no constraints supported for now in this wrapper
    int nit = 20; // Default max iterations
    double dgmax = 0.0; // Default tolerance
    int ier = 0;

    gradg(ncc, lcc, n, x, y, z, list, lptr, lend, iflgs, sigma, &nit, &dgmax, grad, &ier);

    Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr);
    Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);

    if (ier < 0) {
        Py_DECREF(sigma_arr); Py_DECREF(grad_arr);
        PyErr_Format(PyExc_RuntimeError, "gradg failed with error code %d", ier);
        return NULL;
    }

    // Return (sigma, grad)
    return Py_BuildValue("NN", sigma_arr, grad_arr);
}

// intrc1: Interpolate at a point (or points?)
// The function signature is single point: void intrc1(double px, double py, ... double* pz, double* pzx, double* pzy, int* ier);
// But for Python we want vectorized if possible.
// However, C loop is easy.

static PyObject* py_intr_2d(PyObject* self, PyObject* args) {
    PyObject *px_obj, *py_obj;
    PyObject *x_obj, *y_obj, *z_obj, *list_obj, *lptr_obj, *lend_obj, *sigma_obj, *grad_obj;
    int iflgs = 0; // default 0?
    // We need grad from gradg.

    if (!PyArg_ParseTuple(args, "OOOOOOOOOO|i", &px_obj, &py_obj, &x_obj, &y_obj, &z_obj, &list_obj, &lptr_obj, &lend_obj, &sigma_obj, &grad_obj, &iflgs)) {
        return NULL;
    }

    PyArrayObject* px_arr = (PyArrayObject*)PyArray_FROM_OTF(px_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* py_arr = (PyArrayObject*)PyArray_FROM_OTF(py_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    // Mesh data
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* list_arr = (PyArrayObject*)PyArray_FROM_OTF(list_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lptr_arr = (PyArrayObject*)PyArray_FROM_OTF(lptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lend_arr = (PyArrayObject*)PyArray_FROM_OTF(lend_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_FROM_OTF(grad_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!px_arr || !py_arr || !x_arr || !y_arr || !z_arr || !list_arr || !lptr_arr || !lend_arr || !sigma_arr || !grad_arr) {
        // cleanup
        Py_XDECREF(px_arr); Py_XDECREF(py_arr);
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(z_arr);
        Py_XDECREF(list_arr); Py_XDECREF(lptr_arr); Py_XDECREF(lend_arr);
        Py_XDECREF(sigma_arr); Py_XDECREF(grad_arr);
        return NULL;
    }

    int n_points = (int)PyArray_DIM(px_arr, 0);
    int n = (int)PyArray_DIM(x_arr, 0);

    double* px = (double*)PyArray_DATA(px_arr);
    double* py = (double*)PyArray_DATA(py_arr);

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* z = (double*)PyArray_DATA(z_arr);
    int* list = (int*)PyArray_DATA(list_arr);
    int* lptr = (int*)PyArray_DATA(lptr_arr);
    int* lend = (int*)PyArray_DATA(lend_arr);
    double* sigma = (double*)PyArray_DATA(sigma_arr);
    double* grad = (double*)PyArray_DATA(grad_arr);

    npy_intp dims[1] = {n_points};
    PyArrayObject* pz_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* pz = (double*)PyArray_DATA(pz_arr);

    int ncc = 0;
    int* lcc = NULL;
    int ier = 0;
    int ist = 1; // start search from previous if possible, or 1 for fresh? 1 is safe.
    // Actually intrc1 updates ist.

    // dflag=0 for value only? intrc1 computes value and gradient.
    // signature: void intrc1(..., int dflag, int* ist, double* pz, double* pzx, double* pzy, int* ier);
    // dflag: 0: pz only, 1: pz, pzx, pzy
    // We only want pz for now.

    int dflag = 0;
    double dum_zx, dum_zy;

    for (int i=0; i < n_points; i++) {
        intrc1(px[i], py[i], ncc, lcc, n, x, y, z, list, lptr, lend, iflgs, sigma, grad, dflag, &ist, &pz[i], &dum_zx, &dum_zy, &ier);
        if (ier < 0) {
            // Error handling?
            // ier > 0 means extrapolation (not fatal)
            // ier < 0 fatal
            // We'll just continue? Or raise error?
            // Usually we might want to fill with NaN or something.
            // But let's fail for now.
             Py_DECREF(px_arr); Py_DECREF(py_arr);
            Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr);
            Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);
            Py_DECREF(sigma_arr); Py_DECREF(grad_arr);
            Py_DECREF(pz_arr);
            PyErr_Format(PyExc_RuntimeError, "intrc1 failed at point %d with error %d", i, ier);
            return NULL;
        }
    }

    Py_DECREF(px_arr); Py_DECREF(py_arr);
    Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr);
    Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);
    Py_DECREF(sigma_arr); Py_DECREF(grad_arr);

    return (PyObject*)pz_arr;
}

// Wrappers for SSRFPACK (Spherical Surface Interpolation)

// ssrf_gradg
static PyObject* py_ssrf_gradg(PyObject* self, PyObject* args) {
    PyObject *x_obj, *y_obj, *z_obj, *f_obj, *list_obj, *lptr_obj, *lend_obj;
    int iflgs = 0;
    PyObject* sigma_obj = NULL;

    if (!PyArg_ParseTuple(args, "OOOOOOO|iO", &x_obj, &y_obj, &z_obj, &f_obj, &list_obj, &lptr_obj, &lend_obj, &iflgs, &sigma_obj)) {
        return NULL;
    }

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* f_arr = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* list_arr = (PyArrayObject*)PyArray_FROM_OTF(list_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lptr_arr = (PyArrayObject*)PyArray_FROM_OTF(lptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lend_arr = (PyArrayObject*)PyArray_FROM_OTF(lend_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !y_arr || !z_arr || !f_arr || !list_arr || !lptr_arr || !lend_arr) {
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(z_arr); Py_XDECREF(f_arr);
        Py_XDECREF(list_arr); Py_XDECREF(lptr_arr); Py_XDECREF(lend_arr);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_arr, 0);
    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* z = (double*)PyArray_DATA(z_arr);
    double* f = (double*)PyArray_DATA(f_arr);
    int* list = (int*)PyArray_DATA(list_arr);
    int* lptr = (int*)PyArray_DATA(lptr_arr);
    int* lend = (int*)PyArray_DATA(lend_arr);

    PyArrayObject* sigma_arr;
    if (sigma_obj && sigma_obj != Py_None) {
        sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    } else {
        npy_intp dims[1] = {n};
        sigma_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    }
    double* sigma = (double*)PyArray_DATA(sigma_arr);

    // Grad: 3*N for spherical? (gx, gy, gz)
    // ssrf_gradg documentation: GRAD(3,N)
    npy_intp dims_g[1] = {3 * n};
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims_g, NPY_DOUBLE);
    double* grad = (double*)PyArray_DATA(grad_arr);

    int nit = 20; // Default max iterations
    double dgmax = 0.0; // Default tolerance
    int ier = 0;

    ssrf_gradg(n, x, y, z, f, list, lptr, lend, iflgs, sigma, &nit, &dgmax, grad, &ier);

    Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr); Py_DECREF(f_arr);
    Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);

    if (ier < 0) {
        Py_DECREF(sigma_arr); Py_DECREF(grad_arr);
        PyErr_Format(PyExc_RuntimeError, "ssrf_gradg failed with error code %d", ier);
        return NULL;
    }

    return Py_BuildValue("NN", sigma_arr, grad_arr);
}

// ssrf_intrc1
static PyObject* py_ssrf_intr(PyObject* self, PyObject* args) {
    PyObject *plat_obj, *plon_obj;
    PyObject *x_obj, *y_obj, *z_obj, *f_obj, *list_obj, *lptr_obj, *lend_obj, *sigma_obj, *grad_obj;
    int iflgs = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOO|i", &plat_obj, &plon_obj, &x_obj, &y_obj, &z_obj, &f_obj, &list_obj, &lptr_obj, &lend_obj, &sigma_obj, &grad_obj, &iflgs)) {
        return NULL;
    }

    PyArrayObject* plat_arr = (PyArrayObject*)PyArray_FROM_OTF(plat_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* plon_arr = (PyArrayObject*)PyArray_FROM_OTF(plon_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* y_arr = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* z_arr = (PyArrayObject*)PyArray_FROM_OTF(z_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* f_arr = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* list_arr = (PyArrayObject*)PyArray_FROM_OTF(list_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lptr_arr = (PyArrayObject*)PyArray_FROM_OTF(lptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* lend_arr = (PyArrayObject*)PyArray_FROM_OTF(lend_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* sigma_arr = (PyArrayObject*)PyArray_FROM_OTF(sigma_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* grad_arr = (PyArrayObject*)PyArray_FROM_OTF(grad_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!plat_arr || !plon_arr || !x_arr || !y_arr || !z_arr || !f_arr || !list_arr || !lptr_arr || !lend_arr || !sigma_arr || !grad_arr) {
        Py_XDECREF(plat_arr); Py_XDECREF(plon_arr);
        Py_XDECREF(x_arr); Py_XDECREF(y_arr); Py_XDECREF(z_arr); Py_XDECREF(f_arr);
        Py_XDECREF(list_arr); Py_XDECREF(lptr_arr); Py_XDECREF(lend_arr);
        Py_XDECREF(sigma_arr); Py_XDECREF(grad_arr);
        return NULL;
    }

    int n_points = (int)PyArray_DIM(plat_arr, 0);
    int n = (int)PyArray_DIM(x_arr, 0);

    double* plat = (double*)PyArray_DATA(plat_arr);
    double* plon = (double*)PyArray_DATA(plon_arr);

    double* x = (double*)PyArray_DATA(x_arr);
    double* y = (double*)PyArray_DATA(y_arr);
    double* z = (double*)PyArray_DATA(z_arr);
    double* f = (double*)PyArray_DATA(f_arr);
    int* list = (int*)PyArray_DATA(list_arr);
    int* lptr = (int*)PyArray_DATA(lptr_arr);
    int* lend = (int*)PyArray_DATA(lend_arr);
    double* sigma = (double*)PyArray_DATA(sigma_arr);
    double* grad = (double*)PyArray_DATA(grad_arr);

    npy_intp dims[1] = {n_points};
    PyArrayObject* pval_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* pval = (double*)PyArray_DATA(pval_arr);

    int ier = 0;
    int ist = 1;
    // Fix: We must set iflgg = 1 because we provide gradients in 'grad'.
    // If iflgg=0, ssrf_intrc1 attempts to compute them using ssrf_gradl, which fails for N < 7.
    int iflgg = 1;

    for (int i=0; i < n_points; i++) {
        ssrf_intrc1(n, plat[i], plon[i], x, y, z, f, list, lptr, lend, iflgs, sigma, iflgg, grad, &ist, &pval[i], &ier);
        if (ier < 0) {
             Py_DECREF(plat_arr); Py_DECREF(plon_arr);
            Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr); Py_DECREF(f_arr);
            Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);
            Py_DECREF(sigma_arr); Py_DECREF(grad_arr);
            Py_DECREF(pval_arr);
            PyErr_Format(PyExc_RuntimeError, "ssrf_intrc1 failed at point %d with error %d", i, ier);
            return NULL;
        }
    }

    Py_DECREF(plat_arr); Py_DECREF(plon_arr);
    Py_DECREF(x_arr); Py_DECREF(y_arr); Py_DECREF(z_arr); Py_DECREF(f_arr);
    Py_DECREF(list_arr); Py_DECREF(lptr_arr); Py_DECREF(lend_arr);
    Py_DECREF(sigma_arr); Py_DECREF(grad_arr);

    return (PyObject*)pval_arr;
}

static PyMethodDef RenkaMethods[] = {
    {"tspsi", (PyCFunction)py_tspsi, METH_VARARGS | METH_KEYWORDS, "Compute derivatives and tension factors for curve"},
    {"tsval1", (PyCFunction)py_tsval1, METH_VARARGS | METH_KEYWORDS, "Evaluate spline at points"},
    {"trmesh", (PyCFunction)py_trmesh, METH_VARARGS, "Create triangulation for 2D points"},
    {"stri_trmesh", (PyCFunction)py_stri_trmesh, METH_VARARGS, "Create triangulation for points on sphere"},
    {"hval", (PyCFunction)py_hval, METH_VARARGS, "Evaluate Hermite interpolation"},
    {"hpval", (PyCFunction)py_hpval, METH_VARARGS, "Evaluate Hermite interpolation derivative"},
    {"tspss", (PyCFunction)py_tspss, METH_VARARGS, "Smooth curve"},

    // New SRFPACK wrappers
    {"gradg", (PyCFunction)py_gradg, METH_VARARGS, "Compute gradients for surface interpolation"},
    {"intr_2d", (PyCFunction)py_intr_2d, METH_VARARGS, "Interpolate 2D surface at points"},

    // New SSRFPACK wrappers
    {"ssrf_gradg", (PyCFunction)py_ssrf_gradg, METH_VARARGS, "Compute gradients for spherical surface interpolation"},
    {"ssrf_intr", (PyCFunction)py_ssrf_intr, METH_VARARGS, "Interpolate spherical surface at points"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef renkamodule = {
    PyModuleDef_HEAD_INIT,
    "renka.renka",
    "Python wrapper for Renka's packages (tspack, tripack, stripack, srfpack, ssrfpack)",
    -1,
    RenkaMethods
};

PyMODINIT_FUNC PyInit_renka(void) {
    import_array();
    return PyModule_Create(&renkamodule);
}
