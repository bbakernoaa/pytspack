#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "tspack.h"

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

    int lwk = 6 * n;
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

static PyMethodDef PyTsPackMethods[] = {
    {"tspsi", (PyCFunction)py_tspsi, METH_VARARGS | METH_KEYWORDS, "Compute derivatives and tension factors for curve"},
    {"tsval1", (PyCFunction)py_tsval1, METH_VARARGS | METH_KEYWORDS, "Evaluate spline at points"},
    {"hval", (PyCFunction)py_hval, METH_VARARGS, "Evaluate Hermite interpolation"},
    {"hpval", (PyCFunction)py_hpval, METH_VARARGS, "Evaluate Hermite interpolation derivative"},
    {"tspss", (PyCFunction)py_tspss, METH_VARARGS, "Smooth curve"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pytspackmodule = {
    PyModuleDef_HEAD_INIT,
    "pytspack._libpytspack",
    "Python wrapper for Renka's TSPACK",
    -1,
    PyTsPackMethods
};

PyMODINIT_FUNC PyInit__libpytspack(void) {
    import_array();
    return PyModule_Create(&pytspackmodule);
}
