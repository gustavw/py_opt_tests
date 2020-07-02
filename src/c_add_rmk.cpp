#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <cstdint>

/* Holds to elements part of a struct array */
typedef struct Data
{
    float x, y;
} Data;

/* Destructor function for points */
static void del_Data(PyObject *obj) {
    free(PyCapsule_GetPointer(obj,"Data"));
}

/* Get py object pointer object */
static Data *PyData_AsPoint(PyObject *obj) {
    return (Data *) PyCapsule_GetPointer(obj, "Data");
}

/* Create new PyCapsule (keeping mem state) */
static PyObject *PyData_FromPoint(Data *d, int must_free) {
    return PyCapsule_New(d, "Data", must_free ? del_Data : NULL);
}

/* Initiate Mem, struct array, x and y with N length*/
static PyObject *py_method_data(PyObject *self, PyObject *args) {
    Data *d;
    int n;
    if (!PyArg_ParseTuple(args,"i",&n)) {
        return NULL;
    }
    d = (Data *) malloc(n * sizeof(Data));

    return PyData_FromPoint(d, 1);
}

/* Set value to each element pair */
static PyObject *py_method_set(PyObject *self, PyObject *args) {
    Data *d;
    PyObject *py_d;
    int n;
    if (!PyArg_ParseTuple(args, "iO", &n, &py_d)) {
        return NULL;
    }

    if (!(d = PyData_AsPoint(py_d))) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        d[i].x = 1.0f;
        d[i].y = 2.0f;
    }

    Py_RETURN_NONE;
}

/* Add x and y, return error check */
static PyObject *py_method_add(PyObject *self, PyObject *args) {
    Data *d;
    PyObject *py_d;
    int r;
    int n;
    if (!PyArg_ParseTuple(args, "iiO", &r, &n, &py_d)) {
        return NULL;
    }

    if (!(d = PyData_AsPoint(py_d))) {
        return NULL;
    }

    //float result[n];

    for (int i = 0; i < r; i += 1) {
        for (int j = 0; j < n; j += 1) {
            d[j].y =  d[j].x + d[j].y;
        }
    }

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < n; i++)
        maxError = fmax(maxError, fabs(d[i].y-3.0f));

    /* For Demo of Garbage collection */
    //Py_DECREF(py_d);
    //del_Point(py_d);
    return PyLong_FromLong(maxError);
}


static PyMethodDef CMethods[] = {
    {"py_data", py_method_data, METH_VARARGS, "Python C++ API init function for memory"},
    {"py_set", py_method_set, METH_VARARGS, "Python C++ API update values"},
    {"py_add", py_method_add, METH_VARARGS, "Python C++ API add x and y return error check"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef cmodule = {
    PyModuleDef_HEAD_INIT,
    "c_add_rmk",
    "Python C module for C add function library testing",
    -1,
    CMethods
};


PyMODINIT_FUNC PyInit_c_add_rmk(void) {
    return PyModule_Create(&cmodule);
}
