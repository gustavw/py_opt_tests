#include <Python.h>
#include <math.h>
#include <cstdint>


void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

static PyObject *method_c_add(PyObject *self, PyObject *args)
{
    int * NArg = NULL;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "i", &NArg)) {
        return NULL;
    }

    int N = (uintptr_t)NArg;
    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    //std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    free(x);
    free(y);

    return PyLong_FromLong(maxError);
}


static PyMethodDef CMethods[] = {
    {"c_add", method_c_add, METH_VARARGS, "Python C++ API function for add vector testing on CPU"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef cmodule = {
    PyModuleDef_HEAD_INIT,
    "c_add",
    "Python C module for C add function library testing",
    -1,
    CMethods
};


PyMODINIT_FUNC PyInit_c_add(void) {
    return PyModule_Create(&cmodule);
}
