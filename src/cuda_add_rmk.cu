#include <Python.h>
#include <math.h>
#include <cstdint>
#include <stdio.h>

/* Holds to elements part of a struct array */
typedef struct
{
    float x, y;
} Data;
/* Global name variables to seperate CPU from GPU */
static const char CPU[] = "CPU";
static const char GPU[] = "GPU";


__global__
void add(int r, int n, Data *g)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = 0; i < r; i += 1) {
        for (int j = index; j < n; j += stride)
            g[j].y = g[j].x + g[j].y;
    }
}

/* Destructor function for cpu data stucture */
static void del_Data(PyObject *obj) {
	const char *name = PyCapsule_GetName(obj);
	if (strcmp(name, CPU)) {
		printf("%s", name);
    	free(PyCapsule_GetPointer(obj, name));
	}
	else if (strcmp(name, GPU)) {
		printf("%s", name);
		cudaFree(PyCapsule_GetPointer(obj, name));
	}
	else {
		printf("PyCapsule failed, couldn't get capsule name");
	}
}

/* Get py object pointer object */
static Data *PyData_AsPoint(PyObject *obj, const char *name) {
    return (Data *) PyCapsule_GetPointer(obj, name);
}

/* Create new PyCapsule (keeping mem state) */
static PyObject *PyData_FromPoint(Data *d, int must_free, const char *name) {
    return PyCapsule_New(d, name, must_free ? del_Data : NULL);
}

/* Initiate Cuda Mem, struct array, struct x and y with N length*/
static PyObject *method_cuda_allocate(PyObject *self, PyObject *args) {
    Data *gpu_device;
    Data *host_device;
    int n;
    if (!PyArg_ParseTuple(args,"i",&n)) {
        return NULL;
    }

    host_device = (Data *) malloc(n * sizeof(Data));
    cudaMalloc((void**)&gpu_device, n * sizeof(Data));

    return Py_BuildValue("OO", PyData_FromPoint(host_device, 1, CPU), \
        PyData_FromPoint(gpu_device, 1, GPU));
}

/* Set value to each element pair */
static PyObject *method_cuda_set(PyObject *self, PyObject *args) {
    Data *d;
    Data *g;
    PyObject *py_host_device;
    PyObject *py_gpu_device;
    int n;
    if (!PyArg_ParseTuple(args, "iOO", &n, &py_host_device, &py_gpu_device)) {
        return NULL;
    }

    if (!(d = PyData_AsPoint(py_host_device, CPU)) | \
		!(g = PyData_AsPoint(py_gpu_device, GPU))) {

        return NULL;
    }

    for (int i = 0; i < n; i++) {
        d[i].x = 1.0f;
        d[i].y = 2.0f;
    }

  	cudaMemcpy(g, d, n * sizeof(Data), cudaMemcpyHostToDevice);

    Py_RETURN_NONE;
}

/* Add x and y, return error check */
static PyObject *method_cuda_add(PyObject *self, PyObject *args) {
    Data *d;
    Data *g;
    PyObject *py_host_device;
    PyObject *py_gpu_device;
    int n;
    int r;

    if (!PyArg_ParseTuple(args, "iiOO", &r, &n, &py_host_device, &py_gpu_device)) {
        return NULL;
    }

    if (!(d = PyData_AsPoint(py_host_device, CPU)) | \
		!(g = PyData_AsPoint(py_gpu_device, GPU))) {

        return NULL;
    }
    // Run kernel on 1M elements on the GPU
    int devID;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cudaGetDevice(&devID));
    add<<<32 * numSMs, 256>>>(r, n, g);
    // add<<<1, 1>>>(N, x, y);  // Debug

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
  	cudaMemcpy(d, g, n * sizeof(Data), cudaMemcpyDeviceToHost);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        maxError = fmax(maxError, fabs(d[i].y-3.0f));
    }

    return PyLong_FromLong(maxError);
}

static PyMethodDef CudaMethods[] = {
    {"cuda_allocate", method_cuda_allocate, METH_VARARGS, \
        "Python interface for cuda GPU memory allocation"},
    {"cuda_set", method_cuda_set, METH_VARARGS, \
        "Python interface for cuda GPU mem set values"},
    {"cuda_add", method_cuda_add, METH_VARARGS, \
        "Python interface for cuda GPU mem set values"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef cudamodule = {
    PyModuleDef_HEAD_INIT,
    "cuda_add_rmk",
    "Python interface for Cuda library functions",
    -1,
    CudaMethods
};


PyMODINIT_FUNC PyInit_cuda_add_rmk(void) {
    return PyModule_Create(&cudamodule);
}
