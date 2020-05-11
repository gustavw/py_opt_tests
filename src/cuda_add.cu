#include <Python.h>
#include <math.h>
#include <cstdint>


__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

static PyObject *method_cuda_add(PyObject *self, PyObject *args) {
    int * NArg = NULL;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "i", &NArg)) {
        return NULL;
    }

    int N = (uintptr_t)NArg;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    // Time consuming!!
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    // 

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int devID;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, cudaGetDevice(&devID));
    add<<<32 * numSMs, 256>>>(N, x, y);
    // add<<<1, 1>>>(N, x, y);  // Debug

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    //std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return PyLong_FromLong(maxError);
}


static PyMethodDef CudaMethods[] = {
    {"cuda_add", method_cuda_add, METH_VARARGS, "Python interface for cuda GPU handling"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef cudamodule = {
    PyModuleDef_HEAD_INIT,
    "cuda_add",
    "Python interface for Cuda library functions",
    -1,
    CudaMethods
};


PyMODINIT_FUNC PyInit_cuda_add(void) {
    return PyModule_Create(&cudamodule);
}