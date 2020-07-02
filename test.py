#!python3
import timeit


def experiment1():
    """
    Use vanilla pyton and perform "vector" (list) addition. No optimization or \
    performance enhancement what so ever. Kind of c-ish style
    """
    from src.py_add import py_add

    # Declarations for python timit method
    init = "from src.py_add import py_add"
    py_add = "py_add(1000000)"

    # Time function call
    print("Python:     ", '{:1.6f}'.format(timeit.timeit(
        setup = init, stmt = py_add, number = 10)))


def experiment2():
    """
    Use third party optimized library Numpy. Numpy is a c/c++ and fortran \
    compiled library with python bindings. If you are doint any numerical \
    computing, Numpy is your every day goto library
    """
    from src.np_add import np_add

    # Declariontion for python timit method
    init = "from src.np_add import np_add"
    np_add = "np_add(1000000)"

    # Time function call
    print("Numpy:      ", '{:1.6f}'.format(timeit.timeit(
        setup = init, stmt = np_add, number = 10)))


def experiment3():
    """
    In this experiment to optmize even further C/C++ (more to c) is used with \
    python bindings. Still basic functionality is used.
    """
    # src/c_add.cpp

    # Declariontion for python timit method
    init = "from c_add import c_add"
    c_add = "c_add(1000000)"

    print("C/C++:      ", '{:1.6f}'.format(timeit.timeit(
        setup = init, stmt = c_add, number = 10)))

def experiment4():
    """
    Now we take a first step to perform calculations on the GPU. Nvidia GPU is \
    needed with a working Cuda development. This example is your standard Cuda \
    tutorial with added Python bindings.
    """
    # src/cuda_add.cu

    # Declariontion for python timit method
    init = "from cuda_add import cuda_add"
    cuda_add = "cuda_add(1000000)"

    print("Cuda C/C++: ", '{:1.6f}'.format(timeit.timeit(
        setup = init, stmt = cuda_add, number = 10)))


def experiment5():
    """
    In this step we utalize PyCapsule, a way to share memory/pointers. This is \
    a way to share state and to get rid of the re-initialization. One \
    difference in this experiment is that we allocate memory before running \
    the function
    """
    # src/c_add_rmk.cpp

    # Declariontion for python timit method
    init = "from c_add_rmk import py_data, py_set, py_add\nmem = py_data(1000000)"

    # repeat, arr_len, allocated_mem
    c_add = "py_set(1000000, mem)\npy_add(1000, 1000000, mem)"

    print("C++ RMK:  ", '{:1.6f}'.format(timeit.timeit(
        setup = init, stmt = c_add, number = 10)))


def experiment6():
    """
    In this last example we do exactly like what we are doing in experiment5 \
    however this time we add Cuda GPU into the mix
    """
    # src/cuda_add_rmk.cu

    init = "from cuda_add_rmk import cuda_allocate, cuda_set, cuda_add\ncpu_mem, gpu_mem = cuda_allocate(1000000)"

  # repeat, arr_len, allocated_mem
    cuda_add = \
        "cuda_set(1000000, cpu_mem, gpu_mem)\ncuda_add(1000, 1000000, cpu_mem, gpu_mem)"

    print("Cuda RMK:  ", '{:1.6f}'.format(timeit.timeit(
        setup = init, stmt = cuda_add, number = 10)))

