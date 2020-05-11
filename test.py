#!python3
import timeit

# init modules, executed only once
init = """
from cuda_add import cuda_add
from c_add import c_add
from src.py_add import py_add
from src.np_add import np_add

# Allocate in memory before execution
from c_add_rmk import py_data, py_set, py_add as py_add_rmk
mem = py_data(1<<20)
"""

# code snippet whose execution time is to be measured
py_add = "py_add(1<<20)"
np_add = "np_add(1<<20)"
c_add = "c_add(1<<20)"
cuda_add = "cuda_add(1<<20)"
c_add_rmk = "py_set(1<<20, mem)\npy_add_rmk(1<<20, mem)"

# timeit statement
print("Python:   ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = py_add, number = 10)))
print("Numpy:    ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = np_add, number = 10)))
print("C++:      ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = c_add, number = 10)))
print("Cuda C++: ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = cuda_add, number = 10)))

# Allocate in memory during init than calc.
print("C++ RMK:  ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = c_add_rmk, number = 10)))
