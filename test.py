#!python3
import timeit 
  
# init modules, executed only once 
init = """
from cuda_add import cuda_add
from c_add import c_add
from src.py_add import py_add
from src.np_add import np_add
"""
  
# code snippet whose execution time is to be measured 
py_add = "py_add(1<<22)"
np_add = "np_add(1<<22)"
c_add = "c_add(1<<22)"
cuda_add = "cuda_add(1<<22)"

# timeit statement 
print("Python:   ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = py_add, number = 10)))
print("Numpy:    ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = np_add, number = 10)))
print("C++:      ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = c_add, number = 10)))
print("Cuda C++: ", '{:1.6f}'.format(timeit.timeit(setup = init, stmt = cuda_add, number = 10)))