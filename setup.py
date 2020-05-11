# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# Defualt source: https://github.com/rmcgibbo/npcuda-example
def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



CUDA = {'home': '/usr/lib/cuda', 'nvcc': '/usr/bin/nvcc', 'include': '/usr/lib/cuda/include', 'lib64': '/usr/lib/cuda/lib64'}
#locate_cuda()

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = Extension('cuda_add',
        sources = ['src/cuda_add.cu'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args= {
            'gcc': [],
            'nvcc': [
                '-arch=sm_30', '--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
                ]
            },
            include_dirs = [numpy_include, CUDA['include'], 'src']
        )


setup(
    name = 'cuda_add',
    author = 'Gustav Wiberg',
    version = '0.1',
    description="Python interface for Cuda C/C++ library function",
    ext_modules = [ext],

    # Inject our custom trigger
    cmdclass = {'build_ext': custom_build_ext},

    # Since the package has c code, the egg cannot be zipped
    zip_safe = False
)

setup(
    name="c_add",
    version="0.1",
    author="Gustav Wiberg",
    description="Python interface for C library function",
    ext_modules=[Extension("c_add", ["src/c_add.cpp"])]
)

setup(
    name="c_add_rmk",
    version="0.1",
    author="Gustav Wiberg",
    description="Python interface for C library function",
    ext_modules=[Extension("c_add_rmk", ["src/c_add_rmk.cpp"])]
)