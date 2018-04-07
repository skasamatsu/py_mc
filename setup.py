from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_sfmt = Extension("SFMT_cython.sfmt_random",
                sources=["SFMT_cython/sfmt_random.pyx", "SFMT_cython/SFMT.c"],
                define_macros=[('SFMT_MEXP','19937'),('HAVE_SSE',None)]
                )


setup(
 ext_modules=cythonize(['py_mc/mc.pyx',ext_sfmt], #['ising2D_trialcy.pyx'],
                       annotate=True),
    include_dirs=[numpy.get_include()]
    )
    # Cython code file with primes() function

