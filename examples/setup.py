from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy
import py_mc

#print(py_mc.__path__)

myext = Extension("ising2D_cy", sources=['ising2D_cy.pyx'],
                  define_macros=[('SFMT_MEXP','19937'),('HAVE_SSE',None)]
                  )
setup(
 ext_modules=cythonize([myext], #['ising2D_trialcy.pyx'],
                       annotate=True),
    include_dirs=[numpy.get_include(),'/home/kasamatsu/git/py_mc/py_mc']
    )
    # Cython code file with primes() function

