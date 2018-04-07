# cython: profile=True

from libc.stdint cimport uint32_t, uint64_t
from SFMT_cython.sfmt cimport sfmt_t, sfmt_init_gen_rand, sfmt_genrand_uint64, sfmt_genrand_real1
cimport cython

cdef sfmt_t sfmt

cpdef init_state(int seed):
    global sfmt
    sfmt_init_gen_rand(&sfmt, seed)
    return 0

cpdef uint64_t genrand_int():
    return sfmt_genrand_uint64(&sfmt)
    
@cython.cdivision(True)    
cpdef int randint(int low, int high):
    cdef int x
    cdef int i
    while True:
        x = genrand_int()
        if x < 2**64 - 2**64%(high-low):
            break
    return x%(high - low) + low

        
