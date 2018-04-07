from libc.stdint cimport uint32_t, uint64_t
from SFMT_cython.sfmt cimport sfmt_t, sfmt_init_gen_rand, sfmt_genrand_uint64, sfmt_genrand_real1

cdef sfmt_t sfmt

cpdef init_state(int seed)

cpdef uint64_t genrand_int()
        
cpdef int randint(int low, int high)

        
