    
cdef class observer_base:
    cpdef logfunc(self, calc_state)
    cpdef savefuncs(self, calc_state)
    cpdef observe(self, calc_state, outputfi, lprint=*)

