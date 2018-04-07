# cython: profile=True

import numpy as np
import random as randpy
import sys
import cython
#from libc.stdlib cimport rand, RAND_MAX
from py_mc.mc import model, CanonicalMonteCarlo, binning, observer_base
cimport SFMT_cython.sfmt_random as sfmt_random
import SFMT_cython.sfmt_random as sfmt_random

from py_mc.mc cimport observer_base


cdef class dconfig_dE:
    cdef int x, y
    cdef double dE
    def __cinit__(self, int x, int y, double dE):
        self.x = x
        self.y = y
        self.dE = dE

cdef class ising2D:
    '''This class defines the 2D ising model'''
    cdef double J
    model_name = "ising2D"
    
    def __init__(self, J):
        self.J = J
        sfmt_random.init_state(12345)

    def energy(self,ising2D_config):
        ''' Calculate total energy of the 2D Ising model'''
        e = 0.0
        config = ising2D_config.config # config should be 2D numpy array
        for i in range(ising2D_config.lenX):
            for j in range(ising2D_config.lenY):
                e += self.J*config[i,j]*(config[i-1,j] + config[i,j-1])
        return e

    def magnetization(self,ising2D_config):
        '''Calculate magnetization'''
        return ising2D_config.config.sum()
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple trialstep(self, ising2D_config, double energy):
        cdef Py_ssize_t x, y, x1, y1
        cdef Py_ssize_t lenX = ising2D_config.lenX
        cdef Py_ssize_t lenY = ising2D_config.lenY
        x = sfmt_random.randint(0, lenX)
        y = sfmt_random.randint(0, lenY)
        #x = 1 + int(rand()/RAND_MAX*float(lenX))
        #y = 1 + int(rand()/RAND_MAX*float(lenY))
        # energy is just a placeholder and isn't used
        # here
        # choose x,y randomly
        
        #x = rand.randrange(ising2D_config.lenX)
        #y = rand.randrange(ising2D_config.lenY)
        # Position of flipped spin to be used by newconfig():
        dconfig = [x,y]
        #print(dconfig[0],dconfig[1],x,y)
        cdef int [:,:] config = ising2D_config.config

        # Calculate energy change if spin flips at x,y
        cdef int left,up,right,down
        if x == lenX - 1:
            right = 0
        else:
            right = x + 1
        if x == 0:
            left = lenX - 1
        else:
            left = x - 1
        if y == lenY - 1:
            down = 0
        else:
            down = y + 1
        if y == 0:
            up = lenY - 1
        else:
            up = y - 1

        cdef double dE = -2.0*self.J*config[x,y]*(config[left,y] + config[right, y] + config[x, up] + config[x, down])
        #print dconfig, dE
        #print(x,y,dE)
        return dconfig, dE

    cpdef newconfig(self, ising2D_config, list dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        ising2D_config.config[dconfig[0],dconfig[1]] *= -1
        return ising2D_config
        

        
class ising2D_config:
    '''This class defines the 2D Ising model configuration'''

    def __init__(self, lenX, lenY):
        self.lenX = lenX
        self.lenY = lenY
        self.config = np.empty([lenX, lenY], dtype=np.intc)

    def prepare_random(self):
        for i in range(self.lenX):
            for j in range(self.lenY):
                if randpy.random() >= 0.5:
                    self.config[i,j] = 1
                else:
                    self.config[i,j] = -1

    def __str__(self):
        s = ""
        for i in range(self.lenX):
            for j in range(self.lenY):
                if self.config[i,j] < 0:
                    s += "-"
                else:
                    s += "+"
            s += "\n"
        return s

cdef class observer(observer_base):
    cdef public list energy_obs
    cdef public list magnet_obs
    def __init__(self):
        self.energy_obs = []
        self.magnet_obs = []
    cpdef logfunc(self, calc_state):
        energy = calc_state.energy
        magnetization = calc_state.model.magnetization(calc_state.config)
        #self.energy_obs.append(energy)
        absmag = abs(magnetization)
        self.magnet_obs.append(absmag)
        return energy, absmag

#@profile    
def main():
    J = -1.0
    kT = abs(J) * 5.0
    size = 5
    nspin = size*size
    eqsteps = 2**12*6 #nspin*1000
    mcsteps = 2**12*50 #nspin*1000
    sample_frequency = 1 #nspin
    print_frequency = 2**20 #1000
    config = ising2D_config(size,size)
    config.prepare_random()
    model = ising2D(J)
    binning_file = open("binning.dat", "a")
    for kT in [5.0]: #np.linspace(5.0, 0.01, 10):      
        kT = abs(J)*kT
        calc = CanonicalMonteCarlo(model, kT, config)
        calc.run(eqsteps)
        myobserver = observer()
        obs = calc.run(mcsteps,sample_frequency,print_frequency,myobserver)
        # binning analysis
        error_estimate = binning(np.asarray(myobserver.magnet_obs)/nspin,12)
        binning_file.write("\n".join([str(x) for x in error_estimate])+"\n\n\n")
        print(kT,"\t", "\t".join([str(x/nspin) for x in obs]), np.max(error_estimate))
        sys.stdout.flush()
        binning_file.flush()
        model = calc.model
        config = calc.config
        #print(config)
    
    
if __name__ == "__main__":
    main()
    '''
    J = -1.0
    kT = abs(J) * 5.0
    size = 5
    nspin = size*size
    eqsteps = 2**14*6 #nspin*1000
    mcsteps = 2**14*30 #nspin*1000
    sample_frequency = 1 #nspin
    print_frequency = 1000
    config = ising2D_config(size,size)
    config.prepare_random()
    model = ising2D(J)
    binning_file = open("binning.dat", "a")
    for kT in np.linspace(5.0, 0.01, 10):      
        kT = abs(J)*kT
        calc = CanonicalMonteCarlo(model, kT, config)
        calc.run(eqsteps)
        myobserver = observer()
        obs = calc.run(mcsteps,sample_frequency,print_frequency,myobserver)
        # binning analysis
        error_estimate = binning(np.asarray(myobserver.magnet_obs)/nspin,13)
        binning_file.write("\n".join([str(x) for x in error_estimate])+"\n\n\n")
        print(kT,"\t", "\t".join([str(x/nspin) for x in obs]), np.max(error_estimate))
        sys.stdout.flush()
        binning_file.flush()
        model = calc.model
        config = calc.config
        #print(config)
'''        
    
        
