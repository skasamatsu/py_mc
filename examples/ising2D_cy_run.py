import numpy as np
import random as randpy
import sys
#import cython
#from libc.stdlib cimport rand, RAND_MAX
from py_mc.mc import model, CanonicalMonteCarlo, binning, observer_base




from ising2D_cy import ising2D_config, ising2D, observer
    
if __name__ == "__main__":
    J = -1.0
    kT = abs(J) * 5.0
    size = 5
    nspin = size*size
    eqsteps = 2**12*6 #nspin*1000
    mcsteps = 2**12*50 #nspin*1000
    sample_frequency = 1 #nspin
    print_frequency = 1000 #2**20 #1000
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
    
        
