import numpy as np
import random as rand
import sys
import copy
import time
from timeit import default_timer as timer

from mc import model, CanonicalMonteCarlo, MultiProcessReplicaRun

class ising2D(model):
    '''This class defines the 2D ising model'''

    model_name = "ising2D"
    
    def __init__(self, J):
        self.J = J

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

    def trialstep(self, ising2D_config):
        # choose x,y randomly
        x = rand.randrange(ising2D_config.lenX)
        y = rand.randrange(ising2D_config.lenY)
        # Position of flipped spin to be used by newconfig():
        dconfig = [x,y]
        config = ising2D_config.config

        # Calculate energy change if spin flips at x,y
        x1 = x + 1
        if x == ising2D_config.lenX - 1:
            x1 = 0
        y1 = y + 1
        if y == ising2D_config.lenY - 1:
            y1 = 0
        dE = -2.0*self.J*config[x,y]*(config[x-1,y] + config[x1, y] + config[x, y-1] + config[x, y1])
        #print dconfig, dE
        return dconfig, dE

    def newconfig(self, ising2D_config, dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        ising2D_config.config[dconfig[0],dconfig[1]] *= -1
        return ising2D_config
        

        
class ising2D_config:
    '''This class defines the 2D Ising model configuration'''

    def __init__(self, lenX, lenY):
        self.lenX = lenX
        self.lenY = lenY
        self.config = np.empty([lenX, lenY])

    def prepare_random(self):
        for i in range(self.lenX):
            for j in range(self.lenY):
                if rand.random() >= 0.5:
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


if __name__ == "__main__":
    J = -1.0
    kT = abs(J) * 1.0
    size = 100
    eqsteps = 100000
    mcsteps = 1000000
    nreplicas = 8
    nprocs = 8
    sample_frequency = size*size
    config = ising2D_config(size,size)
    config.prepare_random()
    configs = [copy.deepcopy(config) for i in range(nreplicas)]
    model = ising2D(J)

    kT = abs(J)*3.0
    
    calc_list = [CanonicalMonteCarlo(model, kT, configs[i])
                 for i in range(nreplicas)
                 ]
    nsteps = 100000
    
    #time.sleep(10)
    start = timer()
    calc_list = MultiProcessReplicaRun(calc_list, nsteps, nprocs)
    end = timer()
    print nprocs, " procs, ",  nsteps, " steps each:", end - start
    
    start = timer()
    calc_list[i].run(nsteps*nprocs)
    end = timer()
    print "1 proc, ", nsteps*nprocs, " steps:", end - start
    sys.exit()

    for kT in np.arange(5, 0.5, -0.05):
        energy_expect = 0
        magnet_expect = 0
        
        kT = abs(J)*kT        

        #print config        
        calc = CanonicalMonteCarlo(model, kT, config)
        calc.run(eqsteps)

        mcloop = mcsteps/sample_frequency
        for i in range(mcloop):
            calc.run(sample_frequency)
            #print model.energy(config), model.magnetization(config)
            current_config = calc.config
            energy_expect += model.energy(current_config)
            magnet_expect += abs(model.magnetization(current_config))
        print kT, energy_expect/mcloop, magnet_expect/mcloop
        sys.stdout.flush()
    #calc.run(100000)
    #print config
    
        
