import numpy as np
import random as rand
import sys

from py_mc.mc import model, CanonicalMonteCarlo, binning, observer_base

class latticegas(model):
    '''This class defines the 2D lattice gas model'''

    model_name = "latticegas"
    
    def __init__(self, Eads,J=0,mu=0):
        self.J = J
        self.Eads = Eads
        self.mu = mu

    def energy(self,latgas_config):
        ''' Calculate total energy of the 2D lattice gas model'''
        e = 0.0
        config = latgas_config.config # config should be 2D numpy array
        for i in range(latgas_config.lenX):
            for j in range(latgas_config.lenY):
                e += self.J*config[i,j]*(config[i-1,j] + config[i,j-1])
        e +=config.sum()*(self.Eads - self.mu)
        return e

    def density(self,latgas_config):
        '''Calculate number of particles'''
        return latgas_config.config.sum()

    def trialstep(self, latgas_config, energy):
        config = latgas_config.config
        nsites = latgas_config.lenX*latgas_config.lenY
        nocc = config.sum()
        # move or add/delete?
        if rand.random() < 0.5 or nocc==nsites or nocc==0:
            # energy is just a placeholder and isn't used
            # here
            # choose x,y randomly
            x = rand.randrange(latgas_config.lenX)
            y = rand.randrange(latgas_config.lenY)
            # Position of flipped occupancy to be used by newconfig():
            dconfig = [x,y]


            # Calculate energy change if occupation flips at x,y
            x1 = x + 1
            if x == latgas_config.lenX - 1:
                x1 = 0
            y1 = y + 1
            if y == latgas_config.lenY - 1:
                y1 = 0
            dE = -(2*config[x,y]-1)*(self.J*(config[x-1,y] + config[x1, y] +
                                             config[x, y-1] + config[x, y1])
                                     + self.Eads - self.mu)
        else:
            # Move a particle
            # Randomly choose an occupied/empty site 
            xocc, yocc = rand.choice(np.transpose(np.where(config==1)))
            xemp, yemp = rand.choice(np.transpose(np.where(config==0)))

            # Calculate energy change of removing particle at xocc yocc
            x1 = xocc + 1
            if xocc == latgas_config.lenX - 1:
                x1 = 0
            y1 = yocc + 1
            if yocc == latgas_config.lenY - 1:
                y1 = 0
            dE = -(self.J*(config[xocc-1,yocc] + config[x1, yocc] +
                           config[xocc, yocc-1] + config[xocc, y1])
                   + self.Eads - self.mu)

            # Calculate energy change of adding particle at xemp yemp
            x1 = xemp + 1
            if xemp == latgas_config.lenX - 1:
                x1 = 0
                y1 = yemp + 1
            if yemp == latgas_config.lenY - 1:
                y1 = 0
            dE = (self.J*(config[xemp-1,yemp] + config[x1, yemp] +
                          config[xemp, yemp-1] + config[xemp, y1])
                  + self.Eads - self.mu)
            dconfig = [xocc, yocc, xemp, yemp]
            
        return dconfig, dE

    def newconfig(self, latgas_config, dconfig):
        '''Construct the new configuration after the trial step is accepted'''
        latgas_config.config[dconfig[0],dconfig[1]] = 1- latgas_config.config[
            dconfig[0],dconfig[1]]
        if len(dconfig) == 4:
            latgas_config.config[dconfig[2],dconfig[3]] = 1- latgas_config.config[
                dconfig[2],dconfig[3]]
            
        return latgas_config
        

        
class latgas_config:
    '''This class defines the lattice gas model configuration'''

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
                    self.config[i,j] = 0

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

class observer(observer_base):
    def __init__(self):
        self.energy_obs = []
        self.density_obs = []
    def logfunc(self, calc_state):
        energy = calc_state.energy
        num = calc_state.model.density(calc_state.config)
        #self.energy_obs.append(energy)
        self.density_obs.append(num)
        return energy, num

if __name__ == "__main__":
    kb = 1.38064852e-23
    m = 1.0/6.023*1e-26
    T = 300
    h = 6.62607004e-34
    size = 5
    nspin = size*size
    eqsteps = nspin*1000
    mcsteps = nspin*1000
    sample_frequency = 1 #nspin
    config = latgas_config(size,size)
    config.prepare_random()
    binning_file = open("binning.dat", "a")
    mu0 = -kb*T*np.log((2.0*np.pi*m*kb*T/h**2)**(3/2)*kb*T)
    Eads = -0.4 * 1.602e-19
    for p in np.linspace(1e4, 0.0001, 20):
        mu = mu0 + kb*T*np.log(p)
        model = latticegas(Eads=Eads, J=0, mu=mu)
        kT = kb*T
        calc = CanonicalMonteCarlo(model, kT, config)
        calc.run(eqsteps)
        myobserver = observer()
        obs = calc.run(mcsteps,sample_frequency,myobserver)
        print(p,"\t", "\t".join([str(x/nspin) for x in obs]))
        # binning analysis
        error_estimate = binning(myobserver.density_obs,10)
        binning_file.write("\n".join([str(x) for x in error_estimate])+"\n\n")
        sys.stdout.flush()
        model = calc.model
        config = calc.config
        #print(config)
        
    
        
